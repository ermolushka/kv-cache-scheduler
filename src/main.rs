mod block_pool;
mod block_table;
mod eviction;
mod sequence;
mod prefix_cache;

use block_pool::BlockPool;
use block_table::BlockTable;
use sequence::{Scheduler, TokenId};

fn main() {
    let mut pool = BlockPool::new(4);
    let mut table = BlockTable::new(2); // block_size = 2

    // append 5 tokens — should allocate 3 blocks (tokens 0-1, 2-3, 4)
    for _ in 0..5 {
        table.append_token(&mut pool).unwrap();
    }
    println!("blocks allocated: {}", table.len()); // expect 3

    // logical_to_physical: token 3 -> block 1, offset 1
    let (block_id, offset) = table.logical_to_physical(3);
    println!("token 3 -> block {:?}, offset {}", block_id, offset);

    // fork: both tables now share the same blocks
    let forked = table.fork(&mut pool);
    println!("forked table blocks: {}", forked.len()); // expect 3
    let shared_block = table.last_block().unwrap();
    println!(
        "shared block ref count: {}",
        pool.get_ref_count(shared_block)
    ); // expect 2

    println!("\n--- Phase 3: Scheduler ---");
    let mut scheduler = Scheduler::new(8, 2);

    // enqueue two requests with different prompt lengths
    let tokens_a: Vec<TokenId> = (0..4).map(TokenId).collect();
    let tokens_b: Vec<TokenId> = (0..6).map(TokenId).collect();
    let seq_a = scheduler.add_request(tokens_a);
    let seq_b = scheduler.add_request(tokens_b);
    println!("waiting queue length: {}", scheduler.waiting.len()); // expect 2

    // prefill both
    scheduler.prefill(seq_a);
    scheduler.prefill(seq_b);
    println!("running count: {}", scheduler.running.len()); // expect 2
    println!("waiting count: {}", scheduler.waiting.len()); // expect 0

    // check blocks allocated per sequence
    let blocks_a = scheduler.sequences[&seq_a].block_table.len();
    let blocks_b = scheduler.sequences[&seq_b].block_table.len();
    println!(
        "seq_a blocks: {} (expect 2 for 4 tokens, block_size=2)",
        blocks_a
    );
    println!(
        "seq_b blocks: {} (expect 3 for 6 tokens, block_size=2)",
        blocks_b
    );

    // run 3 decode steps
    for i in 0..3 {
        scheduler.step();
        println!(
            "step {}: seq_a tokens={}",
            i + 1,
            scheduler.sequences[&seq_a].token_ids.len()
        );
    }

    println!("\n--- Phase 4: LRU Eviction ---");
    // small pool to force eviction: 4 blocks, block_size=2
    // seq_c uses 2 blocks (4 tokens), seq_d uses 2 blocks (4 tokens) = pool full
    // one more step should trigger eviction
    let mut sched = Scheduler::new(4, 2);
    let seq_c = sched.add_request((0..4).map(TokenId).collect());
    let seq_d = sched.add_request((0..4).map(TokenId).collect());
    sched.prefill(seq_c);
    sched.prefill(seq_d);
    println!("pool full — running count: {}", sched.running.len()); // expect 2

    // step will need a new block but pool is full — should evict seq_c (LRU)
    sched.step();
    println!(
        "after eviction step — waiting: {}, running: {}",
        sched.waiting.len(),
        sched.running.len()
    );

    println!("\n--- Phase 5: Prefix Cache ---");
    // 8 blocks, block_size=2
    // Both requests share a 4-token prefix [0,1,2,3], then diverge.
    // seq_e prefills normally; seq_f should reuse 2 blocks from the cache.
    let mut sched5 = Scheduler::new(8, 2);
    let prefix: Vec<TokenId> = (0..4).map(TokenId).collect();
    let suffix_e: Vec<TokenId> = (10..12).map(TokenId).collect();
    let suffix_f: Vec<TokenId> = (20..22).map(TokenId).collect();

    let tokens_e: Vec<TokenId> = prefix.iter().chain(&suffix_e).copied().collect();
    let tokens_f: Vec<TokenId> = prefix.iter().chain(&suffix_f).copied().collect();

    let seq_e = sched5.add_request(tokens_e);
    let seq_f = sched5.add_request(tokens_f);

    sched5.prefill(seq_e);
    println!(
        "seq_e: {} blocks allocated (expect 3)",
        sched5.sequences[&seq_e].block_table.len()
    );

    sched5.prefill(seq_f);
    println!(
        "seq_f: {} block table entries (expect 3: 2 shared + 1 new)",
        sched5.sequences[&seq_f].block_table.len()
    );
    println!(
        "physical blocks in pool: {} (expect 4: seq_e's 3 + 1 new for seq_f suffix)",
        sched5.pool_used_blocks_physical()
    );
    println!(
        "prefix hit rate: {:.0}% (expect 33%: 4 matched of 12 total prompt tokens)",
        sched5.metrics.hit_rate() * 100.0
    );

    println!("\n--- Phase 6: Copy-on-Write ---");
    // parent: 3 tokens, block_size=2 → b0=[0,1] full, b1=[2,_] partial
    // Two forks share all parent blocks. step_cow on each triggers CoW on b1.
    // Sealed block b0 stays shared; each sequence gets its own copy of b1.
    let mut sched6 = Scheduler::new(16, 2);
    let parent_tokens: Vec<TokenId> = (0..3).map(TokenId).collect();
    let parent = sched6.add_request(parent_tokens);
    sched6.prefill(parent);

    let b0 = sched6.sequences[&parent].block_table.blocks()[0];
    let b1 = sched6.sequences[&parent].block_table.blocks()[1];

    let fork_a = sched6.fork_sequence(parent);
    let fork_b = sched6.fork_sequence(parent);
    println!("after 2 forks — b0 ref_count: {} (expect 3)", sched6.pool.get_ref_count(b0));
    println!("after 2 forks — b1 ref_count: {} (expect 3)", sched6.pool.get_ref_count(b1));

    // step_cow: parent copies b1 first (rc 3→2), fork_a copies it next (rc 2→1),
    // fork_b ends up with the last reference and keeps b1 without copying.
    sched6.step_cow();

    let parent_last = sched6.sequences[&parent].block_table.last_block().unwrap();
    let fork_a_last = sched6.sequences[&fork_a].block_table.last_block().unwrap();
    let fork_b_last = sched6.sequences[&fork_b].block_table.last_block().unwrap();

    println!("b0 ref_count: {} (expect 3 — sealed, still shared)", sched6.pool.get_ref_count(b0));
    println!("b1 ref_count: {} (expect 1 — last owner kept it)", sched6.pool.get_ref_count(b1));
    println!("all last blocks distinct: {} (expect true)", {
        let mut ids = [parent_last, fork_a_last, fork_b_last];
        ids.sort_by_key(|b| b.0);
        ids[0] != ids[1] && ids[1] != ids[2]
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use block_pool::BlockID;
    use sequence::{Scheduler, TokenId};

    #[test]
    fn alloc_all_then_exhausted() {
        let mut pool = BlockPool::new(3);
        let _b0 = pool.alloc();
        let _b1 = pool.alloc();
        let _b2 = pool.alloc();
        assert!(pool.alloc().is_none());
    }

    #[test]
    fn decref_returns_block_to_pool() {
        let mut pool = BlockPool::new(2);
        let b0 = pool.alloc().unwrap();
        pool.decref(b0);
        assert!(pool.alloc().is_some());
    }

    #[test]
    fn ref_count_after_alloc_is_one() {
        let mut pool = BlockPool::new(2);
        let b0 = pool.alloc().unwrap();
        assert_eq!(pool.get_ref_count(b0), 1);
    }

    #[test]
    fn incref_decref_shared_block() {
        let mut pool = BlockPool::new(2);
        let b0 = pool.alloc().unwrap();
        pool.incref(b0);
        assert_eq!(pool.get_ref_count(b0), 2);
        pool.decref(b0);
        assert_eq!(pool.get_ref_count(b0), 1);
        assert!(!pool.is_free(b0));
        pool.decref(b0);
        assert!(pool.is_free(b0));
    }

    #[test]
    #[should_panic]
    fn double_free_panics() {
        let mut pool = BlockPool::new(2);
        let b0 = pool.alloc().unwrap();
        pool.decref(b0);
        pool.decref(b0);
    }

    #[test]
    fn empty_pool_returns_none() {
        let mut pool = BlockPool::new(0);
        assert!(pool.alloc().is_none());
    }

    #[test]
    fn reallocated_block_has_ref_count_one() {
        let mut pool = BlockPool::new(1);
        let b0 = pool.alloc().unwrap();
        pool.decref(b0);
        let b0 = pool.alloc().unwrap();
        assert_eq!(pool.get_ref_count(b0), 1);
    }

    #[test]
    fn is_free_reflects_alloc_state() {
        let mut pool = BlockPool::new(1);
        let b0 = pool.alloc().unwrap();
        assert!(!pool.is_free(b0));
        pool.decref(b0);
        assert!(pool.is_free(b0));
    }

    #[test]
    #[should_panic]
    fn decref_unallocated_block_panics() {
        let mut pool = BlockPool::new(2);
        pool.decref(BlockID(0));
    }

    #[test]
    fn append_tokens_allocates_blocks_at_boundaries() {
        let mut pool = BlockPool::new(4);
        let mut table = BlockTable::new(2);
        table.append_token(&mut pool).unwrap();
        assert_eq!(table.len(), 1);
        table.append_token(&mut pool).unwrap();
        assert_eq!(table.len(), 1);
        table.append_token(&mut pool).unwrap();
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn logical_to_physical_correct_offset() {
        let mut pool = BlockPool::new(4);
        let mut table = BlockTable::new(2);
        for _ in 0..4 {
            table.append_token(&mut pool).unwrap();
        }
        let (_, offset) = table.logical_to_physical(0);
        assert_eq!(offset, 0);
        let (_, offset) = table.logical_to_physical(1);
        assert_eq!(offset, 1);
        let (_, offset) = table.logical_to_physical(2);
        assert_eq!(offset, 0);
        let (_, offset) = table.logical_to_physical(3);
        assert_eq!(offset, 1);
    }

    #[test]
    fn is_last_block_full_correct() {
        let mut pool = BlockPool::new(4);
        let mut table = BlockTable::new(2);
        table.append_token(&mut pool).unwrap();
        assert!(!table.is_last_block_full());
        table.append_token(&mut pool).unwrap();
        assert!(table.is_last_block_full());
    }

    #[test]
    fn fork_increfs_all_blocks() {
        let mut pool = BlockPool::new(4);
        let mut table = BlockTable::new(2);
        for _ in 0..4 {
            table.append_token(&mut pool).unwrap();
        }
        let _forked = table.fork(&mut pool);
        let (b0, _) = table.logical_to_physical(0);
        let (b2, _) = table.logical_to_physical(2);
        assert_eq!(pool.get_ref_count(b0), 2);
        assert_eq!(pool.get_ref_count(b2), 2);
    }

    #[test]
    fn append_token_fails_when_pool_exhausted() {
        let mut pool = BlockPool::new(1);
        let mut table = BlockTable::new(2);
        table.append_token(&mut pool).unwrap();
        table.append_token(&mut pool).unwrap();
        let result = table.append_token(&mut pool);
        assert!(result.is_err());
    }

    // --- Phase 4: LRU Eviction tests ---

    #[test]
    fn eviction_triggered_when_pool_full() {
        // 4 blocks, block_size=4 — each sequence uses 1 block after prefill
        // pool has 2 free blocks; after 2 decode steps per sequence pool fills up
        let mut sched = Scheduler::new(4, 4);
        let seq_a = sched.add_request((0..4).map(TokenId).collect());
        let seq_b = sched.add_request((0..4).map(TokenId).collect());
        sched.prefill(seq_a);
        sched.prefill(seq_b);
        // 4 decode steps fills remaining 2 blocks; 5th step forces eviction
        for _ in 0..4 {
            sched.step();
        }
        sched.step();
        assert_eq!(sched.waiting.len(), 1);
    }

    #[test]
    fn evicted_sequence_block_table_cleared() {
        let mut sched = Scheduler::new(4, 4);
        let seq_a = sched.add_request((0..4).map(TokenId).collect());
        let seq_b = sched.add_request((0..4).map(TokenId).collect());
        sched.prefill(seq_a);
        sched.prefill(seq_b);
        for _ in 0..4 {
            sched.step();
        }
        sched.step();
        let evicted_id = *sched.waiting.back().unwrap();
        assert_eq!(sched.sequences[&evicted_id].block_table.len(), 0);
    }

    #[test]
    fn evicted_blocks_returned_to_pool() {
        let mut sched = Scheduler::new(4, 4);
        let seq_a = sched.add_request((0..4).map(TokenId).collect());
        let seq_b = sched.add_request((0..4).map(TokenId).collect());
        sched.prefill(seq_a);
        sched.prefill(seq_b);
        for _ in 0..4 {
            sched.step();
        }
        sched.step();
        assert!(sched.running.len() > 0);
    }

    // --- Phase 3: Scheduler tests ---

    #[test]
    fn add_request_enqueues_to_waiting() {
        let mut scheduler = Scheduler::new(8, 2);
        let tokens: Vec<TokenId> = (0..4).map(TokenId).collect();
        scheduler.add_request(tokens);
        assert_eq!(scheduler.waiting.len(), 1);
        assert_eq!(scheduler.running.len(), 0);
    }

    #[test]
    fn prefill_moves_sequence_to_running() {
        let mut scheduler = Scheduler::new(8, 2);
        let tokens: Vec<TokenId> = (0..4).map(TokenId).collect();
        let seq_id = scheduler.add_request(tokens);
        scheduler.prefill(seq_id);
        assert_eq!(scheduler.waiting.len(), 0);
        assert_eq!(scheduler.running.len(), 1);
    }

    #[test]
    fn prefill_allocates_correct_blocks() {
        let mut scheduler = Scheduler::new(8, 2);
        let tokens: Vec<TokenId> = (0..6).map(TokenId).collect();
        let seq_id = scheduler.add_request(tokens);
        scheduler.prefill(seq_id);
        // 6 tokens with block_size=2 -> 3 blocks
        assert_eq!(scheduler.sequences[&seq_id].block_table.len(), 3);
    }

    #[test]
    fn multiple_sequences_independent_blocks() {
        let mut scheduler = Scheduler::new(16, 2);
        let seq_a = scheduler.add_request((0..4).map(TokenId).collect());
        let seq_b = scheduler.add_request((0..4).map(TokenId).collect());
        scheduler.prefill(seq_a);
        scheduler.prefill(seq_b);
        // each has 2 blocks, no sharing — 4 total pool blocks used
        assert_eq!(scheduler.sequences[&seq_a].block_table.len(), 2);
        assert_eq!(scheduler.sequences[&seq_b].block_table.len(), 2);
    }

    #[test]
    fn step_appends_tokens_and_allocates_blocks() {
        let mut scheduler = Scheduler::new(16, 2);
        let seq_id = scheduler.add_request((0..4).map(TokenId).collect());
        scheduler.prefill(seq_id);
        let tokens_before = scheduler.sequences[&seq_id].token_ids.len();
        scheduler.step();
        assert_eq!(
            scheduler.sequences[&seq_id].token_ids.len(),
            tokens_before + 1
        );
    }

    // --- Phase 5: Prefix Cache tests ---

    fn make_tokens(ids: &[i32]) -> Vec<TokenId> {
        ids.iter().copied().map(TokenId).collect()
    }

    #[test]
    fn shared_prefix_reuses_blocks() {
        // seq_a prefills [0,1,2,3]; seq_b shares same prefix — should reuse 2 blocks.
        let mut sched = Scheduler::new(8, 2);
        let seq_a = sched.add_request(make_tokens(&[0, 1, 2, 3]));
        let seq_b = sched.add_request(make_tokens(&[0, 1, 2, 3, 4, 5]));
        sched.prefill(seq_a);
        sched.prefill(seq_b);
        // seq_b should have 3 blocks: 2 shared + 1 new
        assert_eq!(sched.sequences[&seq_b].block_table.len(), 3);
        // physical blocks: 2 (seq_a) + 1 new (seq_b suffix) = 3
        assert_eq!(sched.pool_used_blocks_physical(), 3);
    }

    #[test]
    fn shared_prefix_increfs_blocks() {
        // The 2 shared blocks must have ref_count == 2.
        let mut sched = Scheduler::new(8, 2);
        let seq_a = sched.add_request(make_tokens(&[0, 1, 2, 3]));
        let seq_b = sched.add_request(make_tokens(&[0, 1, 2, 3]));
        sched.prefill(seq_a);
        sched.prefill(seq_b);
        for &block_id in sched.sequences[&seq_a].block_table.blocks() {
            assert_eq!(sched.pool.get_ref_count(block_id), 2);
        }
    }

    #[test]
    fn no_prefix_match_allocates_independently() {
        // Completely different tokens — no sharing, both allocate their own blocks.
        let mut sched = Scheduler::new(8, 2);
        let seq_a = sched.add_request(make_tokens(&[0, 1, 2, 3]));
        let seq_b = sched.add_request(make_tokens(&[10, 11, 12, 13]));
        sched.prefill(seq_a);
        sched.prefill(seq_b);
        assert_eq!(sched.pool_used_blocks_physical(), 4);
    }

    #[test]
    fn prefix_hit_rate_nonzero_after_shared_prefill() {
        let mut sched = Scheduler::new(8, 2);
        let seq_a = sched.add_request(make_tokens(&[0, 1, 2, 3]));
        let seq_b = sched.add_request(make_tokens(&[0, 1, 2, 3]));
        sched.prefill(seq_a);
        sched.prefill(seq_b);
        assert!(sched.metrics.hit_rate() > 0.0);
    }

    #[test]
    fn hit_rate_zero_with_no_prefix_overlap() {
        let mut sched = Scheduler::new(8, 2);
        let seq_a = sched.add_request(make_tokens(&[0, 1, 2, 3]));
        let seq_b = sched.add_request(make_tokens(&[10, 11, 12, 13]));
        sched.prefill(seq_a);
        sched.prefill(seq_b);
        assert_eq!(sched.metrics.hit_rate(), 0.0);
    }

    #[test]
    fn evicted_block_invalidated_in_prefix_cache() {
        // Pool: 2 blocks, block_size=2. seq_a fills both.
        // step() needs a 3rd block, triggers eviction of seq_a and calls prefix_cache.evict.
        // seq_b with the same tokens should then get 0 prefix hits.
        let mut sched = Scheduler::new(2, 2);
        let seq_a = sched.add_request(make_tokens(&[0, 1, 2, 3]));
        sched.prefill(seq_a);
        sched.step(); // pool full → evicts seq_a, invalidates its cache entries
        assert_eq!(sched.waiting.len(), 1);

        let hits_before = sched.metrics.prefix_hits;
        let seq_b = sched.add_request(make_tokens(&[0, 1, 2, 3]));
        sched.prefill(seq_b);
        assert_eq!(sched.metrics.prefix_hits, hits_before);
    }

    #[test]
    fn partial_prefix_match_only_full_blocks_reused() {
        // block_size=4. Prefix of 6 tokens: only 1 full block (4 tokens) can be cached,
        // the remaining 2 tokens of the second block don't complete a block — not cached.
        let mut sched = Scheduler::new(8, 4);
        let seq_a = sched.add_request(make_tokens(&[0, 1, 2, 3, 4, 5]));
        let seq_b = sched.add_request(make_tokens(&[0, 1, 2, 3, 4, 5, 6, 7]));
        sched.prefill(seq_a);
        sched.prefill(seq_b);
        // seq_b should hit 1 block (tokens 0-3), not 2 (tokens 4-5 were partial in seq_a).
        assert_eq!(sched.metrics.prefix_hits, 4); // 4 tokens from 1 shared block
    }

    // --- Phase 6: Copy-on-Write tests ---

    #[test]
    fn fork_increfs_all_parent_blocks() {
        // parent: 4 tokens → 2 full blocks. After fork, both blocks have ref_count 2.
        let mut sched = Scheduler::new(8, 2);
        let parent = sched.add_request(make_tokens(&[0, 1, 2, 3]));
        sched.prefill(parent);
        let fork_a = sched.fork_sequence(parent);
        let _ = fork_a;
        for &block_id in sched.sequences[&parent].block_table.blocks() {
            assert_eq!(sched.pool.get_ref_count(block_id), 2);
        }
    }

    #[test]
    fn fork_appears_in_running() {
        let mut sched = Scheduler::new(8, 2);
        let parent = sched.add_request(make_tokens(&[0, 1, 2, 3]));
        sched.prefill(parent);
        let fork_a = sched.fork_sequence(parent);
        assert!(sched.running.contains(&fork_a));
    }

    #[test]
    fn cow_unshares_last_block_on_write() {
        // parent: 3 tokens (b0 full, b1 partial). fork shares both blocks.
        // After step_cow on fork only: fork's last block should be unshared (rc == 1).
        let mut sched = Scheduler::new(8, 2);
        let parent = sched.add_request(make_tokens(&[0, 1, 2]));
        sched.prefill(parent);
        // Remove parent from running so step_cow only advances the fork.
        let fork_a = sched.fork_sequence(parent);
        sched.running.retain(|&id| id == fork_a);

        sched.step_cow();

        let fork_last = sched.sequences[&fork_a].block_table.last_block().unwrap();
        assert_eq!(sched.pool.get_ref_count(fork_last), 1);
    }

    #[test]
    fn cow_sealed_blocks_stay_shared() {
        // b0 is sealed (full) — CoW should not affect it. It stays shared after a step.
        let mut sched = Scheduler::new(8, 2);
        let parent = sched.add_request(make_tokens(&[0, 1, 2]));
        sched.prefill(parent);
        let b0 = sched.sequences[&parent].block_table.blocks()[0];
        let fork_a = sched.fork_sequence(parent);
        sched.running.retain(|&id| id == fork_a);

        sched.step_cow();

        assert_eq!(sched.pool.get_ref_count(b0), 2); // parent + fork_a still share b0
    }

    #[test]
    fn multiple_forks_each_get_distinct_last_block() {
        // 3-way fork from a partial-block parent. After step_cow, each ends up with a
        // different last block (two CoW copies + the last holder keeps the original).
        let mut sched = Scheduler::new(16, 2);
        let parent = sched.add_request(make_tokens(&[0, 1, 2]));
        sched.prefill(parent);
        let fork_a = sched.fork_sequence(parent);
        let fork_b = sched.fork_sequence(parent);

        sched.step_cow(); // advances parent, fork_a, fork_b

        let last_blocks: std::collections::HashSet<_> = [parent, fork_a, fork_b]
            .iter()
            .map(|id| sched.sequences[id].block_table.last_block().unwrap())
            .collect();
        assert_eq!(last_blocks.len(), 3); // all distinct
    }

    #[test]
    fn cow_noop_when_block_already_exclusive() {
        // If last block already has ref_count == 1, ensure_unshared should not allocate.
        let mut sched = Scheduler::new(8, 2);
        let parent = sched.add_request(make_tokens(&[0, 1, 2]));
        sched.prefill(parent);
        let b1_before = sched.sequences[&parent].block_table.last_block().unwrap();
        // No fork — parent owns b1 exclusively.
        sched.step_cow();
        let b1_after = sched.sequences[&parent].block_table.last_block().unwrap();
        assert_eq!(b1_before, b1_after); // same block, no copy made
    }

    #[test]
    fn fork_then_step_cow_does_not_corrupt_parent_blocks() {
        // Parent and fork decode one token each. Parent's block table should be unchanged
        // except for its own last-block copy; fork's table is independent.
        let mut sched = Scheduler::new(16, 2);
        let parent = sched.add_request(make_tokens(&[0, 1, 2, 3])); // 2 full blocks
        sched.prefill(parent);
        let b0_parent = sched.sequences[&parent].block_table.blocks()[0];
        let fork_a = sched.fork_sequence(parent);
        let _ = fork_a;
        sched.step_cow();
        // b0 is sealed and must still match in both tables
        assert_eq!(sched.sequences[&parent].block_table.blocks()[0], b0_parent);
        assert_eq!(sched.sequences[&fork_a].block_table.blocks()[0], b0_parent);
    }
}
