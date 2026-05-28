mod block_pool;
mod block_table;
mod eviction;
mod sequence;

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
}
