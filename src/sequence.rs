use crate::block_pool::{BlockPool, BlockPoolConfig};
use crate::eviction::{EvictionPolicy, LRUEviction};
use crate::prefix_cache::PrefixCache;
use crate::{block_pool::PoolFull, block_table::BlockTable};
use std::collections::{HashMap, VecDeque};

// Errors that can occur at the scheduler level.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("pool is full and no blocks are eligible for eviction")]
    NothingEvictable,
}

pub enum SequenceState {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

#[derive(Debug, Copy, PartialEq, Hash, Clone, Eq)]
pub struct SequenceId(pub i32);

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct TokenId(pub i32);

pub struct Sequence {
    pub id: SequenceId,
    // All token IDs so far: prompt tokens followed by generated tokens.
    pub token_ids: Vec<TokenId>,
    pub block_table: BlockTable,
    pub state: SequenceState,
}

impl Sequence {
    pub fn new(id: SequenceId, token_ids: Vec<TokenId>, block_size: usize) -> Sequence {
        Sequence {
            id,
            token_ids,
            block_table: BlockTable::new(block_size),
            state: SequenceState::Waiting,
        }
    }
}

// Aggregate metrics collected over a scheduler's lifetime.
pub struct Metrics {
    // Tokens matched from the prefix cache (saved allocations).
    pub prefix_hits: usize,
    // Total prompt tokens submitted to the prefix cache.
    pub prefix_total: usize,
    // Number of sequences preempted due to pool exhaustion.
    pub blocks_evicted: usize,
}

impl Metrics {
    fn new() -> Self {
        Metrics { prefix_hits: 0, prefix_total: 0, blocks_evicted: 0 }
    }

    // Fraction of prompt tokens served from the prefix cache.
    pub fn hit_rate(&self) -> f64 {
        if self.prefix_total == 0 { return 0.0; }
        self.prefix_hits as f64 / self.prefix_total as f64
    }
}

// Top-level scheduler: owns the block pool, sequence state, eviction policy,
// and prefix cache. Drives the prefill → decode loop.
pub struct Scheduler {
    pub pool: BlockPool,
    pub sequences: HashMap<SequenceId, Sequence>,
    // Sequences waiting to be prefilled.
    pub waiting: VecDeque<SequenceId>,
    // Sequences currently in the decode loop.
    pub running: Vec<SequenceId>,
    pub block_size: usize,
    pub next_id: i32,
    pub eviction: LRUEviction,
    pub prefix_cache: PrefixCache,
    pub metrics: Metrics,
}

impl Scheduler {
    pub fn new(num_blocks: u32, block_size: usize) -> Scheduler {
        Scheduler {
            pool: BlockPool::new(num_blocks),
            block_size,
            sequences: HashMap::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            next_id: 0,
            eviction: LRUEviction::new(),
            prefix_cache: PrefixCache::new(block_size),
            metrics: Metrics::new(),
        }
    }

    // Alternative constructor from a [`BlockPoolConfig`].
    pub fn with_config(config: BlockPoolConfig) -> Scheduler {
        Scheduler::new(config.num_blocks, config.block_size)
    }

    // Number of unique physical blocks currently in use across all sequences.
    pub fn pool_used_blocks_physical(&self) -> usize {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for seq in self.sequences.values() {
            for &block_id in seq.block_table.blocks() {
                seen.insert(block_id);
            }
        }
        seen.len()
    }

    pub fn pool_used_blocks(&self) -> usize {
        self.sequences.values().map(|s| s.block_table.len()).sum()
    }

    pub fn add_request(&mut self, tokens: Vec<TokenId>) -> SequenceId {
        let seq_id = SequenceId(self.next_id);
        self.sequences.insert(seq_id, Sequence::new(seq_id, tokens, self.block_size));
        self.waiting.push_back(seq_id);
        self.next_id += 1;
        seq_id
    }

    // Allocates blocks for the full prompt, consulting the prefix cache first.
    // Moves the sequence from waiting to running.
    pub fn prefill(&mut self, seq_id: SequenceId) {
        let token_ids: Vec<TokenId> = self.sequences[&seq_id].token_ids.clone();
        let token_count = token_ids.len();

        self.metrics.prefix_total += token_count;

        let (_matched_tokens, matched_blocks) = self.prefix_cache.match_prefix(&token_ids);
        let cache_block_count = matched_blocks.len();

        for &block_id in &matched_blocks {
            self.pool.incref(block_id);
            self.sequences.get_mut(&seq_id).unwrap().block_table.push_full_block(block_id);
            self.eviction.on_access(&block_id);
        }

        let prefix_token_count = cache_block_count * self.block_size;
        self.metrics.prefix_hits += prefix_token_count;

        for _ in prefix_token_count..token_count {
            self.sequences.get_mut(&seq_id).unwrap().block_table.append_token(&mut self.pool).unwrap();
            let block_id = self.sequences[&seq_id].block_table.last_block().unwrap();
            self.eviction.on_access(&block_id);
        }

        let all_blocks: Vec<_> = self.sequences[&seq_id].block_table.blocks().to_vec();
        self.prefix_cache.insert(&token_ids, &all_blocks);

        self.sequences.get_mut(&seq_id).unwrap().state = SequenceState::Decoding;
        self.waiting.retain(|id| id != &seq_id);
        self.running.push(seq_id);
    }

    // Advances every running sequence by one decode token using standard (non-CoW) allocation.
    pub fn step(&mut self) {
        let running = self.running.clone();
        for seq_id in &running {
            if !self.running.contains(seq_id) { continue; }
            self.sequences.get_mut(seq_id).unwrap().token_ids.push(TokenId(0));
            match self.sequences.get_mut(seq_id).unwrap().block_table.append_token(&mut self.pool) {
                Ok(_) => {
                    let block_id = self.sequences[seq_id].block_table.last_block().unwrap();
                    self.eviction.on_access(&block_id);
                }
                Err(PoolFull) => {
                    self.handle_pool_full();
                    if self.running.contains(seq_id) {
                        self.sequences.get_mut(seq_id).unwrap().block_table.append_token(&mut self.pool).unwrap();
                        let block_id = self.sequences[seq_id].block_table.last_block().unwrap();
                        self.eviction.on_access(&block_id);
                    }
                }
            }
        }
    }

    // Like [`step`](Scheduler::step) but uses CoW allocation — triggers a block copy when
    // writing into a shared last block rather than corrupting shared state.
    pub fn step_cow(&mut self) {
        let running = self.running.clone();
        for seq_id in &running {
            if !self.running.contains(seq_id) { continue; }
            self.sequences.get_mut(seq_id).unwrap().token_ids.push(TokenId(0));
            match self.sequences.get_mut(seq_id).unwrap().block_table.append_token_cow(&mut self.pool) {
                Ok(_) => {
                    let block_id = self.sequences[seq_id].block_table.last_block().unwrap();
                    self.eviction.on_access(&block_id);
                }
                Err(PoolFull) => {
                    self.handle_pool_full();
                    if self.running.contains(seq_id) {
                        self.sequences.get_mut(seq_id).unwrap().block_table.append_token_cow(&mut self.pool).unwrap();
                        let block_id = self.sequences[seq_id].block_table.last_block().unwrap();
                        self.eviction.on_access(&block_id);
                    }
                }
            }
        }
    }

    // Creates a new sequence sharing the parent's block table (all blocks incref'd).
    // Both parent and fork are placed in running and decode independently from this point.
    pub fn fork_sequence(&mut self, seq_id: SequenceId) -> SequenceId {
        let new_id = SequenceId(self.next_id);
        self.next_id += 1;

        let token_ids = self.sequences[&seq_id].token_ids.clone();
        let forked_table = self.sequences.get_mut(&seq_id).unwrap().block_table.fork(&mut self.pool);

        for &block_id in forked_table.blocks() {
            self.eviction.on_access(&block_id);
        }

        let new_seq = Sequence {
            id: new_id,
            token_ids,
            block_table: forked_table,
            state: SequenceState::Decoding,
        };
        self.sequences.insert(new_id, new_seq);
        self.running.push(new_id);
        new_id
    }

    // Marks a sequence as finished: decrefs all its blocks and removes it from running.
    // Call this when a sequence has generated its full completion.
    pub fn finish_sequence(&mut self, seq_id: SequenceId) {
        let blocks: Vec<_> = self.sequences[&seq_id].block_table.blocks().to_vec();
        for block_id in &blocks {
            self.pool.decref(*block_id);
            self.eviction.on_free(block_id);
        }
        self.sequences.get_mut(&seq_id).unwrap().block_table.clear();
        self.sequences.get_mut(&seq_id).unwrap().state = SequenceState::Finished;
        self.running.retain(|id| id != &seq_id);
    }

    // Evicts the LRU-eligible sequence to free blocks when the pool is full.
    // Panics via [`SchedulerError::NothingEvictable`] if no sequence can be preempted.
    pub fn handle_pool_full(&mut self) {
        let victim_id = self.eviction.select_victim(&self.pool)
            .ok_or(SchedulerError::NothingEvictable)
            .expect("pool full and no blocks eligible for eviction");

        let victim_seq_id = self.sequences
            .iter()
            .find(|(_, seq)| seq.block_table.blocks().contains(&victim_id))
            .map(|(seq_id, _)| *seq_id)
            .unwrap();

        let blocks: Vec<_> = self.sequences[&victim_seq_id].block_table.blocks().to_vec();
        for block_id in &blocks {
            self.pool.decref(*block_id);
            self.eviction.on_free(block_id);
            self.prefix_cache.evict(*block_id);
        }

        self.metrics.blocks_evicted += 1;
        self.sequences.get_mut(&victim_seq_id).unwrap().state = SequenceState::Waiting;
        self.sequences.get_mut(&victim_seq_id).unwrap().block_table.clear();
        self.waiting.push_back(victim_seq_id);
        self.running.retain(|id| id != &victim_seq_id);
    }
}
