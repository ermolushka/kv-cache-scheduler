use crate::block_pool::BlockPool;
use crate::eviction::{EvictionPolicy, LRUEviction};
use crate::{block_pool::PoolFull, block_table::BlockTable};
use std::collections::{HashMap, VecDeque};

pub enum SequenceState {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

#[derive(Debug, Copy, PartialEq, Hash, Clone, Eq)]
pub struct SequenceId(i32);

#[derive(Debug, Copy, Clone)]
pub struct TokenId(pub i32);

pub struct Sequence {
    pub id: SequenceId,
    // all tokens so far (prompt + generated)
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

pub struct Scheduler {
    pub pool: BlockPool,
    pub sequences: HashMap<SequenceId, Sequence>,
    // not yet started
    pub waiting: VecDeque<SequenceId>,
    // running
    pub running: Vec<SequenceId>,
    pub block_size: usize,
    pub next_id: i32,
    pub eviction: LRUEviction,
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
        }
    }
    pub fn add_request(&mut self, tokens: Vec<TokenId>) -> SequenceId {
        let seq_id = SequenceId(self.next_id);
        let sequence = Sequence::new(seq_id, tokens, self.block_size);
        self.sequences.insert(seq_id, sequence);
        self.waiting.push_back(seq_id);
        self.next_id += 1;
        seq_id
    }
    pub fn prefill(&mut self, seq_id: SequenceId) {
        let token_count = self.sequences.get(&seq_id).unwrap().token_ids.len();
        for _ in 0..token_count {
            self.sequences
                .get_mut(&seq_id)
                .unwrap()
                .block_table
                .append_token(&mut self.pool)
                .unwrap();
            let block_id = self
                .sequences
                .get(&seq_id)
                .unwrap()
                .block_table
                .last_block()
                .unwrap();
            self.eviction.on_access(&block_id);
        }
        self.sequences.get_mut(&seq_id).unwrap().state = SequenceState::Decoding;
        self.waiting.retain(|id| id != &seq_id);
        self.running.push(seq_id);
    }
    pub fn step(&mut self) {
        let running = self.running.clone();
        for seq_id in &running {
            if !self.running.contains(seq_id) {
                continue;
            }
            self.sequences
                .get_mut(&seq_id)
                .unwrap()
                .token_ids
                .push(TokenId(0));
            match self
                .sequences
                .get_mut(seq_id)
                .unwrap()
                .block_table
                .append_token(&mut self.pool)
            {
                Ok(_) => {
                    let block_id = self
                        .sequences
                        .get(&seq_id)
                        .unwrap()
                        .block_table
                        .last_block()
                        .unwrap();
                    self.eviction.on_access(&block_id);
                }
                Err(PoolFull) => {
                    self.handle_pool_full();
                    if self.running.contains(seq_id) {
                        self.sequences
                            .get_mut(seq_id)
                            .unwrap()
                            .block_table
                            .append_token(&mut self.pool)
                            .unwrap();
                        let block_id = self
                            .sequences
                            .get(seq_id)
                            .unwrap()
                            .block_table
                            .last_block()
                            .unwrap();
                        self.eviction.on_access(&block_id);
                    }
                }
            }
        }
    }
    pub fn handle_pool_full(&mut self) {
        let victim_id = match self.eviction.select_victim(&self.pool) {
            None => panic!("pool full, nothing evictable"),
            Some(id) => id,
        };
        let victim_seq_id = self
            .sequences
            .iter()
            .find(|(_, seq)| seq.block_table.blocks().contains(&victim_id))
            .map(|(seq_id, _)| *seq_id)
            .unwrap();
        let blocks: Vec<_> = self.sequences[&victim_seq_id].block_table.blocks().to_vec();
        for block_id in &blocks {
            self.pool.decref(*block_id);
            self.eviction.on_free(block_id);
        }
        self.sequences.get_mut(&victim_seq_id).unwrap().state = SequenceState::Waiting;
        // clear block table
        self.sequences
            .get_mut(&victim_seq_id)
            .unwrap()
            .block_table
            .clear();
        self.waiting.push_back(victim_seq_id);
        self.running.retain(|id| id != &victim_seq_id);
    }
}
