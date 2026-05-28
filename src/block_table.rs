use crate::block_pool::{BlockID, BlockPool, PoolFull};

pub struct BlockTable {
    blocks: Vec<BlockID>,
    block_size: usize,
    num_tokens: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> BlockTable {
        BlockTable {
            blocks: Vec::new(),
            block_size,
            num_tokens: 0,
        }
    }
    pub fn logical_to_physical(&self, token_index: usize) -> (BlockID, usize) {
        let block_idx: usize = token_index / self.block_size;
        let offset: usize = token_index % self.block_size;
        (self.blocks[block_idx], offset)
    }
    pub fn append_token(&mut self, pool: &mut BlockPool) -> Result<(), PoolFull> {
        let block_boundary: usize = self.num_tokens % self.block_size;
        match block_boundary {
            0 => {
                let new_block = pool.alloc();
                match new_block {
                    Some(block) => {
                        self.blocks.push(block);
                        self.num_tokens += 1;
                        Ok(())
                    }
                    _ => Err(PoolFull),
                }
            }
            _ => {
                self.num_tokens += 1;
                Ok(())
            }
        }
    }
    pub fn last_block(&self) -> Option<BlockID> {
        self.blocks.last().copied()
    }
    pub fn is_last_block_full(&self) -> bool {
        !self.blocks.is_empty() && self.num_tokens % self.block_size == 0
    }
    pub fn len(&self) -> usize {
        self.blocks.len()
    }
    pub fn blocks(&self) -> &[BlockID] {
        &self.blocks
    }
    pub fn clear(&mut self) {
        self.num_tokens = 0;
        self.blocks.clear();
    }
    pub fn fork(&mut self, pool: &mut BlockPool) -> BlockTable {
        let blocks_forked = self.blocks.clone();
        for block in &blocks_forked {
            pool.incref(*block);
        }
        BlockTable {
            blocks: blocks_forked,
            block_size: self.block_size,
            num_tokens: self.num_tokens,
        }
    }
}
