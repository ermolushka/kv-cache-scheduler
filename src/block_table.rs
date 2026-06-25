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

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }
    // Insert a pre-existing shared block (from prefix cache) without allocating from the pool
    // Caller is responsible for having already incref'd the block.
    pub fn push_full_block(&mut self, block_id: BlockID) {
        self.blocks.push(block_id);
        self.num_tokens += self.block_size;
    }

    pub fn clear(&mut self) {
        self.num_tokens = 0;
        self.blocks.clear();
    }
    // If the block at `slot` is shared (ref_count > 1), allocates a private copy,
    // decrefs the old block, and updates the table entry. No-op if already exclusive.
    pub fn ensure_unshared(&mut self, slot: usize, pool: &mut BlockPool) -> Result<BlockID, PoolFull> {
        let block_id = self.blocks[slot];
        if pool.get_ref_count(block_id) > 1 {
            let new_block = pool.alloc().ok_or(PoolFull)?;
            pool.decref(block_id);
            self.blocks[slot] = new_block;
            Ok(new_block)
        } else {
            Ok(block_id)
        }
    }

    // Like append_token but triggers a CoW copy when writing into a shared last block.
    pub fn append_token_cow(&mut self, pool: &mut BlockPool) -> Result<(), PoolFull> {
        if self.num_tokens % self.block_size == 0 {
            // Crossing into a new block — always unshared from birth.
            let new_block = pool.alloc().ok_or(PoolFull)?;
            self.blocks.push(new_block);
            self.num_tokens += 1;
            Ok(())
        } else {
            // Writing within the last block — unshare it first if needed.
            let last_slot = self.blocks.len() - 1;
            self.ensure_unshared(last_slot, pool)?;
            self.num_tokens += 1;
            Ok(())
        }
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
