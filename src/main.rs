use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct BlockID(u32);

pub struct BlockPool {
    free_list: VecDeque<BlockID>,
    ref_count: Vec<u32>,
}

#[derive(Debug)]
pub struct PoolFull;

pub struct BlockTable {
    blocks: Vec<BlockID>,
    block_size: usize,
    num_tokens: usize, // how many tokens appended so far
}

impl BlockTable {
    pub fn logical_to_physical(&self, token_index: usize) -> (BlockID, usize) {
        // we return block_id and offset within that block for the given token
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
    pub fn new(block_size: usize) -> BlockTable {
        BlockTable {
            blocks: Vec::new(),
            block_size,
            num_tokens: 0,
        }
    }
}

impl BlockPool {
    pub fn new(num_blocks: u32) -> Self {
        let ref_count = vec![0; num_blocks as usize];
        let mut free_list = VecDeque::new();
        for i in 0..num_blocks {
            free_list.push_back(BlockID(i))
        }
        BlockPool {
            free_list,
            ref_count,
        }
    }
    pub fn alloc(&mut self) -> Option<BlockID> {
        let free_block = self.free_list.pop_back();
        match free_block {
            Some(block) => {
                self.ref_count[block.0 as usize] += 1;
                Some(block)
            }
            None => None,
        }
    }
    fn free(&mut self, block_id: BlockID) {
        // we always check it has only 1 ref count
        assert!(self.ref_count[block_id.0 as usize] == 1);
        self.ref_count[block_id.0 as usize] = 0;
        self.free_list.push_front(block_id);
    }
    pub fn incref(&mut self, block_id: BlockID) {
        self.ref_count[block_id.0 as usize] += 1;
    }
    pub fn decref(&mut self, block_id: BlockID) {
        match self.ref_count[block_id.0 as usize] {
            1 => self.free(block_id),
            _ => {
                assert!(self.ref_count[block_id.0 as usize] > 0);
                self.ref_count[block_id.0 as usize] -= 1;
            }
        }
    }
    pub fn get_ref_count(&self, id: BlockID) -> u32 {
        self.ref_count[id.0 as usize]
    }
    pub fn is_free(&self, id: BlockID) -> bool {
        self.ref_count[id.0 as usize] == 0
    }
}

fn main() {
    let mut pool = BlockPool::new(4);
    let mut table = BlockTable::new(2); // block_size = 2

    // append 5 tokens — should allocate 3 blocks (tokens 0-1, 2-3, 4)
    for _ in 0..5 {
        table.append_token(&mut pool).unwrap();
    }
    println!("blocks allocated: {}", table.len()); // expect 3
    println!("pool blocks used: {}", 4 - pool.free_list.len()); // expect 3

    // logical_to_physical: token 3 -> block 2, offset 1
    let (block_id, offset) = table.logical_to_physical(3);
    println!("token 3 -> block {:?}, offset {}", block_id, offset); // expect block idx 1, offset 1

    // fork: both tables now share the same blocks
    let forked = table.fork(&mut pool);
    println!("forked table blocks: {}", forked.len()); // expect 3
    let shared_block = table.last_block().unwrap();
    println!(
        "shared block ref count: {}",
        pool.get_ref_count(shared_block)
    ); // expect 2
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // --- Phase 2: BlockTable tests ---

    #[test]
    fn append_tokens_allocates_blocks_at_boundaries() {
        let mut pool = BlockPool::new(4);
        let mut table = BlockTable::new(2);
        // first token -> needs a new block
        table.append_token(&mut pool).unwrap();
        assert_eq!(table.len(), 1);
        // second token -> same block
        table.append_token(&mut pool).unwrap();
        assert_eq!(table.len(), 1);
        // third token -> needs a new block
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
        // token 0 -> block 0, offset 0
        let (_, offset) = table.logical_to_physical(0);
        assert_eq!(offset, 0);
        // token 1 -> block 0, offset 1
        let (_, offset) = table.logical_to_physical(1);
        assert_eq!(offset, 1);
        // token 2 -> block 1, offset 0
        let (_, offset) = table.logical_to_physical(2);
        assert_eq!(offset, 0);
        // token 3 -> block 1, offset 1
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
        // 2 blocks allocated, each has ref count 1
        let _forked = table.fork(&mut pool);
        // after fork, each block should have ref count 2
        let (b0, _) = table.logical_to_physical(0);
        let (b2, _) = table.logical_to_physical(2);
        assert_eq!(pool.get_ref_count(b0), 2);
        assert_eq!(pool.get_ref_count(b2), 2);
    }

    #[test]
    fn append_token_fails_when_pool_exhausted() {
        let mut pool = BlockPool::new(1);
        let mut table = BlockTable::new(2);
        table.append_token(&mut pool).unwrap(); // uses the only block
        table.append_token(&mut pool).unwrap(); // fills it
        let result = table.append_token(&mut pool); // needs new block, pool empty
        assert!(result.is_err());
    }
}
