use std::collections::VecDeque;

// Opaque handle to a physical block. Wraps a `u32` index into the pool.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct BlockID(pub u32);

// Returned by [`BlockPool::alloc`] and [`BlockTable::append_token`] when no free blocks remain.
#[derive(Debug, thiserror::Error)]
#[error("block pool exhausted")]
pub struct PoolFull;

// Capacity and block-size settings used to construct a [`BlockPool`] and [`Scheduler`].
pub struct BlockPoolConfig {
    // Total number of physical blocks in the pool.
    pub num_blocks: u32,
    // Number of tokens each block can hold. Smaller values reduce wasted space at sequence
    // boundaries; larger values reduce per-block bookkeeping overhead.
    pub block_size: usize,
}

impl BlockPoolConfig {
    pub fn new(num_blocks: u32, block_size: usize) -> Self {
        BlockPoolConfig { num_blocks, block_size }
    }
}

// Fixed-size allocator for physical KV-cache blocks.
//
// Blocks are identified by [`BlockID`] (a pool index). Shared ownership is tracked via
// reference counts: `incref`/`decref` instead of direct `free`. A block returns to the
// free list only when its ref count reaches zero.
pub struct BlockPool {
    free_list: VecDeque<BlockID>,
    ref_count: Vec<u32>,
    // Monotonically increasing count of successful [`alloc`](BlockPool::alloc) calls.
    // Useful for measuring total allocation pressure over a simulation run.
    pub total_allocs: usize,
}

impl BlockPool {
    pub fn new(num_blocks: u32) -> Self {
        let ref_count = vec![0; num_blocks as usize];
        let mut free_list = VecDeque::new();
        for i in 0..num_blocks {
            free_list.push_back(BlockID(i));
        }
        BlockPool { free_list, ref_count, total_allocs: 0 }
    }

    // Allocates one block, sets its ref count to 1, and returns its ID.
    // Returns `None` if the pool is exhausted.
    pub fn alloc(&mut self) -> Option<BlockID> {
        let block = self.free_list.pop_back()?;
        self.ref_count[block.0 as usize] = 1;
        self.total_allocs += 1;
        Some(block)
    }

    fn free(&mut self, block_id: BlockID) {
        assert_eq!(self.ref_count[block_id.0 as usize], 1);
        self.ref_count[block_id.0 as usize] = 0;
        self.free_list.push_front(block_id);
    }

    // Increments the ref count. Call when a block is shared (prefix cache hit, CoW fork).
    pub fn incref(&mut self, block_id: BlockID) {
        self.ref_count[block_id.0 as usize] += 1;
    }

    // Decrements the ref count. Frees the block when it reaches zero.
    pub fn decref(&mut self, block_id: BlockID) {
        match self.ref_count[block_id.0 as usize] {
            0 => panic!("decref on unallocated block {:?}", block_id),
            1 => self.free(block_id),
            _ => self.ref_count[block_id.0 as usize] -= 1,
        }
    }

    pub fn get_ref_count(&self, id: BlockID) -> u32 {
        self.ref_count[id.0 as usize]
    }

    pub fn is_free(&self, id: BlockID) -> bool {
        self.ref_count[id.0 as usize] == 0
    }

    // Number of blocks currently available for allocation. O(1).
    pub fn available(&self) -> usize {
        self.free_list.len()
    }
}
