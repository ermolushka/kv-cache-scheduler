use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct BlockID(u32);

pub struct BlockPool {
    free_list: VecDeque<BlockID>,
    ref_count: Vec<u32>,
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
    println!("Hello, world!");
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
}
