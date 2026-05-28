use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct BlockID(pub u32);

#[derive(Debug)]
pub struct PoolFull;

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
