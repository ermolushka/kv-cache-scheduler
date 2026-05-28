use crate::block_pool::{BlockID, BlockPool};
use linked_hash_map::LinkedHashMap;

pub trait EvictionPolicy {
    fn select_victim(&mut self, pool: &BlockPool) -> Option<BlockID>;
}

pub struct LRUEviction {
    pub blocks_map: LinkedHashMap<BlockID, ()>,
}

impl LRUEviction {
    pub fn new() -> LRUEviction {
        LRUEviction {
            blocks_map: LinkedHashMap::new(),
        }
    }
    pub fn on_access(&mut self, block_id: &BlockID) {
        // move block in front
        // by removing and adding to the
        // ordered map
        self.blocks_map.remove(block_id);
        self.blocks_map.insert(*block_id, ());
    }
    pub fn on_free(&mut self, block_id: &BlockID) {
        self.blocks_map.remove(block_id);
    }
}

impl EvictionPolicy for LRUEviction {
    fn select_victim(&mut self, pool: &BlockPool) -> Option<BlockID> {
        let block_ids = self.blocks_map.keys();
        for block in block_ids {
            let ref_count = pool.get_ref_count(*block);
            if ref_count == 1 {
                return Some(*block);
            }
        }
        None
    }
}
