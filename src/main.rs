mod block_pool;
mod block_table;

use block_pool::BlockPool;
use block_table::BlockTable;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use block_pool::BlockID;

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
}
