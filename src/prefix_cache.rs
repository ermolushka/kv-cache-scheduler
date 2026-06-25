use std::collections::HashMap;

use crate::{block_pool::BlockID, sequence::TokenId};


pub struct TrieNode {
    pub children: HashMap<TokenId, TrieNode>,
    block_id: Option<BlockID>
}
pub struct PrefixCache {
    pub root: TrieNode,
    pub block_size: usize
}

impl TrieNode {
    fn new() -> Self {
        TrieNode { children: HashMap::new(), block_id: None }
    }

    fn evict_recursive(&mut self, block_id: BlockID) {
        if self.block_id == Some(block_id) {
            self.block_id = None;
        }
        for child in self.children.values_mut() {
            child.evict_recursive(block_id);
        }
    }
}

impl PrefixCache {
    pub fn new(block_size: usize) -> Self {
        PrefixCache { root: TrieNode::new(), block_size }
    }

    pub fn insert(&mut self, tokens: &[TokenId], blocks: &[BlockID]) {
        let mut current_node = &mut self.root;
        for (i, token) in tokens.iter().enumerate() {
            current_node = current_node.children.entry(*token).or_insert_with(TrieNode::new);
            if (i + 1) % self.block_size == 0 {
                current_node.block_id = Some(blocks[(i + 1) / self.block_size - 1]);
            }
        }
    }

    pub fn match_prefix(&self, tokens: &[TokenId]) -> (usize, Vec<BlockID>) {
        let mut matched_tokens = 0;
        let mut matched_blocks = Vec::new();
        let mut current_node = &self.root;

        for token in tokens {
            if let Some(child) = current_node.children.get(token) {
                current_node = child;
                matched_tokens += 1;
                if matched_tokens % self.block_size == 0 {
                    if let Some(id) = current_node.block_id {
                        matched_blocks.push(id);
                    } else {
                        break;
                    }
                }
            } else {
                break;
            }
        }

        (matched_tokens, matched_blocks)
    }

    pub fn evict(&mut self, block_id: BlockID) {
        self.root.evict_recursive(block_id);
    }
}