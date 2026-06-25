# kv-cache-scheduler

[![crates.io](https://img.shields.io/crates/v/kv-cache-scheduler.svg)](https://crates.io/crates/kv-cache-scheduler)

A [PagedAttention](https://arxiv.org/abs/2309.06180)-style KV cache block manager for LLM inference, written in pure Rust.

Instead of allocating contiguous memory per request, this crate manages a fixed pool of fixed-size blocks - the same idea as virtual memory paging. Multiple requests can share blocks (prefix caching), and beam search branches can fork without copying (copy-on-write).

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
kv-cache-scheduler = "0.1.0"
```

Or run:

```
cargo add kv-cache-scheduler
```

## Features

- **Block pool** - fixed-size allocator with reference counting; blocks shared across requests are never double-freed
- **Block table** - per-request page table mapping logical token positions to physical block IDs
- **Sequence manager** - prefill/decode lifecycle with a scheduler that drives multiple sequences
- **LRU eviction** - preempts the least-recently-used sequence when the pool is full
- **Prefix cache** - radix trie keyed on token IDs; reuses physical blocks across requests that share a common prefix (system prompts, few-shot examples)
- **Copy-on-write** - beam search forks share the parent's blocks; a private copy is made only on the first write

## Benchmark

Synthetic trace: 300 requests, 512 pool blocks, `block_size=16`, 16 concurrent sequences.

| Scenario | Blocks allocated | Hit rate | Avg utilization |
|---|---|---|---|
| Shared prefix (64-token system prompt) | 1 643 | 69.9% | 12.9% |
| No sharing (unique tokens per request) | 2 070 | 0.0% | 16.1% |

**~20% fewer block allocations** from prefix caching on this trace.

Run it yourself:

```
cargo run --example benchmark
```

## Quick start

```rust
use kv_cache_scheduler::block_pool::BlockPoolConfig;
use kv_cache_scheduler::sequence::{Scheduler, TokenId};

// 128 blocks, 16 tokens per block
let mut sched = Scheduler::with_config(BlockPoolConfig::new(128, 16));

// Enqueue and prefill a prompt
let tokens: Vec<TokenId> = (0..32).map(TokenId).collect();
let seq_id = sched.add_request(tokens);
sched.prefill(seq_id);

// Decode loop
for _ in 0..64 {
    sched.step();
}

// Release blocks when done
sched.finish_sequence(seq_id);
println!("hit rate: {:.1}%", sched.metrics.hit_rate() * 100.0);
```

## Copy-on-write (beam search)

```rust
let parent = sched.add_request(prompt_tokens);
sched.prefill(parent);

// Fork into K branches - all share the parent's blocks
let branch_a = sched.fork_sequence(parent);
let branch_b = sched.fork_sequence(parent);

// step_cow triggers a block copy only when a branch writes to a shared block
sched.step_cow();
```

## Design

Three core concepts:

1. **BlockPool** - a fixed array of `N` blocks. `alloc()` pops from a free list; `decref()` returns a block when its ref count hits zero.
2. **BlockTable** - per-request page table: `Vec<BlockID>` where index `i` covers tokens `[i*block_size .. (i+1)*block_size)`.
3. **PrefixCache** - a radix trie keyed on `TokenId` sequences. On a cache hit, matched blocks are incref'd and inserted directly into the new request's block table - no allocation needed.

## License

[MIT](LICENSE)
