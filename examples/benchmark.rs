use kv_cache_scheduler::block_pool::BlockPoolConfig;
use kv_cache_scheduler::sequence::{Scheduler, SequenceId, TokenId};
use std::collections::{HashMap, VecDeque};

// Minimal seeded PRNG — avoids pulling in the rand crate.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize % (hi - lo + 1))
    }
}

struct RequestRecord {
    tokens: Vec<TokenId>,
    completion_len: usize,
}

// Requests that share a fixed system-prompt prefix — the high-hit-rate scenario.
fn traces_shared_prefix(rng: &mut Rng, n: usize, system_prompt_len: usize) -> Vec<RequestRecord> {
    let system_tokens: Vec<TokenId> = (0..system_prompt_len)
        .map(|i| TokenId(i as i32 + 5000))
        .collect();
    (0..n)
        .map(|_| {
            let user_len = rng.range(8, 48);
            let completion_len = rng.range(16, 96);
            let mut tokens = system_tokens.clone();
            tokens.extend((0..user_len).map(|_| TokenId(rng.range(0, 4999) as i32)));
            RequestRecord { tokens, completion_len }
        })
        .collect()
}

// Requests with entirely unique token IDs — no prefix sharing possible.
fn traces_no_sharing(rng: &mut Rng, n: usize) -> Vec<RequestRecord> {
    (0..n)
        .map(|i| {
            let prompt_len = rng.range(32, 64);
            let completion_len = rng.range(16, 96);
            let tokens = (0..prompt_len)
                .map(|j| TokenId((i * 10_000 + j) as i32))
                .collect();
            RequestRecord { tokens, completion_len }
        })
        .collect()
}

struct BenchResult {
    blocks_allocated: usize,
    evictions: usize,
    hit_rate: f64,
    avg_utilization: f64,
    avg_fragmentation: f64,
    steps: usize,
}

fn run_simulation(
    traces: Vec<RequestRecord>,
    config: BlockPoolConfig,
    max_concurrent: usize,
) -> BenchResult {
    let num_blocks = config.num_blocks;
    let block_size = config.block_size;
    let mut sched = Scheduler::with_config(config);
    let mut pending: VecDeque<RequestRecord> = traces.into();
    let mut completion_targets: HashMap<SequenceId, usize> = HashMap::new();

    let mut steps = 0usize;
    let mut util_sum = 0.0f64;
    let mut frag_sum = 0.0f64;

    loop {
        // Admit requests up to max_concurrent.
        while sched.running.len() < max_concurrent {
            match pending.pop_front() {
                None => break,
                Some(record) => {
                    let target = record.tokens.len() + record.completion_len;
                    let seq_id = sched.add_request(record.tokens);
                    sched.prefill(seq_id);
                    completion_targets.insert(seq_id, target);
                }
            }
        }

        if sched.running.is_empty() && pending.is_empty() {
            break;
        }

        sched.step();
        steps += 1;

        // Retire sequences that have reached their completion length.
        let finished: Vec<SequenceId> = sched
            .running
            .iter()
            .filter(|&&id| sched.sequences[&id].token_ids.len() >= completion_targets[&id])
            .copied()
            .collect();
        for id in finished {
            sched.finish_sequence(id);
            completion_targets.remove(&id);
        }

        // Pool utilization: fraction of physical blocks in use. O(1) via available().
        let used = (num_blocks as usize).saturating_sub(sched.pool.available());
        util_sum += used as f64 / num_blocks as f64;

        // Fragmentation: wasted token slots in partially-filled last blocks.
        let (wasted, capacity) = sched.running.iter().fold((0usize, 0usize), |(w, c), &id| {
            let seq = &sched.sequences[&id];
            let partial = seq.token_ids.len() % block_size;
            let cap = seq.block_table.len() * block_size;
            (w + if partial > 0 { block_size - partial } else { 0 }, c + cap)
        });
        frag_sum += if capacity > 0 { wasted as f64 / capacity as f64 } else { 0.0 };
    }

    BenchResult {
        blocks_allocated: sched.pool.total_allocs,
        evictions: sched.metrics.blocks_evicted,
        hit_rate: sched.metrics.hit_rate(),
        avg_utilization: if steps > 0 { util_sum / steps as f64 } else { 0.0 },
        avg_fragmentation: if steps > 0 { frag_sum / steps as f64 } else { 0.0 },
        steps,
    }
}

fn print_row(label: &str, r: &BenchResult) {
    println!(
        "  {:<26} {:>10}  {:>10}  {:>9.1}%  {:>9.1}%  {:>9.1}%  {:>7}",
        label,
        r.blocks_allocated,
        r.evictions,
        r.hit_rate * 100.0,
        r.avg_utilization * 100.0,
        r.avg_fragmentation * 100.0,
        r.steps,
    );
}

fn main() {
    const N_REQUESTS: usize = 300;
    const NUM_BLOCKS: u32 = 512;
    const BLOCK_SIZE: usize = 16;
    const MAX_CONCURRENT: usize = 16;
    const SYSTEM_PROMPT_LEN: usize = 64; // 4 shared blocks per request

    println!("=== KV Cache Scheduler Benchmark ===");
    println!(
        "Config: {} pool blocks, block_size={}, max_concurrent={}, {} requests\n",
        NUM_BLOCKS, BLOCK_SIZE, MAX_CONCURRENT, N_REQUESTS
    );

    let mut rng = Rng::new(42);

    let shared = traces_shared_prefix(&mut rng, N_REQUESTS, SYSTEM_PROMPT_LEN);
    let no_share = traces_no_sharing(&mut rng, N_REQUESTS);

    let r_shared = run_simulation(
        shared,
        BlockPoolConfig::new(NUM_BLOCKS, BLOCK_SIZE),
        MAX_CONCURRENT,
    );
    let r_no_share = run_simulation(
        no_share,
        BlockPoolConfig::new(NUM_BLOCKS, BLOCK_SIZE),
        MAX_CONCURRENT,
    );

    println!(
        "  {:<26} {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>7}",
        "Scenario", "Alloc'd", "Evictions", "Hit rate", "Avg util", "Avg frag", "Steps"
    );
    println!("  {}", "-".repeat(88));
    print_row(&format!("shared prefix ({}t)", SYSTEM_PROMPT_LEN), &r_shared);
    print_row("no sharing", &r_no_share);

    let savings = if r_no_share.blocks_allocated > 0 {
        (1.0 - r_shared.blocks_allocated as f64 / r_no_share.blocks_allocated as f64) * 100.0
    } else {
        0.0
    };
    println!("\nPrefix cache savings: {:.1}% fewer block allocations", savings);
}
