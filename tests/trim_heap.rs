#![cfg(target_os = "linux")]

use hark::scheduler::trim_heap;

fn rss_kib() -> u64 {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap();
    statm
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u64>()
        .unwrap()
        * 4
}

#[test]
fn trim_heap_returns_fragmented_free_memory_to_the_os() {
    let mut blocks: Vec<Vec<u8>> = (0..20_000).map(|_| vec![1u8; 16 * 1024]).collect();
    let mut i = 0;
    blocks.retain(|_| {
        i += 1;
        i % 2 == 0
    });
    let before = rss_kib();
    trim_heap();
    let after = rss_kib();
    assert!(
        before.saturating_sub(after) > 50 * 1024,
        "rss {before} KiB before, {after} KiB after"
    );
    drop(blocks);
}
