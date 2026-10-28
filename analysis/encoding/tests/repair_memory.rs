//! Memory of the mojibake repair, measured with a counting allocator.
//!
//! The repair keeps the tracked text (each character with its source
//! span) and the text each step builds; the badness heuristic and the
//! UTF-8 clue search look characters up as they go, with no per-scan
//! table of character classes. The test checks that bound on mixed text,
//! where no code page fits the whole text and the UTF-8 clue path runs.
//!
//! `measure_10_mib` is ignored by default: it repairs 10 MiB of mixed
//! mojibake and prints the wall time and peak memory, for comparing
//! builds (`cargo test --release --test repair_memory -- --ignored
//! --nocapture`, with `REPAIR_MODE=lines` for line-by-line repair).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use turbo_analysis::ViewBuilder;
use turbo_encoding::repair::{fix_encoding, repair_lines};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// One piece of mixed text: Japanese that no single-byte code page can
/// encode, so the whole text is never re-encoded at once, around UTF-8
/// read as windows-1252, which the clue path fixes run by run.
const UNIT: &str = "日本 naÃ¯ve cafÃ© â€œquotedâ€\u{9D} rÃ©sumÃ© ";

/// About `bytes` of mixed text, with a newline every eight units when
/// `lines` is set.
fn mixed(bytes: usize, lines: bool) -> String {
    let mut s = String::with_capacity(bytes + UNIT.len() + 1);
    let mut n = 0;
    while s.len() < bytes {
        s.push_str(UNIT);
        n += 1;
        if lines && n % 8 == 0 {
            s.push('\n');
        }
    }
    s
}

/// Peak bytes allocated while running `f`, above what was live before.
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let out = f();
    (out, PEAK.load(Ordering::Relaxed) - base)
}

#[test]
fn clue_path_needs_no_class_table_per_scan() {
    let text = mixed(1 << 20, false);
    let chars = text.chars().count();
    let (r, peak) = peak_during(|| fix_encoding(&text));
    assert!(r.text.starts_with("日本 naïve café \u{201C}quoted\u{201D} résumé "));
    // The tracked input and the text one step builds are 12 bytes a
    // character each (a char and a two-u32 source span), each sized up
    // front. A table of classes for each scan would add 8 bytes a
    // character on top, twice when the UTF-8 clue search and its repair
    // scan separately, and letting the vectors double as they grow would
    // add up to as much again; the bound leaves no room for any of these
    // (the code before measured 52 bytes a character here).
    let per_char = peak / chars;
    assert!(per_char <= 30, "{per_char} bytes a character at the peak");
}

#[test]
#[ignore]
fn measure_10_mib() {
    let lines = std::env::var("REPAIR_MODE").is_ok_and(|m| m == "lines");
    let text = mixed(10 << 20, lines);
    let start = Instant::now();
    let (len, peak) = peak_during(|| {
        if lines {
            let mut out = ViewBuilder::new(&text);
            repair_lines(&text, &mut out);
            out.finish().0.len()
        } else {
            fix_encoding(&text).text.len()
        }
    });
    let elapsed = start.elapsed();
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let hwm = status.lines().find(|l| l.starts_with("VmHWM")).unwrap_or("VmHWM: unknown");
    println!(
        "mode={} input={} output={len} wall={:.3}s heap_peak={:.1}MiB {hwm}",
        if lines { "lines" } else { "whole" },
        text.len(),
        elapsed.as_secs_f64(),
        peak as f64 / (1 << 20) as f64
    );
}
