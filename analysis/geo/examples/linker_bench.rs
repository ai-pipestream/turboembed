//! Times `GeoEntityLinker::link_mentions` on made up documents of 10, 100 and
//! 1000 location mentions over a small gazetteer whose names repeat across
//! clustered places, so mentions share candidate Geohashes, country codes and
//! containment names.
//!
//! `cargo run --release -p turbo-geo --example linker_bench -- [ROUNDS]`.
//! ROUNDS (default 5) is how many times each size runs; the median is
//! printed with the allocations of one run (counted by a wrapping global
//! allocator), the peak resident set of the process so far (`VmHWM`, Linux
//! only) and a checksum of every score's bits, which two builds can compare.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use turbo_analysis::Span;
use turbo_geo::{EntryParts, FEATURE_CLASS_CITY, GazetteerEntry, GeoEntityLinker, GeoPoint, InMemoryGazetteer};

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const NAMES: [&str; 12] = [
    "Springfield",
    "Riverton",
    "Fairview",
    "Georgetown",
    "Clinton",
    "Salem",
    "Madison",
    "Franklin",
    "Ashland",
    "Milton",
    "Oxford",
    "Burlington",
];
const PROVINCES: [&str; 4] = ["Illinois", "Normandy", "Bavaria", "Kent"];
const COUNTRIES: [(&str, &str); 4] = [("US", "United States"), ("FR", "France"), ("DE", "Germany"), ("GB", "England")];
const CLUSTERS: [(f64, f64); 4] = [(39.78, -89.65), (49.18, -0.37), (48.14, 11.58), (51.28, 1.08)];

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

fn gazetteer() -> InMemoryGazetteer {
    let mut rng = Lcg(7);
    let mut entries = Vec::new();
    for (n, name) in NAMES.iter().enumerate() {
        for k in 0..6 {
            let c = (n + k) % CLUSTERS.len();
            let (lat, lon) = CLUSTERS[c];
            let jitter = |r: &mut Lcg| (r.below(200) as f64 - 100.0) / 100.0;
            entries.push(
                GazetteerEntry::new(EntryParts {
                    source: "geonames".into(),
                    record_id: format!("{n}-{k}"),
                    name: (*name).into(),
                    alternate_names: Vec::new(),
                    location: GeoPoint::new(lat + jitter(&mut rng), lon + jitter(&mut rng)).unwrap(),
                    bounding_box: None,
                    country_code: Some(COUNTRIES[c].0.into()),
                    containment: vec![PROVINCES[c].into()],
                    population: 1_000 + rng.below(500_000) as i64,
                    feature_class: Some(FEATURE_CLASS_CITY.into()),
                    attributes: Vec::new(),
                })
                .unwrap(),
            );
        }
    }
    InMemoryGazetteer::from_entries(&entries).unwrap()
}

fn document(mentions: usize) -> (String, Vec<Span>) {
    let mut rng = Lcg(mentions as u64);
    let mut text = String::new();
    let mut spans = Vec::with_capacity(mentions);
    for _ in 0..mentions {
        text.push_str("We drove to ");
        let start = text.len();
        text.push_str(NAMES[rng.below(NAMES.len())]);
        spans.push(Span::new(start as u32, text.len() as u32));
        let c = rng.below(COUNTRIES.len());
        text.push_str(" in ");
        text.push_str(PROVINCES[c]);
        text.push_str(", ");
        text.push_str(COUNTRIES[c].1);
        text.push_str(". ");
    }
    (text, spans)
}

fn peak_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn main() {
    let rounds: usize = std::env::args().nth(1).map_or(5, |s| s.parse().expect("ROUNDS is a number"));
    let linker = GeoEntityLinker::new(gazetteer());
    println!("| Mentions | Median ms | Allocations | Allocated bytes | Peak RSS KB | Checksum |");
    println!("|---|---|---|---|---|---|");
    for mentions in [10usize, 100, 1000] {
        let (text, spans) = document(mentions);
        let mut times = Vec::with_capacity(rounds);
        let mut allocs = 0;
        let mut bytes = 0;
        let mut sum = turbo_analysis::Fnv1a::new();
        for round in 0..rounds {
            let (a0, b0) = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
            let t = Instant::now();
            let linked = linker.link_mentions(&text, &spans).unwrap();
            times.push(t.elapsed().as_secs_f64() * 1e3);
            if round == 0 {
                allocs = ALLOCS.load(Ordering::Relaxed) - a0;
                bytes = BYTES.load(Ordering::Relaxed) - b0;
                for c in linked.iter().flatten() {
                    for bits in [c.score, c.fuzzy_score, c.geohash_score, c.country_score, c.province_score]
                        .map(f64::to_bits)
                        .into_iter()
                        .chain(c.entry.record_id().bytes().map(u64::from))
                    {
                        sum.fold_u64(bits);
                    }
                }
            }
            drop(linked);
        }
        times.sort_by(f64::total_cmp);
        let rss = peak_rss_kb().map_or_else(|| "n/a".to_string(), |kb| kb.to_string());
        println!("| {mentions} | {:.3} | {allocs} | {bytes} | {rss} | {:016x} |", times[times.len() / 2], sum.finish());
    }
}
