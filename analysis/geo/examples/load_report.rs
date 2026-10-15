//! Measures how long the gazetteers take to load and how much memory they
//! hold.
//!
//! ```text
//! cargo run --release -p turbo-geo --example load_report -- bundled
//! cargo run --release -p turbo-geo --example load_report -- synthesize <rows> <file>
//! cargo run --release -p turbo-geo --example load_report -- geonames <file> [<snapshot file>]
//! ```
//!
//! `bundled` parses and indexes the bundled table ten times. `synthesize`
//! writes a table in the GeoNames main format with `<rows>` made up rows
//! whose names, alternate name counts and scripts are spread roughly as in
//! GeoNames' `allCountries.txt`, for measuring a table of that size when
//! the real file cannot be downloaded. `geonames` loads a GeoNames table,
//! writes its snapshot to the second file when one is named and reads the
//! snapshot back. Each step reports the wall time, the bytes the index
//! holds on the heap and the process's peak resident memory.

use std::time::Instant;

use turbo_geo::{BundledGazetteer, Gazetteer, GeoNamesGazetteer};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["bundled"] => bundled(),
        ["synthesize", rows, file] => synthesize(rows.parse().expect("a row count"), file),
        ["geonames", file] => geonames(file, None),
        ["geonames", file, snapshot] => geonames(file, Some(snapshot)),
        _ => {
            eprintln!("usage: load_report bundled | synthesize <rows> <file> | geonames <file> [<snapshot file>]");
            std::process::exit(2);
        }
    }
}

/// The peak resident set of this process, from `/proc/self/status`.
fn peak_rss_mib() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).map(|l| l.to_string()))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|kb| kb.parse::<u64>().ok()))
        .map_or_else(|| "unknown".to_string(), |kb| format!("{:.1} MiB", kb as f64 / 1024.0))
}

fn mib(bytes: usize) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn bundled() {
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..10 {
        let start = Instant::now();
        let g = BundledGazetteer::parse().expect("the bundled table loads");
        times.push(start.elapsed());
        last = Some(g);
    }
    let g = last.expect("ten loads");
    times.sort();
    println!(
        "bundled: {} records, {} keys, load median {:.2?} (min {:.2?}), index heap {}, peak RSS {}",
        g.len(),
        g.key_count(),
        times[times.len() / 2],
        times[0],
        mib(g.heap_bytes()),
        peak_rss_mib()
    );
    lookup_rate(&g, &["Paris", "springfield", "São Paulo", "nowhere at all"]);
}

/// Times a million lookups spread over `queries`.
fn lookup_rate(g: &impl Gazetteer, queries: &[&str]) {
    let n = 1_000_000;
    let start = Instant::now();
    let mut found = 0usize;
    for i in 0..n {
        found += g.lookup(queries[i % queries.len()]).len();
    }
    let elapsed = start.elapsed();
    println!(
        "  {n} lookups in {elapsed:.2?} ({:.0} ns each, {found} candidates)",
        elapsed.as_nanos() as f64 / n as f64
    );
}

fn geonames(file: &str, snapshot: Option<&str>) {
    let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
    let start = Instant::now();
    let g = GeoNamesGazetteer::load(file).expect("the table loads");
    let elapsed = start.elapsed();
    println!(
        "geonames: {} ({}) -> {} records, {} keys, load {:.2?}, index heap {}, peak RSS {}",
        file,
        mib(size as usize),
        g.len(),
        g.key_count(),
        elapsed,
        mib(g.heap_bytes()),
        peak_rss_mib()
    );
    lookup_rate(&g, &["Berlin", "springfield", "東京", "nowhere at all"]);
    let Some(path) = snapshot else { return };
    let start = Instant::now();
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).expect("the snapshot file"));
    g.write_snapshot(&mut out).expect("the snapshot is written");
    drop(out);
    println!("  snapshot written in {:.2?}, {}", start.elapsed(), mib(std::fs::metadata(path).unwrap().len() as usize));
    drop(g);
    let start = Instant::now();
    let back = GeoNamesGazetteer::load_snapshot(path).expect("the snapshot reads back");
    println!(
        "  snapshot read in {:.2?}, {} records, index heap {}, peak RSS {}",
        start.elapsed(),
        back.len(),
        mib(back.heap_bytes()),
        peak_rss_mib()
    );
}

/// A small deterministic generator (SplitMix64).
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn chance(&mut self, per_mille: u64) -> bool {
        self.below(1000) < per_mille
    }
}

/// Syllables per script, so the made up names fold, segment and hash as
/// names in those scripts do.
const LATIN: &[&str] = &[
    "ber", "lin", "san", "ta", "mar", "ia", "de", "los", "an", "ge", "les", "kö", "nig", "stadt", "dorf", "ville",
    "sur", "mer", "pó", "lis", "gra", "do", "rio", "la", "ña", "ço", "ø", "ł", "ő", "vá", "ros", "ham", "ton", "burg",
    "ka",
];
const CYRILLIC: &[&str] = &["мо", "ск", "ва", "пе", "тер", "бург", "но", "во", "си", "бирск", "ки", "ев", "ль", "вов"];
const HAN: &[&str] = &["東", "京", "北", "上", "海", "广", "州", "山", "川", "市", "村", "町", "大", "阪"];
const ARABIC: &[&str] = &["ال", "قا", "هر", "ة", "دم", "شق", "بغ", "داد", "مد", "ينة"];
const GREEK: &[&str] = &["αθ", "ή", "να", "θεσ", "σα", "λο", "νί", "κη", "πά", "τρα"];
const DEVANAGARI: &[&str] = &["मुं", "बई", "दि", "ल्ली", "पु", "णे", "नग", "र"];

fn name(r: &mut Random, syllables: &[&str], words: u64) -> String {
    let mut s = String::new();
    for w in 0..words {
        if w > 0 {
            s.push(' ');
        }
        let n = 1 + r.below(3);
        for i in 0..n {
            let syl = syllables[r.below(syllables.len() as u64) as usize];
            if i == 0 && w == 0 && syllables == LATIN {
                let mut c = syl.chars();
                let first = c.next().expect("a syllable");
                s.extend(first.to_uppercase());
                s.push_str(c.as_str());
            } else {
                s.push_str(syl);
            }
        }
    }
    s
}

/// A name in a script chosen with GeoNames' rough mix of alternate names.
fn alternate(r: &mut Random) -> String {
    let script = match r.below(100) {
        0..=54 => LATIN,
        55..=69 => CYRILLIC,
        70..=79 => HAN,
        80..=87 => ARABIC,
        88..=93 => GREEK,
        _ => DEVANAGARI,
    };
    let words = 1 + r.below(2);
    name(r, script, words)
}

fn synthesize(rows: u64, file: &str) {
    use std::io::Write;
    let mut out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(file).expect("the output file"));
    let mut r = Random(221);
    let classes = ["P", "A", "H", "L", "R", "S", "T", "U", "V"];
    let mut line = String::new();
    for id in 1..=rows {
        line.clear();
        let words = 1 + r.below(3);
        let canonical = name(&mut r, LATIN, words);
        // Most records have no alternate names; a few have hundreds.
        let count = if r.chance(780) {
            0
        } else if r.chance(990) {
            1 + r.below(6)
        } else {
            20 + r.below(300)
        };
        let alternates: Vec<String> = (0..count).map(|_| alternate(&mut r)).collect();
        let class = classes[if r.chance(400) { 0 } else { r.below(classes.len() as u64) as usize }];
        let population = if class == "P" && r.chance(300) { r.below(2_000_000) } else { 0 };
        let ascii: String = canonical.chars().filter(char::is_ascii).collect();
        let lat = r.below(180_000_00) as f64 / 100_000.0 - 90.0;
        let lon = r.below(360_000_00) as f64 / 100_000.0 - 180.0;
        let cc = format!("{}{}", char::from(b'A' + r.below(26) as u8), char::from(b'A' + r.below(26) as u8));
        line.push_str(&format!(
            "{id}\t{canonical}\t{ascii}\t{}\t{lat:.5}\t{lon:.5}\t{class}\tPPL\t{cc}\t\t01\t\t\t\t{population}\t\t123\tEurope/Berlin\t2026-01-01\n",
            alternates.join(",")
        ));
        out.write_all(line.as_bytes()).expect("the output file is written");
    }
    out.flush().expect("the output file is written");
    println!("wrote {rows} rows to {file} ({})", mib(std::fs::metadata(file).unwrap().len() as usize));
}
