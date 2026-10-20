//! Times the mapped automaton against the in-heap one on the same
//! glossary and text: building, writing and loading the image, and
//! scanning and finding, the two matchers interleaved run by run.
//!
//! `cargo run --release -p turbo-glossary --example mapped_bench -- TERMS
//! [insensitive]`. The glossary is TERMS terms of one to three
//! pseudo-words from a vocabulary of 20,000, the text about 8 MB of the
//! same vocabulary with a quarter of its words glossary terms.

use std::time::Instant;

use turbo_glossary::{
    Boundaries, Case, EntryIx, Glossary, ImageBuf, MappedGlossaryMatcher, MappedGlossaryWriter, Scratch, TextMatcher,
};

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(terms) = args.first().and_then(|a| a.parse::<usize>().ok()) else {
        eprintln!("usage: mapped_bench TERMS [insensitive]");
        std::process::exit(2);
    };
    let case = if args.get(1).is_some_and(|a| a == "insensitive") { Case::Insensitive } else { Case::Sensitive };
    let mut rng = Lcg(202);
    const LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDE";
    let words: Vec<String> = (0..20_000)
        .map(|_| (0..3 + rng.below(8)).map(|_| LETTERS[rng.below(LETTERS.len())] as char).collect())
        .collect();
    let mut glossary = Glossary::new();
    for i in 0..terms {
        let term: Vec<&str> = (0..1 + rng.below(3)).map(|_| words[rng.below(words.len())].as_str()).collect();
        glossary.push(&format!("t{i}"), &term.join(" ")).unwrap();
    }
    let mut text = String::new();
    while text.len() < 8 << 20 {
        let w = if rng.below(4) == 0 {
            glossary.term(EntryIx(rng.below(terms) as u32))
        } else {
            &words[rng.below(words.len())]
        };
        text.push_str(w);
        text.push_str(if rng.below(10) == 0 { ", " } else { " " });
    }
    let mut cuts = Vec::new();
    turbo_tokenize::word_boundaries(&text, &mut cuts);
    let boundaries = Boundaries::from_offsets(text.len() as u32, cuts);

    let t = Instant::now();
    let heap = TextMatcher::new(&glossary, case).unwrap();
    let build_ms = ms(t);
    // Writing builds its own in-heap automaton, so the write time is the
    // whole time less the build time above.
    let t = Instant::now();
    let bytes = MappedGlossaryWriter::write(&glossary, case).unwrap();
    let write_ms = ms(t) - build_ms;
    let image = ImageBuf::copy_from(&bytes);
    let mut load = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        let m = MappedGlossaryMatcher::from_bytes(image.bytes()).unwrap();
        load.push(ms(t));
        assert_eq!(m.entries(), terms);
    }
    let mapped = MappedGlossaryMatcher::from_bytes(image.bytes()).unwrap();

    let mut scratch = Scratch::new();
    let (mut out_h, mut out_m) = (Vec::new(), Vec::new());
    let (mut scan_h, mut scan_m, mut find_h, mut find_m) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut hits_h, mut hits_m) = (0u64, 0u64);
    for _ in 0..7 {
        let t = Instant::now();
        heap.scan(&text, |s, e| hits_h = hits_h.wrapping_mul(31).wrapping_add(u64::from(s.start ^ s.end ^ e.0)));
        scan_h.push(ms(t));
        let t = Instant::now();
        mapped.scan(&text, &mut scratch, |s, e| {
            hits_m = hits_m.wrapping_mul(31).wrapping_add(u64::from(s.start ^ s.end ^ e.0))
        });
        scan_m.push(ms(t));
        out_h.clear();
        let t = Instant::now();
        heap.find(&text, &boundaries, &mut out_h);
        find_h.push(ms(t));
        out_m.clear();
        let t = Instant::now();
        mapped.find(&text, &boundaries, &mut scratch, &mut out_m);
        find_m.push(ms(t));
    }
    assert_eq!(hits_h, hits_m, "the scans differ");
    assert_eq!(out_h, out_m, "the matches differ");
    let mb = text.len() as f64 / (1 << 20) as f64;
    let (sh, sm, fh, fm) = (median(scan_h), median(scan_m), median(find_h), median(find_m));
    println!(
        "terms={terms} case={case:?} text_mb={mb:.1} matches={} states={} cells={} heap_mb={:.1} image_mb={:.1} \
         build_ms={build_ms:.0} write_ms={write_ms:.0} load_ms={:.2} scan_heap_ms={sh:.1} scan_mapped_ms={sm:.1} \
         find_heap_ms={fh:.1} find_mapped_ms={fm:.1} scan_mapped_mb_s={:.0} find_mapped_mb_s={:.0}",
        out_m.len(),
        heap.states(),
        mapped.cells(),
        heap.heap_bytes() as f64 / (1 << 20) as f64,
        bytes.len() as f64 / (1 << 20) as f64,
        median(load),
        mb / sm * 1e3,
        mb / fm * 1e3,
    );
}
