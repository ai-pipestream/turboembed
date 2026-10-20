//! Times finding a gazetteer's names in text with its mapped key image
//! against the in-heap gazetteer: looking up every run of word tokens up
//! to the longest name with `Gazetteer::lookup`, which finds the same
//! names. The two alternate run by run.
//!
//! `cargo run --release -p turbo-geo --example names_bench -- [GEONAMES_TABLE
//! [ROUNDS]]`. Without a table it uses the bundled one; ROUNDS (default 5)
//! is how many times each side runs. The text is about 2 MB of the
//! gazetteer's own names, written with capitals and hyphens, between
//! common words.

use std::time::Instant;

use turbo_geo::{BundledGazetteer, Gazetteer, GeoNamesGazetteer, NameFinder, NameScratch};
use turbo_glossary::ImageBuf;
use turbo_glossary_layers::EntryIx;

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

fn run<G: Gazetteer>(name: &str, g: &G, image_bytes: Vec<u8>, image_ms: f64, heap: usize, rounds: usize) {
    let image = ImageBuf::copy_from(&image_bytes);
    let mut load = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        NameFinder::from_bytes(image.bytes()).unwrap();
        load.push(ms(t));
    }
    let finder = NameFinder::from_bytes(image.bytes()).unwrap();
    let longest = (0..finder.keys() as u32).map(|k| finder.key(EntryIx(k)).unwrap().split(' ').count()).max().unwrap();
    let mut rng = Lcg(5);
    const FILLER: [&str; 12] = ["the", "of", "in", "and", "to", "a", "from", "near", "river", "road", "north", "via"];
    let mut text = String::new();
    while text.len() < 2 << 20 {
        if rng.below(4) == 0 {
            let key = finder.key(EntryIx(rng.below(finder.keys()) as u32)).unwrap();
            for (i, w) in key.split(' ').enumerate() {
                text.push_str(if i == 0 {
                    ""
                } else if rng.below(3) == 0 {
                    "-"
                } else {
                    " "
                });
                let mut c = w.chars();
                text.extend(c.next().into_iter().flat_map(char::to_uppercase));
                text.push_str(c.as_str());
            }
        } else {
            text.push_str(FILLER[rng.below(FILLER.len())]);
        }
        text.push_str(if rng.below(8) == 0 { ", " } else { " " });
    }
    let (mut spans, mut types) = (Vec::new(), Vec::new());
    turbo_tokenize::word_tokens(&text, turbo_tokenize::WordTokenizer::DEFAULT_MAX_TOKEN_LENGTH, &mut spans, &mut types);
    let mut scratch = NameScratch::new();
    let mut found = Vec::new();
    let (mut heap_ms, mut mapped_ms) = (Vec::new(), Vec::new());
    let (mut heap_n, mut mapped_n) = (0, 0);
    for _ in 0..rounds {
        let t = Instant::now();
        heap_n = 0;
        for i in 0..spans.len() {
            for j in i..spans.len().min(i + longest) {
                if !g.lookup(&text[spans[i].start as usize..spans[j].end as usize]).is_empty() {
                    heap_n += 1;
                }
            }
        }
        heap_ms.push(ms(t));
        let t = Instant::now();
        found.clear();
        finder.find_all(&text, &mut scratch, &mut found);
        mapped_ms.push(ms(t));
        mapped_n = found.len();
    }
    assert_eq!(heap_n, mapped_n, "the two disagree");
    let mb = text.len() as f64 / (1 << 20) as f64;
    let (h, m) = (median(heap_ms), median(mapped_ms));
    println!(
        "gazetteer={name} keys={} longest_tokens={longest} heap_mb={:.1} image_mb={:.1} image_build_ms={image_ms:.0} \
         image_load_ms={:.2} text_mb={mb:.1} tokens={} names={mapped_n} lookup_windows_ms={h:.1} mapped_ms={m:.1} \
         mapped_mb_s={:.1}",
        finder.keys(),
        heap as f64 / (1 << 20) as f64,
        image_bytes.len() as f64 / (1 << 20) as f64,
        median(load),
        spans.len(),
        mb / m * 1e3,
    );
}

fn main() {
    let rounds = std::env::args().nth(2).map_or(5, |r| r.parse().unwrap());
    match std::env::args().nth(1) {
        Some(path) => {
            let t = Instant::now();
            let g = GeoNamesGazetteer::load(&path).unwrap();
            eprintln!("loaded {} records in {:.0} ms", g.len(), ms(t));
            let t = Instant::now();
            let image = g.name_image().unwrap();
            run("geonames", &g, image, ms(t), g.heap_bytes(), rounds);
        }
        None => {
            let g = BundledGazetteer::get();
            let t = Instant::now();
            let image = g.name_image().unwrap();
            run("bundled", g, image, ms(t), g.heap_bytes(), rounds);
        }
    }
}
