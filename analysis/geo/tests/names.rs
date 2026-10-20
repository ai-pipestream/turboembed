//! Gazetteer names found in text with the mapped key automaton, against
//! the in-heap gazetteer: every run of word tokens whose lookup finds a
//! record, and nothing else, with the same records.

use std::collections::BTreeSet;

use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_geo::{
    BundledGazetteer, EntryParts, FEATURE_CLASS_CITY, Folder, Gazetteer, GazetteerEntry, GeoPoint, InMemoryGazetteer,
    NameAnnotator, NameFinder, NameImageError, NameScratch,
};
use turbo_glossary::{Case, ImageBuf, ImageError, MappedGlossaryWriter};
use turbo_glossary_layers::{EntryIx, Glossary};

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

/// Every run of up to `longest` UAX #29 word tokens of `text` whose lookup
/// in `g` finds a record, with its key, by folding each run's text on its
/// own.
fn naive<G: Gazetteer>(g: &G, text: &str, longest: usize) -> BTreeSet<(Span, String)> {
    let (mut spans, mut types) = (Vec::new(), Vec::new());
    turbo_tokenize::word_tokens(text, turbo_tokenize::WordTokenizer::DEFAULT_MAX_TOKEN_LENGTH, &mut spans, &mut types);
    let mut folder = Folder::new();
    let mut out = BTreeSet::new();
    for i in 0..spans.len() {
        for j in i..spans.len().min(i + longest) {
            let span = Span::new(spans[i].start, spans[j].end);
            let name = &text[span.range()];
            if !g.lookup(name).is_empty() {
                out.insert((span, folder.fold(name)));
            }
        }
    }
    out
}

fn ids<G: Gazetteer>(g: &G, name: &str) -> Vec<String> {
    g.lookup(name).iter().map(|e| format!("{}/{}", e.source(), e.record_id())).collect()
}

/// Checks the finder on `text` against a naive search over `g`, and that
/// every match's key finds the records its text does.
fn same(g: &InMemoryOrBundled<'_>, finder: &NameFinder<'_>, text: &str, scratch: &mut NameScratch) -> usize {
    let mut found = Vec::new();
    finder.find_all(text, scratch, &mut found);
    let got: BTreeSet<(Span, String)> =
        found.iter().map(|m| (m.span, finder.key(m.key).unwrap().to_string())).collect();
    assert_eq!(got.len(), found.len(), "a match twice in {text:?}");
    assert_eq!(got, g.naive(text, 12), "in {text:?}");
    for m in &found {
        let key = finder.key(m.key).unwrap();
        assert_eq!(g.ids_of_key(key), g.ids(&text[m.span.range()]), "{key:?} in {text:?}");
    }
    // The non-overlapping selection: leftmost, then longest, from the full set.
    let mut want = Vec::new();
    let mut sorted = found.clone();
    sorted.sort_by_key(|m| (m.span.start, std::cmp::Reverse(m.span.end), m.key));
    let mut end = 0;
    for m in sorted {
        if want.is_empty() || m.span.start >= end {
            end = m.span.end;
            want.push(m);
        }
    }
    let mut kept = Vec::new();
    finder.find(text, scratch, &mut kept);
    assert_eq!(kept, want, "selection in {text:?}");
    found.len()
}

enum InMemoryOrBundled<'g> {
    Memory(&'g InMemoryGazetteer),
    Bundled(&'g BundledGazetteer),
}

impl InMemoryOrBundled<'_> {
    fn naive(&self, text: &str, longest: usize) -> BTreeSet<(Span, String)> {
        match self {
            InMemoryOrBundled::Memory(g) => naive(*g, text, longest),
            InMemoryOrBundled::Bundled(g) => naive(*g, text, longest),
        }
    }
    fn ids(&self, name: &str) -> Vec<String> {
        match self {
            InMemoryOrBundled::Memory(g) => ids(*g, name),
            InMemoryOrBundled::Bundled(g) => ids(*g, name),
        }
    }
    fn ids_of_key(&self, key: &str) -> Vec<String> {
        let entries = match self {
            InMemoryOrBundled::Memory(g) => g.lookup_key(key),
            InMemoryOrBundled::Bundled(g) => g.lookup_key(key),
        };
        entries.iter().map(|e| format!("{}/{}", e.source(), e.record_id())).collect()
    }
}

/// A name or a key written the way a text might: capitals, and hyphens,
/// commas or runs of spaces between its words.
fn written(rng: &mut Lcg, key: &str) -> String {
    let mut out = String::new();
    for (i, w) in key.split(' ').enumerate() {
        if i > 0 {
            out.push_str([" ", "-", ", ", "  ", " "][rng.below(5)]);
        }
        match rng.below(3) {
            0 => out.push_str(&w.to_uppercase()),
            1 => {
                let mut c = w.chars();
                out.extend(c.next().into_iter().flat_map(char::to_uppercase));
                out.push_str(c.as_str());
            }
            _ => out.push_str(w),
        }
    }
    out
}

fn image_buf(bytes: Vec<u8>) -> &'static ImageBuf {
    Box::leak(Box::new(ImageBuf::copy_from(&bytes)))
}

#[test]
fn the_bundled_names_are_found_as_the_gazetteer_looks_them_up() {
    let g = BundledGazetteer::get();
    let image = image_buf(g.name_image().unwrap());
    let finder = NameFinder::from_bytes(image.bytes()).unwrap();
    assert_eq!(finder.keys(), g.key_count());
    let mut rng = Lcg(3);
    let filler =
        ["the", "of", "in", "and", "to", "a", "city", "near", "river", "north", "é", "東", "\u{2014}", "St.", "2024"];
    let mut scratch = NameScratch::new();
    let mut found = 0;
    for _ in 0..300 {
        let mut text = String::new();
        for _ in 0..1 + rng.below(25) {
            let piece = if rng.below(3) == 0 {
                let k = EntryIx(rng.below(finder.keys()) as u32);
                written(&mut rng, finder.key(k).unwrap())
            } else {
                filler[rng.below(filler.len())].to_string()
            };
            text.push_str(&piece);
            text.push_str([" ", ", ", ". ", " ("][rng.below(4)]);
        }
        found += same(&InMemoryOrBundled::Bundled(g), &finder, &text, &mut scratch);
    }
    assert!(found > 1000, "{found} names");
}

fn place(id: usize, name: &str, alternates: Vec<String>) -> GazetteerEntry {
    GazetteerEntry::new(EntryParts {
        source: "test".into(),
        record_id: format!("r{id}"),
        name: name.into(),
        alternate_names: alternates,
        location: GeoPoint::new(0.0, 0.0).unwrap(),
        bounding_box: None,
        country_code: None,
        containment: Vec::new(),
        population: (id * 7 % 13) as i64,
        feature_class: Some(FEATURE_CLASS_CITY.into()),
        attributes: Vec::new(),
    })
    .unwrap()
}

#[test]
fn random_names_are_found_as_the_gazetteer_looks_them_up() {
    const PIECES: &[&str] = &[
        "São",
        "Paulo",
        "Zürich",
        "Łódź",
        "Straße",
        "ΑΘΗΝΑ",
        "東京",
        "New",
        "York",
        "St.",
        "John's",
        "e\u{301}",
        "Ærø",
    ];
    let mut rng = Lcg(17);
    let mut scratch = NameScratch::new();
    for _ in 0..150 {
        let name = |rng: &mut Lcg| -> String {
            let mut s = String::new();
            for i in 0..1 + rng.below(3) {
                if i > 0 {
                    s.push_str([" ", "-", " "][rng.below(3)]);
                }
                s.push_str(PIECES[rng.below(PIECES.len())]);
            }
            s
        };
        let entries: Vec<GazetteerEntry> = (0..1 + rng.below(20))
            .map(|i| {
                let alternates = (0..rng.below(3)).map(|_| name(&mut rng)).collect();
                place(i, &name(&mut rng), alternates)
            })
            .collect();
        let g = InMemoryGazetteer::from_entries(&entries).unwrap();
        let image = image_buf(g.name_image().unwrap());
        let finder = NameFinder::from_bytes(image.bytes()).unwrap();
        for _ in 0..5 {
            let mut text = String::new();
            for _ in 0..rng.below(14) {
                let n = name(&mut rng);
                text.push_str(&written(&mut rng, &n));
                text.push_str([" ", ", ", "; ", " - "][rng.below(4)]);
            }
            same(&InMemoryOrBundled::Memory(&g), &finder, &text, &mut scratch);
        }
    }
}

#[test]
fn bad_images_fail_to_load() {
    let g = BundledGazetteer::get();
    let bytes = g.name_image().unwrap();
    let cut = ImageBuf::copy_from(&bytes[..bytes.len() - 8]);
    assert!(matches!(NameFinder::from_bytes(cut.bytes()), Err(NameImageError::Image(ImageError::Length { .. }))));
    let mut flipped = bytes.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 1;
    let flipped = ImageBuf::copy_from(&flipped);
    assert_eq!(NameFinder::from_bytes(flipped.bytes()).err(), Some(NameImageError::Image(ImageError::Checksum)));
    let mut glossary = Glossary::new();
    glossary.push("0", "paris").unwrap();
    let folding = ImageBuf::copy_from(&MappedGlossaryWriter::write(&glossary, Case::Insensitive).unwrap());
    assert_eq!(NameFinder::from_bytes(folding.bytes()).err(), Some(NameImageError::Insensitive));
    // No truncation of a small image loads.
    let small = InMemoryGazetteer::from_entries(&[place(0, "Lyon", vec!["Lugdunum".into()])]).unwrap();
    let bytes = small.name_image().unwrap();
    for n in 0..bytes.len() {
        let b = ImageBuf::copy_from(&bytes[..n]);
        assert!(matches!(NameFinder::from_bytes(b.bytes()), Err(NameImageError::Image(_))), "{n}");
    }
}

#[test]
fn the_step_writes_location_entities_the_geocoder_reads() {
    let g = InMemoryGazetteer::from_entries(&[
        place(0, "New York", vec!["NYC".into()]),
        place(1, "York", Vec::new()),
        place(2, "São Paulo", Vec::new()),
    ])
    .unwrap();
    let finder = NameFinder::from_bytes(image_buf(g.name_image().unwrap()).bytes()).unwrap();
    let text = "From new-york to SAO PAULO, not York.";
    let mut doc = Document::new(text);
    let (mut spans, mut types) = (Vec::new(), Vec::new());
    turbo_tokenize::word_tokens(text, turbo_tokenize::WordTokenizer::DEFAULT_MAX_TOKEN_LENGTH, &mut spans, &mut types);
    let mut token_spans = Spans::new(ViewId::SOURCE);
    for s in spans {
        token_spans.push(s);
    }
    doc.add(&Key::DEFAULT, Tokens { spans: token_spans, sentence: None }, Provenance::by("test")).unwrap();
    let analyzer = Analyzer::builder().add(NameAnnotator::new(finder)).given(Key::<Tokens>::DEFAULT).build().unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();
    let entities = doc.get::<Entities>().unwrap();
    let got: Vec<(&str, &str, usize)> = (0..entities.spans.len())
        .map(|i| {
            let ix = turbo_analysis::Ix::new(i as u32);
            let s = entities.spans.get(ix);
            (&text[s.range()], entities.kinds.label(entities.kind[i]), entities.tokens[i].len())
        })
        .collect();
    assert_eq!(got, [("new-york", "location", 2), ("SAO PAULO", "location", 2), ("York", "location", 1)]);
}
