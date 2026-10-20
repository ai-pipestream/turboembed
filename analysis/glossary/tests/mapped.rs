//! The mapped matcher against the in-heap one: the same hits in the same
//! order, and the same matches, on random glossaries and texts and on the
//! JVM goldens' glossary and inputs.

use std::borrow::Cow;

use turbo_analysis::Span;
use turbo_glossary::{
    Boundaries, Case, DelimitedReader, FirstRecord, Glossary, GlossaryMatch, ImageBuf, ImageError,
    MappedGlossaryMatcher, MappedGlossaryWriter, Scratch, TextMatcher,
};
use turbo_glossary_layers::EntryIx;

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

fn hits_heap(m: &TextMatcher, text: &str) -> Vec<(Span, EntryIx)> {
    let mut out = Vec::new();
    m.scan(text, |s, e| out.push((s, e)));
    out
}

fn hits_mapped(m: &MappedGlossaryMatcher<'_>, text: &str, scratch: &mut Scratch) -> Vec<(Span, EntryIx)> {
    let mut out = Vec::new();
    m.scan(text, scratch, |s, e| out.push((s, e)));
    out
}

fn words(text: &str) -> Boundaries {
    let mut b = Vec::new();
    turbo_tokenize::word_boundaries(text, &mut b);
    Boundaries::from_offsets(text.len() as u32, b)
}

fn every(text: &str) -> Boundaries {
    Boundaries::from_offsets(text.len() as u32, 0..=text.len() as u32)
}

/// Checks both matchers on `text`, hits in order and matches.
fn same(heap: &TextMatcher, mapped: &MappedGlossaryMatcher<'_>, text: &str, scratch: &mut Scratch) {
    assert_eq!(hits_mapped(mapped, text, scratch), hits_heap(heap, text), "hits in {text:?}");
    for b in [words(text), every(text)] {
        let (mut want, mut got) = (Vec::<GlossaryMatch>::new(), Vec::new());
        heap.find(text, &b, &mut want);
        mapped.find(text, &b, scratch, &mut got);
        assert_eq!(got, want, "matches in {text:?}");
    }
}

#[test]
fn random_glossaries_match_as_the_in_heap_matcher_does() {
    const ALPHABET: &[&str] = &["a", "b", "A", "ab", " ", "é", "É", "東", "🙂", "İ", "i", "ß", "-", "Σ", "ς", "σ"];
    let mut rng = Lcg(11);
    let mut scratch = Scratch::new();
    for round in 0..600 {
        let mut g = Glossary::new();
        for i in 0..1 + rng.below(30) {
            let term: String = (0..1 + rng.below(5)).map(|_| ALPHABET[rng.below(ALPHABET.len())]).collect();
            if !term.trim().is_empty() {
                g.push(&format!("t{i}"), &term).unwrap();
            }
        }
        if g.is_empty() {
            continue;
        }
        let text: String = (0..rng.below(80)).map(|_| ALPHABET[rng.below(ALPHABET.len())]).collect();
        for case in [Case::Sensitive, Case::Insensitive] {
            let heap = TextMatcher::new(&g, case).unwrap();
            let image = ImageBuf::copy_from(&MappedGlossaryWriter::write(&g, case).unwrap());
            let mapped = MappedGlossaryMatcher::from_bytes(image.bytes()).unwrap();
            assert_eq!(
                (mapped.case(), mapped.states(), mapped.entries()),
                (case, heap.states(), g.len()),
                "round {round}"
            );
            same(&heap, &mapped, &text, &mut scratch);
            for (ix, e) in g.iter() {
                assert_eq!((mapped.id(ix), mapped.term(ix)), (Some(e.id), Some(e.term)));
            }
            assert_eq!(mapped.id(EntryIx(g.len() as u32)), None);
        }
    }
}

#[test]
fn the_goldens_glossary_matches_their_inputs_as_the_in_heap_matcher_does() {
    let data = turbo_goldens::data_dir();
    let bytes = std::fs::read(data.join("fixtures").join("glossary-terms.tsv")).unwrap();
    let g = DelimitedReader::tsv(FirstRecord::Header).read(&bytes).unwrap().glossary;
    let dir = data.join("inputs").join("glossary");
    let mut texts: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with("text-"))
        .map(|p| std::fs::read_to_string(p).unwrap())
        .collect();
    texts.sort();
    assert!(texts.len() >= 3, "{} texts", texts.len());
    let mut scratch = Scratch::new();
    let fold = |t: &str| -> Cow<'_, str> { Cow::Owned(t.split_whitespace().collect::<Vec<_>>().join(" ")) };
    for case in [Case::Sensitive, Case::Insensitive] {
        let heap = TextMatcher::new(&g, case).unwrap();
        let image = ImageBuf::copy_from(&MappedGlossaryWriter::write(&g, case).unwrap());
        let mapped = MappedGlossaryMatcher::from_bytes(image.bytes()).unwrap();
        let folded = TextMatcher::with_normalizer(&g, case, fold).unwrap();
        let folded_image = ImageBuf::copy_from(&MappedGlossaryWriter::write_normalized(&g, case, fold).unwrap());
        let folded_mapped = MappedGlossaryMatcher::from_bytes(folded_image.bytes()).unwrap();
        assert_eq!(folded_mapped.fingerprint(), u64::from_le_bytes(folded_image.bytes()[56..64].try_into().unwrap()));
        let mut found = 0;
        for text in &texts {
            same(&heap, &mapped, text, &mut scratch);
            same(&folded, &folded_mapped, text, &mut scratch);
            found += hits_heap(&heap, text).len();
        }
        assert!(found > 20, "{found} hits");
    }
}

#[test]
fn an_image_is_the_same_bytes_every_time() {
    let mut g = Glossary::new();
    for (id, term) in [("1", "New York"), ("2", "York"), ("3", "東京"), ("4", "ork")] {
        g.push(id, term).unwrap();
    }
    let a = MappedGlossaryWriter::write(&g, Case::Insensitive).unwrap();
    assert_eq!(a, MappedGlossaryWriter::write(&g, Case::Insensitive).unwrap());
    assert_ne!(a, MappedGlossaryWriter::write(&g, Case::Sensitive).unwrap());
    assert!(a.len().is_multiple_of(8));
    let path = std::env::temp_dir().join(format!("turbo-glossary-{}.tglac", std::process::id()));
    std::fs::write(&path, &a).unwrap();
    let read = ImageBuf::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(read.bytes(), &a[..]);
    let m = MappedGlossaryMatcher::from_bytes(read.bytes()).unwrap();
    let mut scratch = Scratch::new();
    let hits = hits_mapped(&m, "NEW YORK", &mut scratch);
    assert_eq!(hits.iter().map(|h| (h.0.start, h.0.end, h.1.0)).collect::<Vec<_>>(), [(0, 8, 0), (4, 8, 1), (5, 8, 3)]);
}

#[test]
fn a_slice_that_does_not_start_at_a_multiple_of_four_is_refused() {
    let mut g = Glossary::new();
    g.push("1", "cat").unwrap();
    let image = MappedGlossaryWriter::write(&g, Case::Sensitive).unwrap();
    let mut shifted = vec![0u64; image.len() / 8 + 1];
    // SAFETY: the buffer holds image.len() + 8 bytes.
    let bytes = unsafe { std::slice::from_raw_parts_mut(shifted.as_mut_ptr().cast::<u8>(), image.len() + 8) };
    bytes[1..1 + image.len()].copy_from_slice(&image);
    assert_eq!(MappedGlossaryMatcher::from_bytes(&bytes[1..1 + image.len()]).err(), Some(ImageError::Misaligned));
    bytes[4..4 + image.len()].copy_from_slice(&image);
    assert!(MappedGlossaryMatcher::from_bytes(&bytes[4..4 + image.len()]).is_ok());
}

#[cfg(feature = "mmap")]
#[test]
fn a_mapped_file_matches_as_the_in_heap_matcher_does() {
    let mut g = Glossary::new();
    for (id, term) in [("1", "hot dog"), ("2", "dog"), ("3", "Größe")] {
        g.push(id, term).unwrap();
    }
    let path = std::env::temp_dir().join(format!("turbo-glossary-map-{}.tglac", std::process::id()));
    std::fs::write(&path, MappedGlossaryWriter::write(&g, Case::Insensitive).unwrap()).unwrap();
    // SAFETY: the file is this test's own and is not changed while mapped.
    let file = unsafe { turbo_glossary::MappedFile::open(&path) }.unwrap();
    let m = MappedGlossaryMatcher::from_bytes(file.bytes()).unwrap();
    let heap = TextMatcher::new(&g, Case::Insensitive).unwrap();
    same(&heap, &m, "a HOT DOG of GRÖSSE and größe", &mut Scratch::new());
    drop(file);
    std::fs::remove_file(&path).unwrap();
}
