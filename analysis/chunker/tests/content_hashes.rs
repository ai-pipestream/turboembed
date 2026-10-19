//! The content-hash builders over a real pipeline: IDs are stable
//! across an unrelated edit elsewhere in the document, heading
//! paths come from the structure, and the levels builder hashes
//! every unit at its own kind.

use turbo_analysis::*;
use turbo_analysis_layers::{Block, Boundaries, Chunks, Cut, Level, Levels, Sentences, Structure, Subwords, Unit};
use turbo_chunker::levels::LevelBuilder;
use turbo_chunker::{Budget, ChunkId, Chunker, ContentHashBuilder, ContentHashes, LevelHashBuilder};

fn toy_model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

/// Two sections, each opened by a heading sentence, with a heading
/// block over the first sentence of each section.
fn document(text: &str, section_starts: &[usize]) -> Document {
    let mut doc = Document::new(text);
    let (mut sentences, mut pieces) = (Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE));
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if c == '.' {
            sentences.push(Span::new(start, i as u32 + 1));
            start = i as u32 + 2;
        }
    }
    let mut at = 0;
    for word in text.split(' ') {
        let w = word.trim_end_matches('.');
        pieces.push(Span::new(at, at + w.len() as u32));
        if w.len() < word.len() {
            pieces.push(Span::new(at + w.len() as u32, at + word.len() as u32));
        }
        at += word.len() as u32 + 1;
    }
    let ids = vec![0; pieces.len()];
    let n = sentences.len();
    let by = Provenance::by("t");
    let mut blocks: Vec<(Span, Block, Option<Ix<Structure>>)> = Vec::new();
    for (k, &s) in section_starts.iter().enumerate() {
        let end = section_starts.get(k + 1).copied().unwrap_or(n);
        let section_ix = Ix::new(blocks.len() as u32);
        let first = sentences.get(Ix::new(s as u32));
        let last = sentences.get(Ix::new(end as u32 - 1));
        blocks.push((Span::new(first.start, last.end), Block::Section, None));
        blocks.push((first, Block::Heading, Some(section_ix)));
    }
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(s, ..) in &blocks {
        spans.push(s);
    }
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, Subwords { spans: pieces, ids, model: toy_model(), token: None }, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, Boundaries { cost: vec![1000; n], reason: vec![Cut::Sentence; n] }, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Structure {
            spans,
            kind: blocks.iter().map(|b| b.1).collect(),
            parent: blocks.iter().map(|b| b.2).collect(),
            depth: vec![0; blocks.len()],
        },
        by,
    )
    .unwrap();
    doc
}

fn run(text: &str) -> Document {
    let mut docs = vec![document(text, &[0, 3])];
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Boundaries>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(Chunker { budget: Budget { max: 7, target: 6, size_weight: 1000, overhead: 0 } })
        .add(LevelBuilder)
        .add(ContentHashBuilder)
        .add(LevelHashBuilder)
        .build()
        .unwrap();
    assert!(analyzer.run(&mut docs)[0].is_ok());
    docs.pop().unwrap()
}

#[test]
fn ids_follow_level_heading_path_and_text() {
    let doc = run("Cats head. Cats purr. Cats nap. Rust head. Rust builds. Rust ships.");
    let chunks = doc.get::<Chunks>().unwrap();
    let hashes = doc.get::<ContentHashes<Chunks>>().unwrap();
    assert_eq!(hashes.len(), chunks.spans.len());

    // Every ID is the documented digest over its row's level,
    // heading path and span text.
    for (i, span) in chunks.spans.iter() {
        let level = if chunks.level[i.get()] == 0 { Level::Chunk } else { Level::Section };
        let heading = if span.start < 32 { "Cats head." } else { "Rust head." };
        assert_eq!(hashes.get(i), ChunkId::of(level, &[heading], &doc.source()[span.range()]), "row {}", i.get());
    }

    // Two chunks of identical text under different headings differ.
    let same_text_other_path = ChunkId::of(Level::Chunk, &["Rust head."], "Cats purr. Cats nap.");
    assert!(hashes.id.iter().all(|id| *id != same_text_other_path));

    // The levels: document, sections and chunks at their own kinds.
    let levels = doc.get::<Levels>().unwrap();
    let level_hashes = doc.get::<ContentHashes<Levels>>().unwrap();
    assert_eq!(level_hashes.len(), levels.spans.len());
    assert_eq!(level_hashes.get(Ix::new(0)), ChunkId::of(Level::Document, &[], doc.source()));
    for i in levels.of(Level::Chunk) {
        let Unit::Chunk(row) = levels.unit[i.get()] else { unreachable!() };
        assert_eq!(level_hashes.get(i), hashes.get(row), "a chunk unit hashes as its chunk row");
    }
}

#[test]
fn an_edit_elsewhere_keeps_untouched_ids() {
    let before = run("Cats head. Cats purr. Cats nap. Rust head. Rust builds. Rust ships.");
    let after = run("Cats head. Cats purr loudly. Cats nap. Rust head. Rust builds. Rust ships.");
    let ids = |d: &Document| d.get::<ContentHashes<Chunks>>().unwrap().id.clone();
    let (a, b) = (ids(&before), ids(&after));
    assert_eq!(a.len(), b.len());
    // The first section's parent and edited chunk changed; every
    // other ID, including the whole second section, is identical.
    let changed: Vec<usize> = a.iter().zip(&b).enumerate().filter(|(_, (x, y))| x != y).map(|(i, _)| i).collect();
    assert!(!changed.is_empty() && changed.iter().all(|&i| i < 3), "changed rows: {changed:?}");
}

#[test]
fn the_builder_hashes_discontinuous_fragments_not_envelopes() {
    let build = |text: &str, fragments: &[Span]| {
        let mut doc = Document::new(text);
        let by = Provenance::by("t");
        let mut sentences = Spans::new(ViewId::SOURCE);
        sentences.push(Span::new(0, text.len() as u32));
        doc.add(&Key::DEFAULT, Sentences { spans: sentences }, by.clone()).unwrap();
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push_fragments(fragments);
        doc.add(
            &Key::DEFAULT,
            Chunks {
                spans,
                level: vec![0],
                parent: vec![None],
                sentences: vec![IxRange::new(0, 1)],
                cut: vec![Cut::DocumentStart],
                cost: vec![0],
                model_tokens: vec![1],
                over_budget: vec![false],
            },
            by,
        )
        .unwrap();
        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Chunks>::DEFAULT)
            .add(ContentHashBuilder)
            .build()
            .unwrap();
        let mut docs = vec![doc];
        assert!(analyzer.run(&mut docs)[0].is_ok());
        docs.pop().unwrap().get::<ContentHashes<Chunks>>().unwrap().get(Ix::new(0))
    };
    let gap = [Span::new(0, 1), Span::new(2, 3)];
    assert_eq!(build("aXb.", &gap), build("aYb.", &gap), "a gap-only edit keeps the ID");
    assert_ne!(build("aXb.", &gap), build("aXb.", &[Span::new(0, 2), Span::new(3, 4)]));
}
