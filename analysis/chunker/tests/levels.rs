//! The chunk, section and document levels: from Markdown through the
//! structure detector, the boundary costs and the chunker, and from
//! hand-made chunks and sections for the cases the chunker does not make
//! on its own. Every unit is checked by its exact span, read back as text.

use turbo_analysis::*;
use turbo_analysis_layers::{Block, Chunks, Cut, Level, Levels, Sentences, Structure, Subwords, Tokens, Unit};
use turbo_chunker::levels::LevelBuilder;
use turbo_chunker::terms::{TermCounter, TermIndexer, TermStats, Vocabulary};
use turbo_chunker::{Budget, Chunker, Costs, StructureBoundaries};
use turbo_structure::{StructureAnnotator, Syntax};

fn toy_model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

/// The words of `text` and the periods after them, each a span.
fn words(text: &str) -> Spans<Tokens> {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = None;
    for (i, c) in text.char_indices().chain(std::iter::once((text.len(), ' '))) {
        let i = i as u32;
        match (at, c.is_whitespace() || c == '.') {
            (Some(a), true) => {
                spans.push(Span::new(a, i));
                at = None;
            }
            (None, false) => at = Some(i),
            _ => {}
        }
        if c == '.' {
            spans.push(Span::new(i, i + 1));
        }
    }
    spans
}

/// A document whose sentences are `sentences`, found in `text` in order,
/// with a subword and a token for every word and every period.
fn document(text: &str, sentences: &[&str]) -> Document {
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut from = 0;
    for s in sentences {
        let a = from + text[from..].find(s).unwrap_or_else(|| panic!("{s:?} is not in the text"));
        spans.push(Span::new(a as u32, (a + s.len()) as u32));
        from = a + s.len();
    }
    let w = words(text);
    let n = w.len();
    let mut pieces = Spans::new(ViewId::SOURCE);
    for (_, s) in w.iter() {
        pieces.push(s);
    }
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("t")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Subwords { spans: pieces, ids: vec![0; n], model: toy_model(), token: None },
        Provenance::by("t"),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: w, sentence: None }, Provenance::by("t")).unwrap();
    doc
}

/// Structure from Markdown, boundary costs, the chunker, the levels and
/// their term counts.
fn pipeline(max: u32) -> Analyzer {
    Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(StructureAnnotator::new(Syntax::Markdown))
        .add(StructureBoundaries { costs: Costs::default() })
        .add(Chunker { budget: Budget { max, target: max, size_weight: 1000, overhead: 0 } })
        .add(LevelBuilder)
        .add(TermIndexer)
        .add(TermCounter::<Levels>::new())
        .add(TermCounter::<View>::new())
        .build()
        .unwrap()
}

/// The levels from hand-made chunks and, when given, structure.
fn levels_of(mut doc: Document, chunks: Chunks, structure: Option<Structure>) -> Result<Document, Error> {
    doc.add(&Key::DEFAULT, chunks, Provenance::by("t")).unwrap();
    let mut b =
        Analyzer::builder().given(Key::<View>::DEFAULT).given(Key::<Sentences>::DEFAULT).given(Key::<Chunks>::DEFAULT);
    if let Some(st) = structure {
        doc.add(&Key::DEFAULT, st, Provenance::by("t")).unwrap();
        b = b.given(Key::<Structure>::DEFAULT);
    }
    let mut docs = vec![doc];
    let status = b.add(LevelBuilder).build().unwrap().run(&mut docs);
    status.into_iter().next().unwrap().map(|_| docs.pop().unwrap())
}

fn run(text: &str, sentences: &[&str], max: u32) -> Document {
    let mut docs = vec![document(text, sentences)];
    let status = pipeline(max).run(&mut docs);
    assert_eq!(status[0], Ok(()));
    docs.pop().unwrap()
}

/// Each unit as (level, its text, its parent's row).
fn rows(doc: &Document) -> Vec<(Level, &str, Option<u32>)> {
    let l = doc.get::<Levels>().unwrap();
    l.spans
        .iter()
        .map(|(i, s)| (l.level(i), &doc.source()[s.range()], l.parent[i.get()].map(|p| p.get() as u32)))
        .collect()
}

/// Chunks at level 0 over sentence ranges, spans from the first
/// sentence's start to the last one's end.
fn chunks(doc: &Document, ranges: &[(u32, u32)]) -> Chunks {
    let s = &doc.get::<Sentences>().unwrap().spans;
    let mut c = Chunks {
        spans: Spans::new(ViewId::SOURCE),
        level: Vec::new(),
        parent: Vec::new(),
        sentences: Vec::new(),
        cut: Vec::new(),
        cost: Vec::new(),
        model_tokens: Vec::new(),
        over_budget: Vec::new(),
        prompt_overhead: 0,
    };
    for &(i, j) in ranges {
        let span = if i < j {
            Span::new(s.get(Ix::new(i)).start, s.get(Ix::new(j - 1)).end)
        } else {
            let at = s.get(Ix::new(i)).start;
            Span::new(at, at)
        };
        c.spans.push(span);
        c.level.push(0);
        c.parent.push(None);
        c.sentences.push(IxRange::new(i, j));
        c.cut.push(Cut::Sentence);
        c.cost.push(0);
        c.model_tokens.push(0);
        c.over_budget.push(false);
    }
    c
}

/// Sections from (start, end, parent), as byte offsets.
fn sections(list: &[(u32, u32, Option<u32>)]) -> Structure {
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b, _) in list {
        spans.push(Span::new(a, b));
    }
    Structure {
        spans,
        kind: vec![Block::Section; list.len()],
        parent: list.iter().map(|l| l.2.map(Ix::new)).collect(),
        depth: vec![1; list.len()],
    }
}

const MARKDOWN: &str = "Notes first. More notes.\n\n# Cats\n\nCats purr. Cats nap.\n\n## Kittens\n\nKittens play. Kittens sleep.\n\n# Rust\n\nRust compiles.\n";

const MARKDOWN_SENTENCES: [&str; 10] = [
    "Notes first.",
    "More notes.",
    "# Cats",
    "Cats purr.",
    "Cats nap.",
    "## Kittens",
    "Kittens play.",
    "Kittens sleep.",
    "# Rust",
    "Rust compiles.",
];

#[test]
fn markdown_gives_every_level_with_exact_offsets() {
    let doc = run(MARKDOWN, &MARKDOWN_SENTENCES, 7);
    use Level::*;
    assert_eq!(
        rows(&doc),
        [
            (Document, MARKDOWN, None),
            // Text before the first heading is the document's.
            (Chunk, "Notes first. More notes.", Some(0)),
            (Section, "# Cats\n\nCats purr. Cats nap.\n\n## Kittens\n\nKittens play. Kittens sleep.", Some(0)),
            (Chunk, "# Cats\n\nCats purr.", Some(2)),
            (Chunk, "Cats nap.", Some(2)),
            (Section, "## Kittens\n\nKittens play. Kittens sleep.", Some(2)),
            (Chunk, "## Kittens\n\nKittens play.", Some(5)),
            (Chunk, "Kittens sleep.", Some(5)),
            (Section, "# Rust\n\nRust compiles.", Some(0)),
            (Chunk, "# Rust\n\nRust compiles.", Some(8)),
        ]
    );
    let l = doc.get::<Levels>().unwrap();
    assert_eq!(l.check(), Ok(()));
    // Each unit names the row it comes from.
    let st = doc.get::<Structure>().unwrap();
    let ch = doc.get::<Chunks>().unwrap();
    for (i, s) in l.spans.iter() {
        match l.unit[i.get()] {
            Unit::Document => assert_eq!((s.start, s.end), (0, MARKDOWN.len() as u32)),
            Unit::Section(b) => {
                assert_eq!(st.kind[b.get()], Block::Section);
                assert_eq!(st.spans.get(b), s);
            }
            Unit::Chunk(c) => {
                assert_eq!(ch.level[c.get()], 0);
                assert_eq!(ch.spans.get(c), s);
            }
        }
    }
    assert_eq!(l.of(Level::Chunk).count(), ch.level.iter().filter(|&&x| x == 0).count());
    // The provenance says the structure was read.
    let reads = &doc.provenance(Key::<Levels>::DEFAULT.id()).unwrap().reads;
    assert_eq!(reads.last(), Some(&Key::<Structure>::DEFAULT.id()));
}

#[test]
fn every_level_gets_term_counts() {
    let doc = run(MARKDOWN, &MARKDOWN_SENTENCES, 7);
    let l = doc.get::<Levels>().unwrap();
    let stats = doc.get::<TermStats<Levels>>().unwrap();
    let v = doc.get::<Vocabulary>().unwrap();
    assert_eq!(stats.len(), l.len());
    let tf = |i: u32, t: &str| stats.tf(Ix::new(i), v.lookup(t).unwrap());
    // The document's row is the view's own count.
    let whole = doc.get::<TermStats<View>>().unwrap();
    assert_eq!(stats.unit(Ix::new(0)), whole.unit(Ix::new(0)));
    assert_eq!(stats.length[0], whole.length[0]);
    // "Kittens" three times in the section around it and in its own
    // section, twice in the first chunk of that; not in the preamble.
    assert_eq!([tf(2, "Kittens"), tf(5, "Kittens"), tf(6, "Kittens"), tf(1, "Kittens")], [3, 3, 2, 0]);
    assert_eq!([tf(0, "Rust"), tf(8, "Rust"), tf(9, "Rust"), tf(2, "Rust")], [2, 2, 2, 0]);
    // A term's df counts the units of every level that hold it.
    assert_eq!(stats.df[v.lookup("Cats").unwrap().get()], 4);
    assert_eq!(stats.check(), Ok(()));
}

#[test]
fn without_structure_chunks_hang_on_the_document() {
    let text = "Cats purr. Cats nap. Dogs bark.";
    let doc = document(text, &["Cats purr.", "Cats nap.", "Dogs bark."]);
    let c = chunks(&doc, &[(0, 2), (2, 3)]);
    let doc = levels_of(doc, c, None).unwrap();
    assert_eq!(
        rows(&doc),
        [
            (Level::Document, text, None),
            (Level::Chunk, "Cats purr. Cats nap.", Some(0)),
            (Level::Chunk, "Dogs bark.", Some(0))
        ]
    );
    // No section points into a structure, so none is indexed.
    assert_eq!(doc.get::<Levels>().unwrap().indexes().len(), 1);

    // No chunks and no structure: the document alone, whitespace and all.
    let text = "  \n";
    let doc = levels_of(document(text, &[]), chunks(&document(text, &[]), &[]), None).unwrap();
    assert_eq!(rows(&doc), [(Level::Document, text, None)]);
    let doc = levels_of(document("", &[]), chunks(&document("", &[]), &[]), None).unwrap();
    assert_eq!(rows(&doc), [(Level::Document, "", None)]);
}

/// Hand-made: a section with two nested ones, the chunks of the outer
/// one cut across the edge between the nested ones, and a chunk after
/// every section.
#[test]
fn a_chunk_across_a_nested_section_belongs_to_the_section_around_both() {
    //          0         1         2         3         4         5
    //          0123456789012345678901234567890123456789012345678901234
    let text = "A one. A two. B one. B two. C one. Tail.";
    let doc = document(text, &["A one.", "A two.", "B one.", "B two.", "C one.", "Tail."]);
    // Outer 0..34 holds A (0..13) and B (14..27); C is outer's own text.
    let st = sections(&[(0, 34, None), (0, 13, Some(0)), (14, 27, Some(0))]);
    // Chunks: "A one. A two. B one." crosses A's end; "B two. C one."
    // crosses B's end; "Tail." is outside every section.
    let c = chunks(&doc, &[(0, 3), (3, 5), (5, 6)]);
    let doc = levels_of(doc, c, Some(st)).unwrap();
    use Level::*;
    assert_eq!(
        rows(&doc),
        [
            (Document, text, None),
            (Section, "A one. A two. B one. B two. C one.", Some(0)),
            (Section, "A one. A two.", Some(1)),
            (Chunk, "A one. A two. B one.", Some(1)),
            (Section, "B one. B two.", Some(1)),
            (Chunk, "B two. C one.", Some(1)),
            (Chunk, "Tail.", Some(0)),
        ]
    );
    // The chunk and the nested section beside it share bytes; the layer
    // allows that between siblings, never between a unit and its parent.
    assert_eq!(doc.get::<Levels>().unwrap().check(), Ok(()));
}

#[test]
fn a_chunk_inside_a_nested_section_belongs_to_it() {
    let text = "A one. A two. B one. B two.";
    let doc = document(text, &["A one.", "A two.", "B one.", "B two."]);
    // Three levels of sections: outer, A and B in it, and B.1 in B.
    let st = sections(&[(0, 27, None), (0, 13, Some(0)), (14, 27, Some(0)), (21, 27, Some(2))]);
    let c = chunks(&doc, &[(0, 2), (2, 3), (3, 4)]);
    let doc = levels_of(doc, c, Some(st)).unwrap();
    use Level::*;
    assert_eq!(
        rows(&doc),
        [
            (Document, text, None),
            (Section, text, Some(0)),
            (Section, "A one. A two.", Some(1)),
            (Chunk, "A one. A two.", Some(2)),
            (Section, "B one. B two.", Some(1)),
            (Chunk, "B one.", Some(4)),
            (Section, "B two.", Some(4)),
            (Chunk, "B two.", Some(6)),
        ]
    );
}

#[test]
fn a_sentence_past_a_sections_end_widens_it() {
    // The section ends inside "B one.", which starts in it: the chunker
    // gives that sentence to the section, and the section's span grows to
    // the chunk's end. The next section keeps its own start.
    let text = "A one. B one. C one.";
    let doc = document(text, &["A one.", "B one.", "C one."]);
    let st = sections(&[(0, 10, None), (10, 20, None)]);
    let c = chunks(&doc, &[(0, 2), (2, 3)]);
    let doc = levels_of(doc, c, Some(st)).unwrap();
    let l = doc.get::<Levels>().unwrap();
    let spans: Vec<(u32, u32)> = l.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
    assert_eq!(spans, [(0, 20), (0, 13), (0, 13), (10, 20), (14, 20)]);
    assert_eq!(
        l.parent.iter().map(|p| p.map(|p| p.get())).collect::<Vec<_>>(),
        [None, Some(0), Some(1), Some(0), Some(3)]
    );
    assert_eq!(l.check(), Ok(()));
}

#[test]
fn empty_units_keep_their_place() {
    // An empty section at the end of the text and an empty chunk (no
    // sentences) inside the first section.
    let text = "A one. A two.";
    let doc = document(text, &["A one.", "A two."]);
    let st = sections(&[(0, 13, None), (13, 13, None)]);
    let c = chunks(&doc, &[(0, 1), (1, 1), (1, 2)]);
    let doc = levels_of(doc, c, Some(st)).unwrap();
    use Level::*;
    assert_eq!(
        rows(&doc),
        [
            (Document, text, None),
            (Section, text, Some(0)),
            (Chunk, "A one.", Some(1)),
            (Chunk, "", Some(1)),
            (Chunk, "A two.", Some(1)),
            (Section, "", Some(0)),
        ]
    );
    let l = doc.get::<Levels>().unwrap();
    assert_eq!((l.spans.get(Ix::new(3)).start, l.spans.get(Ix::new(5)).start), (7, 13));
}

#[test]
fn the_same_layers_give_the_same_levels() {
    let a = run(MARKDOWN, &MARKDOWN_SENTENCES, 7);
    let b = run(MARKDOWN, &MARKDOWN_SENTENCES, 7);
    let (x, y) = (a.get::<Levels>().unwrap(), b.get::<Levels>().unwrap());
    assert_eq!(x.spans.columns(), y.spans.columns());
    assert_eq!((&x.unit, &x.parent), (&y.unit, &y.parent));
    let (s, t) = (a.get::<TermStats<Levels>>().unwrap(), b.get::<TermStats<Levels>>().unwrap());
    assert_eq!((&s.row, &s.term, &s.tf, &s.length, &s.df), (&t.row, &t.term, &t.tf, &t.length, &t.df));
}

#[test]
fn malformed_chunks_fail_the_document() {
    let text = "A one. A two.";
    let doc = document(text, &["A one.", "A two."]);
    // Overlapping chunks.
    let c = chunks(&doc, &[(0, 2), (1, 2)]);
    let e = levels_of(doc.clone(), c, None).err().unwrap();
    assert!(e.to_string().contains("starts before the chunk before it ends"), "{e}");
    // A chunk over sentences that are not there.
    let mut c = chunks(&doc, &[(0, 2)]);
    c.sentences[0] = IxRange::new(0, 3);
    // The document refuses the chunks before the builder sees them.
    let mut d = doc.clone();
    assert!(d.add(&Key::DEFAULT, c, Provenance::by("t")).is_err());
    // Rows the chunker puts above the chunks are not repeated.
    let mut c = chunks(&doc, &[(0, 2), (0, 1), (1, 2)]);
    c.level[0] = 1;
    c.parent[1] = Some(Ix::new(0));
    c.parent[2] = Some(Ix::new(0));
    let d = levels_of(doc, c, None).unwrap();
    let l = d.get::<Levels>().unwrap();
    assert_eq!(l.unit, [Unit::Document, Unit::Chunk(Ix::new(1)), Unit::Chunk(Ix::new(2))]);
}

#[test]
fn a_nested_sections_span_does_not_depend_on_the_chunking() {
    // N ends inside "B one.", which starts in it: N runs to that
    // sentence's end whether or not a chunk of N's own holds it.
    let text = "A one. B one. C one.";
    let doc = document(text, &["A one.", "B one.", "C one."]);
    let st = || sections(&[(0, 20, None), (0, 10, Some(0))]);
    let section_texts = |ranges: &[(u32, u32)]| {
        let d = levels_of(doc.clone(), chunks(&doc, ranges), Some(st())).unwrap();
        let l = d.get::<Levels>().unwrap();
        assert_eq!(l.check(), Ok(()));
        let texts: Vec<String> = l.of(Level::Section).map(|i| text[l.spans.get(i).range()].to_owned()).collect();
        texts
    };
    let one = section_texts(&[(0, 3)]);
    assert_eq!(one, [text, "A one. B one."]);
    assert_eq!(section_texts(&[(0, 2), (2, 3)]), one);
    assert_eq!(section_texts(&[(0, 1), (1, 2), (2, 3)]), one);
}
