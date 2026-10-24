//! Local and contextual vectors for the chunk, section and document
//! levels, on the small BERT of testdata/tiny-bert-bundle on the CPU
//! backend. Every vector is held to the same unit pooled by hand: each
//! window built here from the unit's (or its parent's) pieces, each range
//! run alone in a session of its own, and parts weighted by their pieces.
//! Then the levels in windows shorter than a section, units with no
//! pieces, determinism, the two named instances and the provider.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo::turbo_token_range;
use turbo_analysis::registry::{Registry, Spec};
use turbo_analysis::*;
use turbo_analysis_embed::window::pieces_of;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{
    Block, Chunks, Cut, Level, Levels, Sentences, Structure, Subwords, Unit, VectorKind, Vectors,
};
use turbo_chunker::levels::LevelBuilder;

fn bert() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn open(max_seq: u32) -> Arc<VectorAnnotator> {
    let o =
        VectorOptions { encoder: EncoderOptions { max_seq, ..EncoderOptions::default() }, ..VectorOptions::default() };
    Arc::new(VectorAnnotator::open(&bert(), &o).unwrap_or_else(|e| panic!("{e}")))
}

const TEXT: &str = "Some notes before any heading.\n\n# Embedding\n\nEmbedding models turn a passage of text into a vector. \
Passages with similar meaning land near each other.\n\n## Search\n\nA search system embeds every document once. \
It stores the vectors in an index for later.\n\n# Queries\n\nEach query is embedded as it arrives.\n";

const SENTENCES: [&str; 9] = [
    "Some notes before any heading.",
    "# Embedding",
    "Embedding models turn a passage of text into a vector.",
    "Passages with similar meaning land near each other.",
    "## Search",
    "A search system embeds every document once.",
    "It stores the vectors in an index for later.",
    "# Queries",
    "Each query is embedded as it arrives.",
];

/// Where `s` is in `TEXT`, from byte `from` on.
fn find(s: &str, from: usize) -> Range<u32> {
    let a = from + TEXT[from..].find(s).unwrap();
    a as u32..(a + s.len()) as u32
}

/// The text with its sentences, chunks over `ranges` of them, and the
/// sections Embedding (with Search nested in it) and Queries when
/// `structure` is set; empty sections are added after the text.
fn document(ranges: &[(u32, u32)], structure: bool, empty_section: bool) -> Document {
    let mut d = Document::new(TEXT);
    let mut s = Spans::new(ViewId::SOURCE);
    let mut from = 0;
    for x in SENTENCES {
        let r = find(x, from);
        from = r.end as usize;
        s.push(Span::new(r.start, r.end));
    }
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
            Span::new(s.get(Ix::new(i)).start, s.get(Ix::new(i)).start)
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
    d.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("test")).unwrap();
    d.add(&Key::DEFAULT, c, Provenance::by("test")).unwrap();
    if structure {
        let embedding = find("# Embedding", 0).start..find("for later.", 0).end;
        let search = find("## Search", 0).start..embedding.end;
        let queries = find("# Queries", 0).start..TEXT.trim_end().len() as u32;
        let mut blocks = vec![(embedding, None), (search, Some(0)), (queries, None)];
        if empty_section {
            blocks.push((TEXT.len() as u32..TEXT.len() as u32, None));
        }
        let mut spans = Spans::new(ViewId::SOURCE);
        for (r, _) in &blocks {
            spans.push(Span::new(r.start, r.end));
        }
        let n = blocks.len();
        let st = Structure {
            spans,
            kind: vec![Block::Section; n],
            parent: blocks.iter().map(|b| b.1.map(Ix::new)).collect(),
            depth: blocks.iter().map(|b| 1 + u8::from(b.1.is_some())).collect(),
        };
        d.add(&Key::DEFAULT, st, Provenance::by("test")).unwrap();
    }
    d
}

/// One chunk per sentence run: the preamble, the Embedding text, the
/// Search text, the Queries text.
const CHUNKS: [(u32, u32); 4] = [(0, 1), (1, 4), (4, 7), (7, 9)];

/// The subwords, the levels, and the level vectors.
fn run(a: &Arc<VectorAnnotator>, docs: &mut [Document]) {
    let mut b =
        Analyzer::builder().given(Key::<View>::DEFAULT).given(Key::<Sentences>::DEFAULT).given(Key::<Chunks>::DEFAULT);
    if docs[0].get::<Structure>().is_some() {
        b = b.given(Key::<Structure>::DEFAULT);
    }
    let analyzer = b
        .add(SubwordAnnotator::open(&bert()).unwrap())
        .add(LevelBuilder)
        .step(LevelVectors(a.clone()).bound())
        .build()
        .unwrap();
    for r in analyzer.run(docs) {
        r.unwrap_or_else(|e| panic!("{e}"));
    }
}

fn vectors(d: &Document, kind: VectorKind) -> &Vectors<Levels> {
    d.get_key(&kind.key()).unwrap_or_else(|| panic!("{kind:?} vectors"))
}

/// Windows of at most `width` pieces over `[lo, hi)`, from `lo`.
fn tiles(lo: u32, hi: u32, width: u32) -> Vec<Range<u32>> {
    (lo..hi).step_by(width as usize).map(|a| a..(a + width).min(hi)).collect()
}

/// Every unit's vector of `kind`, pooled by hand: the windows are the
/// tiles of the unit's own pieces (local) or its parent's (contextual,
/// the document's own for the document), each range of the unit in a
/// tile run alone, the parts summed times their pieces over the unit's
/// pieces, then normalized.
fn by_hand(a: &VectorAnnotator, d: &Document, kind: VectorKind) -> Vec<f32> {
    let sub = d.get::<Subwords>().unwrap();
    let l = d.get::<Levels>().unwrap();
    let info = a.info();
    let mut enc = Encoder::open(
        &bert(),
        &EncoderOptions { max_seq: info.max_seq, max_batch: 1, max_ranges: 1, ..EncoderOptions::default() },
    )
    .unwrap();
    let dim = a.dim() as usize;
    let (starts, ends) = sub.spans.columns();
    let pieces = |i: Ix<Levels>| {
        let s = l.spans.get(i);
        pieces_of(starts, ends, s.start, s.end)
    };
    let (head, tail) = a.row_ends();
    let mut out = vec![0f32; l.len() * dim];
    for (i, _) in l.spans.iter() {
        let own = pieces(i);
        if own.is_empty() {
            continue;
        }
        let scope = match (kind, l.parent[i.get()]) {
            (VectorKind::Contextual, Some(p)) => pieces(p),
            _ => own.clone(),
        };
        let dst = &mut out[i.get() * dim..][..dim];
        for w in tiles(scope.start, scope.end, a.width()) {
            let (lo, hi) = (own.start.max(w.start), own.end.min(w.end));
            if lo >= hi {
                continue;
            }
            let mut rows = Rows::new(info.max_seq, 0);
            rows.push(&[head, &sub.ids[w.start as usize..w.end as usize], tail]);
            let h = head.len() as u32;
            let r = turbo_token_range { row: 0, start: h + lo - w.start, end: h + hi - w.start };
            let mut v = Vec::new();
            enc.run_ranges(&rows, &[r], &mut v).unwrap();
            for (x, y) in dst.iter_mut().zip(&v) {
                *x += y * (hi - lo) as f32;
            }
        }
        for x in dst.iter_mut() {
            *x /= own.len() as f32;
        }
        if info.l2 {
            normalize(dst);
        }
    }
    out
}

fn close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-5 * (1.0 + w.abs()), "{what}[{i}]: {g} vs {w}");
    }
}

fn differ(x: &[f32], y: &[f32]) -> bool {
    x.iter().zip(y).any(|(a, b)| (a - b).abs() > 1e-3)
}

#[test]
fn every_unit_gets_a_local_and_a_contextual_vector() {
    let a = open(0);
    let mut docs = [document(&CHUNKS, true, false)];
    run(&a, &mut docs);
    let d = &docs[0];
    let l = d.get::<Levels>().unwrap();
    // Document, preamble chunk, Embedding, its chunk, Search, its chunk,
    // Queries, its chunk.
    let levels: Vec<Level> = l.unit.iter().map(|u| u.level()).collect();
    use Level::*;
    assert_eq!(levels, [Document, Chunk, Section, Chunk, Section, Chunk, Section, Chunk]);
    assert!(a.width() >= 60, "the whole text fits one window");
    for kind in VectorKind::ALL {
        let v = vectors(d, kind);
        assert_eq!((v.dim, Layer::len(v), v.check()), (32, l.len(), Ok(())), "{kind:?}");
        assert!(v.empty.is_empty());
        close(&format!("{kind:?}"), &v.data, &by_hand(&a, d, kind));
    }
    let (local, contextual) = (vectors(d, VectorKind::Local), vectors(d, VectorKind::Contextual));
    let pieces = |i: u32| {
        let (s, e) = d.get::<Subwords>().unwrap().spans.columns();
        let span = l.spans.get(Ix::new(i));
        pieces_of(s, e, span.start, span.end)
    };
    // The document has no surroundings: both are the same.
    close("document", contextual.row(Ix::new(0)), local.row(Ix::new(0)));
    // A unit whose parent holds more than it reads that text too, and its
    // vector moves. The chunks of Search and of Queries hold every piece
    // of their sections, so the parent adds nothing and both are the same.
    for i in 1..l.len() as u32 {
        let parent = l.parent[i as usize].unwrap().get() as u32;
        let (x, y) = (local.row(Ix::new(i)), contextual.row(Ix::new(i)));
        if pieces(parent) == pieces(i) {
            assert!(matches!(i, 5 | 7), "unit {i}");
            close("a chunk that is its whole section", y, x);
        } else {
            assert!(differ(x, y), "unit {i}");
        }
    }
    // The section Search and its chunk hold the same pieces: alone they
    // are the same vector, in context they are not, as the section is
    // pooled inside Embedding.
    assert_eq!(pieces(4), pieces(5));
    close("Search alone", local.row(Ix::new(4)), local.row(Ix::new(5)));
    assert!(differ(contextual.row(Ix::new(4)), contextual.row(Ix::new(5))));
}

/// Rows of 24 tokens: the document and the Embedding section are longer
/// than a window, so their local vectors and their children's contextual
/// ones are pooled in parts.
#[test]
fn units_longer_than_a_window_are_pooled_in_parts() {
    let a = open(24);
    let mut docs = [document(&CHUNKS, true, false)];
    run(&a, &mut docs);
    let d = &docs[0];
    let sub = d.get::<Subwords>().unwrap();
    let l = d.get::<Levels>().unwrap();
    let (s, e) = sub.spans.columns();
    let width = a.width();
    let long: Vec<u32> = l
        .spans
        .iter()
        .filter(|(_, x)| pieces_of(s, e, x.start, x.end).len() as u32 > width)
        .map(|(i, _)| i.get() as u32)
        .collect();
    assert!(long.contains(&0) && long.contains(&2), "{long:?}, a window of {width}");
    for kind in VectorKind::ALL {
        close(&format!("{kind:?}"), &vectors(d, kind).data, &by_hand(&a, d, kind));
    }
}

#[test]
fn a_document_with_no_structure_has_chunks_under_the_document() {
    let a = open(0);
    let mut docs = [document(&CHUNKS, false, false)];
    run(&a, &mut docs);
    let d = &docs[0];
    let l = d.get::<Levels>().unwrap();
    assert_eq!(l.of(Level::Section).count(), 0);
    assert!(l.parent[1..].iter().all(|&p| p == Some(Ix::new(0))));
    for kind in VectorKind::ALL {
        close(&format!("{kind:?}"), &vectors(d, kind).data, &by_hand(&a, d, kind));
    }
    // No chunks at all: the document alone.
    let mut docs = [document(&[], false, false)];
    run(&a, &mut docs);
    let d = &docs[0];
    assert_eq!(d.get::<Levels>().unwrap().unit, [Unit::Document]);
    close(
        "document",
        vectors(d, VectorKind::Contextual).row(Ix::new(0)),
        vectors(d, VectorKind::Local).row(Ix::new(0)),
    );
}

#[test]
fn units_with_no_pieces_get_the_zero_vector() {
    let a = open(0);
    // An empty chunk in the Embedding section and an empty section at the
    // end of the text.
    let mut docs = [document(&[(0, 1), (1, 3), (3, 3), (3, 4), (4, 7), (7, 9)], true, true)];
    run(&a, &mut docs);
    let d = &docs[0];
    let l = d.get::<Levels>().unwrap();
    let empty: Vec<Ix<Levels>> = l.spans.iter().filter(|(_, s)| s.start == s.end).map(|(i, _)| i).collect();
    assert_eq!(empty.len(), 2);
    for kind in VectorKind::ALL {
        let v = vectors(d, kind);
        assert_eq!(v.empty, empty, "{kind:?}");
        assert_eq!(v.check(), Ok(()));
        for &i in &empty {
            assert!(v.row(i).iter().all(|&x| x == 0.0));
        }
        close(&format!("{kind:?}"), &v.data, &by_hand(&a, d, kind));
    }
}

#[test]
fn the_same_input_gives_the_same_bits() {
    let a = open(24);
    let mut first = [document(&CHUNKS, true, false)];
    let mut second = [document(&CHUNKS, true, false)];
    run(&a, &mut first);
    run(&open(24), &mut second);
    let bits = |d: &Document, k| vectors(d, k).data.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for kind in VectorKind::ALL {
        assert_eq!(bits(&first[0], kind), bits(&second[0], kind), "{kind:?}");
    }
}

#[test]
fn the_two_kinds_are_named_instances_with_provenance() {
    let a = open(0);
    let mut docs = [document(&CHUNKS, true, false)];
    run(&a, &mut docs);
    let d = &docs[0];
    assert!(d.get::<Vectors<Levels>>().is_none(), "no default instance");
    for kind in VectorKind::ALL {
        let p = d.provenance(kind.key().id()).unwrap();
        assert_eq!(p.annotator, "turbo:level-vectors");
        assert_eq!(p.model.as_ref(), Some(&d.get::<Subwords>().unwrap().model));
        assert_eq!(p.config, Annotator::config_hash(&LevelVectors(a.clone())));
    }
    // Both outputs under the default name are one instance twice.
    let refused = Analyzer::builder()
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Levels>::DEFAULT)
        .add(LevelVectors(a.clone()))
        .build();
    assert!(refused.is_err());
}

#[test]
fn the_provider_builds_the_level_step() {
    let mut r: Registry<dyn turbo_analysis::annotate::Step> = Registry::default();
    r.register(SubwordsProvider);
    r.register(VectorsProvider);
    let sub = Spec { location: Some(bert()), options: Vec::new() };
    let vec = Spec { location: Some(bert()), options: Vec::new() }.with("layers", "levels");
    let a = r.pipeline(&[("subwords", sub), ("vectors", vec)], |b| b.given(Key::<Levels>::DEFAULT)).unwrap();
    assert_eq!(a.plan(), vec![vec!["turbo:subwords"], vec!["turbo:level-vectors"]]);
}
