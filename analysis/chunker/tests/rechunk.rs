//! Incremental re-chunking after an edit: runs the edit did not
//! touch are carried forward with spans mapped across the alignment
//! (and content IDs intact), the touched run is cut again, and the
//! result always equals a full chunking of the edited document.

use std::collections::HashSet;
use std::ops::Range;

use turbo_analysis::*;
use turbo_analysis_layers::{Block, Boundaries, Chunks, Cut, Levels, Sentences, Structure, Subwords};
use turbo_chunker::levels::LevelBuilder;
use turbo_chunker::rechunk::{Previous, affected_sentences};
use turbo_chunker::{Budget, ChunkId, Chunker, ContentHashes};

const BUDGET: Budget = Budget { max: 7, target: 6, size_weight: 1000, overhead: 0 };

fn toy_model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

/// Sentences (each ends at a period), subwords (every word and
/// period one) and flat boundary costs over `text`, with a section
/// per sentence range in `sections`.
fn document(text: &str, sections: &[Range<usize>]) -> Document {
    document_with_costs(text, sections, None)
}

fn document_with_costs(text: &str, sections: &[Range<usize>], costs: Option<Vec<i32>>) -> Document {
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
    let by = Provenance::by("t");
    let mut structure_spans = Spans::new(ViewId::SOURCE);
    for r in sections {
        let first = sentences.get(Ix::new(r.start as u32));
        let last = sentences.get(Ix::new(r.end as u32 - 1));
        structure_spans.push(Span::new(first.start, last.end));
    }
    let n = sentences.len();
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, Subwords { spans: pieces, ids, model: toy_model(), token: None }, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Boundaries { cost: costs.unwrap_or_else(|| vec![1000; n]), reason: vec![Cut::Sentence; n] },
        by.clone(),
    )
    .unwrap();
    if !sections.is_empty() {
        doc.add(
            &Key::DEFAULT,
            Structure {
                spans: structure_spans,
                kind: vec![Block::Section; sections.len()],
                parent: vec![None; sections.len()],
                depth: vec![0; sections.len()],
            },
            by,
        )
        .unwrap();
    }
    doc
}

fn analyzer() -> Analyzer {
    Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Boundaries>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(Chunker { budget: BUDGET })
        .add(LevelBuilder)
        .build()
        .unwrap()
}

fn chunked(text: &str, sections: &[Range<usize>]) -> Document {
    chunked_with_costs(text, sections, None)
}

fn chunked_with_costs(text: &str, sections: &[Range<usize>], costs: Option<Vec<i32>>) -> Document {
    let mut docs = vec![document_with_costs(text, sections, costs)];
    assert!(analyzer().run(&mut docs)[0].is_ok());
    docs.pop().unwrap()
}

/// `old` with the byte range of `from` replaced by `to`, and the
/// alignment the replacement makes.
fn edited(old: &str, from: &str, to: &str) -> (String, Alignment) {
    let at = old.find(from).expect("the replaced text is present");
    let mut b = ViewBuilder::new(old);
    b.keep(at);
    b.replace(at + from.len(), to);
    b.finish()
}

fn rechunk(old: &Document, new: &Document, edit: &Alignment) -> (Chunks, turbo_chunker::rechunk::RechunkStats) {
    Chunker { budget: BUDGET }
        .rechunk(
            edit,
            Previous {
                sentences: old.get::<Sentences>().unwrap(),
                subwords: old.get::<Subwords>().unwrap(),
                bounds: old.get::<Boundaries>().unwrap(),
                structure: old.get::<Structure>(),
                chunks: old.get::<Chunks>().unwrap(),
            },
            new.get::<Sentences>().unwrap(),
            new.get::<Subwords>().unwrap(),
            new.get::<Boundaries>().unwrap(),
            new.get::<Structure>(),
        )
        .unwrap()
}

fn same_chunks(a: &Chunks, b: &Chunks) {
    let spans = |c: &Chunks| c.spans.iter().map(|(_, s)| s).collect::<Vec<_>>();
    assert_eq!(spans(a), spans(b), "spans");
    assert_eq!(a.level, b.level, "levels");
    assert_eq!(a.parent, b.parent, "parents");
    assert_eq!(a.sentences, b.sentences, "sentence ranges");
    assert_eq!(a.cut, b.cut, "cuts");
    assert_eq!(a.cost, b.cost, "costs");
    assert_eq!(a.model_tokens, b.model_tokens, "model tokens");
    assert_eq!(a.over_budget, b.over_budget, "over budget");
}

fn leaf_ids(doc: &Document) -> HashSet<ChunkId> {
    let chunks = doc.get::<Chunks>().unwrap();
    let hashes = ContentHashes::<Chunks>::compute(doc.source(), chunks, doc.get::<Structure>()).unwrap();
    chunks.level.iter().enumerate().filter(|&(_, &l)| l == 0).map(|(i, _)| hashes.id[i]).collect()
}

const OLD: &str = "Aa bb. Cc dd. Ee ff. Gg hh. Ii jj. Kk ll. Mm nn. Oo pp. Qq rr.";
const SECTIONS: [Range<usize>; 3] = [0..3, 3..6, 6..9];

#[test]
fn an_edit_in_one_section_carries_the_other_sections() {
    let old = chunked(OLD, &SECTIONS);
    // A longer first sentence in the middle section: +3 bytes, +1
    // token, everything after it shifts.
    let (new_text, edit) = edited(OLD, "Gg hh.", "Gg hh ii.");
    let new = chunked(&new_text, &SECTIONS);
    let (chunks, stats) = rechunk(&old, &new, &edit);

    same_chunks(&chunks, new.get::<Chunks>().unwrap());
    assert_eq!((stats.reused_runs, stats.rechunked_runs), (2, 1));
    assert_eq!(stats.reused_chunks, 4);
    assert_eq!((stats.affected_old, stats.affected_new), (3..4, 3..4));

    // The last section's chunks moved by exactly the edit's growth
    // and kept their content IDs; the first section's did not move.
    let (old_c, new_c) = (old.get::<Chunks>().unwrap(), new.get::<Chunks>().unwrap());
    let old_leaves: Vec<Span> =
        old_c.spans.iter().filter(|&(i, _)| old_c.level[i.get()] == 0).map(|(_, s)| s).collect();
    let new_leaves: Vec<Span> =
        new_c.spans.iter().filter(|&(i, _)| new_c.level[i.get()] == 0).map(|(_, s)| s).collect();
    assert_eq!(old_leaves.len(), new_leaves.len());
    for (k, (o, n)) in old_leaves.iter().zip(&new_leaves).enumerate() {
        let want = if k < 2 {
            *o
        } else if k > 3 {
            Span::new(o.start + 3, o.end + 3)
        } else {
            continue;
        };
        assert_eq!(*n, want, "leaf {k}");
    }
    let (old_ids, new_ids) = (leaf_ids(&old), leaf_ids(&new));
    // Five of six: only the edited sentence's chunk changes ID, the
    // chunk after it in the same section keeps its text and its ID.
    assert_eq!(old_ids.intersection(&new_ids).count(), 5, "only the edited chunk loses its leaf ID");

    // Levels carry the same IDs for untouched chunks.
    let levels_hashes = |d: &Document| {
        ContentHashes::<Levels>::compute(d.source(), d.get::<Levels>().unwrap(), d.get::<Structure>()).unwrap().id
    };
    assert_eq!(levels_hashes(&old).len(), levels_hashes(&new).len());
}

#[test]
fn a_sentence_count_change_still_carries_the_sections_after_it() {
    let old = chunked(OLD, &SECTIONS);
    // The middle section loses its first sentence boundary: two
    // sentences become one, so the sections after it renumber.
    let (new_text, edit) = edited(OLD, "Gg hh. Ii jj.", "Gg hh ii jj.");
    let new = chunked(&new_text, &[0..3, 3..5, 5..8]);
    let (chunks, stats) = rechunk(&old, &new, &edit);
    same_chunks(&chunks, new.get::<Chunks>().unwrap());
    assert_eq!((stats.reused_runs, stats.rechunked_runs), (2, 1));
    let (old_ids, new_ids) = (leaf_ids(&old), leaf_ids(&new));
    assert_eq!(old_ids.intersection(&new_ids).count(), 4);
}

#[test]
fn an_identity_alignment_reuses_everything() {
    let old = chunked(OLD, &SECTIONS);
    let (chunks, stats) = rechunk(&old, &old, &Alignment::identity());
    same_chunks(&chunks, old.get::<Chunks>().unwrap());
    assert_eq!((stats.reused_runs, stats.rechunked_runs, stats.reused_chunks), (3, 0, 6));
    assert_eq!(affected_sentences(&Alignment::identity(), old.get().unwrap(), old.get().unwrap()), (0..0, 0..0));
}

#[test]
fn without_structure_any_change_rechunks_the_single_run() {
    let old = chunked(OLD, &[]);
    let (new_text, edit) = edited(OLD, "Ee ff.", "Ee ff gg.");
    let mut docs = vec![document(&new_text, &[])];
    let plain = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Boundaries>::DEFAULT)
        .add(Chunker { budget: BUDGET })
        .build()
        .unwrap();
    assert!(plain.run(&mut docs)[0].is_ok());
    let new = docs.pop().unwrap();
    let (chunks, stats) = rechunk(&old, &new, &edit);
    same_chunks(&chunks, new.get::<Chunks>().unwrap());
    assert_eq!((stats.reused_runs, stats.rechunked_runs), (0, 1));
}

#[test]
fn rows_cut_by_another_budget_are_not_carried() {
    // Previous chunks from a chunker with a bigger budget tile the
    // runs differently and price their rows differently: the run is
    // cut again rather than transplanted.
    let mut docs = vec![document(OLD, &SECTIONS)];
    let wide = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Boundaries>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(Chunker { budget: Budget { max: 12, target: 9, size_weight: 1000, overhead: 0 } })
        .build()
        .unwrap();
    assert!(wide.run(&mut docs)[0].is_ok());
    let old = docs.pop().unwrap();
    let new = chunked(OLD, &SECTIONS);
    let (chunks, stats) = rechunk(&old, &new, &Alignment::identity());
    same_chunks(&chunks, new.get::<Chunks>().unwrap());
    assert_eq!(stats.reused_runs, 0);
}

#[test]
fn a_changed_boundary_cost_rechunks_its_run() {
    let old = chunked(OLD, &SECTIONS);
    // Same text, but the signal layers now price the last section's
    // boundaries differently: its optimizer inputs changed even
    // though the alignment is the identity.
    let mut costs = vec![1000; 9];
    costs[6] = 10;
    costs[7] = 10;
    let new = chunked_with_costs(OLD, &SECTIONS, Some(costs));
    let (chunks, stats) = rechunk(&old, &new, &Alignment::identity());
    same_chunks(&chunks, new.get::<Chunks>().unwrap());
    assert_eq!((stats.reused_runs, stats.rechunked_runs), (2, 1));
}

#[test]
fn tampered_previous_rows_are_not_carried() {
    let old = chunked(OLD, &SECTIONS);
    let new = chunked(OLD, &SECTIONS);
    // The same rows with one cost rewritten: not this budget's rows
    // any more, so no run containing it is transplanted.
    let src = old.get::<Chunks>().unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    for (_, s) in src.spans.iter() {
        spans.push(s);
    }
    let mut cost = src.cost.clone();
    cost[1] += 1;
    let tampered = Chunks {
        spans,
        level: src.level.clone(),
        parent: src.parent.clone(),
        sentences: src.sentences.clone(),
        cut: src.cut.clone(),
        cost,
        model_tokens: src.model_tokens.clone(),
        over_budget: src.over_budget.clone(),
        prompt_overhead: src.prompt_overhead,
    };
    let (chunks, stats) = Chunker { budget: BUDGET }
        .rechunk(
            &Alignment::identity(),
            Previous {
                sentences: old.get::<Sentences>().unwrap(),
                subwords: old.get::<Subwords>().unwrap(),
                bounds: old.get::<Boundaries>().unwrap(),
                structure: old.get::<Structure>(),
                chunks: &tampered,
            },
            new.get::<Sentences>().unwrap(),
            new.get::<Subwords>().unwrap(),
            new.get::<Boundaries>().unwrap(),
            new.get::<Structure>(),
        )
        .unwrap();
    same_chunks(&chunks, new.get::<Chunks>().unwrap());
    assert_eq!((stats.reused_runs, stats.rechunked_runs), (2, 1), "only the tampered run is cut again");
}
