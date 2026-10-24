//! The chunker's hard limits over hand-made layers: no cut inside a
//! sentence, a table row, a code block or a quote unless that unit alone is
//! over the budget, the structural boundary costs, and chunk offsets that
//! land on the exact source bytes.

use turbo_analysis::*;
use turbo_analysis_layers::{Block, Boundaries, Chunks, Cut, Sentences, Structure, Subwords};
use turbo_chunker::{Budget, Chunker, Costs, StructureBoundaries};

fn toy_model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

/// The sentences of a view (each runs to a period, from the next
/// non-space character) and its subwords (each word, and each period, is
/// one), as a chunker reads them.
fn add_sentences(doc: &mut Document, view: ViewId) {
    let text = doc.text(view).to_owned();
    let (mut sentences, mut pieces) = (Spans::new(view), Spans::new(view));
    let mut start = None;
    for (i, c) in text.char_indices() {
        if start.is_none() && !c.is_whitespace() {
            start = Some(i as u32);
        }
        if c == '.' {
            sentences.push(Span::new(start.take().unwrap(), i as u32 + 1));
        }
    }
    let mut at = 0;
    for word in text.split(char::is_whitespace) {
        let w = word.trim_end_matches('.');
        if !w.is_empty() {
            pieces.push(Span::new(at, at + w.len() as u32));
        }
        if w.len() < word.len() {
            pieces.push(Span::new(at + w.len() as u32, at + word.len() as u32));
        }
        at += word.len() as u32 + 1;
    }
    let ids = vec![0; pieces.len()];
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, Subwords { spans: pieces, ids, model: toy_model(), token: None }, Provenance::by("t"))
        .unwrap();
}

fn plain(text: &str) -> Document {
    let mut doc = Document::new(text);
    add_sentences(&mut doc, ViewId::SOURCE);
    doc
}

/// Blocks over sentences `[a, b)`, each with its parent's index.
fn add_structure(doc: &mut Document, blocks: &[(usize, usize, Block, Option<u32>)]) {
    let s = doc.get::<Sentences>().unwrap();
    let bytes: Vec<(u32, u32, Block, Option<u32>)> = blocks
        .iter()
        .map(|&(a, b, kind, parent)| {
            (s.spans.get(Ix::new(a as u32)).start, s.spans.get(Ix::new(b as u32 - 1)).end, kind, parent)
        })
        .collect();
    doc.add(&Key::DEFAULT, structure(doc.get::<Sentences>().unwrap().spans.view(), &bytes), Provenance::by("parser"))
        .unwrap();
}

/// Blocks over bytes `[a, b)`, each with its parent's index.
fn structure(view: ViewId, blocks: &[(u32, u32, Block, Option<u32>)]) -> Structure {
    let mut spans = Spans::new(view);
    for &(a, b, ..) in blocks {
        spans.push(Span::new(a, b));
    }
    Structure {
        spans,
        kind: blocks.iter().map(|b| b.2).collect(),
        parent: blocks.iter().map(|b| b.3.map(Ix::new)).collect(),
        depth: vec![0; blocks.len()],
    }
}

/// A unit's cost after every sentence, but `cheap` after the ones named.
fn add_costs(doc: &mut Document, cheap: &[usize]) {
    let n = doc.get::<Sentences>().unwrap().spans.len();
    let mut cost = vec![Boundaries::UNIT; n];
    let mut reason = vec![Cut::Sentence; n];
    for &g in cheap {
        cost[g] = -5 * Boundaries::UNIT;
        reason[g] = Cut::TopicShift;
    }
    doc.add(&Key::DEFAULT, Boundaries { cost, reason }, Provenance::by("t")).unwrap();
}

fn run(doc: Document, budget: Budget) -> Document {
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Boundaries>::DEFAULT)
        .add(Chunker { budget })
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    docs.pop().unwrap()
}

/// The rows of the chunks the optimizer cut, level 0. With structure
/// each run is also a row one level up, the parent of those.
fn leaves(doc: &Document) -> Vec<usize> {
    let c = doc.get::<Chunks>().unwrap();
    (0..c.level.len()).filter(|&i| c.level[i] == 0).collect()
}

/// Each cut chunk's sentences, `[a, b)`.
fn sentences_of(doc: &Document) -> Vec<(usize, usize)> {
    let c = doc.get::<Chunks>().unwrap();
    leaves(doc).into_iter().map(|i| (c.sentences[i].start.get(), c.sentences[i].end.get())).collect()
}

/// Why each cut chunk starts where it does.
fn cuts(doc: &Document) -> Vec<Cut> {
    leaves(doc).into_iter().map(|i| doc.get::<Chunks>().unwrap().cut[i]).collect()
}

/// Whether each cut chunk is over the budget.
fn over(doc: &Document) -> Vec<bool> {
    leaves(doc).into_iter().map(|i| doc.get::<Chunks>().unwrap().over_budget[i]).collect()
}

/// Each cut chunk's length in model tokens.
fn tokens(doc: &Document) -> Vec<u32> {
    leaves(doc).into_iter().map(|i| doc.get::<Chunks>().unwrap().model_tokens[i]).collect()
}

/// Each cut chunk's span, in the sentences' view.
fn spans(doc: &Document) -> Vec<Span> {
    leaves(doc).into_iter().map(|i| doc.get::<Chunks>().unwrap().spans.get(Ix::new(i as u32))).collect()
}

/// Three tokens each: two words and a period.
const SIX: &str = "One one. Two two. Three three. Four four. Five five. Six six.";

/// A chunk of three sentences at most, two at best.
const B: Budget = Budget { max: 9, target: 6, size_weight: 1000, overhead: 0 };

/// The costs say to cut between sentences 2 and 3, inside a unit over
/// them that fits the budget; the chunker never does.
fn a_unit_that_fits_is_never_cut(kind: Block) {
    let mut free = plain(SIX);
    add_costs(&mut free, &[2]);
    let mut held = free.clone();
    add_structure(&mut held, &[(2, 4, kind, None)]);

    // Without the unit the cheap cut is taken.
    assert_eq!(sentences_of(&run(free, B)), [(0, 3), (3, 6)]);
    let held = run(held, B);
    assert_eq!(sentences_of(&held), [(0, 2), (2, 4), (4, 6)]);
    assert_eq!(cuts(&held), [Cut::DocumentStart, Cut::Sentence, Cut::Sentence]);
    assert_eq!(tokens(&held), [6, 6, 6]);
    // The whole text, with no section in it, is the run above them.
    let c = held.get::<Chunks>().unwrap();
    assert_eq!((c.level[0], c.sentences[0], c.model_tokens[0]), (1, IxRange::new(0, 6), 18));
}

#[test]
fn a_table_row_that_fits_is_never_cut() {
    a_unit_that_fits_is_never_cut(Block::TableRow);
}

#[test]
fn a_code_block_that_fits_is_never_cut() {
    a_unit_that_fits_is_never_cut(Block::Code);
}

#[test]
fn a_quote_that_fits_is_never_cut() {
    a_unit_that_fits_is_never_cut(Block::Quote);
}

#[test]
fn a_row_in_a_table_is_held_and_the_table_is_not() {
    // A table over sentences 1 to 4, rows of two; the costs say to cut
    // inside the first row and between the rows.
    let mut doc = plain(SIX);
    add_costs(&mut doc, &[1, 2]);
    add_structure(
        &mut doc,
        &[(1, 5, Block::Table, None), (1, 3, Block::TableRow, Some(0)), (3, 5, Block::TableRow, Some(0))],
    );
    let doc = run(doc, B);
    assert_eq!(sentences_of(&doc), [(0, 3), (3, 6)]);
    assert_eq!(cuts(&doc), [Cut::DocumentStart, Cut::TopicShift]);
}

/// A unit over sentences 1 to 4, twelve tokens against a budget of nine,
/// is cut, and the chunk that starts inside it says why.
fn a_unit_over_the_budget_is_cut_and_says_why(kind: Block) {
    let mut doc = plain(SIX);
    add_costs(&mut doc, &[]);
    add_structure(&mut doc, &[(1, 5, kind, None)]);
    let doc = run(doc, B);
    assert_eq!(sentences_of(&doc), [(0, 3), (3, 6)]);
    assert_eq!(cuts(&doc), [Cut::DocumentStart, Cut::Inside(kind)]);
    assert!(tokens(&doc).iter().all(|&t| t <= B.max));
}

#[test]
fn a_table_row_over_the_budget_is_cut_and_says_why() {
    a_unit_over_the_budget_is_cut_and_says_why(Block::TableRow);
}

#[test]
fn a_code_block_over_the_budget_is_cut_and_says_why() {
    a_unit_over_the_budget_is_cut_and_says_why(Block::Code);
}

#[test]
fn a_quote_over_the_budget_is_cut_and_says_why() {
    a_unit_over_the_budget_is_cut_and_says_why(Block::Quote);
}

#[test]
fn a_unit_inside_one_over_the_budget_is_still_held() {
    // A quote over every sentence (eighteen tokens) holds a code block over
    // 2 and 3 (six). The quote is cut; the code block is not, though the
    // costs say to cut inside it.
    let mut doc = plain(SIX);
    add_costs(&mut doc, &[2]);
    add_structure(&mut doc, &[(0, 6, Block::Quote, None), (2, 4, Block::Code, Some(0))]);
    let doc = run(doc, B);
    assert_eq!(sentences_of(&doc), [(0, 2), (2, 4), (4, 6)]);
    assert_eq!(cuts(&doc), [Cut::DocumentStart, Cut::Inside(Block::Quote), Cut::Inside(Block::Quote)]);
}

#[test]
fn a_sentence_is_never_cut_and_one_over_the_budget_stands_alone() {
    // The third sentence is twelve tokens, inside a code block that is
    // over the budget with it. Every chunk starts and ends on a sentence's
    // edge, and the long sentence is a chunk of its own, marked as over
    // the budget, and still says why it starts there.
    let text = "One one. Two two. A b c d e f g h i j k. Four four. Five five.";
    let mut doc = plain(text);
    add_costs(&mut doc, &[]);
    add_structure(&mut doc, &[(1, 4, Block::Code, None)]);
    let doc = run(doc, B);
    assert_eq!(sentences_of(&doc), [(0, 2), (2, 3), (3, 5)]);
    assert_eq!(cuts(&doc), [Cut::DocumentStart, Cut::Inside(Block::Code), Cut::Inside(Block::Code)]);
    assert_eq!(over(&doc), [false, true, false]);
    assert_eq!(tokens(&doc), [6, 12, 6]);
    let s = doc.get::<Sentences>().unwrap();
    let starts: Vec<u32> = s.spans.iter().map(|(_, s)| s.start).collect();
    let ends: Vec<u32> = s.spans.iter().map(|(_, s)| s.end).collect();
    assert!(spans(&doc).iter().all(|x| starts.contains(&x.start) && ends.contains(&x.end)));
}

#[test]
fn every_chunk_is_the_exact_source_bytes_of_its_sentences() {
    // Two- to four-byte characters, and a code block that fits.
    let text = "Ünï cödé. Ça va. 日本 語. Ωμέγα ἄλφα. 🦀 rust. Ende gut.";
    let mut doc = plain(text);
    add_costs(&mut doc, &[]);
    add_structure(&mut doc, &[(1, 3, Block::Code, None)]);
    let doc = run(doc, B);
    let c = doc.get::<Chunks>().unwrap();
    let s = doc.get::<Sentences>().unwrap();
    let texts: Vec<&str> = spans(&doc).iter().map(|x| &doc.source()[x.range()]).collect();
    assert_eq!(texts, ["Ünï cödé. Ça va. 日本 語.", "Ωμέγα ἄλφα. 🦀 rust. Ende gut."]);
    // Every row, the run above them too, is the bytes of its sentences.
    for (i, x) in c.spans.iter() {
        let r = c.sentences[i.get()];
        assert_eq!(x.start, s.spans.get(r.start).start);
        assert_eq!(x.end, s.spans.get(Ix::new(r.end.get() as u32 - 1)).end);
    }
}

#[test]
fn chunks_of_a_normalized_view_map_back_to_the_source() {
    // The view expands the "fi" ligature, so its offsets run ahead of the
    // source's; the chunks are cut in the view and mapped back.
    let mut doc = Document::new("The ﬁrst ﬁle. Two two. Three ﬁve. Four four.");
    let src = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(src.text());
    for (i, c) in src.text().char_indices() {
        if c == 'ﬁ' {
            b.keep(i);
            b.replace(i + c.len_utf8(), "fi");
        }
    }
    let (text, alignment) = b.finish();
    let folded = Key::named("folded");
    doc.add(&folded, View::derived(&src, text, alignment), Provenance::by("fold")).unwrap();
    let view = doc.get_key(&folded).unwrap().id();
    add_sentences(&mut doc, view);
    add_costs(&mut doc, &[]);
    add_structure(&mut doc, &[(0, 2, Block::Quote, None)]);
    let doc = run(doc, Budget { max: 7, target: 7, ..B });
    let c = doc.get::<Chunks>().unwrap();
    assert_eq!(c.spans.view(), view);
    let source: Vec<&str> = spans(&doc).iter().map(|&x| &doc.source()[doc.to_source(view, x).range()]).collect();
    assert_eq!(source, ["The ﬁrst ﬁle. Two two.", "Three ﬁve. Four four."]);
    let viewed: Vec<&str> = spans(&doc).iter().map(|x| &doc.text(view)[x.range()]).collect();
    assert_eq!(viewed, ["The first file. Two two.", "Three five. Four four."]);
}

#[test]
fn the_structure_sets_the_boundary_costs() {
    // A heading, a paragraph of two sentences, a heading, a paragraph of two.
    let text = "Cat facts. Cats purr. Cats nap. Dog facts. Dogs bark. Dogs run.";
    let mut doc = plain(text);
    add_structure(
        &mut doc,
        &[
            (0, 1, Block::Heading, None),
            (1, 3, Block::Paragraph, None),
            (3, 4, Block::Heading, None),
            (4, 6, Block::Paragraph, None),
        ],
    );
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .add(StructureBoundaries { costs: Costs::default() })
        .add(Chunker { budget: Budget { max: 12, target: 9, size_weight: 1000, overhead: 0 } })
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let doc = &docs[0];

    let b = doc.get::<Boundaries>().unwrap();
    let c = Costs::default();
    let after_heading = c.paragraph + c.after_heading;
    assert_eq!(b.cost, [after_heading, c.sentence, c.heading, after_heading, c.sentence, c.paragraph]);
    let p = Cut::Paragraph;
    assert_eq!(b.reason, [p, Cut::Sentence, Cut::Heading, p, Cut::Sentence, p]);

    // Each heading stays with its paragraph.
    assert_eq!(sentences_of(doc), [(0, 3), (3, 6)]);
    assert_eq!(cuts(doc), [Cut::DocumentStart, Cut::Heading]);
}

#[test]
fn without_structure_every_boundary_is_a_sentence() {
    let mut docs = vec![plain(SIX)];
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(StructureBoundaries { costs: Costs::default() })
        .build()
        .unwrap();
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let b = docs[0].get::<Boundaries>().unwrap();
    assert_eq!(b.cost, [Boundaries::UNIT; 6]);
    assert_eq!(b.reason, [Cut::Sentence; 6]);
}

#[test]
fn units_that_share_sentences_and_chain_over_the_budget_are_cut() {
    // Three rows of a table, apart in bytes, whose edges fall inside
    // sentences 1 and 2: each holds two sentences (eight tokens, which
    // fit), and together they hold all four (sixteen, which do not).
    let mut doc = plain("Aa bb cc. Dd ee ff. Gg hh ii. Jj kk ll.");
    add_costs(&mut doc, &[]);
    let rows = [
        (0, 39, Block::Table, None),
        (0, 14, Block::TableRow, Some(0)),
        (14, 24, Block::TableRow, Some(0)),
        (24, 39, Block::TableRow, Some(0)),
    ];
    doc.add(&Key::DEFAULT, structure(ViewId::SOURCE, &rows), Provenance::by("parser")).unwrap();
    let doc = run(doc, Budget { max: 10, target: 8, ..B });
    assert_eq!(sentences_of(&doc), [(0, 2), (2, 4)]);
    assert_eq!(cuts(&doc), [Cut::DocumentStart, Cut::Overlap(Block::TableRow)]);
    assert_eq!(over(&doc), [false, false]);
}

#[test]
fn a_section_edge_is_a_cut_even_inside_a_unit() {
    // The second section starts inside sentence 2, which belongs to the
    // first by its start; a code block from there holds sentences 2 and 3
    // and fits. The section wins: the unit is cut at its edge.
    let text = "One one. Two two. Three three. Four four. Five five.";
    let mut doc = plain(text);
    add_costs(&mut doc, &[]);
    let blocks = [(0, 24, Block::Section, None), (24, 52, Block::Section, None), (24, 41, Block::Code, Some(1))];
    doc.add(&Key::DEFAULT, structure(ViewId::SOURCE, &blocks), Provenance::by("parser")).unwrap();
    let doc = run(doc, B);
    assert_eq!(sentences_of(&doc), [(0, 3), (3, 5)]);
    assert_eq!(cuts(&doc), [Cut::DocumentStart, Cut::Section]);
}

#[test]
fn a_chunk_over_the_budget_at_a_section_start_keeps_the_section() {
    let text = "One one. Two two. A b c d e f g h i j k. Four four.";
    let mut doc = plain(text);
    add_costs(&mut doc, &[]);
    add_structure(&mut doc, &[(0, 2, Block::Section, None), (2, 4, Block::Section, None)]);
    let doc = run(doc, B);
    assert_eq!(sentences_of(&doc), [(0, 2), (2, 3), (3, 4)]);
    assert_eq!(cuts(&doc), [Cut::DocumentStart, Cut::Section, Cut::Sentence]);
    assert_eq!(over(&doc), [false, true, false]);
    // The second section is over the budget as a whole, as a row above.
    let c = doc.get::<Chunks>().unwrap();
    assert_eq!((c.level[2], c.over_budget[2], c.model_tokens[2]), (1, true, 15));
}

#[test]
fn chunks_nest_inside_their_parents_at_a_level_above() {
    let chunks = |spans: [(u32, u32); 2], level: [u8; 2]| {
        let mut c = Chunks {
            spans: Spans::new(ViewId::SOURCE),
            level: level.to_vec(),
            parent: vec![None, Some(Ix::new(0))],
            sentences: vec![IxRange::new(0, 2), IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart; 2],
            cost: vec![0; 2],
            model_tokens: vec![0; 2],
            over_budget: vec![false; 2],
            prompt_overhead: 0,
        };
        for (a, b) in spans {
            c.spans.push(Span::new(a, b));
        }
        c
    };
    assert_eq!(chunks([(0, 20), (0, 9)], [1, 0]).check(), Ok(()));
    assert!(chunks([(0, 20), (0, 9)], [0, 0]).check().unwrap_err().contains("level below"));
    assert!(chunks([(0, 20), (0, 9)], [1, 2]).check().unwrap_err().contains("level below"));
    assert!(chunks([(0, 9), (0, 20)], [1, 0]).check().unwrap_err().contains("outside its parent"));
    let mut short = chunks([(0, 20), (0, 9)], [1, 0]);
    short.over_budget.pop();
    assert!(short.check().is_err());
}

#[test]
fn a_section_start_inside_a_sentence_is_one_gap_for_costs_and_runs() {
    // The second section starts inside sentence 2, which starts in the
    // first. Both the boundary costs and the runs give that sentence to
    // the first section, so the section edge is the gap after it, and
    // every chunk marked Section starts a run.
    let text = "One one. Two two. Three three. Four four.";
    let mut doc = plain(text);
    let blocks = [(0, 24, Block::Section, None), (24, 41, Block::Section, None)];
    doc.add(&Key::DEFAULT, structure(ViewId::SOURCE, &blocks), Provenance::by("parser")).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .add(StructureBoundaries { costs: Costs::default() })
        .add(Chunker { budget: Budget { max: 6, target: 4, size_weight: 1000, overhead: 0 } })
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let doc = &docs[0];
    let reasons = &doc.get::<Boundaries>().unwrap().reason;
    assert_eq!(reasons[..3], [Cut::Sentence, Cut::Sentence, Cut::Section]);
    let c = doc.get::<Chunks>().unwrap();
    let runs: Vec<(usize, usize)> = (0..c.level.len())
        .filter(|&i| c.level[i] == 1)
        .map(|i| (c.sentences[i].start.get(), c.sentences[i].end.get()))
        .collect();
    assert_eq!(runs, [(0, 3), (3, 4)]);
    for (&(a, _), cut) in sentences_of(doc).iter().zip(cuts(doc)) {
        assert_eq!(cut == Cut::Section, runs.iter().skip(1).any(|r| r.0 == a), "chunk at sentence {a}");
    }
}
