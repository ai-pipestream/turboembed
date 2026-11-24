//! A chain step in a document against the same steps as separate views, on
//! fixed and random texts.

use super::*;

// Through a Document

/// Adds one `NormalizeStep` for `kind`, reading the view `from` and
/// writing the view `to` ("" is the source).
fn add_step(
    builder: turbo_analysis::analyzer::Builder,
    kind: Kind,
    from: &'static str,
    to: &'static str,
) -> turbo_analysis::analyzer::Builder {
    macro_rules! step {
        ($n:expr) => {
            builder.step(Bound::new(NormalizeStep::new($n)).read(0, from).write(0, to))
        };
    }
    match kind {
        Kind::Ligature => step!(Ligature),
        Kind::Spaces => step!(Spaces),
        Kind::SoftHyphen => step!(SoftHyphen),
        Kind::AfterPeriod => step!(AfterPeriod::default()),
        Kind::Upper => step!(Upper),
        Kind::AbToX => step!(AbToX),
        Kind::Identity => step!(Identity),
        Kind::DeleteAll => step!(DeleteAll),
        Kind::ReplaceAll => step!(ReplaceAll),
    }
}

const VIEW_NAMES: [&str; 4] = ["v1", "v2", "v3", "v4"];

/// Runs `kinds` as separate views, one step each, and returns the document
/// and the name of the last view.
fn run_separately(kinds: &[Kind], text: &str) -> (Document, &'static str) {
    let mut builder = Analyzer::builder();
    let mut from = "";
    for (k, &kind) in kinds.iter().enumerate() {
        builder = add_step(builder, kind, from, VIEW_NAMES[k]);
        from = VIEW_NAMES[k];
    }
    let mut docs = vec![Document::new(text)];
    let status = builder.build().unwrap().run(&mut docs);
    assert!(status[0].is_ok(), "{status:?}");
    (docs.pop().unwrap(), from)
}

/// Runs `kinds` as one `Chain` step writing the view "v1".
pub(super) fn run_chained(kinds: &[Kind], text: &str) -> Document {
    let analyzer =
        Analyzer::builder().step(Bound::new(NormalizeStep::new(chain_of(kinds))).write(0, "v1")).build().unwrap();
    let mut docs = vec![Document::new(text)];
    let status = analyzer.run(&mut docs);
    assert!(status[0].is_ok(), "{status:?}");
    docs.pop().unwrap()
}

pub(super) fn check_document(kinds: &[Kind], text: &str) {
    let case = format!("{kinds:?} on {text:?}");
    let (separate, last) = run_separately(kinds, text);
    let chained = run_chained(kinds, text);
    let sep_view = separate.get_key(&Key::<View>::named(last)).unwrap();
    let chain_view = chained.get_key(&Key::<View>::named("v1")).unwrap();
    assert_eq!(sep_view.text(), chain_view.text(), "text of {case}");
    let stages: Vec<&Alignment> = VIEW_NAMES[..kinds.len()]
        .iter()
        .map(|n| separate.get_key(&Key::<View>::named(n)).unwrap().alignment())
        .collect();
    // The chain's one alignment against the views' alignments in turn.
    assert_mapping_agrees(
        &case,
        text,
        chain_view.text(),
        chain_view.alignment(),
        |s| walk_back(stages.iter().copied(), s).0,
        |s| walk_forward(stages.iter().copied(), s).0,
    );
    // `Document` walks the same alignments; it agrees with the walk above
    // wherever no empty span turns up half way (it cannot map those).
    for s in non_empty_spans_of(sep_view.text()) {
        let (stepwise, plain) = walk_back(stages.iter().copied(), s);
        if plain {
            assert_eq!(separate.to_source(sep_view.id(), s), stepwise, "to_source {s:?} of {case}");
        }
        assert_eq!(chained.to_source(chain_view.id(), s), chain_view.alignment().span_to_parent(s), "{case}");
    }
    for s in non_empty_spans_of(text) {
        let (stepwise, plain) = walk_forward(stages.iter().copied(), s);
        if plain {
            assert_eq!(separate.from_source(sep_view.id(), s), stepwise, "from_source {s:?} of {case}");
        }
        assert_eq!(chained.from_source(chain_view.id(), s), chain_view.alignment().span_to_child(s), "{case}");
    }
}

#[test]
fn a_chain_step_and_separate_views_give_the_same_source_spans() {
    for &text in INPUTS {
        for a in KINDS {
            check_document(&[a], text);
            for b in KINDS {
                check_document(&[a, b], text);
            }
        }
    }
}

#[test]
fn three_separate_views_and_a_triple_chain_give_the_same_source_spans() {
    for &text in INPUTS {
        for a in KINDS {
            for b in KINDS {
                for c in KINDS {
                    check_document(&[a, b, c], text);
                }
            }
        }
    }
}

#[test]
fn the_view_a_chain_writes_has_the_source_as_parent() {
    let doc = run_chained(&[Kind::Ligature, Kind::Spaces], "o\u{FB01}ce  x");
    let v = doc.get_key(&Key::<View>::named("v1")).unwrap();
    assert_eq!(v.text(), "ofice x");
    assert_eq!(v.parent(), Some(doc.get_key(&Key::<View>::DEFAULT).unwrap().id()));
    // One edit list straight to the source, not one per step.
    assert_eq!(v.alignment().edits().len(), 2);
}

// 3. Property loop

/// xorshift64: deterministic, no dependencies.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Characters of 1 to 4 bytes, plus every trigger of the toys.
const ALPHABET: &[&str] =
    &["a", "b", "a", "b", "c", "x", " ", " ", ".", "\u{AD}", "\u{FB01}", "\u{E9}", "\u{20AC}", "\u{1F600}", "A"];

fn random_text(rng: &mut Rng) -> String {
    let n = rng.below(14);
    (0..n).map(|_| ALPHABET[rng.below(ALPHABET.len())]).collect()
}

fn random_kinds(rng: &mut Rng) -> Vec<Kind> {
    let n = 1 + rng.below(4);
    (0..n).map(|_| KINDS[rng.below(KINDS.len())]).collect()
}

#[test]
fn random_chains_on_random_text_match_the_steps_run_in_sequence() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..4000 {
        let text = random_text(&mut rng);
        let kinds = random_kinds(&mut rng);
        check_chain(&kinds, &text);
    }
}

#[test]
fn random_chains_match_separate_views_in_a_document() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for _ in 0..600 {
        let text = random_text(&mut rng);
        let kinds = random_kinds(&mut rng);
        check_document(&kinds, &text);
    }
}

#[test]
fn a_chain_alignment_is_well_formed() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    for _ in 0..2000 {
        let text = random_text(&mut rng);
        let kinds = random_kinds(&mut rng);
        let (out, a) = normalize(&chain_of(&kinds), &text);
        assert_well_formed(&format!("{kinds:?} on {text:?}"), &text, &out, &a);
    }
}
