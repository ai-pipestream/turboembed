//! `ClaimVerificationAnnotator` on the inputs #1322 added: a reused
//! `Tokens` layer with punctuation tokens, mismatched token layers, and
//! scorers that return non-finite logits.

mod claim_support;

use claim_support::{RuleNli, rows};
use turbo_analysis::{Annotator, Document, Error, Key, ModelRef, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{ClaimRelationKind as K, Sentences, Strs, Terms, Tokens};
use turbo_doccat::{ClaimVerificationAnnotator, NliScorer, ZeroShotError};

type Edges = Result<Vec<(usize, usize, K)>, Error>;

fn run<S: NliScorer + Send + Sync + 'static>(
    verifier: &ClaimVerificationAnnotator<S, Sentences>,
    d: &Document,
) -> Edges {
    run_with(verifier, d, d.get::<Tokens>(), d.get::<Terms>())
}

fn run_with<S: NliScorer + Send + Sync + 'static>(
    verifier: &ClaimVerificationAnnotator<S, Sentences>,
    d: &Document,
    tokens: Option<&Tokens>,
    terms: Option<&Terms>,
) -> Edges {
    let view = d.require(&Key::<View>::DEFAULT)?;
    let (edges,) = verifier.annotate((view, d.require(&Key::<Sentences>::DEFAULT)?, None, None, tokens, terms))?;
    Ok(rows(&edges))
}

/// Sentences joined by spaces, tokenized like a word tokenizer: words
/// and each punctuation mark are separate tokens.
fn doc(sentences: &[&str], with_tokens: bool) -> Document {
    let text = sentences.join(" ");
    let mut sents = Spans::new(ViewId(0));
    let mut at = 0;
    for s in sentences {
        sents.push(Span::new(at, at + s.len() as u32));
        at += s.len() as u32 + 1;
    }
    let mut tokens = Spans::new(ViewId(0));
    let mut start = None;
    for (i, c) in text.char_indices().chain([(text.len(), ' ')]) {
        if c.is_alphanumeric() {
            start.get_or_insert(i);
            continue;
        }
        if let Some(s) = start.take() {
            tokens.push(Span::new(s as u32, i as u32));
        }
        if !c.is_whitespace() {
            tokens.push(Span::new(i as u32, (i + c.len_utf8()) as u32));
        }
    }
    let mut d = Document::new(text.as_str());
    d.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sents }, Provenance::by("test:sentences")).unwrap();
    if with_tokens {
        d.add(&Key::<Tokens>::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("test:tokens"))
            .unwrap();
    }
    d
}

#[test]
fn punctuation_tokens_do_not_make_candidates() {
    let verifier = ClaimVerificationAnnotator::new(RuleNli::three());
    for with_tokens in [false, true] {
        run(&verifier, &doc(&["It rained.", "Cats purr.", "Dogs bark, loudly!"], with_tokens)).unwrap();
    }
    assert_eq!(verifier.scorer().seen(), 0, "unrelated sentences share only punctuation");
    for with_tokens in [false, true] {
        let edges = run(&verifier, &doc(&["Acme bought Beta, in March.", "Acme bought Beta."], with_tokens)).unwrap();
        assert_eq!(edges, [(0, 1, K::Entails)]);
    }
    assert_eq!(verifier.scorer().seen(), 4, "content words still pair, with and without tokens");
}

#[test]
fn folded_punctuation_terms_do_not_make_candidates() {
    let mut d = doc(&["It rained.", "Cats purr."], true);
    let n = d.get::<Tokens>().unwrap().spans.len();
    let mut forms = Strs::default();
    let text = d.text(ViewId(0)).to_string();
    for (_, span) in d.get::<Tokens>().unwrap().spans.iter() {
        forms.push(&text[span.range()].to_lowercase());
    }
    let terms = Terms { surface: forms.clone(), folded: forms, stem: None, lemma: None };
    d.add(&Key::<Terms>::DEFAULT, terms, Provenance::by("test:terms")).unwrap();
    assert_eq!(n, 6);
    let verifier = ClaimVerificationAnnotator::new(RuleNli::three());
    run(&verifier, &d).unwrap();
    assert_eq!(verifier.scorer().seen(), 0);
}

#[test]
fn mismatched_tokens_and_terms_are_refused() {
    let verifier = ClaimVerificationAnnotator::new(RuleNli::three());
    let d = doc(&["Acme grew.", "Zoë grew."], true);
    for (view, span) in [(ViewId(1), Span::new(0, 4)), (ViewId(0), Span::new(0, 40)), (ViewId(0), Span::new(14, 15))] {
        let mut spans = Spans::new(view);
        spans.push(span);
        let err = run_with(&verifier, &d, Some(&Tokens { spans, sentence: None }), None).unwrap_err();
        assert!(format!("{err:?}").contains("not in the units' view"), "{span:?} {err:?}");
    }
    let mut forms = Strs::default();
    forms.push("acme");
    let terms = Terms { surface: forms.clone(), folded: forms, stem: None, lemma: None };
    let err = run_with(&verifier, &d, d.get::<Tokens>(), Some(&terms)).unwrap_err();
    assert!(format!("{err:?}").contains("one entry per token"), "{err:?}");
    assert_eq!(verifier.scorer().seen(), 0);
}

/// Returns `row` for every pair, over `contradiction, entailment, neutral`.
struct FixedNli {
    rule: RuleNli,
    row: [f32; 3],
}

impl NliScorer for FixedNli {
    fn outcome_labels(&self) -> &[String] {
        self.rule.outcome_labels()
    }

    fn model_ref(&self) -> ModelRef {
        self.rule.model_ref()
    }

    fn max_pair_tokens(&self) -> u32 {
        self.rule.max_pair_tokens()
    }

    fn pair_special_tokens(&self) -> usize {
        self.rule.pair_special_tokens()
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.rule.count_tokens(text)
    }

    fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<Vec<f32>>, ZeroShotError> {
        Ok(vec![self.row.to_vec(); pairs.len()])
    }
}

#[test]
fn non_finite_logits_write_no_edge_instead_of_panicking() {
    let d = doc(&["Acme bought Beta.", "Acme bought Beta."], true);
    for row in [[f32::NAN, 4.0, 0.0], [0.0, f32::INFINITY, 0.0], [f32::NAN; 3]] {
        let verifier = ClaimVerificationAnnotator::new(FixedNli { rule: RuleNli::three(), row });
        assert_eq!(run(&verifier, &d).unwrap(), [], "{row:?}");
    }
    let verifier =
        ClaimVerificationAnnotator::new(FixedNli { rule: RuleNli::three(), row: [f32::NEG_INFINITY, 4.0, 0.0] });
    assert_eq!(run(&verifier, &d).unwrap(), [(0, 1, K::Redundant)], "-inf is probability 0, as before");
}
