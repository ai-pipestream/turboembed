use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis::{Annotator, Document, Error, Ix, IxRange, Span, Spans, TagSet, View, ViewId};
use turbo_analysis_layers::{PhraseChunks, Sentences, Subwords, Tokens};

use super::*;
use crate::encoder::EncoderOptions;
use crate::heads::{ClassifierHead, HeadMode};
use crate::subwords::SubwordAnnotator;
use crate::vectors::{VectorAnnotator, VectorOptions};

fn bert_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn open_test_vectors() -> (SubwordAnnotator, Arc<VectorAnnotator>) {
    let dir = bert_dir();
    let sub = SubwordAnnotator::open(&dir).unwrap();
    let opts = VectorOptions {
        encoder: EncoderOptions { max_seq: 64, ..EncoderOptions::default() },
        ..VectorOptions::default()
    };
    let vec_ann = Arc::new(VectorAnnotator::open(&dir, &opts).unwrap());
    (sub, vec_ann)
}

#[test]
fn negated_opinion_flips_opinion_polarity_once_not_twice() {
    let (sub_ann, vec_ann) = open_test_vectors();
    let text = "service is not helpful";
    let doc = Document::new(text.to_string());
    let view = doc.get::<View>().unwrap();

    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, text.len() as u32));
    let sentences = Sentences { spans: sent_spans };

    let mut tok_spans = Spans::new(ViewId::SOURCE);
    tok_spans.push(Span::new(0, 7)); // "service"
    tok_spans.push(Span::new(8, 10)); // "is"
    tok_spans.push(Span::new(11, 14)); // "not"
    tok_spans.push(Span::new(15, 22)); // "helpful"
    let tokens = Tokens { spans: tok_spans, sentence: Some(vec![Ix::new(0); 4]) };

    let (subwords,) = sub_ann.annotate((view, Some(&tokens))).unwrap();

    // Fit on positive opinion ("helpful") at token 3..4 and aspect ("service") at token 0..1.
    let train_doc = LabeledAbsaDoc {
        subwords: &subwords,
        sentences: &sentences,
        tokens: &tokens,
        aspects: &[(IxRange::new(0, 1), "service", "positive")],
        opinions: &[(IxRange::new(3, 4), "positive")],
    };
    let annotator =
        LearnedAspectSentiment::fit(Arc::clone(&vec_ann), &["service"], &[train_doc], 1e-4, 200, 2.0).unwrap();

    let (opinions, aspects) = annotator.annotate((view, &sentences, &tokens, &subwords, None, None, None)).unwrap();
    assert_eq!(opinions.len(), 1);
    let op = opinions.get(Ix::new(0));
    assert!(op.negated, "opinion 'helpful' preceded by 'not' must be marked negated");
    assert!(
        op.polarity < 0.0,
        "negated positive opinion must have negative polarity, got {} (double-negation bug)",
        op.polarity
    );
    assert_eq!(aspects.len(), 1);
    let asp = aspects.get(Ix::new(0));
    assert!(
        asp.polarity.negative > asp.polarity.positive,
        "linked aspect must also reflect negative polarity after negation: {:?}",
        asp.polarity
    );
}

#[test]
fn annotate_rejects_mismatched_views_and_invalid_token_spans() {
    let (sub_ann, vec_ann) = open_test_vectors();
    let dim = vec_ann.dim() as usize;
    let model = vec_ann.model().clone();
    let bio_tags = bio_aspect_opinion_tagset();
    let pol_tags = TagSet::new("sentiment", &["positive", "negative", "neutral"]);
    let bio_head = ClassifierHead::linear(
        model.clone(),
        dim as u32,
        bio_tags,
        HeadMode::SingleLabel,
        vec![0.0; 5 * dim],
        vec![0.0; 5],
    )
    .unwrap();
    let pol_head =
        ClassifierHead::linear(model, dim as u32, pol_tags, HeadMode::SingleLabel, vec![0.0; 3 * dim], vec![0.0; 3])
            .unwrap();
    let annotator = LearnedAspectSentiment::new(vec_ann, bio_head, pol_head, None);

    let doc = Document::new("service is helpful".to_string());
    let alt_view_id = ViewId(99);
    let view = doc.get::<View>().unwrap();

    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, 18));
    let sentences = Sentences { spans: sent_spans };

    let mut tok_spans = Spans::new(ViewId::SOURCE);
    tok_spans.push(Span::new(0, 7));
    tok_spans.push(Span::new(8, 10));
    tok_spans.push(Span::new(11, 18));
    let tokens = Tokens { spans: tok_spans, sentence: None };

    let (subwords,) = sub_ann.annotate((view, Some(&tokens))).unwrap();

    // 1. Mismatched sentences view
    let bad_sentences = Sentences { spans: Spans::new(alt_view_id) };
    let err = annotator.annotate((view, &bad_sentences, &tokens, &subwords, None, None, None)).unwrap_err();
    assert!(
        matches!(&err, Error::Annotator { annotator: LearnedAspectSentiment::ID, detail } if detail == "sentences are not in the given view"),
        "unexpected error: {err:?}"
    );

    // 2. Mismatched tokens view
    let bad_tokens = Tokens { spans: Spans::new(alt_view_id), sentence: None };
    let err = annotator.annotate((view, &sentences, &bad_tokens, &subwords, None, None, None)).unwrap_err();
    assert!(
        matches!(&err, Error::Annotator { annotator: LearnedAspectSentiment::ID, detail } if detail == "tokens are not in the given view"),
        "unexpected error: {err:?}"
    );

    // 3. Mismatched subwords view
    let bad_subwords =
        Subwords { spans: Spans::new(alt_view_id), ids: Vec::new(), token: None, model: subwords.model.clone() };
    let err = annotator.annotate((view, &sentences, &tokens, &bad_subwords, None, None, None)).unwrap_err();
    assert!(
        matches!(&err, Error::Annotator { annotator: LearnedAspectSentiment::ID, detail } if detail == "subwords are not in the given view"),
        "unexpected error: {err:?}"
    );

    // 4. Mismatched phrase_chunks view
    let chunk_kinds = TagSet::new("chunks", &["NP"]);
    let bad_chunks =
        PhraseChunks { spans: Spans::new(alt_view_id), tokens: Vec::new(), kind: Vec::new(), kinds: chunk_kinds };
    let err = annotator.annotate((view, &sentences, &tokens, &subwords, None, None, Some(&bad_chunks))).unwrap_err();
    assert!(
        matches!(&err, Error::Annotator { annotator: LearnedAspectSentiment::ID, detail } if detail == "phrase chunks are not in the given view"),
        "unexpected error: {err:?}"
    );

    // 5. Out-of-bounds token span
    let mut oob_tok_spans = Spans::new(ViewId::SOURCE);
    oob_tok_spans.push(Span::new(0, 999));
    let oob_tokens = Tokens { spans: oob_tok_spans, sentence: None };
    let err = annotator.annotate((view, &sentences, &oob_tokens, &subwords, None, None, None)).unwrap_err();
    assert!(
        matches!(&err, Error::Annotator { annotator: LearnedAspectSentiment::ID, detail } if detail.contains("out of bounds or not on a UTF-8 char boundary")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn fit_rejects_empty_inputs_and_out_of_bounds_ranges() {
    let (sub_ann, vec_ann) = open_test_vectors();
    let doc = Document::new("battery is good".to_string());
    let view = doc.get::<View>().unwrap();

    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, 15));
    let sentences = Sentences { spans: sent_spans };

    let mut tok_spans = Spans::new(ViewId::SOURCE);
    tok_spans.push(Span::new(0, 7));
    tok_spans.push(Span::new(8, 10));
    tok_spans.push(Span::new(11, 15));
    let tokens = Tokens { spans: tok_spans, sentence: None };
    let (subwords,) = sub_ann.annotate((view, Some(&tokens))).unwrap();

    // Empty category labels
    assert!(LearnedAspectSentiment::fit(Arc::clone(&vec_ann), &[], &[], 1e-4, 10, 0.1).is_err());

    // Empty docs
    assert!(LearnedAspectSentiment::fit(Arc::clone(&vec_ann), &["battery"], &[], 1e-4, 10, 0.1).is_err());

    // Out-of-bounds aspect token range
    let bad_asp = LabeledAbsaDoc {
        subwords: &subwords,
        sentences: &sentences,
        tokens: &tokens,
        aspects: &[(IxRange::new(0, 99), "battery", "positive")],
        opinions: &[],
    };
    assert!(LearnedAspectSentiment::fit(Arc::clone(&vec_ann), &["battery"], &[bad_asp], 1e-4, 10, 0.1).is_err());

    // Out-of-bounds opinion token range
    let bad_op = LabeledAbsaDoc {
        subwords: &subwords,
        sentences: &sentences,
        tokens: &tokens,
        aspects: &[],
        opinions: &[(IxRange::new(2, 99), "positive")],
    };
    assert!(LearnedAspectSentiment::fit(Arc::clone(&vec_ann), &["battery"], &[bad_op], 1e-4, 10, 0.1).is_err());
}
