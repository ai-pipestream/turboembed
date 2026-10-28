//! The tagger annotators end to end on the small BERT in testdata, given
//! a part-of-speech head and an entity head with seeded weights: the
//! wiring from tokens to layers, the window plan for long sentences,
//! batching, determinism, and refusals.

mod common;

use common::*;
use serde_json::json;
use turbo::head::tagger::Work;
use turbo::manifest::WordPooling;
use turbo_analysis::{Annotator, Document, Key, Layer, Span, Spans, View, ViewId};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::PosTags;
use turbo_tagger::words::plan;
use turbo_tagger::{EntityTagger, HeadRunner, PosTagger, WordEncoder};

#[test]
fn the_window_plan_covers_every_word_from_a_window_that_holds_it() {
    let cases: [(&[usize], usize); 5] =
        [(&[1, 1, 1], 8), (&[2, 3, 1, 4, 2, 2, 1, 3, 2, 1], 6), (&[1; 40], 7), (&[5, 5, 5, 5], 5), (&[], 4)];
    for (lens, width) in cases {
        let (spans, owner) = plan(lens, width);
        assert_eq!(owner.len(), lens.len());
        if lens.is_empty() {
            assert!(spans.is_empty());
            continue;
        }
        assert_eq!(spans[0].0, 0);
        assert_eq!(spans.last().unwrap().1, lens.len());
        for (i, &(a, b)) in spans.iter().enumerate() {
            assert!(a < b && lens[a..b].iter().sum::<usize>() <= width, "{lens:?} {width}: {spans:?}");
            if i > 0 {
                assert!(spans[i - 1].0 < a && a <= spans[i - 1].1, "{lens:?}: windows advance and touch: {spans:?}");
            }
        }
        for (w, &o) in owner.iter().enumerate() {
            assert!(spans[o].0 <= w && w < spans[o].1, "{lens:?}: word {w} owned by {o}: {spans:?}");
        }
    }
    // Forty one-piece words in windows of seven: a middle word is read
    // with at least two words of context on each side.
    let (spans, owner) = plan(&[1; 40], 7);
    for w in 3..37 {
        let (a, b) = spans[owner[w]];
        assert!(w - a >= 2 && b - 1 - w >= 2, "word {w} in {a}..{b}");
    }
}

#[test]
fn features_do_not_depend_on_batching() {
    let dir = bundle("batching", |_| {});
    let words: Vec<&str> = TEXT.split(' ').collect();
    let seqs: Vec<Vec<&str>> = (0..5).map(|k| words[k..].to_vec()).collect();
    let mut wide = WordEncoder::open(&dir, &EncoderOptions::default()).unwrap();
    let mut narrow = WordEncoder::open(&dir, &EncoderOptions { max_batch: 2, ..EncoderOptions::default() }).unwrap();
    for pooling in [WordPooling::First, WordPooling::Mean] {
        let together = wide.features(&seqs, pooling).unwrap();
        let alone: Vec<Vec<f32>> =
            seqs.iter().map(|s| narrow.features(std::slice::from_ref(s), pooling).unwrap().remove(0)).collect();
        assert_eq!(together, alone, "{pooling:?}");
        for (s, f) in seqs.iter().zip(&together) {
            assert_eq!(f.len(), s.len() * 32);
        }
    }
}

#[test]
fn a_long_sentence_is_read_in_windows_and_a_short_one_is_not_changed_by_it() {
    let dir = bundle("windows", |_| {});
    let words: Vec<&str> = TEXT.split(' ').collect();
    // Rows of 8 tokens: 6 pieces beside [CLS] and [SEP].
    let mut small = WordEncoder::open(&dir, &EncoderOptions { max_seq: 8, ..EncoderOptions::default() }).unwrap();
    let mut big = WordEncoder::open(&dir, &EncoderOptions::default()).unwrap();
    assert_eq!(small.width(), 6);
    let long = small.features(std::slice::from_ref(&words), WordPooling::Mean).unwrap().remove(0);
    assert_eq!(long.len(), words.len() * 32);
    assert!(long.iter().all(|v| v.is_finite()));
    let short = &words[..3];
    let a = small.features(&[short.to_vec()], WordPooling::Mean).unwrap();
    let b = big.features(&[short.to_vec()], WordPooling::Mean).unwrap();
    for (x, y) in a[0].iter().zip(&b[0]) {
        assert!((x - y).abs() <= 1e-5, "{x} {y}");
    }
}

#[test]
fn the_annotators_write_one_tag_per_token_and_entities_on_token_bounds() {
    let dir = bundle("annotate", |_| {});
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let pos = PosTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    let (tags,) = pos.annotate((view, &toks, None, None)).unwrap();
    tags.check().unwrap();
    assert_eq!(tags.tags.len(), toks.spans.len());
    assert!(tags.tags.iter().any(|t| *t != tags.tags[0]), "the seeded head gives more than one tag");
    let prob = tags.prob.as_ref().unwrap();
    assert!(prob.iter().all(|p| (0.0..=1.0).contains(p)));

    let ner = EntityTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    assert_eq!(ner.kinds().len(), 2);
    let (ents,) = ner.annotate((view, &toks, None, None)).unwrap();
    assert!(!ents.spans.is_empty(), "the seeded head marks some entities");
    let (starts, ends) = toks.spans.columns();
    for ((_, s), r) in ents.spans.iter().zip(&ents.tokens) {
        assert_eq!(s.start, starts[r.start.get()]);
        assert_eq!(s.end, ends[r.end.get() - 1]);
    }
    // The same document again gives the same layers.
    let (again,) = ner.annotate((view, &toks, None, None)).unwrap();
    assert_eq!(again.tokens, ents.tokens);
    assert_eq!(again.score, ents.score);
    assert_eq!(pos.annotate((view, &toks, None, None)).unwrap().0.tags, tags.tags);

    // The tags are the core head's decode of the word features.
    let run = HeadRunner::open(&dir, Some("pos"), PosTags::ID, &EncoderOptions::default()).unwrap();
    let words: Vec<&str> = TEXT.split(' ').collect();
    let mut enc = WordEncoder::open(&dir, &EncoderOptions::default()).unwrap();
    let f = enc.features(std::slice::from_ref(&words), WordPooling::Mean).unwrap().remove(0);
    let ids: Vec<u32> = (0..words.len() as u32).collect();
    let (mut scores, mut out) = (Vec::new(), Vec::new());
    run.tagger().scores(&f, &ids, &mut scores, &mut Work::default()).unwrap();
    run.tagger().decode(&scores, words.len(), &mut out).unwrap();
    assert_eq!(tags.tags.iter().map(|t| t.0 as u32).collect::<Vec<_>>(), out);
}

#[test]
fn heads_that_do_not_fit_are_refused() {
    let dir = bundle("refusals", |_| {});
    let o = EncoderOptions::default();
    // The entity head is BIO: not a part-of-speech head.
    assert!(PosTagger::open(&dir, Some("ner"), &o).is_err());
    assert!(EntityTagger::open(&dir, Some("pos"), &o).is_err());
    assert!(PosTagger::open(&dir, Some("lemma"), &o).is_err());
    let two = bundle("two-pos", |m| {
        let mut again = m["heads"][0].clone();
        again["name"] = json!("pos2");
        m["heads"].as_array_mut().unwrap().push(again);
    });
    assert!(PosTagger::open(&two, None, &o).is_err());
    assert!(PosTagger::open(&two, Some("pos2"), &o).is_ok());
}

#[test]
fn sentences_come_from_the_token_column_else_the_sentence_layer_else_the_view() {
    use turbo_analysis::Ix;
    use turbo_analysis_layers::Sentences;
    let text = "Ann met Bob . Bob slept .";
    let doc = Document::new(text);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let mut toks = tokens(text);
    assert_eq!(HeadRunner::sequences(view, &toks, None).unwrap(), [(0..7).collect::<Vec<_>>()]);
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 13));
    spans.push(Span::new(14, 25));
    let sentences = Sentences { spans };
    let by_layer = HeadRunner::sequences(view, &toks, Some(&sentences)).unwrap();
    assert_eq!(by_layer, [vec![0, 1, 2, 3], vec![4, 5, 6]]);
    toks.sentence = Some([0, 0, 0, 0, 1, 1, 1].iter().map(|&i| Ix::new(i)).collect());
    assert_eq!(HeadRunner::sequences(view, &toks, None).unwrap(), by_layer);

    // A sentence is tagged as it would be on its own.
    let dir = bundle("sentences", |_| {});
    let pos = PosTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    let (both,) = pos.annotate((view, &toks, None, None)).unwrap();
    let alone_text = "Bob slept .";
    let alone_doc = Document::new(alone_text);
    let alone_view = alone_doc.require(&Key::<View>::DEFAULT).unwrap();
    let (alone,) = pos.annotate((alone_view, &tokens(alone_text), None, None)).unwrap();
    assert_eq!(both.tags[4..], alone.tags[..]);
}

#[test]
fn a_head_names_its_layer_in_the_turboembed_namespace() {
    use turbo_analysis_layers::Entities;
    use turbo_tagger::annotate::same_layer;
    assert!(same_layer("turboembed:pos", PosTags::ID));
    assert!(same_layer("turboembed:entities", Entities::ID));
    assert!(same_layer("turboembed:pos", "turboembed:pos"));
    assert!(!same_layer("turboembed:pos", Entities::ID));
}
