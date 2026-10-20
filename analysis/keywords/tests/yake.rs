//! The YAKE extractor and annotator end to end on real text: a
//! hand-verified ranking, the design's worked example, occurrence
//! offsets, deduplication, comma tokens, interior stop words, and the
//! boundary cases.

use turbo_analysis::{Analyzer, Bound, Document, Error, Key, Layer, Provenance, Span, Spans, Status, ViewId};
use turbo_analysis_layers::{Sentences, Strs, Terms, Tokens};
use turbo_keywords::{Dedup, Keywords, PhraseSimilarity, Stopwords, YAKE, YakeAnnotator};
use turbo_stopword::StopwordLanguage;

/// One document from `sentences`: tokens at the single spaces, each
/// sentence's tokens space-joined with ". " between sentences, folded
/// forms lowercased. Real text, a deterministic lay-down.
fn document(sentences: &[Vec<&str>]) -> Document {
    let mut text = String::new();
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut surface = Strs::default();
    let mut folded = Strs::default();
    for (si, tokens) in sentences.iter().enumerate() {
        if si > 0 {
            text.push(' ');
        }
        for token in tokens {
            if !text.ends_with(' ') && !text.is_empty() {
                text.push(' ');
            }
            let start = text.len() as u32;
            text.push_str(token);
            spans.push(Span::new(start, text.len() as u32));
            surface.push(token);
            folded.push(&token.to_lowercase());
        }
    }
    let mut doc = Document::new(text);
    let mut s_spans = Spans::new(ViewId::SOURCE);
    let mut cursor = 0usize;
    for sent in sentences {
        let start = cursor;
        for token in sent {
            cursor += token.len() + 1;
        }
        s_spans.push(Span::new(start as u32, cursor.saturating_sub(1) as u32));
    }
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, Terms { surface, folded, stem: None, lemma: None }, by).unwrap();
    doc
}

fn run(doc: &mut Document, annotator: YakeAnnotator) -> Keywords {
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Terms>::DEFAULT)
        .step(Bound::new(annotator))
        .build()
        .unwrap();
    let mut docs = vec![std::mem::replace(doc, Document::new(""))];
    let results = analyzer.run(&mut docs);
    results[0].as_ref().unwrap();
    *doc = docs.pop().unwrap();
    let layer = doc.get::<Keywords>().expect("the keywords layer");
    layer.check().unwrap();
    clone_keywords(layer)
}

/// Keywords is not Clone; copy the columns the tests read.
fn clone_keywords(k: &Keywords) -> Keywords {
    let mut out = Keywords::new(k.spans.view());
    let yake = turbo_keywords::algorithm_tagset().tag(YAKE).unwrap();
    for (span, item) in k.iter() {
        out.push(span, item.phrase, item.score, yake, item.terms);
    }
    out
}

fn none() -> Stopwords {
    Stopwords::custom(["qwerty"]).unwrap()
}

/// The hand-verified fixture of the crate's unit tests, as a document.
fn fish_doc() -> Document {
    document(&[vec!["red", "Fish", "blue", "fish"], vec!["swim", "deep", "red", "swim"], vec!["Fish", "blue", "red"]])
}

#[test]
fn ranks_the_fixture_hand_verified() {
    let mut doc = fish_doc();
    // Window 2, the window the hand computation used.
    let kws = run(&mut doc, YakeAnnotator::new(none()).with_window_size(2));
    let text = doc.source();
    // Six items, in document order: one per occurrence. The containment
    // rule drops every phrase nested in a better-scoring survivor: fish
    // blue and fish die inside fish blue fish, blue fish with them,
    // red fish inside red fish blue, and the single red, blue, deep
    // with the multi-word survivors that hold them.
    let phrases: Vec<(u32, u32, &str, f64, u16)> =
        kws.iter().map(|(span, item)| (span.start, span.end, item.phrase, item.score, item.terms)).collect();
    let salience = |s: &str| -> f64 {
        let mut found = phrases.iter().filter(|(_, _, p, _, _)| *p == s).map(|(_, _, _, sc, _)| *sc);
        let first = found.next().unwrap();
        assert!(found.all(|sc| (sc - first).abs() < 1e-12), "one salience per phrase");
        first
    };
    assert_eq!(phrases.len(), 6);
    let expected: [(u32, u32, &str); 6] = [
        (0, 13, "red fish blue"),
        (4, 18, "fish blue fish"),
        (19, 23, "swim"),
        (24, 32, "deep red"),
        (33, 37, "swim"),
        (43, 51, "blue red"),
    ];
    for ((s, e, p), (ws, we, wp, _, wt)) in expected.iter().zip(&phrases) {
        assert_eq!((*s, *e, *p), (*ws, *we, *wp));
        assert_eq!(*wt, wp.matches(' ').count() as u16 + 1, "term count of {wp}");
        assert_eq!(&text[*s as usize..*e as usize].to_lowercase(), *p, "the span slices its phrase");
    }
    // The survivors of dedup, with saliences from the hand computation
    // of the raw scores (best fish blue fish 0.023873960, worst deep
    // red 0.516476708; red fish blue and blue red survive on their own
    // merits, at 0.099170488 and 0.445564983).
    let want = [
        ("fish blue fish", 1.0),
        ("red fish blue", 0.847145538),
        ("swim", 0.401500577),
        ("blue red", 0.143953165),
        ("deep red", 0.0),
    ];
    for (phrase, salience_want) in want {
        assert!((salience(phrase) - salience_want).abs() < 1e-9, "{phrase}: {} != {salience_want}", salience(phrase));
    }
    // Nothing that dedup drops survived: the nested singles, the
    // reversed duplicates, the word-reorder twin of red fish blue.
    for dropped in ["fish", "red", "blue", "deep", "fish blue", "blue fish", "red fish", "fish blue red"] {
        assert!(!phrases.iter().any(|(_, _, p, _, _)| *p == dropped), "{dropped}");
    }
}

#[test]
fn the_worked_example_ranks_google_kaggle() {
    // The design's worked example, with OpenNLP's bundled English list
    // (174 words; smaller than Spark NLP's, so "us" and "tell" are not
    // stop words here and "sources tell us" is a legitimate candidate
    // the formulas score first). Values below are the hand-verified
    // literals of the independent reference computation.
    let mut doc = document(&[
        vec![
            "Sources",
            "tell",
            "us",
            "that",
            "Google",
            "is",
            "acquiring",
            "Kaggle",
            ",",
            "a",
            "platform",
            "that",
            "hosts",
            "data",
            "science",
            "and",
            "machine",
            "learning",
            "competitions",
            ".",
        ],
        vec![
            "Google",
            "is",
            "hosting",
            "its",
            "Cloud",
            "Next",
            "conference",
            "in",
            "San",
            "Francisco",
            "this",
            "week",
            ".",
        ],
        vec!["Kaggle", "co-founder", "CEO", "Anthony", "Goldbloom", "declined", "to", "comment", "."],
    ]);
    let text = doc.source().to_owned();
    let kws = run(&mut doc, YakeAnnotator::new(Stopwords::bundled(StopwordLanguage::English)).with_top_n(10));
    let mut phrases: Vec<(String, f64)> = Vec::new();
    for (_, item) in kws.iter() {
        if !phrases.iter().any(|(p, _)| p == item.phrase) {
            phrases.push((item.phrase.to_owned(), item.score));
        }
    }
    phrases.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    assert_eq!(
        phrases.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(),
        [
            "sources tell us",
            "hosts data science",
            "machine learning competitions",
            "cloud next conference",
            "science and machine",
            "ceo anthony goldbloom",
            "san francisco",
            "acquiring kaggle",
            "platform that hosts",
            "google",
        ]
    );
    let salience = |s: &str| phrases.iter().find(|(p, _)| p == s).unwrap().1;
    for (phrase, want) in [
        ("sources tell us", 1.0),
        ("hosts data science", 0.944879858),
        ("machine learning competitions", 0.944879858),
        ("cloud next conference", 0.602613987),
        ("san francisco", 0.426050792),
        ("acquiring kaggle", 0.242701338),
        ("google", 0.0),
    ] {
        assert!((salience(phrase) - want).abs() < 1e-9, "{phrase}: {} != {want}", salience(phrase));
    }
    // Every occurrence span slices the text its phrase folds from.
    for (span, item) in kws.iter() {
        assert_eq!(&text[span.start as usize..span.end as usize].to_lowercase(), item.phrase);
    }
    // Google appears twice, once per sentence, at exact spans; the
    // multi-word "acquiring kaggle" carries sentence 1's occurrence.
    let google_spans: Vec<_> = kws.iter().filter(|(_, i)| i.phrase == "google").map(|(s, _)| s).collect();
    assert_eq!(google_spans.len(), 2);
    assert_eq!(&text[google_spans[0].start as usize..google_spans[0].end as usize], "Google");
    assert_eq!(&text[google_spans[1].start as usize..google_spans[1].end as usize], "Google");
    assert!(google_spans[0].start < google_spans[1].start);
    // No phrase starts or ends on an English function word.
    let en = turbo_stopword::StopwordList::bundled(StopwordLanguage::English);
    for (_, item) in kws.iter() {
        let (first, last) = (item.phrase.split(' ').next().unwrap(), item.phrase.rsplit(' ').next().unwrap());
        assert!(!en.contains(first) && !en.contains(last), "{first}/{last} of {}", item.phrase);
    }
}

#[test]
fn repeated_phrase_two_spans() {
    // The number the offset story proves: the same phrase in two
    // sentences is two layer items at different exact spans.
    let mut doc = document(&[vec!["big", "cats"], vec!["big", "cats"]]);
    let text = doc.source().to_owned();
    let kws = run(&mut doc, YakeAnnotator::new(none()).with_ngrams(2, 2));
    let spans: Vec<_> = kws.iter().filter(|(_, i)| i.phrase == "big cats").map(|(s, _)| s).collect();
    assert_eq!(spans.len(), 2);
    assert_eq!(&text[spans[0].start as usize..spans[0].end as usize], "big cats");
    assert_eq!(&text[spans[1].start as usize..spans[1].end as usize], "big cats");
    assert_ne!(spans[0], spans[1]);
}

#[test]
fn a_comma_token_survives_as_one_token() {
    let mut doc = document(&[vec!["alpha", "hello,world", "omega"]]);
    let kws = run(&mut doc, YakeAnnotator::new(none()).with_ngrams(1, 1));
    let phrases: Vec<_> = kws.iter().map(|(_, i)| (i.phrase.to_owned(), i.terms)).collect();
    assert!(phrases.contains(&("hello,world".to_owned(), 1)), "{phrases:?}");
    // Candidates are keyed by their token list, so the comma token was
    // never re-split: no "hello" or "world" phrase exists.
    assert!(!phrases.iter().any(|(p, _)| p == "hello"));
    assert!(!phrases.iter().any(|(p, _)| p == "world"));
}

#[test]
fn interior_stop_words_get_the_correction() {
    // "wheels of steel" with "of" a stop word: the edge rule passes
    // (wheels, steel are not stop words), the interior stop word is
    // corrected instead of dropped, and the neighbours' co-occurrence
    // probabilities shape the score.
    let mut doc = document(&[vec!["wheels", "of", "steel", "turn"]]);
    let text = doc.source().to_owned();
    let stop = Stopwords::custom(["of"]).unwrap();
    let kws = run(&mut doc, YakeAnnotator::new(stop).with_ngrams(2, 3));
    let phrase: Vec<_> = kws.iter().filter(|(_, i)| i.phrase == "wheels of steel").collect();
    assert_eq!(phrase.len(), 1, "wheels of steel survives once");
    assert_eq!(&text[phrase[0].0.start as usize..phrase[0].0.end as usize], "wheels of steel");
    // "of steel" and "wheels of" are rejected by the edge rule.
    assert!(!kws.iter().any(|(_, i)| i.phrase.starts_with("of ")));
    assert!(!kws.iter().any(|(_, i)| i.phrase.ends_with(" of")));
}

#[test]
fn dedup_keeps_the_lowest_scoring_of_three() {
    // "google cloud", "google cloud platform" and "cloud platform",
    // one per sentence; containment and overlap leave exactly the
    // lowest-scoring of the nested three.
    let mut doc = document(&[vec!["google", "cloud", "platform"], vec!["google", "cloud"], vec!["cloud", "platform"]]);
    let kws = run(&mut doc, YakeAnnotator::new(none()).with_ngrams(2, 3));
    let survivors: Vec<_> = kws.iter().map(|(_, i)| i.phrase).collect();
    let trio = ["google cloud", "google cloud platform", "cloud platform"];
    let kept = trio.iter().filter(|t| survivors.contains(t)).count();
    assert_eq!(kept, 1, "{survivors:?}");
    assert_eq!(survivors[0], "google cloud platform", "the lowest-scoring of the nested three");
}

#[test]
fn threshold_and_top_n_cut_the_ranking() {
    // Window 2, the hand-computed fixture: raw scores at most 0.1 leave
    // fish blue fish and red fish blue after dedup.
    let mut doc = fish_doc();
    let kws = run(&mut doc, YakeAnnotator::new(none()).with_window_size(2).with_top_n(2));
    let distinct: std::collections::HashSet<_> = kws.iter().map(|(_, i)| i.phrase).collect();
    assert_eq!(distinct.len(), 2, "top_n 2 keeps the two best survivors");
    let mut doc = fish_doc();
    let kws = run(&mut doc, YakeAnnotator::new(none()).with_window_size(2).with_threshold(0.1));
    let distinct: std::collections::HashSet<_> = kws.iter().map(|(_, i)| i.phrase).collect();
    assert_eq!(distinct.len(), 2, "raw scores at most 0.1: the first two after dedup");
    assert!(distinct.contains("fish blue fish") && distinct.contains("red fish blue"));
}

#[test]
fn the_german_list_stays_german() {
    let mut doc = document(&[vec!["Der", "Hund", "und", "die", "Katze", "spielen"]]);
    let kws = run(&mut doc, YakeAnnotator::new(Stopwords::bundled(StopwordLanguage::German)).with_ngrams(1, 3));
    let phrases: Vec<_> = kws.iter().map(|(_, i)| i.phrase).collect();
    assert!(phrases.contains(&"hund"));
    assert!(phrases.iter().any(|p| p.contains("katze")), "{phrases:?}");
    let de = turbo_stopword::StopwordList::bundled(StopwordLanguage::German);
    for phrase in &phrases {
        let (first, last) = (phrase.split(' ').next().unwrap(), phrase.rsplit(' ').next().unwrap());
        assert!(!de.contains(first) && !de.contains(last), "{first}/{last} of {phrase}");
    }
}

#[test]
fn empty_tiny_and_stopword_only_documents() {
    // Empty document: the layer is present and empty, never skipped.
    let mut doc = document(&[]);
    let kws = run(&mut doc, YakeAnnotator::new(none()));
    assert_eq!(kws.len(), 0);
    assert_eq!(kws.spans.len(), 0);

    // Single token.
    let mut doc = document(&[vec!["hi"]]);
    let kws = run(&mut doc, YakeAnnotator::new(none()));
    assert_eq!(kws.len(), 1);
    assert_eq!(kws.iter().next().unwrap().1.phrase, "hi");

    // No candidates at all: every token is a stop word.
    let mut doc = document(&[vec!["the", "and", "the"]]);
    let kws = run(&mut doc, YakeAnnotator::new(Stopwords::bundled(StopwordLanguage::English)));
    assert_eq!(kws.len(), 0, "present-but-empty, not skipped");
}

#[test]
fn tokens_outside_sentences_fail_the_document() {
    // A sentences layer that is present but empty: the merge-walk has
    // nowhere to put the token, and the document fails with the reason.
    let mut doc = Document::new("lonely token");
    let by = Provenance::by("test");
    let mut s_spans = Spans::new(ViewId::SOURCE);
    let _ = &mut s_spans;
    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, by.clone()).unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 12));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, by.clone()).unwrap();
    let mut surface = Strs::default();
    let mut folded = Strs::default();
    surface.push("lonely");
    folded.push("lonely");
    let rest = Terms { surface, folded, stem: None, lemma: None };
    doc.add(&Key::DEFAULT, rest, by).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Terms>::DEFAULT)
        .step(Bound::new(YakeAnnotator::new(none())))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let results = analyzer.run(&mut docs);
    let key = Key::<Keywords>::DEFAULT.id();
    match &results[0] {
        Err(Error::Annotator { detail, .. }) => assert!(detail.contains("outside every sentence"), "{detail}"),
        other => panic!("expected the merge-walk failure, got {other:?}"),
    }
    assert!(matches!(docs[0].status(key), Some(Status::Failed(Error::Annotator { .. }))));
}

#[test]
fn broken_spans_and_foreign_views_are_rejected() {
    use turbo_analysis::{Annotator, View};
    let doc = fish_doc();
    let view = doc.get::<View>().unwrap();
    let annotator = YakeAnnotator::new(none());
    let sentences = doc.get::<Sentences>().unwrap();
    let terms = doc.get::<Terms>().unwrap();

    let mut foreign = Spans::new(ViewId(7));
    foreign.push(Span::new(0, 3));
    assert!(matches!(
        annotator.annotate((view, sentences, &Tokens { spans: foreign, sentence: None }, terms)),
        Err(Error::Annotator { annotator: YakeAnnotator::ID, .. })
    ));
    for (s, e) in [(5, 3), (0, 99), (4, 5)] {
        let mut spans = Spans::new(view.id());
        spans.push(Span { start: s, end: e });
        let tokens = Tokens { spans, sentence: None };
        assert!(
            matches!(
                annotator.annotate((view, sentences, &tokens, terms)),
                Err(Error::Annotator { annotator: YakeAnnotator::ID, .. })
            ),
            "span {s}..{e}"
        );
    }
    // A terms layer not aligned with the tokens.
    let tokens = doc.get::<Tokens>().unwrap();
    let short = Terms { surface: Strs::default(), folded: Strs::default(), stem: None, lemma: None };
    assert!(matches!(
        annotator.annotate((view, sentences, tokens, &short)),
        Err(Error::Annotator { annotator: YakeAnnotator::ID, .. })
    ));
}

#[test]
fn configurations_hash_apart() {
    use turbo_analysis::Annotator;
    let base = YakeAnnotator::new(none());
    let variants = [
        base.clone().with_window_size(2),
        base.clone().with_ngrams(2, 2),
        base.clone().with_top_n(5),
        base.clone().with_threshold(0.5),
        base.clone().with_dedup(Dedup { similarity: PhraseSimilarity::TrigramJaccard, threshold: 0.8 }),
        base.clone().with_dedup(Dedup { similarity: PhraseSimilarity::TokenOverlap, threshold: 0.9 }),
        YakeAnnotator::new(Stopwords::bundled(StopwordLanguage::English)),
    ];
    let base_hash = base.config_hash();
    for v in &variants {
        assert_ne!(v.config_hash(), base_hash, "a configuration hashes like the default");
    }
    for (i, a) in variants.iter().enumerate() {
        assert!(!variants[i + 1..].iter().any(|b| a.config_hash() == b.config_hash()));
    }
}
