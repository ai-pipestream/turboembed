//! #2398: `PiiTagger` gives a name its head reads two ways as two
//! candidates on one span, the `city` reading and the person reading,
//! each at its own probability.

mod common;

use common::bundle;
use serde_json::json;
use turbo_analysis::{Analyzer, Annotator, Document, Key, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Pii, PiiCandidates, PiiKind, Subwords, Tokens};
use turbo_pii::ModelCandidate;
use turbo_pii::{Pack, PiiDetector};
use turbo_tagger::PiiTagger;
use turbo_tagger::pii::{SECOND_READING_MIN, second_reading_config_hash};
use turbo_tokenize::SimpleTokenizer;

const TEXT: &str = "Paris won the race on Saturday, and Columbus drove home after the shift. We saw Madison at \
                    lunch, and they called Salem twice before Dayton answered the phone in Verona.";

/// The head's candidates on [`TEXT`] at `temperature`, from a seeded
/// head of the three name types.
fn candidates(name: &str, temperature: f32) -> (PiiTagger, Vec<ModelCandidate>) {
    let dir = bundle(name, |m| {
        m["heads"] = json!([{
            "name": "pii",
            "layer": "turboembed:pii",
            "kind": "HEAD_TOKEN_TAGGER",
            "labels": ["O", "B-city", "I-city", "B-first_name", "I-first_name", "B-last_name", "I-last_name"],
            "scheme": "SCHEME_BIO",
            "word_pooling": "WORD_POOLING_FIRST",
            "decode": "DECODE_ARGMAX",
            "artifact": "weights-f32",
            "tensor_prefix": "heads.pii"
        }]);
    });
    let tagger =
        PiiTagger::open(&dir, None, &EncoderOptions::default()).unwrap().with_temperature(temperature).unwrap();
    let analyzer = Analyzer::builder().add(SimpleTokenizer::default()).build().unwrap();
    let mut docs = vec![Document::new(TEXT)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = &docs[0];
    let got = tagger
        .model_candidates(doc.get::<View>().unwrap(), doc.get::<Tokens>().unwrap(), doc.get::<Subwords>())
        .unwrap();
    (tagger, got)
}

#[test]
fn a_name_read_two_ways_gives_two_candidates_on_one_span() {
    // A soft head: the types' shares lie close together.
    let (tagger, soft) = candidates("pii-two-soft", 4.0);
    let kinds = tagger.custom_kinds();
    let person = [kinds.find("first-name").unwrap(), kinds.find("last-name").unwrap()];
    let mut seconds = 0;
    for pair in soft.windows(2) {
        let (first, second) = (pair[0], pair[1]);
        if first.span != second.span {
            continue;
        }
        seconds += 1;
        // The two readings are a city and a person, either way round.
        assert!(
            first.kind == PiiKind::City && person.contains(&second.kind)
                || person.contains(&first.kind) && second.kind == PiiKind::City,
            "{first:?} then {second:?}"
        );
        assert!(second.probability >= SECOND_READING_MIN && second.probability < 0.5, "{second:?}");
        // The first is the likelier as scored, so no bar admits the
        // second and turns the first down.
        assert!(second.probability < first.probability, "{first:?} then {second:?}");
    }
    assert!(seconds > 0, "a soft head reads some name two ways: {soft:?}");
    // No span has more than two candidates.
    assert!(soft.windows(3).all(|w| w[0].span != w[2].span));

    // A sharp head is sure of each reading: the same spans, each once.
    let (_, sharp) = candidates("pii-two-sharp", 0.01);
    assert!(sharp.windows(2).all(|w| w[0].span != w[1].span), "{sharp:?}");
    let firsts: Vec<_> =
        soft.iter().enumerate().filter(|&(i, c)| i == 0 || soft[i - 1].span != c.span).map(|(_, c)| c).collect();
    assert_eq!(firsts.len(), sharp.len());
    assert!(firsts.iter().zip(&sharp).all(|(a, b)| (a.span, a.kind) == (b.span, b.kind)));
}

/// Kimi's finding on #2411: the configuration hash did not change with
/// the second reading, so an incremental run reused layers an earlier
/// build wrote with one reading a span. A document whose PII layers are
/// recorded under the earlier hash is computed again, and one recorded
/// under the tagger's own hash is not.
#[test]
fn layers_cached_before_the_second_reading_are_computed_again() {
    let (tagger, _) = candidates("pii-two-cache", 4.0);
    let (current, before) = (tagger.config_hash(), tagger.single_reading_config_hash());
    assert_ne!(current, before);
    // The hash holds the bar of the second reading: another bar, another hash.
    assert_eq!(current, second_reading_config_hash(before, SECOND_READING_MIN));
    assert_ne!(current, second_reading_config_hash(before, SECOND_READING_MIN + 0.01));
    let analyzer = Analyzer::builder().add(SimpleTokenizer::default()).add(tagger).build().unwrap();
    let mut doc = Document::new(TEXT);
    assert!(analyzer.analyze_incremental(&mut doc).unwrap().was_executed(PiiTagger::ID));
    assert!(analyzer.analyze_incremental(&mut doc).unwrap().was_skipped(PiiTagger::ID));

    // Layers as an earlier build recorded them, here with nothing in them.
    let (pii_key, candidates_key) = (Key::<Pii>::DEFAULT, Key::<PiiCandidates>::DEFAULT);
    let mut pii_prov = doc.provenance(pii_key.id()).unwrap().clone();
    let mut candidates_prov = doc.provenance(candidates_key.id()).unwrap().clone();
    assert_eq!((pii_prov.config, candidates_prov.config), (current, current));
    pii_prov.config = before;
    candidates_prov.config = before;
    let view = doc.get::<View>().unwrap().id();
    let found = doc.get::<PiiCandidates>().unwrap().candidates().count();
    assert!(found > 0);
    doc.remove(pii_key.id());
    doc.remove(candidates_key.id());
    doc.add(&pii_key, Pii::new(view), pii_prov).unwrap();
    doc.add(&candidates_key, PiiCandidates::new(view), candidates_prov).unwrap();

    assert!(analyzer.analyze_incremental(&mut doc).unwrap().was_executed(PiiTagger::ID));
    assert_eq!(doc.provenance(candidates_key.id()).unwrap().config, current);
    assert_eq!(doc.get::<PiiCandidates>().unwrap().candidates().count(), found);
    assert!(analyzer.analyze_incremental(&mut doc).unwrap().was_skipped(PiiTagger::ID));
}

/// The seeded head of [`candidates`] with the Places rules and a bar.
fn with_places(name: &str, bar: Option<f32>) -> (String, Vec<(u32, PiiKind)>) {
    let (tagger, _) = candidates(name, 4.0);
    let tagger = match bar {
        Some(bar) => tagger.with_score_threshold(bar).unwrap(),
        None => tagger,
    }
    .with_rules(PiiDetector::pack(Pack::Places));
    let analyzer = Analyzer::builder().add(SimpleTokenizer::default()).add(tagger).build().unwrap();
    let text = "When Denton reached Windsor the rain had stopped.";
    let mut docs = vec![Document::new(text)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let pii = docs[0].get::<Pii>().unwrap();
    (text.to_owned(), pii.mentions().map(|m| (m.span.start, m.kind)).collect())
}

/// Cursor's finding on #2411: the second reading's share of a whole type
/// could lie above the first reading's probability of one label. The
/// seeded head reads `Windsor` as `city` at 0.313 with a person's share
/// of 0.319, so a bar of 0.316 turned the city down, kept the person and
/// took the rule's city away with it (#2312). The second reading is now
/// given only below the first, and the city stands.
#[test]
fn a_second_reading_never_outscores_the_first_and_takes_a_city_away() {
    let (text, pii) = with_places("pii-two-windsor", Some(0.316));
    let windsor = text.find("Windsor").unwrap() as u32;
    let on_windsor: Vec<PiiKind> = pii.iter().filter(|m| m.0 == windsor).map(|m| m.1).collect();
    assert_eq!(on_windsor, [PiiKind::City]);
}

/// With no bar every candidate of the head is a mention, a second
/// reading too: the name carries both kinds, as two rules' kinds on one
/// span do. A bar between the two keeps the likelier alone.
#[test]
fn with_no_bar_both_readings_are_mentions() {
    let (text, pii) = with_places("pii-two-no-bar", None);
    let denton = text.find("Denton").unwrap() as u32;
    let on_denton: Vec<PiiKind> = pii.iter().filter(|m| m.0 == denton).map(|m| m.1).collect();
    assert_eq!(on_denton.len(), 2, "{on_denton:?}");
    assert!(on_denton.contains(&PiiKind::City));
}
