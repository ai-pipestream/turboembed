//! Case folding does not change which place names are found (#2115, #2141).
//! The place-name step matches text as written, so in a chain that folds case
//! it reads the cased view: the normalizer chain with the fold steps left out.
//! Tokens and subwords still read the folded view, and every place is a span
//! of the source.

mod common;

use tokio_stream::StreamExt as _;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Document, Key, View, ViewId};
use turbo_analysis_layers::{Entities, Sentences, Subwords, Tokens};
use turbo_kserve::progressive::compile_analyzer;
use turbo_kserve::proto::result::AnalyzePipelineRequest;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::results::document::codecs;

/// Places with non-ASCII case pairs, and three spellings a full fold turns
/// into a place name the text does not write: the ligatures of `finland`
/// and `florida`, and the long s of `Nis`.
const TEXT: &str = "Zo\u{eb} flew from S\u{e3}o Paulo to \u{130}stanbul, then Z\u{fc}rich, \u{c5}rhus, \u{141}\u{f3}d\u{17a}, \
                    Gie\u{df}en and K\u{d6}LN. Not places: \u{fb01}nland, \u{fb02}orida, Ni\u{17f}.";

/// Places written with an invisible character (`Z\u{fc}\u{200b}rich`),
/// decomposed non-NFC forms (`S\u{61}\u{303}o Paulo`, `\u{41}\u{30a}rhus`),
/// and extra whitespace (`New   York`), alongside non-ASCII case pairs and
/// the ligature/long-s spellings (`\u{fb01}nland`, `\u{fb02}orida`, `Ni\u{17f}`).
const DIRTY: &str = "Zo\u{eb} flew from S\u{61}\u{303}o Paulo to \u{130}stanbul, then Z\u{fc}\u{200b}rich, \
                     \u{41}\u{30a}rhus, New   York, Gie\u{df}en and K\u{d6}LN. \
                     Not places: \u{fb01}nland, \u{fb02}orida, Ni\u{17f}.";

const LEXICAL: &[&str] = &["turbo-tokenize:uax29-untyped", "turbo-geo:names"];

fn analyzed_text(text: &str, norms: &[&str]) -> Document {
    let mut doc = Document::new(text);
    let steps: Vec<&str> = norms.iter().chain(LEXICAL).copied().collect();
    let analyzer = compile_analyzer(&doc, &steps).unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().for_each(Result::unwrap);
    doc
}

fn analyzed(norms: &[&str]) -> Document {
    analyzed_text(TEXT, norms)
}

/// Each place as a slice of `text`: `get` refuses a span that is not
/// on character boundaries.
fn places_in<'a>(text: &'a str, doc: &Document) -> Vec<&'a str> {
    let entities = doc.get::<Entities>().expect("the place-name layer");
    assert_eq!(entities.spans.view(), ViewId::SOURCE);
    let slice = |start: u32, end: u32| text.get(start as usize..end as usize).expect("a span on char boundaries");
    entities.spans.iter().map(|(_, s)| slice(s.start, s.end)).collect()
}

fn places(doc: &Document) -> Vec<&'static str> {
    places_in(TEXT, doc)
}

fn without_fold<'a>(norms: &[&'a str]) -> Vec<&'a str> {
    norms.iter().copied().filter(|&s| !matches!(s, "turbo:normalize:case-fold" | "case_fold")).collect()
}

#[test]
fn a_fold_first_finds_the_places_of_the_unnormalized_request() {
    let unnormalized = analyzed(&[]);
    let expected = places(&unnormalized);
    for place in
        ["S\u{e3}o Paulo", "\u{130}stanbul", "Z\u{fc}rich", "\u{c5}rhus", "\u{141}\u{f3}d\u{17a}", "Gie\u{df}en"]
    {
        assert!(expected.contains(&place), "{place:?} in {expected:?}");
    }
    assert!(!expected.iter().any(|p| p.contains('\u{fb01}') || p.contains('\u{17f}')), "{expected:?}");
    for norms in [
        &["turbo:normalize:case-fold"][..],
        &["case_fold", "turbo:normalize:whitespace"][..],
        &["turbo:normalize:case-fold", "turbo:normalize:invisible"][..],
    ] {
        let doc = analyzed(norms);
        assert_eq!(places(&doc), expected, "{norms:?}");
        // The fold still reaches the steps that read the last view.
        let last = doc.require(&Key::<View>::named("normalized")).unwrap();
        assert!(last.text().contains("z\u{fc}rich") && last.text().contains("giessen"), "{}", last.text());
        assert_eq!(doc.get::<Tokens>().unwrap().spans.len(), unnormalized.get::<Tokens>().unwrap().spans.len());
    }
}

#[test]
fn a_fold_after_a_normalizer_finds_the_places_of_the_chain_without_it() {
    for before in [&["turbo:normalize"][..], &["turbo:normalize:nfc"][..]] {
        let expected = analyzed(before);
        let with_fold: Vec<&str> = before.iter().copied().chain(["turbo:normalize:case-fold"]).collect();
        let folded_last = analyzed(&with_fold);
        assert_eq!(places(&folded_last), places(&expected), "{with_fold:?}");
        assert!(folded_last.get_key(&Key::<View>::named("cased")).is_none(), "{with_fold:?}");
        let then_more: Vec<&str> = with_fold.iter().copied().chain(["turbo:normalize:whitespace"]).collect();
        assert_eq!(places(&analyzed(&then_more)), places(&expected), "{then_more:?}");
    }
}

#[test]
fn normalizers_after_a_fold_still_reach_place_names() {
    let raw = places_in(DIRTY, &analyzed_text(DIRTY, &[]));
    assert!(!raw.contains(&"Z\u{fc}\u{200b}rich"), "{raw:?}");

    let all_cleaned =
        analyzed_text(DIRTY, &["turbo:normalize:invisible", "turbo:normalize:nfc", "turbo:normalize:whitespace"]);
    let cleaned_places = places_in(DIRTY, &all_cleaned);
    for place in [
        "S\u{61}\u{303}o Paulo",
        "\u{130}stanbul",
        "Z\u{fc}\u{200b}rich",
        "\u{41}\u{30a}rhus",
        "New   York",
        "Gie\u{df}en",
    ] {
        assert!(cleaned_places.contains(&place), "{place:?} in {cleaned_places:?}");
    }
    assert!(!cleaned_places.iter().any(|p| p.contains('\u{fb01}') || p.contains('\u{17f}')), "{cleaned_places:?}");

    for norms in [
        // Fold first.
        &["turbo:normalize:case-fold", "turbo:normalize:invisible"][..],
        &["turbo:normalize:case-fold", "turbo:normalize:nfc"][..],
        &["turbo:normalize:case-fold", "turbo:normalize:whitespace"][..],
        &["turbo:normalize:case-fold", "turbo:normalize"][..],
        &["case_fold", "turbo:normalize:invisible", "turbo:normalize:nfc", "turbo:normalize:whitespace"][..],
        // Fold in the middle.
        &["turbo:normalize:invisible", "turbo:normalize:case-fold", "turbo:normalize:nfc"][..],
        &[
            "turbo:normalize:invisible",
            "turbo:normalize:case-fold",
            "turbo:normalize:nfc",
            "turbo:normalize:whitespace",
        ][..],
        &[
            "turbo:normalize:invisible",
            "turbo:normalize:nfc",
            "turbo:normalize:case-fold",
            "turbo:normalize:whitespace",
        ][..],
        // Fold last.
        &["turbo:normalize:invisible", "turbo:normalize:case-fold"][..],
        &["turbo:normalize:nfc", "turbo:normalize:case-fold"][..],
        &["turbo:normalize:whitespace", "turbo:normalize:case-fold"][..],
        &[
            "turbo:normalize:invisible",
            "turbo:normalize:nfc",
            "turbo:normalize:whitespace",
            "turbo:normalize:case-fold",
        ][..],
    ] {
        let unfolded = without_fold(norms);
        let expected = analyzed_text(DIRTY, &unfolded);
        let got = analyzed_text(DIRTY, norms);
        assert_eq!(places_in(DIRTY, &got), places_in(DIRTY, &expected), "{norms:?}");
        assert!(expected.get_key(&Key::<View>::named("cased")).is_none(), "{unfolded:?}");
        if matches!(norms.last().copied(), Some("turbo:normalize:case-fold" | "case_fold")) {
            assert!(got.get_key(&Key::<View>::named("cased")).is_none(), "{norms:?}");
            assert!(got.get_key(&Key::<View>::named("cased_1")).is_none(), "{norms:?}");
        }
    }

    let full = analyzed_text(
        DIRTY,
        &["case_fold", "turbo:normalize:invisible", "turbo:normalize:nfc", "turbo:normalize:whitespace"],
    );
    let cased = full.require(&Key::<View>::named("cased")).unwrap();
    assert!(
        cased.text().contains("S\u{e3}o Paulo")
            && cased.text().contains("Z\u{fc}rich")
            && cased.text().contains("\u{c5}rhus")
            && cased.text().contains("New York")
            && cased.text().contains("Gie\u{df}en"),
        "{}",
        cased.text()
    );
    let last = full.require(&Key::<View>::named("normalized")).unwrap();
    assert!(
        last.text().contains("s\u{e3}o paulo")
            && last.text().contains("z\u{fc}rich")
            && last.text().contains("new york")
            && last.text().contains("giessen"),
        "{}",
        last.text()
    );
}

#[tokio::test]
async fn tokens_and_subwords_still_read_the_folded_view() {
    let (_s, mut client) = common::serve().await;
    let steps = vec![
        "turbo:normalize:case-fold".into(),
        "turbo:normalize:invisible".into(),
        "turbo:normalize:nfc".into(),
        "turbo:normalize:whitespace".into(),
        "turbo-sentdetect:rules".into(),
        "turbo-tokenize:uax29-untyped".into(),
        "turbo-geo:names".into(),
        "turbo:subwords".into(),
    ];
    let req = AnalyzePipelineRequest {
        doc_id: "fold-first-subwords".into(),
        text: DIRTY.into(),
        steps,
        model: common::NAME.into(),
        ..Default::default()
    };
    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut complete = None;
    while let Some(ev) = stream.next().await {
        if let Some(Event::Complete(c)) = ev.unwrap().event {
            complete = Some(c);
        }
    }
    let complete = complete.expect("complete event");
    let graph = DocumentGraph::from_bytes(&complete.graph).unwrap();
    let doc = import(&graph, codecs()).unwrap();

    let norm_id = Key::<View>::named("normalized").id();
    let cased_id = Key::<View>::named("cased").id();
    assert!(doc.provenance(Key::<Tokens>::DEFAULT.id()).unwrap().reads.contains(&norm_id));
    assert!(doc.provenance(Key::<Subwords>::DEFAULT.id()).unwrap().reads.contains(&norm_id));
    assert!(doc.provenance(Key::<Sentences>::DEFAULT.id()).unwrap().reads.contains(&cased_id));
    assert!(doc.provenance(Key::<Entities>::DEFAULT.id()).unwrap().reads.contains(&cased_id));

    let found = places_in(DIRTY, &doc);
    for place in [
        "S\u{61}\u{303}o Paulo",
        "\u{130}stanbul",
        "Z\u{fc}\u{200b}rich",
        "\u{41}\u{30a}rhus",
        "New   York",
        "Gie\u{df}en",
    ] {
        assert!(found.contains(&place), "{place:?} in {found:?}");
    }
    assert_eq!(doc.get::<Subwords>().unwrap().spans.view(), ViewId::SOURCE);
}
