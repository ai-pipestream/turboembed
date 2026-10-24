use std::collections::HashMap;

use super::tables::{COUNTRY_BY_ENGLISH_NAME, ISO_COUNTRIES};
use super::*;
use crate::entry::{EntryParts, FEATURE_CLASS_CITY, GazetteerEntry, GeoPoint};
use crate::geocode::{GeoResolution, GeocodeAnnotator, GeocodeError, Geocoder};
use turbo_analysis::{Analyzer, Document, IxRange, Key, Layer, Provenance, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::Tokens;

const SHARE_DELTA: f64 = 1e-9;
const TABLE_CONFIDENCE: f64 = 0.8;

fn make_entry(id: &str, name: &str, country: Option<&str>) -> GazetteerEntry {
    GazetteerEntry::new(EntryParts {
        source: "test".into(),
        record_id: id.into(),
        name: name.into(),
        alternate_names: Vec::new(),
        location: GeoPoint::new(0.0, 0.0).unwrap(),
        bounding_box: None,
        country_code: country.map(Into::into),
        containment: Vec::new(),
        population: 1000,
        feature_class: Some(FEATURE_CLASS_CITY.into()),
        attributes: Vec::new(),
    })
    .unwrap()
}

struct TableGeocoder {
    entries: HashMap<String, (Option<String>, f64)>,
}

impl TableGeocoder {
    fn uniform(pairs: &[(&str, &str)], confidence: f64) -> TableGeocoder {
        let entries = pairs.iter().map(|&(k, v)| (k.to_string(), (Some(v.to_string()), confidence))).collect();
        TableGeocoder { entries }
    }

    fn scored(pairs: &[(&str, Option<&str>, f64)]) -> TableGeocoder {
        let entries = pairs.iter().map(|&(k, v, c)| (k.to_string(), (v.map(Into::into), c))).collect();
        TableGeocoder { entries }
    }
}

impl Geocoder for TableGeocoder {
    fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError> {
        let mut out = Vec::new();
        for &m in location_mentions {
            let surface = &text[m.start as usize..m.end as usize];
            if let Some((country, conf)) = self.entries.get(surface) {
                out.push(GeoResolution::new(m, make_entry(surface, surface, country.as_deref()), *conf)?);
            }
        }
        Ok(out)
    }
}

fn doc_with_location_entities(text: &str, names: &[&str]) -> Document {
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None }, Provenance::by("test"))
        .unwrap();
    let kinds = TagSet::new("ner", &["location"]);
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut tokens = Vec::new();
    let mut kind = Vec::new();
    let mut score = Vec::new();
    let mut found: Vec<Span> = names
        .iter()
        .map(|&name| {
            let start = text.find(name).unwrap();
            Span::new(start as u32, (start + name.len()) as u32)
        })
        .collect();
    found.sort_by_key(|s| s.start);
    for span in found {
        spans.push(span);
        tokens.push(IxRange::new(0, 0));
        kind.push(Tag(0));
        score.push(1.0);
    }
    doc.add(&Key::DEFAULT, Entities { spans, tokens, kind, kinds, score }, Provenance::by("test")).unwrap();
    doc
}

fn run_pipeline<G: Geocoder + Send + Sync + 'static>(geocoder: G, mut doc: Document) -> Document {
    let a = Analyzer::builder()
        .add(GeocodeAnnotator::new(geocoder))
        .add(DocumentRegionAnnotator::new())
        .given(Key::<Entities>::DEFAULT)
        .build()
        .unwrap();
    a.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();
    doc
}

#[test]
fn shares_are_the_same_on_every_run() {
    // 0.1 + 0.2 + 0.3 sums to a different double depending on the order
    // the weights are added in, so a ballot summed in hash map order
    // changes from run to run.
    let geocoder =
        || TableGeocoder::scored(&[("Lyon", Some("FR"), 0.1), ("Turin", Some("IT"), 0.2), ("Porto", Some("PT"), 0.3)]);
    let text = "Lyon, Turin and Porto";
    let shares = |doc: &Document| -> Vec<u64> {
        let ballot = doc.get::<Regions>().unwrap();
        (0..ballot.len()).map(|i| ballot.get(i).unwrap().share().to_bits()).collect()
    };
    let first = shares(&run_pipeline(geocoder(), doc_with_location_entities(text, &["Lyon", "Turin", "Porto"])));
    for _ in 0..40 {
        let doc = run_pipeline(geocoder(), doc_with_location_entities(text, &["Lyon", "Turin", "Porto"]));
        assert_eq!(shares(&doc), first);
    }
}

#[test]
fn country_tables_are_sorted() {
    assert!(ISO_COUNTRIES.windows(2).all(|w| w[0] < w[1]));
    assert!(COUNTRY_BY_ENGLISH_NAME.windows(2).all(|w| w[0].0 < w[1].0));
}

#[test]
fn geocoded_mentions_vote_with_confidence() {
    let geocoder =
        TableGeocoder::uniform(&[("Sydney", "AU"), ("Melbourne", "AU"), ("Auckland", "NZ")], TABLE_CONFIDENCE);
    let doc = run_pipeline(
        geocoder,
        doc_with_location_entities(
            "flights from Sydney and Melbourne to Auckland",
            &["Sydney", "Melbourne", "Auckland"],
        ),
    );
    let ballot = doc.get::<Regions>().unwrap();
    assert_eq!(ballot.len(), 2);
    assert_eq!(ballot.get(0).unwrap().country_code(), "AU");
    assert!((ballot.get(0).unwrap().share() - 2.0 / 3.0).abs() < SHARE_DELTA);
    assert_eq!(ballot.get(1).unwrap().country_code(), "NZ");
}

#[test]
fn country_names_vote_when_unresolved_and_resolved_mentions_override_name_table() {
    let doc = run_pipeline(
        TableGeocoder::uniform(&[], TABLE_CONFIDENCE),
        doc_with_location_entities("mining exports from Australia rose", &["Australia"]),
    );
    let ballot = doc.get::<Regions>().unwrap();
    assert_eq!(ballot.len(), 1);
    assert_eq!(ballot.get(0).unwrap().country_code(), "AU");
    assert!((ballot.get(0).unwrap().share() - 1.0).abs() < SHARE_DELTA);

    // Resolved Georgia -> US overrides country name GE.
    let doc_ga = run_pipeline(
        TableGeocoder::uniform(&[("Georgia", "US"), ("Atlanta", "US")], TABLE_CONFIDENCE),
        doc_with_location_entities(
            "Georgia's secretary of state confirmed the recount in Atlanta",
            &["Georgia", "Atlanta"],
        ),
    );
    let ballot_ga = doc_ga.get::<Regions>().unwrap();
    assert_eq!(ballot_ga.len(), 1);
    assert_eq!(ballot_ga.get(0).unwrap().country_code(), "US");

    // Resolved Australia casts one vote at its resolution confidence, tying Auckland.
    let doc_tie = run_pipeline(
        TableGeocoder::uniform(&[("Australia", "AU"), ("Auckland", "NZ")], TABLE_CONFIDENCE),
        doc_with_location_entities("exports from Australia to Auckland rose", &["Australia", "Auckland"]),
    );
    let ballot_tie = doc_tie.get::<Regions>().unwrap();
    assert_eq!(ballot_tie.len(), 2);
    assert_eq!(ballot_tie.get(0).unwrap().country_code(), "AU");
    assert!((ballot_tie.get(0).unwrap().share() - 0.5).abs() < SHARE_DELTA);
    assert_eq!(ballot_tie.get(1).unwrap().country_code(), "NZ");
    assert!((ballot_tie.get(1).unwrap().share() - 0.5).abs() < SHARE_DELTA);

    // Unresolved country name (0.95) outweighs geocoded city (0.80).
    let doc_out = run_pipeline(
        TableGeocoder::uniform(&[("Sydney", "AU")], TABLE_CONFIDENCE),
        doc_with_location_entities("the Sydney office reports to New Zealand headquarters", &["Sydney", "New Zealand"]),
    );
    let ballot_out = doc_out.get::<Regions>().unwrap();
    assert_eq!(ballot_out.len(), 2);
    assert_eq!(ballot_out.get(0).unwrap().country_code(), "NZ");
    assert!((ballot_out.get(0).unwrap().share() - 0.95 / (0.95 + TABLE_CONFIDENCE)).abs() < SHARE_DELTA);
    assert_eq!(ballot_out.get(1).unwrap().country_code(), "AU");
    assert_eq!(DocumentRegionAnnotator::winner(ballot_out, 0.05).unwrap().unwrap().country_code(), "NZ");
    assert_eq!(DocumentRegionAnnotator::winner(ballot_out, 0.1).unwrap(), None);
}

#[test]
fn country_name_spelling_variants_match() {
    let cases = [
        ("Côte d\u{2019}Ivoire", "CI"),
        ("Côte d'Ivoire", "CI"),
        ("Cote d'Ivoire", "CI"),
        ("México", "MX"),
        ("AUSTRALIA", "AU"),
        ("australia", "AU"),
    ];
    for (mention, expected) in cases {
        let text = format!("new offices in {mention} opened");
        let doc =
            run_pipeline(TableGeocoder::uniform(&[], TABLE_CONFIDENCE), doc_with_location_entities(&text, &[mention]));
        let ballot = doc.get::<Regions>().unwrap();
        assert_eq!(ballot.len(), 1, "{mention}");
        assert_eq!(ballot.get(0).unwrap().country_code(), expected, "{mention}");
    }
}

#[test]
fn flag_emojis_vote_and_invalid_flags_are_ignored() {
    // Australia flag U+1F1E6 U+1F1FA.
    let doc = run_pipeline(
        TableGeocoder::uniform(&[], TABLE_CONFIDENCE),
        doc_with_location_entities("shipping update \u{1F1E6}\u{1F1FA} soon", &[]),
    );
    let ballot = doc.get::<Regions>().unwrap();
    assert_eq!(ballot.len(), 1);
    assert_eq!(ballot.get(0).unwrap().country_code(), "AU");

    // Consecutive FR and DE flags segment left-to-right and tie by country code (DE before FR).
    let doc2 = run_pipeline(
        TableGeocoder::uniform(&[], TABLE_CONFIDENCE),
        doc_with_location_entities("match report \u{1F1EB}\u{1F1F7}\u{1F1E9}\u{1F1EA} today", &[]),
    );
    let ballot2 = doc2.get::<Regions>().unwrap();
    assert_eq!(ballot2.len(), 2);
    assert_eq!(ballot2.get(0).unwrap().country_code(), "DE");
    assert_eq!(ballot2.get(1).unwrap().country_code(), "FR");

    // Disagreeing flags blend with geocoded evidence.
    let doc3 = run_pipeline(
        TableGeocoder::uniform(&[("Sydney", "AU")], TABLE_CONFIDENCE),
        doc_with_location_entities("Sydney update \u{1F1EB}\u{1F1F7} \u{1F1E9}\u{1F1EA}", &["Sydney"]),
    );
    let ballot3 = doc3.get::<Regions>().unwrap();
    assert_eq!(ballot3.len(), 3);
    assert_eq!(ballot3.get(0).unwrap().country_code(), "DE");
    assert!((ballot3.get(0).unwrap().share() - 0.95 / 2.7).abs() < SHARE_DELTA);
    assert_eq!(ballot3.get(1).unwrap().country_code(), "FR");
    assert_eq!(ballot3.get(2).unwrap().country_code(), "AU");
    assert!((ballot3.get(2).unwrap().share() - 0.8 / 2.7).abs() < SHARE_DELTA);

    // Lone regional indicators do not vote.
    let doc4 = run_pipeline(
        TableGeocoder::uniform(&[], TABLE_CONFIDENCE),
        doc_with_location_entities("grade \u{1F1E6} list \u{1F1FA}", &[]),
    );
    assert!(doc4.get::<Regions>().unwrap().is_empty());

    // Subdivision tag-sequence flag (England GB-ENG) does not vote.
    let england = "match report \u{1F3F4}\u{E0067}\u{E0062}\u{E0065}\u{E006E}\u{E0067}\u{E007F} today";
    let doc5 = run_pipeline(TableGeocoder::uniform(&[], TABLE_CONFIDENCE), doc_with_location_entities(england, &[]));
    assert!(doc5.get::<Regions>().unwrap().is_empty());

    // Unassigned XX pair does not vote.
    let doc6 = run_pipeline(
        TableGeocoder::uniform(&[], TABLE_CONFIDENCE),
        doc_with_location_entities("status \u{1F1FD}\u{1F1FD} checked", &[]),
    );
    assert!(doc6.get::<Regions>().unwrap().is_empty());
}

#[test]
fn ballot_edge_cases_tie_breaking_zero_confidence_and_null_country() {
    for text in ["trains between London and Paris", "trains between Paris and London"] {
        let doc = run_pipeline(
            TableGeocoder::scored(&[("London", Some("GB"), 0.8), ("Paris", Some("FR"), 0.8)]),
            doc_with_location_entities(text, &["London", "Paris"]),
        );
        let ballot = doc.get::<Regions>().unwrap();
        assert_eq!(ballot.len(), 2);
        assert_eq!(ballot.get(0).unwrap().country_code(), "FR");
        assert_eq!(ballot.get(1).unwrap().country_code(), "GB");
    }

    // One confident mention outvotes two weak ones.
    let doc = run_pipeline(
        TableGeocoder::scored(&[("Nice", Some("FR"), 0.3), ("Nancy", Some("FR"), 0.3), ("Chicago", Some("US"), 0.7)]),
        doc_with_location_entities("flights from Nice and Nancy to Chicago", &["Nice", "Nancy", "Chicago"]),
    );
    let ballot = doc.get::<Regions>().unwrap();
    assert_eq!(ballot.len(), 2);
    assert_eq!(ballot.get(0).unwrap().country_code(), "US");
    assert_eq!(ballot.get(0).unwrap().share(), 0.7 / (0.7 + 0.6));
    assert_eq!(ballot.get(1).unwrap().country_code(), "FR");

    // Zero-confidence resolution casts no vote and does not dilute.
    let doc_zero = run_pipeline(
        TableGeocoder::scored(&[("Bilbao", Some("ES"), 0.0)]),
        doc_with_location_entities("a dispatch from Bilbao", &["Bilbao"]),
    );
    assert!(doc_zero.get::<Regions>().unwrap().is_empty());

    let doc_zero2 = run_pipeline(
        TableGeocoder::scored(&[("Bilbao", Some("ES"), 0.0), ("Sydney", Some("AU"), 0.8)]),
        doc_with_location_entities("flights from Bilbao to Sydney", &["Bilbao", "Sydney"]),
    );
    let ballot_zero2 = doc_zero2.get::<Regions>().unwrap();
    assert_eq!(ballot_zero2.len(), 1);
    assert_eq!(ballot_zero2.get(0).unwrap().country_code(), "AU");
    assert_eq!(ballot_zero2.get(0).unwrap().share(), 1.0);

    // Null country code resolution casts no vote, or falls back to country name if mention is a country name.
    let doc_null = run_pipeline(
        TableGeocoder::scored(&[("Pacific", None, 0.9), ("Sydney", Some("AU"), 0.8)]),
        doc_with_location_entities("across the Pacific to Sydney", &["Pacific", "Sydney"]),
    );
    let ballot_null = doc_null.get::<Regions>().unwrap();
    assert_eq!(ballot_null.len(), 1);
    assert_eq!(ballot_null.get(0).unwrap().country_code(), "AU");

    let doc_country_fallback = run_pipeline(
        TableGeocoder::scored(&[("Australia", None, 0.8)]),
        doc_with_location_entities("mining exports from Australia rose", &["Australia"]),
    );
    let ballot_cf = doc_country_fallback.get::<Regions>().unwrap();
    assert_eq!(ballot_cf.len(), 1);
    assert_eq!(ballot_cf.get(0).unwrap().country_code(), "AU");
}

#[test]
fn document_region_annotator_rejects_invalid_entity_spans_without_panicking() {
    let text = "México exports";
    let annotator = DocumentRegionAnnotator::new();

    for bad_span in [
        Span::new(0, 99),          // past end of text
        Span { start: 5, end: 2 }, // inverted span
        Span::new(0, 2),           // end splits 2-byte UTF-8 'é' (bytes 1..3)
        Span::new(2, 3),           // start splits 2-byte UTF-8 'é' (bytes 1..3)
    ] {
        let doc = Document::new(text);
        let view = doc.get::<View>().unwrap();
        let kinds = TagSet::new("ner", &["location"]);
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(bad_span);
        let entities =
            Entities { spans, tokens: vec![IxRange::new(0, 0)], kind: vec![Tag(0)], kinds, score: vec![1.0] };
        let locations = Locations::new(ViewId::SOURCE);
        let err = annotator.annotate((view, &entities, &locations)).unwrap_err();
        assert!(
            matches!(&err, Error::Annotator { annotator: DocumentRegionAnnotator::ID, detail } if detail.contains("outside the text or off a UTF-8 boundary")),
            "unexpected error for {bad_span:?}: {err:?}"
        );
    }
}

#[test]
fn document_region_annotator_config_hash_reflects_location_types() {
    let default_ann = DocumentRegionAnnotator::new();
    let explicit_default = DocumentRegionAnnotator::with_location_types(["LOCATION"]).unwrap();
    let gpe_ann = DocumentRegionAnnotator::with_location_types(["gpe"]).unwrap();
    let multi_a = DocumentRegionAnnotator::with_location_types(["location", "gpe"]).unwrap();
    let multi_b = DocumentRegionAnnotator::with_location_types(["GPE", "Location"]).unwrap();

    assert_ne!(default_ann.config_hash(), 0);
    assert_eq!(default_ann.config_hash(), explicit_default.config_hash());
    assert_ne!(default_ann.config_hash(), gpe_ann.config_hash());
    assert_ne!(default_ann.config_hash(), multi_a.config_hash());
    assert_eq!(multi_a.config_hash(), multi_b.config_hash());
}
