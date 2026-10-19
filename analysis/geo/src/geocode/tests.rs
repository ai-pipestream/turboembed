use super::*;
use crate::bundled::BundledGazetteer;
use crate::entry::{EntryParts, FEATURE_CLASS_ADMIN, FEATURE_CLASS_CITY};
use crate::in_memory::InMemoryGazetteer;
use crate::user::UserGazetteer;
use turbo_analysis::{Analyzer, Document, Ix, IxRange, Key, Layer, Provenance, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Entities, Tokens};

const ROWS: &[&str] = &[
    "1\tSpringfield\t\t39.79968\t-89.64399\tUS\tCITY\t750000\t\tIllinois",
    "2\tSpringfield\t\t37.18095\t-93.29229\tUS\tCITY\t450000\t\tMissouri",
    "3\tSpringfield\t\t42.10148\t-72.58982\tUS\tCITY\t680000\t\tMassachusetts",
    "4\tSolo\t\t-7.55611\t110.83167\tID\tCITY\t100000\t\tCentral Java",
    "5\tTietown\t\t10.0\t10.0\t\tADMIN\t5000",
    "6\tTietown\t\t11.0\t11.0\t\tCITY\t5000",
    "7\tGhostville\t\t20.0\t20.0\t\tCITY\t0",
    "8\tGhostville\t\t21.0\t21.0\t\tCITY\t0",
];

fn fixture_gazetteer() -> UserGazetteer {
    let mut data = String::new();
    for row in ROWS {
        data.push_str(row);
        data.push('\n');
    }
    UserGazetteer::from_bytes(data.as_bytes(), "naturalearth").unwrap()
}

fn mention_of(text: &str, mention: &str) -> Span {
    let start = text.find(mention).unwrap();
    Span::new(start as u32, (start + mention.len()) as u32)
}

fn place(id: &str, name: &str, lat: f64, lon: f64, country: Option<&str>, population: i64) -> GazetteerEntry {
    GazetteerEntry::new(EntryParts {
        source: "test".into(),
        record_id: id.into(),
        name: name.into(),
        alternate_names: Vec::new(),
        location: GeoPoint::new(lat, lon).unwrap(),
        bounding_box: None,
        country_code: country.map(Into::into),
        containment: Vec::new(),
        population,
        feature_class: Some(FEATURE_CLASS_CITY.into()),
        attributes: Vec::new(),
    })
    .unwrap()
}

struct ReversedGazetteer<G>(G);

impl<G: Gazetteer> Gazetteer for ReversedGazetteer<G> {
    fn lookup(&self, name: &str) -> Vec<Entry<'_>> {
        let mut v = self.0.lookup(name);
        v.reverse();
        v
    }

    fn by_id(&self, source: &str, record_id: &str) -> Option<Entry<'_>> {
        self.0.by_id(source, record_id)
    }

    fn by_region(&self, code: crate::entry::RegionCode) -> Option<Entry<'_>> {
        self.0.by_region(code)
    }

    fn sources(&self) -> Vec<&str> {
        self.0.sources()
    }
}

#[test]
fn population_prior_resolves_ambiguity_and_scores_confidence() {
    let g = PopulationPriorGeocoder::new(fixture_gazetteer());

    let text = "She grew up in Springfield.";
    let mention = mention_of(text, "Springfield");
    let res = g.resolve(text, &[mention]).unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].entry().record_id(), "1");
    assert_eq!(res[0].mention(), mention);
    let expected = 0.5 + 0.4 * (70_000.0 / 1_430_000.0);
    assert_eq!(res[0].confidence(), expected);

    let text2 = "From Springfield to Atlantis.";
    let s = mention_of(text2, "Springfield");
    let a = mention_of(text2, "Atlantis");
    let res2 = g.resolve(text2, &[s, a]).unwrap();
    assert_eq!(res2.len(), 1);
    assert_eq!(res2[0].mention(), s);

    let text3 = "Solo, then Springfield, then Tietown.";
    let solo = mention_of(text3, "Solo");
    let sp = mention_of(text3, "Springfield");
    let tie = mention_of(text3, "Tietown");
    let res3 = g.resolve(text3, &[solo, sp, tie]).unwrap();
    assert_eq!(res3.len(), 3);
    assert_eq!(res3[0].mention(), solo);
    assert_eq!(res3[0].confidence(), 0.9);
    assert_eq!(res3[1].mention(), sp);
    assert_eq!(res3[2].mention(), tie);
    assert_eq!(res3[2].entry().record_id(), "6");
    assert_eq!(res3[2].confidence(), 0.5);

    let text4 = "Ghostville again.";
    let res4 = g.resolve(text4, &[mention_of(text4, "Ghostville")]).unwrap();
    assert_eq!(res4[0].confidence(), 0.5);
    assert_eq!(res4[0].entry().record_id(), "7");

    assert!(g.resolve("Springfield", &[]).unwrap().is_empty());
    assert!(g.resolve("Springfield", &[Span::new(3, 3)]).unwrap().is_empty());
    assert!(g.resolve("abc", &[Span::new(0, 4)]).is_err());
}

#[test]
fn population_prior_re_ranks_reversed_gazetteer_and_handles_max_populations() {
    let small = place("small", "Springfield", 0.0, 0.0, None, 1000);
    let large = place("large", "Springfield", 1.0, 1.0, None, 2000);
    let rev = ReversedGazetteer(InMemoryGazetteer::from_entries(&[small, large]).unwrap());
    let res = PopulationPriorGeocoder::new(rev).resolve("Springfield", &[Span::new(0, 11)]).unwrap();
    assert_eq!(res[0].entry().record_id(), "large");

    let adv1 = place("adv-1", "Overflowville", 0.0, 0.0, None, i64::MAX);
    let adv2 = place("adv-2", "Overflowville", 0.0, 0.0, None, i64::MAX);
    let tie = PopulationPriorGeocoder::new(InMemoryGazetteer::from_entries(&[adv1, adv2]).unwrap())
        .resolve("Overflowville", &[Span::new(0, 13)])
        .unwrap();
    assert_eq!(tie[0].confidence(), 0.5);

    let adv_max = place("adv-max", "Overflowville", 0.0, 0.0, None, i64::MAX);
    let adv_one = place("adv-one", "Overflowville", 0.0, 0.0, None, 1);
    let dom = PopulationPriorGeocoder::new(InMemoryGazetteer::from_entries(&[adv_max, adv_one]).unwrap())
        .resolve("Overflowville", &[Span::new(0, 13)])
        .unwrap();
    assert!(dom[0].confidence() > 0.5 && dom[0].confidence() <= 1.0);
    assert_eq!(dom[0].entry().record_id(), "adv-max");

    let admin = GazetteerEntry::new(EntryParts {
        source: "ext".into(),
        record_id: "admin-1".into(),
        name: "Twinsburg".into(),
        alternate_names: Vec::new(),
        location: GeoPoint::new(41.3, -81.4).unwrap(),
        bounding_box: None,
        country_code: Some("US".into()),
        containment: Vec::new(),
        population: 5000,
        feature_class: Some(FEATURE_CLASS_ADMIN.into()),
        attributes: Vec::new(),
    })
    .unwrap();
    let city = GazetteerEntry::new(EntryParts {
        source: "ext".into(),
        record_id: "city-1".into(),
        name: "Twinsburg".into(),
        alternate_names: Vec::new(),
        location: GeoPoint::new(41.4, -81.5).unwrap(),
        bounding_box: None,
        country_code: Some("US".into()),
        containment: Vec::new(),
        population: 5000,
        feature_class: Some(FEATURE_CLASS_CITY.into()),
        attributes: Vec::new(),
    })
    .unwrap();
    let rev_twins = ReversedGazetteer(InMemoryGazetteer::from_entries(&[admin, city]).unwrap());
    let res_twins = PopulationPriorGeocoder::new(rev_twins).resolve("Twinsburg", &[Span::new(0, 9)]).unwrap();
    assert_eq!(res_twins[0].entry().record_id(), "city-1");
    assert_eq!(res_twins[0].confidence(), 0.5);
}

#[test]
fn population_prior_resolves_against_bundled_table() {
    let text = "The flight goes from Paris to Springfield.";
    let paris = mention_of(text, "Paris");
    let springfield = mention_of(text, "Springfield");
    let g = PopulationPriorGeocoder::new(BundledGazetteer::get());
    let res = g.resolve(text, &[paris, springfield]).unwrap();
    assert_eq!(res.len(), 2);
    assert_eq!(res[0].entry().country_code(), Some("FR"));
    assert_eq!(res[1].entry().country_code(), Some("US"));
    assert_eq!(res[1].entry().containment(), &["Massachusetts".to_string()]);
}

#[test]
fn spatial_coherence_moves_ambiguous_mention_toward_context() {
    let gaz = InMemoryGazetteer::from_entries(&[
        place("paris-fr", "Paris", 48.85, 2.35, Some("FR"), 2_100_000),
        place("paris-tx", "Paris", 33.66, -95.56, Some("US"), 25_000),
        place("dallas", "Dallas", 32.78, -96.80, Some("US"), 1_300_000),
        place("houston", "Houston", 29.76, -95.36, Some("US"), 2_300_000),
        place("london", "London", 51.51, -0.13, Some("GB"), 8_800_000),
    ])
    .unwrap();
    let geocoder = SpatialCoherenceGeocoder::new(&gaz);

    let t1 = "the tour hits Dallas, Paris and Houston";
    let r1 =
        geocoder.resolve(t1, &[mention_of(t1, "Dallas"), mention_of(t1, "Paris"), mention_of(t1, "Houston")]).unwrap();
    assert_eq!(r1.len(), 3);
    assert_eq!(r1[1].entry().record_id(), "paris-tx");
    assert!(r1[1].confidence() > 0.5);

    let t2 = "a week in Paris";
    let r2 = geocoder.resolve(t2, &[mention_of(t2, "Paris")]).unwrap();
    assert_eq!(r2.len(), 1);
    assert_eq!(r2[0].entry().record_id(), "paris-fr");

    let t3 = "flights between London and Paris";
    let r3 = geocoder.resolve(t3, &[mention_of(t3, "London"), mention_of(t3, "Paris")]).unwrap();
    assert_eq!(r3[1].entry().record_id(), "paris-fr");

    let t4 = "Dallas and Atlantis";
    let r4 = geocoder.resolve(t4, &[mention_of(t4, "Dallas"), mention_of(t4, "Atlantis")]).unwrap();
    assert_eq!(r4.len(), 1);
    assert_eq!(r4[0].entry().record_id(), "dallas");
    assert_eq!(r4[0].confidence(), 0.9);

    assert!(geocoder.resolve("x", &[Span::new(0, 5)]).is_err());

    // Antipodal context produces a finite distance.
    let antipodal_gaz = InMemoryGazetteer::from_entries(&[
        place("first", "Choice", -61.23628525861706, -132.98189410882517, Some("US"), 2),
        place("second", "Choice", -61.23628525861706, -132.98189410882517, Some("US"), 1),
        place("context", "Context", 61.2362852585404, 47.01810589109053, Some("AU"), 1),
    ])
    .unwrap();
    let antipodal = SpatialCoherenceGeocoder::new(antipodal_gaz);
    let t5 = "Choice Context";
    let r5 = antipodal.resolve(t5, &[mention_of(t5, "Choice"), mention_of(t5, "Context")]).unwrap();
    assert_eq!(r5[0].entry().record_id(), "first");
}

#[test]
fn geocode_annotator_filters_location_entities_and_validates_returned_spans() {
    let kinds = TagSet::new("ner", &["PLACE", "place", "person"]);
    let text = "from Paris to Lyon with Alice";
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None }, Provenance::by("test"))
        .unwrap();
    let mut entities = Entities {
        spans: Spans::new(ViewId::SOURCE),
        tokens: vec![IxRange::new(0, 0), IxRange::new(0, 0), IxRange::new(0, 0)],
        kind: vec![Tag(0), Tag(1), Tag(2)],
        kinds,
        score: vec![1.0, 1.0, 1.0],
    };
    entities.spans.push(mention_of(text, "Paris"));
    entities.spans.push(mention_of(text, "Lyon"));
    entities.spans.push(mention_of(text, "Alice"));
    doc.add(&Key::DEFAULT, entities, Provenance::by("test")).unwrap();

    let gaz = InMemoryGazetteer::from_entries(&[
        place("paris", "Paris", 48.85, 2.35, Some("FR"), 2_100_000),
        place("lyon", "Lyon", 45.76, 4.83, Some("FR"), 500_000),
    ])
    .unwrap();
    let annotator = GeocodeAnnotator::with_location_types(PopulationPriorGeocoder::new(gaz), ["Place"]).unwrap();
    let analyzer = Analyzer::builder().add(annotator).given(Key::<Entities>::DEFAULT).build().unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();
    let locs = doc.get::<Locations>().unwrap();
    assert_eq!(locs.len(), 2);
    assert_eq!(locs.get(Ix::new(0)).entry().name(), "Paris");
    assert_eq!(locs.get(Ix::new(1)).entry().name(), "Lyon");

    assert!(GeocodeAnnotator::with_location_types(PopulationPriorGeocoder::new(BundledGazetteer::get()), []).is_err());
    assert!(
        GeocodeAnnotator::with_location_types(PopulationPriorGeocoder::new(BundledGazetteer::get()), ["\u{00A0}"])
            .is_err()
    );
}
