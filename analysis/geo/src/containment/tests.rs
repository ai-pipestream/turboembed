use super::*;
use crate::entry::{ATTRIBUTE_KEY_GEONAMES, AttributeValue, EntryParts, FEATURE_CLASS_CITY, GazetteerEntry, GeoPoint};
use crate::geocode::{GeoResolution, GeocodeAnnotator};
use crate::region::DocumentRegionAnnotator;
use turbo_analysis::{Analyzer, Document, Ix, IxRange, Key, Layer, Provenance, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_numeric::{CONVERTED_MONEY, Currency, Date, EcbRates, Money, MoneyConverter, RegionAwareMoneyAnnotator};

fn run_one(a: &Analyzer, mut doc: Document) -> Document {
    a.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();
    doc
}

fn park_slope_spine() -> ContainmentSpine {
    let mut b = ContainmentSpine::builder();
    b.add("85865587", Some("421205765"), "Park Slope", "neighbourhood")
        .unwrap()
        .add("421205765", Some("85977539"), "Brooklyn", "borough")
        .unwrap()
        .add("85977539", Some("85688543"), "New York", "locality")
        .unwrap()
        .add("85688543", Some("85633793"), "New York", "region")
        .unwrap()
        .add("85633793", None, "United States", "country")
        .unwrap();
    b.build().unwrap()
}

fn entry_with_wof(name: &str, wof_id: Option<&str>) -> GazetteerEntry {
    let attributes = match wof_id {
        Some(id) => {
            vec![(ATTRIBUTE_KEY_WHOSONFIRST.to_string(), AttributeValue::new(id, "test", "fixture").unwrap())]
        }
        None => Vec::new(),
    };
    GazetteerEntry::new(EntryParts {
        source: "test".into(),
        record_id: name.into(),
        name: name.into(),
        alternate_names: Vec::new(),
        location: GeoPoint::new(0.0, 0.0).unwrap(),
        bounding_box: None,
        country_code: Some("US".into()),
        containment: Vec::new(),
        population: 1000,
        feature_class: Some(FEATURE_CLASS_CITY.into()),
        attributes,
    })
    .unwrap()
}

#[test]
fn ancestors_walk_outward_and_handle_roots_missing_parents_duplicates_cycles_and_depth_cap() {
    let spine = park_slope_spine();
    let chain = spine.ancestors("85865587").unwrap();
    assert_eq!(chain.len(), 4);
    assert_eq!(chain[0].name(), "Brooklyn");
    assert_eq!(chain[0].place_type(), "borough");
    assert_eq!(chain[1].place_type(), "locality");
    assert_eq!(chain[2].place_type(), "region");
    assert_eq!(chain[3].name(), "United States");
    assert_eq!(chain[3].place_type(), "country");

    assert!(spine.ancestors("999").unwrap().is_empty());
    assert!(spine.ancestors("85633793").unwrap().is_empty());

    // Missing parent reference ends chain.
    let mut b = ContainmentSpine::builder();
    b.add("x", Some("y"), "X", "t").unwrap().add("y", Some("zz"), "Y", "t").unwrap();
    let missing = b.build().unwrap();
    assert_eq!(missing.ancestors("x").unwrap(), vec![PlaceAncestor::new("y", "Y", "t").unwrap()]);
    assert!(missing.ancestors("y").unwrap().is_empty());

    // Duplicate IDs keep last addition.
    let mut b_dup = ContainmentSpine::builder();
    b_dup
        .add("child", Some("dup"), "Child", "t")
        .unwrap()
        .add("dup", None, "First", "t1")
        .unwrap()
        .add("dup", None, "Second", "t2")
        .unwrap();
    assert_eq!(
        b_dup.build().unwrap().ancestors("child").unwrap(),
        vec![PlaceAncestor::new("dup", "Second", "t2").unwrap()]
    );

    // Self-loop and triangle cycles terminate without repeating an identifier.
    let mut b_self = ContainmentSpine::builder();
    b_self.add("a", Some("a"), "A", "t").unwrap();
    assert!(b_self.build().unwrap().ancestors("a").unwrap().is_empty());

    let mut b_tri = ContainmentSpine::builder();
    b_tri
        .add("a", Some("b"), "A", "t")
        .unwrap()
        .add("b", Some("c"), "B", "t")
        .unwrap()
        .add("c", Some("a"), "C", "t")
        .unwrap();
    assert_eq!(
        b_tri.build().unwrap().ancestors("a").unwrap(),
        vec![PlaceAncestor::new("b", "B", "t").unwrap(), PlaceAncestor::new("c", "C", "t").unwrap()]
    );

    // 64-ancestor depth cap.
    let mut b_deep = ContainmentSpine::builder();
    for i in 0..70 {
        let id = format!("p{i}");
        let parent = (i < 69).then(|| format!("p{}", i + 1));
        let name = format!("P{i}");
        b_deep.add(&id, parent.as_deref(), &name, "t").unwrap();
    }
    let deep_chain = b_deep.build().unwrap().ancestors("p0").unwrap();
    assert_eq!(deep_chain.len(), 64);
    assert_eq!(deep_chain[0].id(), "p1");
    assert_eq!(deep_chain[63].id(), "p64");
}

#[test]
fn neutral_table_and_wof_meta_csv_loaders_and_validation() {
    let tsv = "# id\tparent\tname\ttype\nQ123\tQ60\tSoHo\tneighbourhood\nQ60\tQ1384\tNew York City\tcity\nQ1384\t\tNew York\tstate\n";
    let mut b = ContainmentSpine::builder();
    b.add_table_bytes(tsv.as_bytes(), "containment.tsv").unwrap();
    let chain = b.build().unwrap().ancestors("Q123").unwrap();
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].name(), "New York City");
    assert_eq!(chain[1].name(), "New York");

    // WOF CSV with quoted commas, quoted newlines, CRLF, and doubled quotes.
    let csv = "bbox,id,name,parent_id,placetype,source\n\"1,2,3,4\",85977539,Brooklyn,85688543,borough,mz\n\"5,6,7,8\",85688543,\"New York, the Big Apple\",-1,locality,mz\n";
    let mut b_wof = ContainmentSpine::builder();
    b_wof.add_wof_meta_bytes(csv.as_bytes(), "wof.csv").unwrap();
    let wof_chain = b_wof.build().unwrap().ancestors("85977539").unwrap();
    assert_eq!(wof_chain.len(), 1);
    assert_eq!(wof_chain[0].name(), "New York, the Big Apple");

    let crlf_csv = "id,parent_id,name,placetype\r\n999,123,Child,neighbourhood\r\n123,456,\"Sao Paulo\r\nZona Norte\",locality\r\n456,-1,\"The \"\"Big Apple\"\"\",country\r\n";
    let mut b_crlf = ContainmentSpine::builder();
    b_crlf.add_wof_meta_bytes(crlf_csv.as_bytes(), "crlf.csv").unwrap();
    let crlf_chain = b_crlf.build().unwrap().ancestors("999").unwrap();
    assert_eq!(
        crlf_chain,
        vec![
            PlaceAncestor::new("123", "Sao Paulo\nZona Norte", "locality").unwrap(),
            PlaceAncestor::new("456", "The \"Big Apple\"", "country").unwrap(),
        ]
    );

    // Error reporting checks.
    let unterm = "id,parent_id,name,placetype\n123,456,\"Sao Paulo,locality\n456,-1,Brazil,country\n";
    let err = ContainmentSpine::builder().add_wof_meta_bytes(unterm.as_bytes(), "u.csv").unwrap_err().to_string();
    assert!(err.contains("unterminated quoted field") && err.contains("line 2"), "{err}");

    let empty_name = "id,parent_id,name,placetype\n123,456,\"Sao Paulo\nZona Norte\",locality\n456,-1,,country\n";
    let err2 = ContainmentSpine::builder().add_wof_meta_bytes(empty_name.as_bytes(), "en.csv").unwrap_err().to_string();
    assert!(err2.contains("empty name") && err2.contains("line 4"), "{err2}");

    let stray = "id,parent_id,name,placetype\n123,456,Kings\"County,county\n";
    let err3 = ContainmentSpine::builder().add_wof_meta_bytes(stray.as_bytes(), "s.csv").unwrap_err().to_string();
    assert!(err3.contains("stray quote") && err3.contains("line 2"), "{err3}");

    let after_q = "id,parent_id,name,placetype\n123,456,\"Kings\"County,county\n";
    let err4 = ContainmentSpine::builder().add_wof_meta_bytes(after_q.as_bytes(), "a.csv").unwrap_err().to_string();
    assert!(err4.contains("after a closing quote") && err4.contains("line 2"), "{err4}");

    for bad_parent in ["abc", "+1", "-x"] {
        let bad_csv = format!("id,parent_id,name,placetype\n1,{bad_parent},Place,locality\n");
        assert!(ContainmentSpine::builder().add_wof_meta_bytes(bad_csv.as_bytes(), "p.csv").is_err());
    }

    // Malformed UTF-8 rejected as Io error.
    assert!(matches!(
        ContainmentSpine::builder().add_table_bytes(&[b'x', b'\t', b'\t', 0xc3, b'\t', b't', b'\n'], "bad.tsv"),
        Err(HierarchyError::Io(_))
    ));
}

#[test]
fn hierarchy_annotator_expands_resolved_mentions_and_handles_multi_candidate_spans() {
    let mut b = ContainmentSpine::builder();
    b.add("85865587", Some("421205765"), "Park Slope", "neighbourhood")
        .unwrap()
        .add("421205765", Some("85977539"), "Brooklyn", "borough")
        .unwrap()
        .add("85977539", None, "New York", "locality")
        .unwrap();
    let spine = b.build().unwrap();

    let text = "A stroll through Park Slope.";
    let mention = Span::new(17, 27);
    let mut doc = Document::new(text);
    let mut locs = Locations::new(ViewId::SOURCE);
    locs.push(GeoResolution::new(mention, entry_with_wof("Park Slope", Some("85865587")), 0.9).unwrap());
    doc.add(&Key::DEFAULT, locs, Provenance::by("test")).unwrap();

    let analyzer = Analyzer::builder()
        .add(HierarchyAnnotator::new(spine.clone()))
        .given(Key::<Locations>::DEFAULT)
        .build()
        .unwrap();
    let out = run_one(&analyzer, doc);
    let cont = out.get::<Containment>().unwrap();
    assert_eq!(cont.len(), 1);
    assert_eq!(cont.spans.get(Ix::new(0)), mention);
    let names: Vec<&str> = cont.ancestors(Ix::new(0)).map(|a| a.name).collect();
    assert_eq!(names, ["Brooklyn", "New York"]);

    // Unexpandable best candidate suppresses second candidate on same span.
    let mut doc2 = Document::new("A week in Paris.");
    let m2 = Span::new(10, 15);
    let mut locs2 = Locations::new(ViewId::SOURCE);
    locs2.push(GeoResolution::new(m2, entry_with_wof("Paris", None), 0.7).unwrap());
    locs2.push(GeoResolution::new(m2, entry_with_wof("Paris", Some("85865587")), 0.3).unwrap());
    doc2.add(&Key::DEFAULT, locs2, Provenance::by("test")).unwrap();
    let out2 = run_one(&analyzer, doc2);
    assert_eq!(out2.get::<Containment>().unwrap().len(), 0);

    // Custom attribute key joins on that attribute.
    let mut doc3 = Document::new(text);
    let mut locs3 = Locations::new(ViewId::SOURCE);
    let entry_geo = GazetteerEntry::new(EntryParts {
        source: "test".into(),
        record_id: "Park Slope".into(),
        name: "Park Slope".into(),
        alternate_names: Vec::new(),
        location: GeoPoint::new(0.0, 0.0).unwrap(),
        bounding_box: None,
        country_code: Some("US".into()),
        containment: Vec::new(),
        population: 1000,
        feature_class: Some(FEATURE_CLASS_CITY.into()),
        attributes: vec![
            (ATTRIBUTE_KEY_WHOSONFIRST.to_string(), AttributeValue::new("77", "test", "fixture").unwrap()),
            (ATTRIBUTE_KEY_GEONAMES.to_string(), AttributeValue::new("85865587", "test", "fixture").unwrap()),
        ],
    })
    .unwrap();
    locs3.push(GeoResolution::new(mention, entry_geo, 0.9).unwrap());
    doc3.add(&Key::DEFAULT, locs3, Provenance::by("test")).unwrap();

    let custom_ann = HierarchyAnnotator::with_attribute_key(spine, ATTRIBUTE_KEY_GEONAMES).unwrap();
    let custom_analyzer = Analyzer::builder().add(custom_ann).given(Key::<Locations>::DEFAULT).build().unwrap();
    let out3 = run_one(&custom_analyzer, doc3);
    assert_eq!(out3.get::<Containment>().unwrap().len(), 1);
}

#[test]
fn hierarchy_annotator_config_hash_reflects_attribute_key() {
    let spine = park_slope_spine();
    let default_ann = HierarchyAnnotator::new(spine.clone());
    let explicit_wof = HierarchyAnnotator::with_attribute_key(spine.clone(), ATTRIBUTE_KEY_WHOSONFIRST).unwrap();
    let geonames_ann = HierarchyAnnotator::with_attribute_key(spine, ATTRIBUTE_KEY_GEONAMES).unwrap();

    assert_ne!(default_ann.config_hash(), 0);
    assert_eq!(default_ann.config_hash(), explicit_wof.config_hash());
    assert_ne!(default_ann.config_hash(), geonames_ann.config_hash());
}

#[test]
fn end_to_end_pipeline_from_entities_to_converted_money() {
    struct SydneyGeocoder;
    impl crate::geocode::Geocoder for SydneyGeocoder {
        fn resolve(
            &self,
            _text: &str,
            location_mentions: &[Span],
        ) -> Result<Vec<GeoResolution>, crate::geocode::GeocodeError> {
            let entry = GazetteerEntry::new(EntryParts {
                source: "test".into(),
                record_id: "sydney".into(),
                name: "Sydney".into(),
                alternate_names: Vec::new(),
                location: GeoPoint::new(-33.87, 151.21).unwrap(),
                bounding_box: None,
                country_code: Some("AU".into()),
                containment: Vec::new(),
                population: 5_300_000,
                feature_class: Some(FEATURE_CLASS_CITY.into()),
                attributes: Vec::new(),
            })
            .unwrap();
            location_mentions.iter().map(|&m| GeoResolution::new(m, entry.clone(), 0.9)).collect()
        }
    }

    let rates = EcbRates::from_bytes(b"Date,USD,AUD,\n2026-07-10,1.2000,1.8000,\n").unwrap();
    let text = "a Sydney startup raised $3 million this week";
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None }, Provenance::by("test"))
        .unwrap();
    let kinds = TagSet::new("ner", &["location"]);
    let mut spans = Spans::new(ViewId::SOURCE);
    let start = text.find("Sydney").unwrap();
    spans.push(Span::new(start as u32, (start + 6) as u32));
    doc.add(
        &Key::DEFAULT,
        Entities { spans, tokens: vec![IxRange::new(0, 0)], kind: vec![Tag(0)], kinds, score: vec![1.0] },
        Provenance::by("test"),
    )
    .unwrap();

    let usd = Currency::parse("USD").unwrap();
    let as_of = Date::parse("2026-07-10").unwrap();
    let analyzer = Analyzer::builder()
        .add(GeocodeAnnotator::new(SydneyGeocoder))
        .add(DocumentRegionAnnotator::new())
        .add(RegionAwareMoneyAnnotator::new())
        .step(MoneyConverter::new(rates, usd, as_of).bound())
        .given(Key::<Entities>::DEFAULT)
        .build()
        .unwrap();

    let out = run_one(&analyzer, doc);
    let identified = out.get::<Money>().unwrap();
    assert_eq!(identified.len(), 1);
    assert_eq!(identified.get(Ix::new(0)).currency.as_str(), "AUD");
    let id_span = identified.get(Ix::new(0)).span;
    assert_eq!(&out.source()[id_span.start as usize..id_span.end as usize], "$3 million");

    let converted = out.get_key(&CONVERTED_MONEY).unwrap();
    assert_eq!(converted.len(), 1);
    assert_eq!(converted.get(Ix::new(0)).currency.as_str(), "USD");
    assert_eq!(converted.amount(Ix::new(0)).to_plain_string(), "2000000");
    assert_eq!(converted.get(Ix::new(0)).span, identified.get(Ix::new(0)).span);
}
