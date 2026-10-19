use turbo_analysis::export::{Codecs, export, import};
use turbo_analysis::text::{Edit, Side};
use turbo_analysis::{
    Alignment, Analyzer, Bound, Document, Ix, IxRange, Key, Layer, Provenance, Span, Spans, Tag, TagSet, View, ViewId,
};
use turbo_analysis_layers::{Entities, Tokens};

use crate::entry::{EntryParts, FEATURE_CLASS_CITY, GazetteerEntry, GeoPoint, RegionCode};
use crate::export::register_codecs;
use crate::gazetteer::Gazetteer;
use crate::geocode::{DEFAULT_LOCATION_TYPE, Geocoder, Locations};
use crate::in_memory::InMemoryGazetteer;
use crate::linker::{
    CANDIDATE_LINK_LABEL, CountryProximityScorer, EntityLocationLinks, FuzzyMetric, FuzzyStringScorer, GeoEntityLinker,
    GeoHashBinningScorer, LinkerWeights, PRIMARY_LINK_LABEL, ProvinceProximityScorer, encode_geohash, jaro_similarity,
    jaro_winkler, normalized_levenshtein, shared_geohash_prefix_len,
};

#[allow(clippy::too_many_arguments)]
fn place(
    id: &str,
    name: &str,
    alternates: &[&str],
    lat: f64,
    lon: f64,
    country: Option<&str>,
    containment: &[&str],
    population: i64,
) -> GazetteerEntry {
    GazetteerEntry::new(EntryParts {
        source: "geonames".into(),
        record_id: id.into(),
        name: name.into(),
        alternate_names: alternates.iter().map(|&s| s.to_owned()).collect(),
        location: GeoPoint::new(lat, lon).unwrap(),
        bounding_box: None,
        country_code: country.map(Into::into),
        containment: containment.iter().map(|&s| s.to_owned()).collect(),
        population,
        feature_class: Some(FEATURE_CLASS_CITY.into()),
        attributes: Vec::new(),
    })
    .unwrap()
}

fn fixture_gazetteer() -> InMemoryGazetteer {
    InMemoryGazetteer::from_entries(&[
        // Two "Springfield" candidates in US (Illinois vs Missouri)
        place(
            "4250542",
            "Springfield",
            &["Springfield IL"],
            39.7817,
            -89.6501,
            Some("US"),
            &["Illinois", "Sangamon County"],
            114_394,
        ),
        place(
            "4409896",
            "Springfield",
            &["Springfield MO"],
            37.2153,
            -93.2982,
            Some("US"),
            &["Missouri", "Greene County"],
            169_176,
        ),
        // Nearby Illinois cities in the same geohash bin ("dp") as Springfield, IL
        place("4887398", "Chicago", &["Windy City"], 41.8500, -87.6500, Some("US"), &["Illinois"], 2_746_388),
        place("4236835", "Decatur", &[], 39.8403, -88.9548, Some("US"), &["Illinois"], 70_522),
        // Two "Paris" candidates: France vs Texas
        place("2988507", "Paris", &["Lutece"], 48.8534, 2.3488, Some("FR"), &["Ile-de-France"], 2_138_551),
        place("4717560", "Paris", &["Paris TX"], 33.6609, -95.5555, Some("US"), &["Texas", "Lamar County"], 24_476),
        // Nearby French city for context
        place("2996944", "Lyon", &[], 45.7485, 4.8467, Some("FR"), &["Auvergne-Rhone-Alpes"], 522_250),
    ])
    .unwrap()
}

fn mention_of(text: &str, needle: &str) -> Span {
    let start = text.find(needle).unwrap_or_else(|| panic!("needle '{needle}' not found in '{text}'"));
    Span::new(start as u32, (start + needle.len()) as u32)
}

#[test]
fn fuzzy_string_metrics_and_gazetteer_lookup_recover_typos() {
    assert!((normalized_levenshtein("springfield", "springfield") - 1.0).abs() < 1e-12);
    assert_eq!(normalized_levenshtein("", "springfield"), 0.0);
    assert!(normalized_levenshtein("springfeild", "springfield") >= 0.80);
    assert!(jaro_similarity("springfeild", "springfield") >= 0.90);
    assert!(jaro_winkler("springfeild", "springfield") > jaro_similarity("springfeild", "springfield"));

    let gaz = fixture_gazetteer();
    let scorer = FuzzyStringScorer::new(FuzzyMetric::Combined, 0.80);
    let hits = gaz.lookup_fuzzy("Springfeild", &scorer, 0.80, 10);
    assert_eq!(hits.len(), 2, "both Springfield records should match typo 'Springfeild'");
    for (_, sim) in &hits {
        assert!(*sim >= 0.80 && *sim < 1.0);
    }

    // Exact match returns similarity 1.0.
    let exact_hits = gaz.lookup_fuzzy("Springfield", &scorer, 0.80, 10);
    assert_eq!(exact_hits.len(), 2);
    assert!((exact_hits[0].1 - 1.0).abs() < 1e-12);
}

#[test]
fn geohash_binning_favors_spatial_cluster() {
    let pt_sp_il = GeoPoint::new(39.7817, -89.6501).unwrap();
    let pt_dec_il = GeoPoint::new(39.8403, -88.9548).unwrap();
    let pt_chi_il = GeoPoint::new(41.8500, -87.6500).unwrap();
    let pt_sp_mo = GeoPoint::new(37.2153, -93.2982).unwrap();

    let gh_sp_il = encode_geohash(pt_sp_il, 5);
    let gh_dec_il = encode_geohash(pt_dec_il, 5);
    let gh_chi_il = encode_geohash(pt_chi_il, 5);
    let gh_sp_mo = encode_geohash(pt_sp_mo, 5);

    assert_eq!(&gh_sp_il[..2], "dp");
    assert_eq!(&gh_dec_il[..2], "dp");
    assert_eq!(&gh_chi_il[..2], "dp");
    assert_ne!(&gh_sp_il[..2], &gh_sp_mo[..2]);
    assert!(shared_geohash_prefix_len(&gh_sp_il, &gh_dec_il) >= 2);

    let scorer = GeoHashBinningScorer::new(5);
    let all_hashes = vec![vec![gh_sp_il.clone(), gh_sp_mo.clone()], vec![gh_dec_il], vec![gh_chi_il]];
    let score_il = scorer.score_in_context(0, &gh_sp_il, &all_hashes);
    let score_mo = scorer.score_in_context(0, &gh_sp_mo, &all_hashes);
    assert!(
        score_il > score_mo,
        "Springfield IL ({score_il}) must outscore Springfield MO ({score_mo}) in an Illinois spatial cluster"
    );
}

#[test]
fn country_and_province_proximity_disambiguate_candidates() {
    let text = "We drove from Paris, Texas in the United States 🇺🇸.";
    let country_scorer = CountryProximityScorer::default();
    let province_scorer = ProvinceProximityScorer::default();

    let c_signals = country_scorer.extract_text_signals(text);
    let us = RegionCode::parse("US").unwrap();
    let fr = RegionCode::parse("FR").unwrap();
    assert!(c_signals.iter().any(|s| s.country == us));

    let paris_span = mention_of(text, "Paris");
    let c_us = country_scorer.score(paris_span, Some(us), &c_signals);
    let c_fr = country_scorer.score(paris_span, Some(fr), &c_signals);
    assert!(c_us > c_fr, "US country score ({c_us}) must beat FR ({c_fr})");

    let mut folder = crate::fold::Folder::new();
    let folded_doc = folder.fold(text);
    let tx_containment = ProvinceProximityScorer::fold_containment(&mut folder, ["Texas", "Lamar County"]);
    let fr_containment = ProvinceProximityScorer::fold_containment(&mut folder, ["Ile-de-France"]);
    let p_tx = province_scorer.score(paris_span, &tx_containment, &[], &folded_doc);
    let p_fr = province_scorer.score(paris_span, &fr_containment, &[], &folded_doc);
    assert!(p_tx > p_fr, "Texas province score ({p_tx}) must beat Ile-de-France ({p_fr})");
}

#[test]
fn geo_entity_linker_selects_springfield_il_and_emits_typed_edges_and_graph_roundtrip() {
    // Note that Springfield MO has higher population (169k vs 114k), so a pure
    // population prior picks MO, whereas GeoEntityLinker uses Illinois + Decatur + Chicago
    // spatial/province context to pick Springfield IL (4250542).
    let text = "Officials met in Springfield, Illinois after stops in Decatur and Chicago.";
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None }, Provenance::by("test"))
        .unwrap();

    let kinds = TagSet::new("ner", &[DEFAULT_LOCATION_TYPE]);
    let mut entities = Entities {
        spans: Spans::new(ViewId::SOURCE),
        tokens: vec![IxRange::new(0, 0), IxRange::new(0, 0), IxRange::new(0, 0)],
        kind: vec![Tag(0), Tag(0), Tag(0)],
        kinds,
        score: vec![1.0, 1.0, 1.0],
    };
    entities.spans.push(mention_of(text, "Springfield"));
    entities.spans.push(mention_of(text, "Decatur"));
    entities.spans.push(mention_of(text, "Chicago"));
    doc.add(&Key::DEFAULT, entities, Provenance::by("test")).unwrap();

    let gaz = fixture_gazetteer();
    let linker = GeoEntityLinker::new(gaz).with_weights(LinkerWeights::default()).with_max_candidates(3);

    // Also verify Geocoder trait implementation on GeoEntityLinker.
    let resolved = linker
        .resolve(text, &[mention_of(text, "Springfield"), mention_of(text, "Decatur"), mention_of(text, "Chicago")])
        .unwrap();
    assert_eq!(resolved.len(), 3);
    assert_eq!(resolved[0].entry().record_id(), "4250542");

    // Run as Annotator in Analyzer pipeline.
    let analyzer = Analyzer::builder().add(linker).given(Key::<Entities>::DEFAULT).build().unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();

    let locations = doc.get::<Locations>().unwrap();
    let links = doc.get::<EntityLocationLinks>().unwrap();

    let ent0 = Ix::<Entities>::new(0);
    let ent0_range = links.from_source(ent0);
    assert_eq!(ent0_range.len(), 2, "Springfield has 2 candidates in gazetteer");

    let mut primary_loc = None;
    let mut alt_loc = None;
    for edge_ix in ent0_range.iter() {
        let r = edge_ix.get();
        assert_eq!(links.from[r], ent0);
        match links.kinds.label(links.kind[r]) {
            PRIMARY_LINK_LABEL => primary_loc = Some((links.to[r], links.score[r])),
            CANDIDATE_LINK_LABEL => alt_loc = Some((links.to[r], links.score[r])),
            other => panic!("unexpected edge label: {other}"),
        }
    }

    let (primary_ix, primary_score) = primary_loc.expect("primary link edge must exist");
    let (alt_ix, alt_score) = alt_loc.expect("candidate link edge must exist");
    assert_eq!(locations.get(primary_ix).entry().record_id(), "4250542");
    assert_eq!(locations.get(alt_ix).entry().record_id(), "4409896");
    assert!((0.0..=1.0).contains(&primary_score));
    assert!((0.0..=1.0).contains(&alt_score));
    assert!(primary_score > alt_score, "Springfield IL ({primary_score}) must beat Springfield MO ({alt_score})");

    // Verify DocumentGraph round-trip preserves both Locations and Edges<Entities, Locations>.
    let mut codecs = Codecs::new();
    turbo_analysis_layers::export::register_codecs(&mut codecs);
    register_codecs(&mut codecs);

    let graph = export(&doc, &codecs).unwrap();
    let restored = import(&graph, &codecs).unwrap();

    let restored_locs = restored.get::<Locations>().unwrap();
    let restored_links = restored.get::<EntityLocationLinks>().unwrap();
    assert_eq!(restored_locs.len(), locations.len());
    assert_eq!(restored_links.len(), links.len());
    assert_eq!(restored_locs.get(primary_ix).entry().record_id(), "4250542");
}

#[test]
fn geo_entity_linker_fuzzy_fallback_and_derived_view_projection() {
    // Source view has HTML wrapper; derived view strips it. Mention has a typo ("Springfeild").
    let raw = "<p>Springfeild in Illinois near Chicago</p>";
    let clean = "Springfeild in Illinois near Chicago";
    let mut doc = Document::new(raw);
    let alignment = Alignment::from_edits(vec![
        Edit { parent: Span::new(0, 3), child: Span::new(0, 0) },
        Edit { parent: Span::new(39, 43), child: Span::new(36, 36) },
    ])
    .unwrap();
    let derived = View::derived(doc.get::<View>().unwrap(), clean.into(), alignment);
    let v1_key = Key::<View>::named("clean");
    doc.add(&v1_key, derived, Provenance::by("strip-html")).unwrap();
    let v1 = doc.get_key(&v1_key).unwrap().id();

    doc.add(&Key::DEFAULT, Tokens { spans: Spans::new(v1), sentence: None }, Provenance::by("test")).unwrap();
    let kinds = TagSet::new("ner", &[DEFAULT_LOCATION_TYPE]);
    let mut entities = Entities {
        spans: Spans::new(v1),
        tokens: vec![IxRange::new(0, 0), IxRange::new(0, 0)],
        kind: vec![Tag(0), Tag(0)],
        kinds,
        score: vec![1.0, 1.0],
    };
    entities.spans.push(mention_of(clean, "Springfeild"));
    entities.spans.push(mention_of(clean, "Chicago"));
    doc.add(&Key::DEFAULT, entities, Provenance::by("test")).unwrap();

    let gaz = fixture_gazetteer();
    let linker = GeoEntityLinker::new(gaz);
    let step = Bound::new(linker).read(0, "clean");
    let analyzer = Analyzer::builder().given(v1_key).given(Key::<Entities>::DEFAULT).step(step).build().unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();

    let locations = doc.get::<Locations>().unwrap();
    assert_eq!(locations.spans.view(), v1);
    assert_eq!(locations.get(Ix::new(0)).entry().record_id(), "4250542");

    // Verify bi-directional view span mapping between derived view v1 and root view.
    let child_span = locations.get(Ix::new(0)).mention();
    let view_ref = doc.get_key(&v1_key).unwrap();
    assert_eq!(&view_ref.text()[child_span.range()], "Springfeild");
    let root_start = view_ref.alignment().to_parent(child_span.start, Side::Start);
    let root_end = view_ref.alignment().to_parent(child_span.end, Side::End);
    let root_span = Span::new(root_start, root_end);
    assert_eq!(root_span, Span::new(3, 14));
    assert_eq!(&doc.text(ViewId::SOURCE)[root_span.range()], "Springfeild");
    assert_eq!(doc.from_source(v1, root_span), child_span);
}

#[test]
fn overlay_gazetteer_delegates_fuzzy_lookup_and_respects_suppressions() {
    use crate::{OverlayGazetteer, Suppression};

    let base = InMemoryGazetteer::from_entries(&[
        place("4250542", "Springfield", &[], 39.7817, -89.6501, Some("US"), &["Illinois"], 114_394),
        place("2988507", "Paris", &[], 48.8534, 2.3488, Some("FR"), &["Ile-de-France"], 2_138_551),
    ])
    .unwrap();
    let scorer = FuzzyStringScorer::new(FuzzyMetric::Combined, 0.80);

    // Base wrapped with empty additions (and a non-matching suppression) still finds fuzzy hits.
    let empty_additions = InMemoryGazetteer::from_entries(&[]).unwrap();
    let overlay_empty_additions = OverlayGazetteer::new(&base, Some(&empty_additions), Vec::new()).unwrap();
    let hits = overlay_empty_additions.lookup_fuzzy("Springfeild", &scorer, 0.80, 10);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0.record_id(), "4250542");

    // Suppress Springfield in base and add a custom Springfield MA in additions;
    // base Springfield is suppressed while additions Springfield and base Paris remain.
    let additions = InMemoryGazetteer::from_entries(&[place(
        "c-1",
        "Springfield",
        &[],
        42.1015,
        -72.5898,
        Some("US"),
        &["Massachusetts"],
        155_929,
    )])
    .unwrap();
    let overlay = OverlayGazetteer::new(
        &base,
        Some(&additions),
        vec![Suppression::new("Springfield", Some("US"), None).unwrap()],
    )
    .unwrap();
    let hits = overlay.lookup_fuzzy("Springfeild", &scorer, 0.80, 10);
    let ids: Vec<&str> = hits.iter().map(|(e, _)| e.record_id()).collect();
    assert_eq!(ids, vec!["c-1"]);
}

#[test]
fn config_hash_distinguishes_metric_policy_and_decay_settings() {
    use crate::MentionPolicy;
    use turbo_analysis::Annotator;

    let base_linker = || GeoEntityLinker::new(fixture_gazetteer());
    let default_hash = base_linker().config_hash();

    let jw_hash = base_linker().with_fuzzy_scorer(FuzzyStringScorer::new(FuzzyMetric::JaroWinkler, 0.80)).config_hash();
    let lev_hash =
        base_linker().with_fuzzy_scorer(FuzzyStringScorer::new(FuzzyMetric::NormalizedLevenshtein, 0.80)).config_hash();
    let policy_hash = base_linker().with_mention_policy(MentionPolicy::ExcludeSingleAsciiLetters).config_hash();
    let country_hash = base_linker().with_country_scorer(CountryProximityScorer::new(500.0)).config_hash();
    let province_hash = base_linker().with_province_scorer(ProvinceProximityScorer::new(500.0)).config_hash();

    for h in [jw_hash, lev_hash, policy_hash, country_hash, province_hash] {
        assert_ne!(h, default_hash);
    }
    assert_ne!(jw_hash, lev_hash);
    assert_ne!(country_hash, province_hash);
}

#[test]
fn extreme_and_unvalidated_weights_never_produce_nan() {
    let extreme = LinkerWeights::new(f64::MAX, f64::MAX, 0.0, 0.0, 0.0).unwrap();
    let c1 = extreme.combine(1.0, 1.0, 0.0, 0.0, 0.0);
    assert!(!c1.is_nan());
    assert!((c1 - 1.0).abs() < 1e-12);
    let c2 = extreme.combine(1.0, 0.0, 0.0, 0.0, 0.0);
    assert!((c2 - 0.5).abs() < 1e-12);

    // Weights constructed via public fields with invalid values are rejected by link_mentions.
    let bad = LinkerWeights { fuzzy: -1.0, geohash: 1.0, country: 0.0, province: 0.0, population: 0.0 };
    assert!(bad.validate().is_err());
    assert_eq!(bad.combine(1.0, 1.0, 1.0, 1.0, 1.0), 0.0);
    let linker = GeoEntityLinker::new(fixture_gazetteer()).with_weights(bad);
    assert!(linker.link_mentions("Springfield", &[Span::new(0, 11)]).is_err());
}

#[test]
fn overlapping_country_and_province_spans_are_skipped_and_distance_decays() {
    let country_scorer = CountryProximityScorer::default();
    // "Panama City, Florida": "Panama" at 0..6 overlaps "Panama City" at 0..11 and must be skipped.
    let text = "Panama City, Florida";
    let sigs = country_scorer.extract_text_signals(text);
    let mention = Span::new(0, 11);
    let us = RegionCode::new(*b"US").unwrap();
    let pa = RegionCode::new(*b"PA").unwrap();
    assert_eq!(country_scorer.score(mention, Some(us), &sigs), 0.5);
    assert_eq!(country_scorer.score(mention, Some(pa), &sigs), 0.5);

    // Multi-word country with newline ("United\nStates") is extracted.
    let nl_sigs = country_scorer.extract_text_signals("Visited Springfield in the United\nStates.");
    assert!(nl_sigs.iter().any(|s| s.country == us));

    // "I love New York.": "New York" at 7..15 overlaps the mention at 7..15 -> neutral 0.5.
    let prov_scorer = ProvinceProximityScorer::default();
    let ny_containment = vec!["new york".to_string()];
    assert_eq!(prov_scorer.score(Span::new(7, 15), &ny_containment, &[], "I love New York."), 0.5);

    // Adjacent province name scores strictly higher than one 10 KB away.
    let near_text = "Springfield, Illinois";
    let far_text = format!("Springfield {} Illinois", "x ".repeat(5_000));
    let il_containment = vec!["illinois".to_string()];
    let near_score = prov_scorer.score(Span::new(0, 11), &il_containment, &[], near_text);
    let far_score = prov_scorer.score(Span::new(0, 11), &il_containment, &[], &far_text);
    assert!(near_score > 0.95, "near_score={near_score}");
    assert!(far_score < 0.60 && far_score > 0.50, "far_score={far_score}");
}

#[test]
fn geohash_handles_prime_meridian_edge_and_skips_repeated_mentions_and_nan_threshold() {
    // London (51.507, -0.128) is "gcpvj0" and 9 km east (51.48, 0.005) is "u10hb5".
    let london = encode_geohash(GeoPoint::new(51.507, -0.128).unwrap(), 6);
    let greenwich = encode_geohash(GeoPoint::new(51.48, 0.005).unwrap(), 6);
    let sydney = encode_geohash(GeoPoint::new(-33.8688, 151.2093).unwrap(), 6);
    assert_eq!(shared_geohash_prefix_len(&london, &greenwich), 0);

    let gh_scorer = GeoHashBinningScorer::new(6);
    let near_ctx = vec![vec![london.clone()], vec![greenwich]];
    let far_ctx = vec![vec![london.clone()], vec![sydney]];
    let near_support = gh_scorer.score_in_context(0, &london, &near_ctx);
    let far_support = gh_scorer.score_in_context(0, &london, &far_ctx);
    assert!(near_support > 0.95, "Prime Meridian 9 km support={near_support}");
    assert!(far_support < 0.05, "intercontinental support={far_support}");

    // Repeated mentions ("Springfield and Springfield") do not self-reinforce geohash_score to 1.0.
    let gaz = fixture_gazetteer();
    let linker = GeoEntityLinker::new(&gaz);
    let rep_text = "Springfield and Springfield";
    let linked = linker.link_mentions(rep_text, &[Span::new(0, 11), Span::new(16, 27)]).unwrap();
    for c in &linked[0] {
        assert_eq!(c.geohash_score, 0.5);
    }

    // NaN min_similarity returns empty results on lookup_fuzzy, and min_similarity = 0.0 respects limit.
    let nan_scorer = FuzzyStringScorer::new(FuzzyMetric::Combined, f64::NAN);
    assert_eq!(nan_scorer.min_similarity(), 1.0);
    assert!(gaz.lookup_fuzzy("Springfield", &nan_scorer, f64::NAN, 10).is_empty());
    assert_eq!(gaz.lookup_fuzzy("Springfield", &FuzzyStringScorer::default(), 0.0, 2).len(), 2);
}
