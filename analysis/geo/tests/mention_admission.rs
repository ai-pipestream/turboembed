use turbo_analysis::Layer;
use turbo_analysis::{Annotator, IxRange, Span, Spans, Tag, TagSet, View};
use turbo_analysis_layers::Entities;
use turbo_geo::{BundledGazetteer, GeocodeAnnotator, MentionPolicy, PopulationPriorGeocoder};

fn entities(view: &View, spans: &[Span]) -> Entities {
    let mut out = Entities {
        spans: Spans::new(view.id()),
        tokens: vec![IxRange::new(0, 0); spans.len()],
        kind: vec![Tag(0); spans.len()],
        kinds: TagSet::new("test", &["location"]),
        score: vec![1.0; spans.len()],
    };
    for &span in spans {
        out.spans.push(span);
    }
    out
}

#[test]
fn dictionary_letters_do_not_become_cities() {
    let doc = turbo_analysis::Document::new("Dublin’s c C S Sao Paulo Cordoba");
    let view = doc.get::<View>().unwrap();
    let spans = [
        Span::new(0, 6),
        Span::new(9, 10),
        Span::new(11, 12),
        Span::new(13, 14),
        Span::new(15, 16),
        Span::new(17, 26),
        Span::new(27, 34),
    ];
    let mentions = entities(view, &spans);
    let base = GeocodeAnnotator::new(PopulationPriorGeocoder::new(BundledGazetteer::get()));
    let (raw,) = base.annotate((view, &mentions)).unwrap();
    assert_eq!(raw.len(), 7, "fixture must reproduce the original alias resolutions");
    let filtered = base.with_mention_policy(MentionPolicy::ExcludeSingleAsciiLetters);
    let (locations,) = filtered.annotate((view, &mentions)).unwrap();
    assert_eq!(locations.len(), 3);
    let resolved: Vec<_> =
        (0..locations.len()).map(|i| locations.resolution(turbo_analysis::Ix::new(i as u32)).0).collect();
    assert_eq!(resolved, [spans[0], spans[5], spans[6]]);
    assert_eq!(mentions.spans.len(), 7);
}

#[test]
fn admission_precedes_geocoder_and_checks_unicode_spans() {
    struct Echo;
    impl turbo_geo::Geocoder for Echo {
        fn resolve(
            &self,
            text: &str,
            mentions: &[Span],
        ) -> Result<Vec<turbo_geo::GeoResolution>, turbo_geo::GeocodeError> {
            let values: Vec<_> = mentions.iter().map(|s| &text[s.range()]).collect();
            assert_eq!(values, ["Å", "東", "St", "São Paulo"]);
            Ok(vec![])
        }
    }
    let doc = turbo_analysis::Document::new("s Å 東 St São Paulo");
    let view = doc.get::<View>().unwrap();
    let spans: Vec<_> = ["s", "Å", "東", "St", "São Paulo"]
        .into_iter()
        .map(|s| {
            let a = view.text().find(s).unwrap();
            Span::new(a as u32, (a + s.len()) as u32)
        })
        .collect();
    let annotator = GeocodeAnnotator::new(Echo).with_mention_policy(MentionPolicy::ExcludeSingleAsciiLetters);
    let mentions = entities(view, &spans);
    annotator.annotate((view, &mentions)).unwrap();
    for bad in [Span::new(3, 4), Span::new(0, 100)] {
        assert!(annotator.annotate((view, &entities(view, &[bad]))).is_err());
    }
}

#[test]
fn admission_and_entity_types_have_stable_distinct_provenance() {
    let make = || PopulationPriorGeocoder::new(BundledGazetteer::get());
    let all = GeocodeAnnotator::new(make());
    let filtered = GeocodeAnnotator::new(make()).with_mention_policy(MentionPolicy::ExcludeSingleAsciiLetters);
    assert_ne!(all.config_hash(), filtered.config_hash());
    let a = GeocodeAnnotator::with_location_types(make(), ["PLACE", "location"]).unwrap();
    let b = GeocodeAnnotator::with_location_types(make(), ["location", "place"]).unwrap();
    assert_eq!(a.config_hash(), b.config_hash());
    assert_ne!(a.config_hash(), all.config_hash());
}
