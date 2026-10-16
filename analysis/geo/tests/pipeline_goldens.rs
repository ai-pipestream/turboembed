//! Compares `PopulationPriorGeocoder`, `SpatialCoherenceGeocoder`,
//! `GeocodeAnnotator`, `DocumentRegionAnnotator`, `ContainmentSpine`,
//! `HierarchyAnnotator`, and `RegionAwareMoneyAnnotator` against the JVM test
//! vectors in `tests/vectors/geo_pipeline.tsv`, produced by
//! `java/GeoPipelineVectors.java` on the pinned JDK 21 image against OpenNLP
//! commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea`.

use std::collections::BTreeMap;

use turbo_analysis::{Analyzer, Document, Ix, IxRange, Key, Layer, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_geo::{
    ATTRIBUTE_KEY_WHOSONFIRST, AttributeValue, Containment, ContainmentSpine, DEFAULT_LOCATION_TYPE,
    DocumentRegionAnnotator, EntryParts, FEATURE_CLASS_CITY, GazetteerEntry, GeoPoint, GeocodeAnnotator, Geocoder,
    HierarchyAnnotator, InMemoryGazetteer, Locations, PopulationPriorGeocoder, Regions, SpatialCoherenceGeocoder,
};
use turbo_numeric::{Money, RegionAwareMoneyAnnotator};

const VECTORS: &str = include_str!("vectors/geo_pipeline.tsv");

const SPINE_TSV: &str = concat!(
    "# id\tparent_id\tname\tplace_type\n",
    "100\t10\tSpringfield\tlocality\n",
    "101\t11\tSpringfield\tlocality\n",
    "102\t12\tChicago\tlocality\n",
    "103\t20\tParis\tlocality\n",
    "104\t13\tParis\tlocality\n",
    "105\t21\tLyon\tlocality\n",
    "106\t30\tTurin\tlocality\n",
    "107\t40\tPorto\tlocality\n",
    "108\t50\tSydney\tlocality\n",
    "109\t60\tGuadalajara\tlocality\n",
    "110\t10\tBoston\tlocality\n",
    "10\t1\tMassachusetts\tregion\n",
    "11\t1\tIllinois\tregion\n",
    "12\t1\tIllinois\tregion\n",
    "13\t1\tTexas\tregion\n",
    "20\t2\tIle-de-France\tregion\n",
    "21\t2\tAuvergne-Rhone-Alpes\tregion\n",
    "30\t3\tPiedmont\tregion\n",
    "40\t4\tPorto District\tregion\n",
    "50\t5\tNew South Wales\tregion\n",
    "60\t6\tJalisco\tregion\n",
    "1\t\tUnited States\tcountry\n",
    "2\t\tFrance\tcountry\n",
    "3\t\tItaly\tcountry\n",
    "4\t\tPortugal\tcountry\n",
    "5\t\tAustralia\tcountry\n",
    "6\t\tMexico\tcountry\n",
);

const SPINE_WOF_CSV: &str = concat!(
    "id,parent_id,name,placetype\n",
    "200,20,\"Saint-Germain, Paris\",neighbourhood\n",
    "20,-1,Ile-de-France,region\n",
);

fn entry(id: &str, name: &str, lat: f64, lon: f64, country: &str, population: i64, wof_id: &str) -> GazetteerEntry {
    GazetteerEntry::new(EntryParts {
        source: "fixture".into(),
        record_id: id.into(),
        name: name.into(),
        alternate_names: Vec::new(),
        location: GeoPoint::new(lat, lon).unwrap(),
        bounding_box: None,
        country_code: Some(country.into()),
        containment: Vec::new(),
        population,
        feature_class: Some(FEATURE_CLASS_CITY.into()),
        attributes: vec![(ATTRIBUTE_KEY_WHOSONFIRST.into(), AttributeValue::new(wof_id, "wof", "").unwrap())],
    })
    .unwrap()
}

fn fixture_gazetteer() -> InMemoryGazetteer {
    let entries = [
        entry("us-il-springfield", "Springfield", 39.7817, -89.6501, "US", 114_394, "101"),
        entry("us-ma-springfield", "Springfield", 42.1015, -72.5898, "US", 155_929, "100"),
        entry("us-il-chicago", "Chicago", 41.8781, -87.6298, "US", 2_746_388, "102"),
        entry("us-ma-boston", "Boston", 42.3601, -71.0589, "US", 675_647, "110"),
        entry("fr-paris", "Paris", 48.8566, 2.3522, "FR", 2_161_000, "103"),
        entry("us-tx-paris", "Paris", 33.6609, -95.5555, "US", 24_476, "104"),
        entry("fr-lyon", "Lyon", 45.7640, 4.8357, "FR", 522_250, "105"),
        entry("it-turin", "Turin", 45.0703, 7.6869, "IT", 848_748, "106"),
        entry("pt-porto", "Porto", 41.1579, -8.6291, "PT", 231_800, "107"),
        entry("au-sydney", "Sydney", -33.8688, 151.2093, "AU", 5_312_163, "108"),
        entry("mx-guadalajara", "Guadalajara", 20.6597, -103.3496, "MX", 1_385_629, "109"),
        entry("fr-st-germain", "Saint-Germain", 48.8540, 2.3327, "FR", 50_000, "200"),
    ];
    InMemoryGazetteer::from_entries(&entries).unwrap()
}

fn fixture_spine() -> ContainmentSpine {
    let mut b = ContainmentSpine::builder();
    b.add_table_bytes(SPINE_TSV.as_bytes(), "spine.tsv").unwrap();
    b.add_wof_meta_bytes(SPINE_WOF_CSV.as_bytes(), "wof.csv").unwrap();
    b.build().unwrap()
}

struct Scenario {
    id: &'static str,
    text: &'static str,
    mentions: &'static [&'static str],
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            id: "multi-country-ballot",
            text: "Lyon, Turin and Porto met in C\u{00F4}te d\u{2019}Ivoire and Saint Barth\u{00E9}lemy \u{1F1E9}\u{1F1EA} \u{1F1E6}\u{1F1FA} before paying \u{20AC}1.234,56 and $500.",
            mentions: &["Lyon", "Turin", "Porto", "C\u{00F4}te d\u{2019}Ivoire", "Saint Barth\u{00E9}lemy"],
        },
        Scenario {
            id: "spatial-springfield-il",
            text: "Springfield and Chicago discussed a $1,250.50 contract.",
            mentions: &["Springfield", "Chicago"],
        },
        Scenario {
            id: "spatial-springfield-ma",
            text: "Springfield and Boston discussed a $1,250.50 contract.",
            mentions: &["Springfield", "Boston"],
        },
        Scenario {
            id: "guadalajara-boston-mxn",
            text: "Guadalajara outsells Boston this quarter, and lunch there costs $8.",
            mentions: &["Guadalajara", "Boston"],
        },
        Scenario {
            id: "paris-lyon-wof",
            text: "Paris, Lyon and Saint-Germain raised \u{20AC}500 and $200 in France \u{1F1EB}\u{1F1F7}.",
            mentions: &["Paris", "Lyon", "Saint-Germain", "France"],
        },
        Scenario {
            id: "sydney-australia-aud",
            text: "Sydney and Australia \u{1F1E6}\u{1F1FA} closed a $3 million round.",
            mentions: &["Sydney", "Australia"],
        },
    ]
}

fn run_pipeline<G: Geocoder + Send + Sync + 'static>(
    scenario: &Scenario,
    geocoder: G,
    spine: &ContainmentSpine,
) -> BTreeMap<&'static str, String> {
    let mut doc = Document::new(scenario.text);
    doc.add(&Key::DEFAULT, Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None }, Provenance::by("test"))
        .unwrap();
    let kinds = TagSet::new("ner", &[DEFAULT_LOCATION_TYPE]);
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut tokens = Vec::new();
    let mut kind = Vec::new();
    let mut score = Vec::new();
    for m in scenario.mentions {
        let start = scenario.text.find(m).unwrap();
        let end = start + m.len();
        spans.push(Span::new(start as u32, end as u32));
        tokens.push(IxRange::new(0, 0));
        kind.push(Tag(0));
        score.push(1.0);
    }
    doc.add(&Key::DEFAULT, Entities { spans, tokens, kind, kinds, score }, Provenance::by("test")).unwrap();

    let analyzer = Analyzer::builder()
        .add(GeocodeAnnotator::new(geocoder))
        .add(DocumentRegionAnnotator::new())
        .add(HierarchyAnnotator::new(spine.clone()))
        .add(RegionAwareMoneyAnnotator::new())
        .given(Key::<Entities>::DEFAULT)
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();

    let locs = doc.get::<Locations>().unwrap();
    let locations = (0..locs.len())
        .map(|i| {
            let ix = Ix::new(i as u32);
            let span = locs.spans.get(ix);
            let e = &locs.entry[i];
            let loc = e.location();
            format!(
                "{}..{}:{}:{}:{}:{}:{:016x}:{:016x}:{:016x}",
                span.start,
                span.end,
                e.source(),
                e.record_id(),
                e.country_code().unwrap_or(""),
                e.population(),
                loc.latitude().to_bits(),
                loc.longitude().to_bits(),
                locs.confidence[i].to_bits()
            )
        })
        .collect::<Vec<_>>()
        .join(";");

    let regions_layer = doc.get::<Regions>().unwrap();
    let regions = (0..regions_layer.len())
        .map(|i| {
            let v = regions_layer.get(i).unwrap();
            format!("{}:{:016x}:{:?}", v.country_code(), v.share().to_bits(), v.share())
        })
        .collect::<Vec<_>>()
        .join(";");

    let margins = [(0.0, "0.0"), (0.05, "0.05"), (0.25, "0.25"), (0.5, "0.5")];
    let winners = margins
        .iter()
        .map(|&(m, label)| {
            let w = DocumentRegionAnnotator::winner(regions_layer, m)
                .unwrap()
                .map_or_else(|| "-".to_owned(), |v| v.country_code().to_owned());
            format!("{label}={w}")
        })
        .collect::<Vec<_>>()
        .join(";");

    let cont = doc.get::<Containment>().unwrap();
    let containment = (0..cont.len())
        .map(|i| {
            let ix = Ix::new(i as u32);
            let span = cont.spans.get(ix);
            let chain =
                cont.ancestors(ix).map(|a| format!("{}|{}|{}", a.id, a.name, a.kind)).collect::<Vec<_>>().join(">");
            format!("{}..{}:{}", span.start, span.end, chain)
        })
        .collect::<Vec<_>>()
        .join(";");

    let money = doc
        .get::<Money>()
        .map(|layer| {
            (0..layer.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let m = layer.get(ix);
                    format!(
                        "{}..{}:{}:{}",
                        m.span.start,
                        m.span.end,
                        layer.amount(ix).to_plain_string(),
                        m.currency.as_str()
                    )
                })
                .collect::<Vec<_>>()
                .join(";")
        })
        .unwrap_or_default();

    let mut out = BTreeMap::new();
    out.insert("locations", locations);
    out.insert("regions", regions);
    out.insert("winners", winners);
    out.insert("containment", containment);
    out.insert("money", money);
    out
}

#[test]
fn pipeline_matches_jvm_vectors() {
    let mut expected: BTreeMap<(String, String, String), String> = BTreeMap::new();
    for (line_no, raw) in VECTORS.lines().enumerate() {
        let line = raw.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.splitn(4, '\t').collect();
        assert_eq!(cols.len(), 4, "line {}: expected 4 columns", line_no + 1);
        expected.insert((cols[0].to_owned(), cols[1].to_owned(), cols[2].to_owned()), cols[3].to_owned());
    }

    let spine = fixture_spine();
    let mut checked = 0usize;
    for sc in scenarios() {
        let prior_out = run_pipeline(&sc, PopulationPriorGeocoder::new(fixture_gazetteer()), &spine);
        for (layer, actual) in prior_out {
            let key = (sc.id.to_owned(), "prior".to_owned(), layer.to_owned());
            let exp = expected.get(&key).unwrap_or_else(|| panic!("missing vector {key:?}"));
            assert_eq!(&actual, exp, "mismatch for {key:?}");
            checked += 1;
        }

        let spatial_out = run_pipeline(&sc, SpatialCoherenceGeocoder::new(fixture_gazetteer()), &spine);
        for (layer, actual) in spatial_out {
            let key = (sc.id.to_owned(), "spatial".to_owned(), layer.to_owned());
            let exp = expected.get(&key).unwrap_or_else(|| panic!("missing vector {key:?}"));
            assert_eq!(&actual, exp, "mismatch for {key:?}");
            checked += 1;
        }
    }
    assert_eq!(checked, expected.len());
}
