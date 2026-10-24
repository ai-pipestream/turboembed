//! `EventExtractor` on hand-parsed sentences: N-ary financial frames with
//! money, quantity and temporal anchoring, passives with and without
//! subtyped relations, coordinate subjects, copulas, nominal triggers,
//! and the layer through the analyzer and the document graph bytes.

mod event_support;

use event_support::*;
use turbo_analysis::{Analyzer, Annotator, DocumentGraph, Key, export, import};
use turbo_analysis_layers::{
    Dependencies, Entities, Events, Money, Quantities, Sentences, TemporalValue, Temporals, Tokens,
};
use turbo_relation::export::codecs;
use turbo_relation::{EventConfig, EventExtractor};

fn row(t: &str, role: &str, arg: &str, layer: &str) -> (String, String, String, String) {
    (t.into(), role.into(), arg.into(), layer.into())
}

fn revenue() -> turbo_analysis::Document {
    let mut d = parsed(
        "Revenue rose 14% to $120M in Q3 2025 .",
        "2:nsubj 0:root 2:obl:npmod 5:case 2:obl 7:case 2:obl 7:nummod 2:punct",
    );
    add_quantities(&mut d, &[(2, 3)]);
    add_money(&mut d, &[(4, 5)]);
    add_temporals(&mut d, &[((6, 8), TemporalValue::Quarter { year: 2025, quarter: 3 })]);
    d
}

#[test]
fn financial_frame_anchors_measures_and_time() {
    let d = revenue();
    let e = events(&d, EventConfig::default());
    assert_eq!(turbo_analysis::Layer::len(&e), 1);
    assert_eq!(
        rows(&d, &e),
        vec![
            row("rose", "Agent", "Revenue", "None"),
            row("rose", "Measure", "14%", "Quantities"),
            row("rose", "Measure", "$120M", "Money"),
            row("rose", "Time", "Q3 2025", "Temporals"),
        ]
    );
    let f = e.get(0);
    assert_eq!(f.arguments[1].anchored_index, Some(0));
    assert!(f.confidence > 0.8);
    let plain = events(&d, EventConfig { anchor: false, ..Default::default() });
    let roles: Vec<String> = rows(&d, &plain).into_iter().map(|r| r.1).collect();
    assert_eq!(roles, ["Agent", "Measure", "Recipient", "Location"]);
}

#[test]
fn governed_mentions_and_location_entities() {
    // `March 2026` hangs off `funding`, not the verb.
    let mut d = parsed(
        "Acme raised $45M in Series B funding in March 2026 in Austin .",
        "2:nsubj 0:root 2:obj 7:case 7:compound 7:compound 2:obl 9:case 7:nmod 9:nummod 12:case 2:obl 2:punct",
    );
    add_money(&mut d, &[(2, 3)]);
    add_temporals(&mut d, &[((8, 10), TemporalValue::Month { year: 2026, month: 3 })]);
    add_entities(&mut d, &[((0, 1), "organization"), ((11, 12), "location")]);
    let e = events(&d, EventConfig::default());
    assert_eq!(
        rows(&d, &e),
        vec![
            row("raised", "Agent", "Acme", "Names"),
            row("raised", "Measure", "$45M", "Money"),
            row("raised", "Location", "Series B funding in March 2026", "None"),
            row("raised", "Time", "March 2026", "Temporals"),
            row("raised", "Location", "Austin", "Names"),
        ]
    );
}

#[test]
fn passive_with_subtyped_relations() {
    let mut d = parsed(
        "The company was acquired by Globex in 2024 .",
        "2:det 4:nsubj:pass 4:aux:pass 0:root 6:case 4:obl 8:case 4:obl 4:punct",
    );
    add_temporals(&mut d, &[((7, 8), TemporalValue::Year(2024))]);
    let e = events(&d, EventConfig::default());
    assert_eq!(
        rows(&d, &e),
        vec![
            row("acquired", "Patient", "The company", "None"),
            row("acquired", "Agent", "Globex", "None"),
            row("acquired", "Time", "2024", "Temporals"),
        ]
    );
}

#[test]
fn passive_from_tags_without_subtypes() {
    let mut d = parsed("Prices were cut by Acme .", "3:nsubj 3:aux 0:root 5:case 3:obl 3:punct");
    add_pos(&mut d, "NNS VBD VBN IN NNP .");
    let roles: Vec<String> = rows(&d, &events(&d, EventConfig::default())).into_iter().map(|r| r.1).collect();
    assert_eq!(roles, ["Patient", "Agent"]);
    let mut active = parsed("Acme cut prices by 5% .", "2:nsubj 0:root 2:obj 5:case 2:obl 2:punct");
    add_pos(&mut active, "NNP VBD NNS IN CD .");
    let roles: Vec<String> = rows(&active, &events(&active, EventConfig::default())).into_iter().map(|r| r.1).collect();
    assert_eq!(roles, ["Agent", "Patient", "Measure"]);
}

#[test]
fn coordinate_verbs_share_a_subject() {
    let d = parsed("Acme cut costs and raised prices .", "2:nsubj 0:root 2:obj 5:cc 2:conj 5:obj 2:punct");
    let e = events(&d, EventConfig::default());
    assert_eq!(
        rows(&d, &e),
        vec![
            row("cut", "Agent", "Acme", "None"),
            row("cut", "Patient", "costs", "None"),
            row("raised", "Agent", "Acme", "None"),
            row("raised", "Patient", "prices", "None"),
        ]
    );
    let off = events(&d, EventConfig { propagate_conj: false, ..Default::default() });
    assert_eq!(rows(&d, &off).len(), 3);
}

#[test]
fn copula_and_nominal_triggers() {
    let d = parsed("Margins were thin because of costs .", "3:nsubj 3:cop 0:root 6:case 4:fixed 3:obl 3:punct");
    assert_eq!(
        rows(&d, &events(&d, EventConfig::default())),
        vec![
            row("were", "Patient", "Margins", "None"),
            row("were", "Attribute", "thin", "None"),
            row("were", "Cause", "costs", "None"),
        ]
    );
    let d =
        parsed("Globex 's acquisition of Initech closed .", "3:nmod:poss 1:case 6:nsubj 5:case 3:nmod 0:root 6:punct");
    assert_eq!(
        rows(&d, &events(&d, EventConfig::default())),
        vec![
            row("acquisition", "Agent", "Globex", "None"),
            row("acquisition", "Patient", "Initech", "None"),
            row("closed", "Agent", "Globex 's acquisition of Initech", "None"),
        ]
    );
    let none = events(&d, EventConfig { nominal_triggers: Vec::new(), ..Default::default() });
    assert_eq!(turbo_analysis::Layer::len(&none), 1);
}

#[test]
fn purpose_and_instrument() {
    let d = parsed(
        "Acme sold bonds with Barclays to fund growth .",
        "2:nsubj 0:root 2:obj 5:case 2:obl 7:mark 2:advcl 7:obj 2:punct",
    );
    let roles: Vec<String> = rows(&d, &events(&d, EventConfig::default())).into_iter().map(|r| r.1).collect();
    assert_eq!(roles, ["Agent", "Patient", "Instrument", "Purpose", "Patient"]);
}

#[test]
fn config_hash_tracks_every_setting() {
    let base = EventExtractor::new().config_hash();
    assert_eq!(base, EventExtractor::new().config_hash());
    for config in [
        EventConfig { anchor: false, ..Default::default() },
        EventConfig { propagate_conj: false, ..Default::default() },
        EventConfig { nominal_triggers: vec!["merger".into()], ..Default::default() },
    ] {
        assert_ne!(EventExtractor::with_config(config).config_hash(), base);
    }
}

#[test]
fn analyzer_provenance_and_graph_round_trip() {
    let d = revenue();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .given(Key::<Temporals>::DEFAULT)
        .given(Key::<Money>::DEFAULT)
        .given(Key::<Quantities>::DEFAULT)
        .add(EventExtractor::new())
        .build()
        .unwrap();
    let mut batch = vec![d];
    for r in analyzer.run(&mut batch) {
        r.unwrap();
    }
    let d = batch.pop().unwrap();
    let prov = d.provenance(Key::<Events>::DEFAULT.id()).unwrap();
    assert_eq!(prov.annotator, EventExtractor::ID);
    assert!(prov.reads.contains(&Key::<Money>::DEFAULT.id()));
    assert!(!prov.reads.contains(&Key::<Entities>::DEFAULT.id()));
    let e = d.require(&Key::<Events>::DEFAULT).unwrap();
    let codecs = codecs();
    let bytes = export(&d, &codecs).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(back.require(&Key::<Events>::DEFAULT).unwrap(), e);
}
