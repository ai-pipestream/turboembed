//! Bounded, dev-only replay of an add-only city proposal against actual
//! packaged PII analyzers. Does not replace or serialize a serving layer.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::{Value, json};
use turbo_analysis::{Analyzer, Document, Layer, ViewId};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Pii, PiiCandidates, PiiKind};
use turbo_pii::{Pack, PiiDetector};
use turbo_tagger::PiiTagger;
use turbo_tokenize::SimpleTokenizer;

type Bounds = (u32, u32);

#[derive(Clone, Debug, PartialEq, Eq)]
struct Mention {
    bounds: Bounds,
    kind: String,
}

fn open(path: &Path, temp: f32, places: bool) -> Analyzer {
    let session = EncoderOptions { cpu_threads: 6, max_batch: 1, ..EncoderOptions::default() };
    let tagger = PiiTagger::open(path, None, &session)
        .unwrap()
        .with_temperature(temp)
        .unwrap()
        .with_score_threshold(0.5)
        .unwrap();
    let tagger = if places { tagger.with_rules(PiiDetector::pack(Pack::Places)) } else { tagger };
    Analyzer::builder().add(SimpleTokenizer::default()).add(tagger).build().unwrap()
}

fn run(analyzer: &Analyzer, text: &str) -> (Vec<Mention>, BTreeMap<Bounds, Vec<String>>) {
    let mut docs = [Document::new(text)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let pii = docs[0].get::<Pii>().unwrap();
    let candidates = docs[0].get::<PiiCandidates>().unwrap();
    pii.check().unwrap();
    candidates.check().unwrap();
    let mut city_candidates = BTreeMap::<Bounds, Vec<String>>::new();
    for item in candidates.candidates().filter(|c| c.kind == PiiKind::City) {
        city_candidates.entry((item.span.start, item.span.end)).or_default().push(format!("{:?}", item.disposition));
    }
    let found = pii
        .mentions()
        .map(|m| Mention { bounds: (m.span.start, m.span.end), kind: pii.kind_name(m.kind).unwrap().to_owned() })
        .collect();
    (found, city_candidates)
}

fn overlaps(a: Bounds, b: Bounds) -> bool {
    a.0 < b.1 && b.0 < a.1
}

fn extra_city(places: Option<&BTreeSet<Bounds>>, a: &[Mention], b: &[Mention], active: &[Mention]) -> Vec<Bounds> {
    let Some(places) = places else {
        return Vec::new();
    };
    places
        .iter()
        .copied()
        .filter(|&x| {
            b.iter().any(|m| m.kind == "city" && m.bounds == x)
                && active.iter().any(|m| m.kind == "city" && m.bounds == x)
                && !a.iter().any(|m| overlaps(x, m.bounds))
        })
        .collect()
}

struct Paths {
    catalog: Analyzer,
    refit_head: Analyzer,
    refit_places: Analyzer,
}

#[derive(Default, Debug)]
struct Counts {
    scanned: usize,
    serving_rows: usize,
    named: BTreeMap<String, usize>,
}

impl Counts {
    fn track(&mut self, label: &str, gold: bool) {
        *self.named.entry(format!("{label}:{}", if gold { "gold" } else { "not_gold" })).or_default() += 1;
    }
}

fn examine(
    paths: &Paths,
    rules: &PiiDetector,
    text: &str,
    gold: &BTreeSet<Bounds>,
    screen: Option<&BTreeSet<Bounds>>,
    out: &mut Counts,
) {
    out.scanned += 1;
    let independent = rules.layer(text, ViewId::SOURCE);
    let mut place: BTreeSet<_> =
        independent.mentions().filter(|m| m.kind == PiiKind::City).map(|m| (m.span.start, m.span.end)).collect();
    if let Some(eligible) = screen {
        for &x in eligible.difference(&place) {
            let reason = if place.iter().any(|&p| overlaps(x, p)) {
                "independent_places_range_differs"
            } else {
                "not_independent_places"
            };
            out.track(reason, gold.contains(&x));
        }
        place = place.intersection(eligible).copied().collect();
    }
    if place.is_empty() {
        return;
    }
    out.serving_rows += 1;
    let (catalog, candidates) = run(&paths.catalog, text);
    let (refit, _) = run(&paths.refit_head, text);
    let (active, _) = run(&paths.refit_places, text);
    let extra = extra_city(Some(&place), &catalog, &refit, &active);
    assert!(extra_city(None, &catalog, &refit, &active).is_empty(), "standalone path has no overlay");
    for &x in &place {
        let wanted = gold.contains(&x);
        if !refit.iter().any(|m| m.kind == "city" && m.bounds == x) {
            out.track("not_neural_city_at_serving_bar", wanted);
        } else if !active.iter().any(|m| m.kind == "city" && m.bounds == x) {
            out.track("not_admitted_with_refit_places", wanted);
        } else if catalog.iter().any(|m| m.kind == "city" && m.bounds == x) {
            out.track("catalog_already_city", wanted);
        } else if let Some(lock) = catalog.iter().find(|m| overlaps(x, m.bounds)) {
            out.track(&format!("catalog_overlap_{}", lock.kind), wanted);
            if let Some(reasons) = candidates.get(&x) {
                for reason in reasons {
                    out.track(&format!("catalog_city_candidate_{reason}"), wanted);
                }
            } else {
                out.track("catalog_city_candidate_removed_under_overlap", wanted);
            }
        } else {
            assert!(extra.contains(&x));
            out.track("added_city", wanted);
        }
    }
    assert!(extra.iter().all(|&x| !catalog.iter().any(|m| overlaps(m.bounds, x))));
}

fn from_json(value: &Value) -> BTreeSet<Bounds> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let ranged = v.as_array().unwrap();
            (ranged[0].as_u64().unwrap() as u32, ranged[1].as_u64().unwrap() as u32)
        })
        .collect()
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert_eq!(args.len(), 3, "pii_city_rescue_audit <catalog bundle> <refit bundle> <dev shortlist.json>");
    let a = turbo_pins::require(&args[0]);
    let b = turbo_pins::require(&args[1]);
    assert_eq!(turbo_pins::digest(&a).unwrap().1, "19d3fb87daac08f0361390809b747af638bc2dbb93e447695c3f5609fc735b6d");
    assert_eq!(turbo_pins::digest(&b).unwrap().1, "dd73330bc8bef5bc0f1b9c7b6c004cfbf6713fa1dadcc03a6238a159b6ddc3b7");
    let paths = Paths {
        catalog: open(&a, 1.093_758, true),
        refit_head: open(&b, 1.080_714_2, false),
        refit_places: open(&b, 1.080_714_2, true),
    };
    let places = PiiDetector::pack(Pack::Places);
    let data: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let mut publisher = Counts::default();
    for entry in data["entries"].as_array().unwrap() {
        let text = entry["text"].as_str().unwrap();
        examine(&paths, &places, text, &from_json(&entry["gold"]), Some(&from_json(&entry["ranges"])), &mut publisher);
    }
    let mut synthetic = Counts::default();
    for (text, name) in [
        ("Hudson invested in a bus terminal by the waterfront.", Some("Hudson")),
        ("Hudson repaired a guitar at the kitchen counter.", None),
        ("Hudson Electric offered a free wiring inspection.", None),
        ("Bristol widened its walking paths last summer.", Some("Bristol")),
        ("Bristol wrapped a birthday gift in blue paper.", None),
        ("Bristol Works upgraded its payroll software.", None),
        ("Portland, Oregon opened a new public records office.", Some("Portland")),
        ("The bike carrier spoke to Portland about a missing receipt.", None),
        ("Rochester repaired the public pier after the storm.", Some("Rochester")),
        ("Rochester asked for another bowl of oatmeal.", None),
        ("At the depot, the van took up most of the parking space.", None),
        ("A crate near the back door reads al in the stencil.", None),
        ("We visited Ithaca to inspect the reinforced docks.", Some("Ithaca")),
        ("Our producers called Ithaca about their rehearsal time.", None),
        ("A regional council hired a consultant for repairs.", None),
        ("Take the bin to the shed when the rain stops.", None),
    ] {
        let gold = name
            .into_iter()
            .map(|s| {
                let i = text.find(s).unwrap() as u32;
                (i, i + s.len() as u32)
            })
            .collect();
        examine(&paths, &places, text, &gold, None, &mut synthetic);
    }
    let report = |x: Counts| json!({"rows":x.scanned,"actual_serving_rows":x.serving_rows,"counts":x.named});
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "publisher_initial_meta":{"rows":data["rows"],"gold":data["gold"],"cached_options":data["cached_options"]},
            "publisher_shortlist":report(publisher),"fresh_synthetic":report(synthetic),"holdout_scored":false
        }))
        .unwrap()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mention(bounds: Bounds, kind: &str) -> Mention {
        Mention { bounds, kind: kind.into() }
    }

    #[test]
    fn gate_is_places_only_and_preserves_all_catalog_spans() {
        let rules = BTreeSet::from([(0, 4), (7, 12), (15, 20), (23, 28)]);
        let catalog = [mention((0, 4), "city"), mention((6, 13), "first-name"), mention((24, 25), "company-name")];
        let proposed = [
            mention((0, 4), "city"),
            mention((7, 12), "city"),
            mention((15, 20), "city"),
            mention((23, 28), "city"),
            mention((30, 35), "city"),
        ];
        assert_eq!(extra_city(Some(&rules), &catalog, &proposed, &proposed), [(15, 20)]);
        assert!(extra_city(None, &catalog, &proposed, &proposed).is_empty());
    }
}
