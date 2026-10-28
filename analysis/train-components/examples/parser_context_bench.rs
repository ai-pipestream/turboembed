//! Parser inference and training-event timing on the checked-in JVM fixtures.
//! Run in release mode from the workspace root; one warm-up and five timed runs.
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::hint::black_box;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use turbo_parser::{EnglishHeadRules, Parse, ParserEventType, ParserFactory, ParserModel, ParserTool};
use turbo_train::Event;
use turbo_train_components::parser::{chunking, treeinsert};

fn timed<T>(name: &str, mut run: impl FnMut() -> T, digest: impl Fn(&T) -> u64) {
    drop(black_box(run()));
    let mut times = Vec::new();
    let mut expected = None;
    for _ in 0..5 {
        let start = Instant::now();
        let result = black_box(run());
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        let hash = digest(&result);
        assert_eq!(*expected.get_or_insert(hash), hash);
    }
    times.sort_by(f64::total_cmp);
    println!("{name}\t{:.3}\t{:016x}", times[2], expected.unwrap());
}
fn events_hash(events: &[Event]) -> u64 {
    let mut h = DefaultHasher::new();
    for e in events {
        e.outcome().hash(&mut h);
        e.context().hash(&mut h);
        e.values().map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()).hash(&mut h);
    }
    h.finish()
}
fn main() {
    let root = turbo_goldens::data_dir();
    let rules =
        EnglishHeadRules::from_str(&std::fs::read_to_string(root.join("fixtures/en-head-rules.txt")).unwrap()).unwrap();
    let gold: Vec<Parse> = ["treebank", "held-out"]
        .into_iter()
        .flat_map(|name| {
            std::fs::read_to_string(root.join(format!("inputs/parse-treebank/{name}.txt")))
                .unwrap()
                .lines()
                .filter_map(|s| Parse::parse(s).ok())
                .collect::<Vec<_>>()
        })
        .collect();
    let lines: Vec<String> = gold
        .iter()
        .map(|p| p.token_nodes().iter().map(|t| &t.text()[t.span().range()]).collect::<Vec<_>>().join(" "))
        .collect();
    for (kind, types) in [
        ("chunking", vec![ParserEventType::Build, ParserEventType::Check, ParserEventType::Chunk]),
        ("treeinsert", vec![ParserEventType::Build, ParserEventType::Check, ParserEventType::Attach]),
    ] {
        let fixture =
            std::fs::read_to_string(root.join(format!("parse/model-{kind}/parse-treebank/treebank.golden"))).unwrap();
        let mut hex = false;
        let mut bytes = Vec::new();
        for row in fixture.lines() {
            if row.starts_with("layer\tturbo:model-hex\t") {
                hex = true;
                continue;
            }
            if row == "end\tturbo:model-hex" {
                break;
            }
            if hex {
                let (pairs, remainder) = row.as_bytes().as_chunks::<2>();
                assert!(remainder.is_empty(), "model hex has an odd byte count");
                for pair in pairs {
                    bytes.push(u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap());
                }
            }
        }
        let model = Arc::new(ParserModel::from_bytes(bytes).unwrap());
        let parser = ParserFactory::create_with_params(model, 20, 0.95);
        for k in [1, 5] {
            timed(
                &format!("parse-{kind}-k{k}"),
                || lines.iter().flat_map(|line| ParserTool::parse_line(line, &parser, k).unwrap()).collect::<Vec<_>>(),
                |parses| {
                    let mut h = DefaultHasher::new();
                    for p in parses {
                        let mut text = String::new();
                        p.show(&mut text);
                        text.hash(&mut h);
                        p.prob().to_bits().hash(&mut h);
                    }
                    h.finish()
                },
            );
        }
        let generated: Vec<Parse> =
            lines.iter().flat_map(|line| ParserTool::parse_line(line, &parser, 1).unwrap()).collect();
        for (etype, name) in types.into_iter().zip(if kind == "chunking" {
            ["build", "check", "chunk"]
        } else {
            ["build", "check", "attach"]
        }) {
            let generate = |samples: &[Parse]| -> Vec<Event> {
                if kind == "chunking" {
                    chunking::ParserEventStream::generate_events(samples, &rules, etype, None).unwrap()
                } else {
                    treeinsert::ParserEventStream::generate_events(samples, &rules, etype, None).unwrap()
                }
            };
            let valid: Vec<Parse> = gold
                .iter()
                .chain(&generated)
                .filter(|p| {
                    if kind == "chunking" {
                        chunking::ParserEventStream::generate_events(std::slice::from_ref(*p), &rules, etype, None)
                            .is_ok()
                    } else {
                        treeinsert::ParserEventStream::generate_events(std::slice::from_ref(*p), &rules, etype, None)
                            .is_ok()
                    }
                })
                .cloned()
                .collect();
            let samples: Vec<Parse> = (0..20).flat_map(|_| valid.iter().cloned()).collect();
            println!("samples-{kind}-{name}\t{}", samples.len());
            timed(&format!("events-{kind}-{name}"), || generate(&samples), |e| events_hash(e));
        }
    }
}
