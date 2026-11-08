//! The canonical encoding and PolicyDigest of a RedactionPolicy (issue
//! #1882 part A).
//!
//! The golden: `pii_policy/golden.hex` is the canonical encoding of
//! `common::rich()` as lowercase hex, 64 digits a line. Its SHA-256 was
//! computed outside the crate with
//! `tr -d '\n' < server/tests/pii_policy/golden.hex | perl -pe 's/(..)/chr hex $1/ge' | sha256sum`
//! and is [`GOLDEN_SHA256`]. `TURBO_UPDATE_FIXTURE=1` rewrites the hex file;
//! the digest constant is only ever changed by hand, after recomputing it
//! with that command.

#[path = "pii_policy/common.rs"]
mod common;

use std::time::Instant;

use common::*;
use pb::pii_rewrite_strategy::Strategy;
use prost::Message;
use turbo_kserve::pii_policy::{PolicyDigest, RedactionPolicy, canonical};
use turbo_kserve::proto::result as pb;

const GOLDEN_SHA256: &str = "8120dd99f91719043aa0d97c762ddc340859a0b582500cefe9fd7dced48b868f";

fn digest(p: &pb::RedactionPolicy) -> PolicyDigest {
    RedactionPolicy::from_proto(p, &TestRegistry).expect("the policy converts").digest
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::new();
    for line in bytes.chunks(32) {
        for b in line {
            out.push_str(&format!("{b:02x}"));
        }
        out.push('\n');
    }
    out
}

#[test]
fn the_golden_policy_has_the_recorded_encoding_and_digest() {
    let bytes = canonical::encode(&rich());
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pii_policy/golden.hex");
    if std::env::var_os("TURBO_UPDATE_FIXTURE").is_some() {
        std::fs::write(&path, hex(&bytes)).unwrap();
    }
    let want = std::fs::read_to_string(&path).expect("the golden hex exists");
    assert_eq!(hex(&bytes), want, "the canonical encoding changed");
    assert!(bytes.starts_with(&(PolicyDigest::DOMAIN.len() as u64).to_le_bytes()));
    assert_eq!(&bytes[8..8 + PolicyDigest::DOMAIN.len()], PolicyDigest::DOMAIN.as_bytes());
    assert_eq!(digest(&rich()).to_hex(), GOLDEN_SHA256);
    assert_eq!(PolicyDigest::of_canonical(&bytes).to_hex(), GOLDEN_SHA256);
}

/// The message's fields, each encoded on its own and concatenated in
/// reverse field order: protobuf merges them into the same message.
fn reversed_wire(p: &pb::RedactionPolicy) -> Vec<u8> {
    let parts = [
        pb::RedactionPolicy { redact_and_flag_unknown: p.redact_and_flag_unknown, ..Default::default() },
        pb::RedactionPolicy { custom_identifiers: p.custom_identifiers.clone(), ..Default::default() },
        pb::RedactionPolicy { deny: p.deny.clone(), ..Default::default() },
        pb::RedactionPolicy { allow: p.allow.clone(), ..Default::default() },
        pb::RedactionPolicy { default_strategy: p.default_strategy.clone(), ..Default::default() },
        pb::RedactionPolicy { kind_rules: p.kind_rules.clone(), ..Default::default() },
        pb::RedactionPolicy { custom_kinds: p.custom_kinds.clone(), ..Default::default() },
        pb::RedactionPolicy { revision: p.revision, ..Default::default() },
        pb::RedactionPolicy { id: p.id.clone(), ..Default::default() },
    ];
    parts.iter().flat_map(Message::encode_to_vec).collect()
}

#[test]
fn wire_field_order_does_not_change_the_digest() {
    let p = rich();
    let forward = p.encode_to_vec();
    let reversed = reversed_wire(&p);
    assert_ne!(forward, reversed, "the two wire forms differ");
    let q = pb::RedactionPolicy::decode(&*reversed).unwrap();
    assert_eq!(digest(&q), digest(&p));
    assert_eq!(canonical::encode(&q), canonical::encode(&p));
}

/// A map strategy's entries written in the order given.
fn map_wire(entries: &[(&str, &str)]) -> pb::RedactionPolicy {
    let mut table = Vec::new();
    for (k, v) in entries {
        let mut entry = Vec::new();
        prost::encoding::string::encode(1, &(*k).to_owned(), &mut entry);
        prost::encoding::string::encode(2, &(*v).to_owned(), &mut entry);
        prost::encoding::bytes::encode(1, &entry, &mut table);
    }
    let p = one(Strategy::Map(Box::new(pb::PiiMapStrategy {
        table: Some(pb::pii_map_strategy::Table::Entries(pb::PiiMapEntries::decode(&*table).unwrap())),
        case_sensitive: false,
        target: pb::PiiMapTarget::Normalized as i32,
        fallback: None,
    })));
    pb::RedactionPolicy::decode(&*p.encode_to_vec()).unwrap()
}

#[test]
fn map_entry_order_does_not_change_the_digest() {
    let entries = [("a@example.com", "A"), ("b@example.com", "B"), ("c@example.com", "C"), ("d@example.com", "D")];
    let mut reversed = entries;
    reversed.reverse();
    let forward = map_wire(&entries);
    let backward = map_wire(&reversed);
    assert_eq!(digest(&forward), digest(&backward));
    let changed = map_wire(&[("a@example.com", "A"), ("b@example.com", "X")]);
    assert_ne!(digest(&changed), digest(&map_wire(&entries[..2])));
}

#[test]
fn what_the_policy_says_changes_the_digest() {
    let base = digest(&rich());
    let mut p = rich();
    p.revision += 1;
    assert_ne!(digest(&p), base, "the revision is part of the identity");
    let mut p = rich();
    p.id.push('2');
    assert_ne!(digest(&p), base);
    let mut p = rich();
    p.kind_rules.swap(0, 1);
    assert_ne!(digest(&p), base, "repeated fields are digested in order");
    let mut p = rich();
    p.kind_rules[0].strategies.swap(0, 2);
    assert_ne!(digest(&p), base, "strategy order is part of a policy");
    let mut p = rich();
    p.redact_and_flag_unknown = true;
    assert_ne!(digest(&p), base);
    // An empty message field is present; an unset one is not.
    let mut p = one(Strategy::Mask(pb::PiiMaskStrategy::default()));
    let unset = digest(&p);
    p.allow = Some(pb::PiiAllowList::default());
    assert_ne!(digest(&p), unset);
    // -0.0 and 0.0 are one score threshold.
    let score = |min: f32| {
        let mut p = one(Strategy::Mask(pb::PiiMaskStrategy::default()));
        p.kind_rules[0].strategies[0].when = vec![pb::PiiStrategyCondition {
            test: Some(pb::pii_strategy_condition::Test::ScoreAtLeast(pb::PiiScoreCondition {
                kind: pb::PiiScoreKind::Heuristic as i32,
                min,
                calibrated: false,
            })),
        }];
        digest(&p)
    };
    assert_eq!(score(0.0), score(-0.0));
    assert_ne!(score(0.0), score(0.5));
}

fn fifty_rules() -> pb::RedactionPolicy {
    let mut p = rich();
    p.kind_rules.clear();
    for id in 1..=30 {
        let kind = pb::PiiKindId::try_from(id).unwrap();
        let strategies = vec![
            pb::PiiStrategyEntry {
                when: vec![pb::PiiStrategyCondition {
                    test: Some(pb::pii_strategy_condition::Test::LabelPresent(pb::PiiLabelPresentCondition {})),
                }],
                strategy: Some(strategy(keyed("tokens", 1))),
            },
            entry(Strategy::LastN(pb::PiiLastNStrategy { count: 4, ..Default::default() })),
        ];
        p.kind_rules.push(rule(&format!("builtin-{id}"), builtin(kind), strategies));
    }
    for i in 0..20u32 {
        p.custom_kinds.push(format!("custom_{i}"));
    }
    for i in 0..20u32 {
        let index = 2 + i;
        p.kind_rules.push(rule(
            &format!("custom-{i}"),
            custom(index),
            vec![entry(Strategy::Format(pb::PiiFormatStrategy { template: "[%t-%v]".to_owned() }))],
        ));
    }
    assert_eq!(p.kind_rules.len(), 50);
    p
}

#[test]
fn a_fifty_rule_policy_converts_and_digests_quickly() {
    let p = fifty_rules();
    let runs = 200u32;
    let start = Instant::now();
    for _ in 0..runs {
        let c = RedactionPolicy::from_proto(std::hint::black_box(&p), &TestRegistry).unwrap();
        std::hint::black_box(c.digest);
    }
    let per = start.elapsed() / runs;
    eprintln!("50-rule policy: conversion + digest {per:?} per policy over {runs} runs");
    let start = Instant::now();
    for _ in 0..runs {
        std::hint::black_box(PolicyDigest::of_canonical(&canonical::encode(std::hint::black_box(&p))));
    }
    eprintln!("50-rule policy: canonical encoding + digest alone {:?} per policy", start.elapsed() / runs);
}
