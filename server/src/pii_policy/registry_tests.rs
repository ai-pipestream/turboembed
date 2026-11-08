//! Dictionary references resolved through the vocabulary store (#1882
//! part C): pinned by content, typed refusals that name the reference and
//! never the content, the format each use takes, and the cache.

use std::path::PathBuf;
use std::sync::Arc;

use super::*;
use crate::pii_policy::RedactionPolicy;
use crate::proto::result as pb;
use crate::proto::vocabulary::DictionaryArtifactDescriptor;
use crate::vocabulary::{Dictionary, Store};

const SECRET: &str = "zebulon-planted";

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-pii-dict-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn sha(bytes: &[u8]) -> String {
    turbo_maxent::sha256(bytes).to_hex()
}

fn put(store: &Store, id: &str, format: &str, bytes: &[u8]) -> String {
    let descriptor = DictionaryArtifactDescriptor {
        artifact_id: id.into(),
        format_id: format.into(),
        language: String::new(),
        name: format!("{format} test"),
        entry_count: 1,
        byte_size: bytes.len() as u64,
        sha256: sha(bytes),
    };
    store.put_dictionary(Dictionary { descriptor, has_header_row: false }, bytes, None).unwrap().sha256
}

fn resolver(name: &str) -> (ServerResolver, String, String, String) {
    let store = Arc::new(Store::open(&dir(name)).unwrap());
    let words = put(&store, "dict-words", TERMS_FORMAT, format!("\u{feff}  {SECRET}  \r\nAcme Corp\n\n").as_bytes());
    let map = put(&store, "dict-map", MAP_FORMAT, format!("{SECRET}\tJ. Doe\nbob\t B \n").as_bytes());
    let csv = put(&store, "dict-csv", "glossary-csv", b"g1,New York\n");
    (ServerResolver::load(&[], Some(store)).unwrap(), words, map, csv)
}

#[test]
fn terms_and_map_rows_resolve_normalized_and_are_cached_by_content() {
    let (r, words, map, _) = resolver("ok");
    assert_eq!(&*r.terms("dict-words", &words).unwrap(), &[SECRET.to_owned(), "Acme Corp".to_owned()]);
    let rows = r.map_entries("dict-map", &map).unwrap();
    assert_eq!(&*rows, &[("bob".to_owned(), "B".to_owned()), (SECRET.to_owned(), "J. Doe".to_owned())]);
    assert_eq!(r.cached_dictionaries(), 2);
    assert!(Arc::ptr_eq(&r.terms("dict-words", &words).unwrap(), &r.terms("dict-words", &words).unwrap()));
    assert_eq!(r.cached_dictionaries(), 2, "a second resolve is a cache hit");
    let dbg = format!("{r:?}");
    assert!(dbg.contains("dict-words") && !dbg.contains(SECRET) && !dbg.contains("Acme"), "{dbg}");
}

#[test]
fn every_bad_reference_is_a_typed_refusal() {
    let (r, words, map, csv) = resolver("bad");
    let other = sha(b"other");
    assert_eq!(r.terms("dict-words", &other).unwrap_err(), ResolveError::DigestMismatch);
    assert_eq!(r.terms("dict-none", &words).unwrap_err(), ResolveError::NotRegistered);
    let wrong = |f: &str| ResolveError::WrongFormat { format: f.to_owned() };
    assert_eq!(r.terms("dict-map", &map).unwrap_err(), wrong(MAP_FORMAT));
    assert_eq!(r.map_entries("dict-words", &words).unwrap_err(), wrong(TERMS_FORMAT));
    assert_eq!(r.terms("dict-csv", &csv).unwrap_err(), wrong("glossary-csv"));
    assert_eq!(r.map_entries("dict-csv", &csv).unwrap_err(), wrong("glossary-csv"));
    let none = ServerResolver::load(&[], None).unwrap();
    assert_eq!(none.terms("dict-words", &words).unwrap_err(), ResolveError::NoRegistry);
    assert_eq!(r.cached_dictionaries(), 0, "a refusal caches nothing");
}

#[test]
fn a_store_whose_bytes_changed_under_the_descriptor_is_unreadable() {
    let d = dir("tamper");
    let store = Arc::new(Store::open(&d).unwrap());
    let words = put(&store, "dict-words", TERMS_FORMAT, format!("{SECRET}\n").as_bytes());
    std::fs::write(d.join("dictionaries/dict-words.data"), b"swapped\n").unwrap();
    let r = ServerResolver::load(&[], Some(store)).unwrap();
    let e = r.terms("dict-words", &words).unwrap_err();
    assert_eq!(e, ResolveError::Unreadable);
    assert!(!e.reason().contains("dictionaries"), "no path: {}", e.reason());
}

fn deny_policy(terms: pb::PiiTermSource) -> pb::RedactionPolicy {
    pb::RedactionPolicy {
        id: "p".into(),
        revision: 1,
        custom_kinds: vec!["codename".to_owned()],
        deny: vec![pb::PiiDenyList {
            rule_id: "codenames".into(),
            kind: Some(pb::PiiKindRef { kind: Some(pb::pii_kind_ref::Kind::Custom(0)) }),
            terms: Some(terms),
            ..Default::default()
        }],
        default_strategy: Some(pb::PiiDefaultStrategy {
            strategy: Some(pb::pii_default_strategy::Strategy::Mask(pb::PiiMaskStrategy::default())),
        }),
        ..Default::default()
    }
}

fn dictionary(id: &str, sha256: &str) -> pb::PiiTermSource {
    pb::PiiTermSource {
        source: Some(pb::pii_term_source::Source::Dictionary(pb::PiiDictionaryRef {
            artifact_id: id.into(),
            sha256: sha256.into(),
        })),
    }
}

#[test]
fn a_refused_reference_names_the_ref_and_never_the_content() {
    let (r, words, map, _) = resolver("errors");
    for (id, sha256) in [("dict-words", sha(b"x")), ("dict-map", map), ("dict-none", words)] {
        let err = RedactionPolicy::from_proto(&deny_policy(dictionary(id, &sha256)), &r).unwrap_err().to_string();
        assert!(err.contains(id), "{err}");
        assert!(!err.contains(SECRET) && !err.contains("J. Doe"), "{err}");
    }
}

fn map_policy(table: pb::pii_map_strategy::Table) -> pb::RedactionPolicy {
    let map = pb::PiiMapStrategy {
        table: Some(table),
        target: pb::PiiMapTarget::NormalizedOrSurface as i32,
        ..Default::default()
    };
    pb::RedactionPolicy {
        id: "p".into(),
        revision: 1,
        kind_rules: vec![pb::PiiKindRule {
            rule_id: "email".into(),
            kind: Some(pb::PiiKindRef { kind: Some(pb::pii_kind_ref::Kind::Builtin(pb::PiiKindId::Email as i32)) }),
            state: pb::PiiRuleState::Enabled as i32,
            strategies: vec![pb::PiiStrategyEntry {
                when: Vec::new(),
                strategy: Some(pb::PiiRewriteStrategy {
                    strategy: Some(pb::pii_rewrite_strategy::Strategy::Map(Box::new(map))),
                }),
            }],
            ..Default::default()
        }],
        default_strategy: Some(pb::PiiDefaultStrategy {
            strategy: Some(pb::pii_default_strategy::Strategy::Mask(pb::PiiMaskStrategy::default())),
        }),
        ..Default::default()
    }
}

/// Review 6726099150: keys equal once trimmed (or folded) used to pick the
/// last row by file order in a dictionary and by sorted order inline, so
/// the same entries redacted differently. Both now refuse the policy.
#[test]
fn map_keys_equal_once_normalized_are_refused_inline_and_from_a_dictionary() {
    let store = Arc::new(Store::open(&dir("collide")).unwrap());
    let rows = format!(" {SECRET} \tB\n{SECRET}\tA\n");
    let map = put(&store, "dict-collide", MAP_FORMAT, rows.as_bytes());
    let r = ServerResolver::load(&[], Some(store)).unwrap();
    let inline = pb::PiiMapEntries {
        entries: [(format!(" {SECRET} "), "B".to_owned()), (SECRET.to_owned(), "A".to_owned())].into(),
    };
    let folded = pb::PiiMapEntries {
        entries: [(SECRET.to_uppercase(), "B".to_owned()), (SECRET.to_owned(), "A".to_owned())].into(),
    };
    let dict = pb::PiiDictionaryRef { artifact_id: "dict-collide".into(), sha256: map };
    use pb::pii_map_strategy::Table;
    for table in [Table::Entries(inline), Table::Entries(folded), Table::Dictionary(dict)] {
        let err = RedactionPolicy::from_proto(&map_policy(table), &r).unwrap_err().to_string();
        assert!(err.contains("equal after trimming and case folding"), "{err}");
        assert!(!err.contains(SECRET), "{err}");
    }
    // Distinct keys, or equal keys under case_sensitive, still convert.
    let ok = pb::PiiMapEntries {
        entries: [(SECRET.to_uppercase(), "B".to_owned()), (SECRET.to_owned(), "A".to_owned())].into(),
    };
    let mut policy = map_policy(Table::Entries(ok));
    if let Some(pb::pii_rewrite_strategy::Strategy::Map(m)) =
        &mut policy.kind_rules[0].strategies[0].strategy.as_mut().unwrap().strategy
    {
        m.case_sensitive = true;
    }
    RedactionPolicy::from_proto(&policy, &r).unwrap();
}
