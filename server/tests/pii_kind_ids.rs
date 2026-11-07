//! One table-driven check that every PII kind and confidence id is the same
//! number in the Rust registry, the C header, Java, the proto enums and the
//! binary layer codec (#1901).

use std::collections::BTreeMap;

use turbo_analysis::export::{export, import};
use turbo_analysis::{Document, Key, Provenance, Span, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{ChecksumAlgorithm, CustomKinds, Pii, PiiConfidence, PiiEvidence, PiiKind, PiiMention};
use turbo_kserve::proto::result::{PiiConfidenceTier, PiiKindId};

fn read(rel: &str) -> String {
    let path = format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

fn upper(name: &str) -> String {
    name.to_ascii_uppercase().replace('-', "_")
}

/// `#define <prefix><NAME> <n>` lines of the header.
fn header_defines(header: &str, prefix: &str) -> BTreeMap<String, u32> {
    header
        .lines()
        .filter_map(|l| l.strip_prefix("#define "))
        .filter_map(|l| l.strip_prefix(prefix))
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.to_owned(), it.next()?.parse().ok()?))
        })
        .collect()
}

/// `NAME(<n>` entries of a Java enum source.
fn java_codes(source: &str) -> BTreeMap<String, u32> {
    source
        .lines()
        .map(str::trim)
        .filter_map(|l| {
            let (name, rest) = l.split_once('(')?;
            if name.is_empty() || !name.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_') {
                return None;
            }
            let code = rest.split([',', ')']).next()?.trim().parse().ok()?;
            Some((name.to_owned(), code))
        })
        .collect()
}

#[test]
fn every_kind_and_tier_has_one_number_everywhere() {
    let header = read("include/turbo/turbo.h");
    let c_kinds = header_defines(&header, "TURBO_PII_KIND_");
    let c_tiers = header_defines(&header, "TURBO_PII_CONFIDENCE_");
    let java_kinds = java_codes(&read("clients/java-api/src/main/java/ai/pipestream/turboembed/TurboPiiKind.java"));
    let java_tiers =
        java_codes(&read("clients/java-api/src/main/java/ai/pipestream/turboembed/TurboPiiConfidence.java"));

    assert_eq!(PiiKind::TABLE.len(), PiiKind::MAX_BUILTIN_ID as usize);
    assert_eq!(c_kinds.len(), PiiKind::TABLE.len(), "one TURBO_PII_KIND_* per kind, no extras");
    assert_eq!(java_kinds.len(), PiiKind::TABLE.len(), "one TurboPiiKind per kind, no extras");
    for row in PiiKind::TABLE {
        let id = u32::from(row.id);
        let name = upper(row.name);
        assert_eq!(row.kind.builtin_id(), Some(row.id), "{name}: the registry row and the enum agree");
        assert_eq!(row.kind.builtin_name(), Some(row.name), "{name}");
        assert_eq!(PiiKind::from_id(row.id), Some(row.kind), "{name}");
        assert_eq!(PiiKind::parse(row.name), Some(row.kind), "{name}");
        assert_eq!(c_kinds.get(&name), Some(&id), "C TURBO_PII_KIND_{name}");
        assert_eq!(java_kinds.get(&name), Some(&id), "Java TurboPiiKind.{name}");
        let proto = PiiKindId::from_str_name(&format!("PII_KIND_ID_{name}")).expect("a proto PiiKindId value");
        assert_eq!(proto as u32, id, "proto PII_KIND_ID_{name}");
        assert_eq!(row.kind.code(), id, "codec code of {name}");
    }
    assert_eq!(PiiKindId::Unspecified as i32, 0);
    assert_eq!(PiiKind::from_id(0), None, "0 is no kind");

    assert_eq!(c_tiers.len(), PiiConfidence::ALL.len());
    for c in PiiConfidence::ALL {
        let id = u32::from(c.id());
        let name = upper(c.name());
        assert_eq!(PiiConfidence::from_id(id), Some(c));
        assert_eq!(c_tiers.get(&name), Some(&id), "C TURBO_PII_CONFIDENCE_{name}");
        assert_eq!(java_tiers.get(&name), Some(&id), "Java TurboPiiConfidence.{name}");
        let proto = PiiConfidenceTier::from_str_name(&format!("PII_CONFIDENCE_TIER_{name}")).expect("a proto tier");
        assert_eq!(proto as u32, id, "proto PII_CONFIDENCE_TIER_{name}");
    }
    assert_eq!(PiiConfidenceTier::Unspecified as i32, 0);
    assert_eq!(PiiConfidence::from_id(0), None, "0 is no tier");

    // The codec carries those numbers: every kind and tier round-trips.
    let text = "x".repeat(PiiKind::TABLE.len() + 1);
    let custom = CustomKinds::from_names(["account-number"]).unwrap();
    let mut pii = Pii::with_custom_kinds(ViewId::SOURCE, custom.clone());
    let mut kinds: Vec<PiiKind> = PiiKind::builtins().collect();
    kinds.push(custom.find("account-number").unwrap());
    let detector = pii.detector("test", "1").unwrap();
    let label = PiiEvidence::Label { span: Span::new(0, 1), phrase: 0 };
    let luhn = PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: true };
    let evidence: [&[PiiEvidence]; 4] = [&[PiiEvidence::Shape], &[luhn], &[luhn, label], &[PiiEvidence::Shape, label]];
    for (i, &kind) in kinds.iter().enumerate() {
        let evidence = evidence[i % evidence.len()];
        let span = Span::new(i as u32, i as u32 + 1);
        pii.push(PiiMention { span, kind, detector, evidence, score: None, normalized: "v" });
    }
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, pii, Provenance::by("test:pii")).unwrap();
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&graph, &codecs()).unwrap();
    let back = back.get::<Pii>().unwrap();
    assert_eq!(back.kind, kinds);
    assert_eq!(back.custom_kinds, custom);
    assert_eq!(back.get(turbo_analysis::Ix::new(3)).confidence(), PiiConfidence::Labelled);
    assert_eq!(back.get(turbo_analysis::Ix::new(2)).confidence(), PiiConfidence::LabelledChecksum);
}
