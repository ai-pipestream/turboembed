//! `StaticReplacer` matches a layer's custom kind to its own overrides by
//! name through the layer's table, so two tables that give custom index 0
//! to different kinds never share an override.

use turbo_analysis::{Span, ViewId};
use turbo_pii::rewrite::StaticReplacer;
use turbo_pii::{CustomKinds, Pii, PiiConfidence, PiiKind, PiiMention};

const TEXT: &str = "A-1 B-2";

/// A layer over [`TEXT`] on `kinds`: "A-1" is `acct`, "B-2" is `other`.
fn layer(kinds: &CustomKinds) -> Pii {
    let mut pii = Pii::with_custom_kinds(ViewId::SOURCE, kinds.clone());
    for (span, name) in [(Span::new(0, 3), "acct"), (Span::new(4, 7), "other")] {
        let kind = kinds.find(name).unwrap();
        pii.push(PiiMention { span, kind, confidence: PiiConfidence::Format, normalized: "x" });
    }
    pii
}

#[test]
fn an_override_follows_the_kind_name_not_the_custom_index() {
    let a = CustomKinds::from_names(["acct", "other"]).unwrap();
    let b = CustomKinds::from_names(["other", "acct"]).unwrap();
    // Index 0 names a different kind in each table.
    assert_eq!((a.name(a.find("acct").unwrap()), b.name(a.find("acct").unwrap())), (Some("acct"), Some("other")));
    assert_ne!(a.find("acct"), b.find("acct"));
    let replacer =
        StaticReplacer::new("DEFAULT").with_custom_kinds(a.clone()).with(a.find("acct").unwrap(), "ACCT-REPLACED");

    let on_a = replacer.rewrite(TEXT, &layer(&a)).unwrap().into_text();
    let on_b = replacer.rewrite(TEXT, &layer(&b)).unwrap().into_text();
    assert_eq!(on_a, "ACCT-REPLACED DEFAULT");
    assert_eq!(on_b, "ACCT-REPLACED DEFAULT");

    // A kind the replacer's table does not name takes the default.
    let c = CustomKinds::from_names(["acct-number", "other"]).unwrap();
    let mut pii = Pii::with_custom_kinds(ViewId::SOURCE, c.clone());
    pii.push(PiiMention {
        span: Span::new(0, 3),
        kind: c.find("acct-number").unwrap(),
        confidence: PiiConfidence::Format,
        normalized: "x",
    });
    assert_eq!(replacer.rewrite(TEXT, &pii).unwrap().into_text(), "DEFAULT B-2");
}

#[test]
fn a_custom_override_outside_the_replacers_table_is_refused() {
    let a = CustomKinds::from_names(["acct"]).unwrap();
    let stray = PiiKind::Custom(turbo_pii::CustomKindId::new(1));
    assert!(StaticReplacer::new("D").with_custom_kinds(a.clone()).try_with(stray, "X").is_err());
    assert!(StaticReplacer::new("D").try_with(a.find("acct").unwrap(), "X").is_err());
    // Built-in overrides need no table and survive a table change.
    let r = StaticReplacer::new("D").with(PiiKind::Email, "E").with_custom_kinds(a);
    assert_eq!(r.replacement(PiiKind::Email), "E");
}
