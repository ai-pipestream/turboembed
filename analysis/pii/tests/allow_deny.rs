//! Integration tests for allow lists (global and per-kind, exact terms,
//! shape/glob patterns, and caller files), deny lists (Aho-Corasick exact
//! matching, Levenshtein fuzzy edits at distance 0, 1, and 2, case and
//! `capitalised_only` flags, and caller files), `config_hash` sensitivity,
//! and suppressed mention counts in `PiiAuditReport`.

mod support;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use support::*;
use turbo_analysis::{Annotator, Layer, ViewId};
use turbo_pii::rewrite::{HmacTokenizer, PiiAuditReport};
use turbo_pii::{
    AllowList, Case, DenyList, ListError, Pack, Pattern, PatternKind, PiiConfidence, PiiDetector, PiiKind,
};

const KEY: &[u8] = b"0123456789abcdef0123456789abcdef";

struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    fn new(label: &str) -> ScratchDir {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("turbo-pii-allow-deny-{}-{label}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create scratch dir");
        ScratchDir { path }
    }

    fn write(&self, name: &str, content: &str) -> PathBuf {
        let file = self.path.join(name);
        std::fs::write(&file, content).expect("write scratch file");
        file
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn per_kind_ignore_leaves_other_kinds_alone() {
    // "943 476 5919" is both a valid UK NHS number and a 10-digit phone shape
    // on the exact same span. Ignoring it for `Phone` only keeps `UkNhs`.
    let text = "Patient NHS 943 476 5919, email help@example.com, card 4111 1111 1111 1111.";
    let baseline = layer(&wide(), text);
    assert_eq!(
        baseline.mentions().map(|m| m.kind).collect::<Vec<_>>(),
        [PiiKind::UkNhs, PiiKind::Phone, PiiKind::Email, PiiKind::Card]
    );

    let phone_ignored = wide().allow_kind_term(PiiKind::Phone, "943 476 5919").unwrap();
    let (pii, suppressed) = phone_ignored.layer_with_suppressed(text, ViewId::SOURCE);
    assert_eq!(pii.check(), Ok(()));
    assert_eq!(pii.mentions().map(|m| m.kind).collect::<Vec<_>>(), [PiiKind::UkNhs, PiiKind::Email, PiiKind::Card]);
    assert_eq!(suppressed, [(PiiKind::Phone, 1)]);

    // Ignoring `help@example.com` under `Card` does not suppress the `Email` mention.
    let wrong_kind = wide().allow_kind_term(PiiKind::Card, "help@example.com").unwrap();
    let pii_wrong = layer(&wrong_kind, text);
    assert_eq!(
        pii_wrong.mentions().map(|m| m.kind).collect::<Vec<_>>(),
        [PiiKind::UkNhs, PiiKind::Phone, PiiKind::Email, PiiKind::Card]
    );

    // Ignoring `help@example.com` under `Email` drops only the email.
    let email_ignored = wide().allow_kind_term(PiiKind::Email, "help@example.com").unwrap();
    let (pii_email, sup_email) = email_ignored.layer_with_suppressed(text, ViewId::SOURCE);
    assert_eq!(
        pii_email.mentions().map(|m| m.kind).collect::<Vec<_>>(),
        [PiiKind::UkNhs, PiiKind::Phone, PiiKind::Card]
    );
    assert_eq!(sup_email, [(PiiKind::Email, 1)]);
}

#[test]
fn global_ignore_suppresses_covered_text_and_normalized_form_across_kinds() {
    let text = "NHS 943 476 5919, call (555) 123-4567, card 4111 1111 1111 1111, mail desk@example.com.";

    // Matching by normalized form ("9434765919" and "4111111111111111") or
    // by covered surface text ("(555) 123-4567") globally suppresses all
    // matching kinds, including both `UkNhs` and `Phone` on "943 476 5919".
    let detector = wide().allow_terms(["9434765919", "(555) 123-4567", "4111111111111111"]).unwrap();
    let (pii, suppressed) = detector.layer_with_suppressed(text, ViewId::SOURCE);
    assert_eq!(pii.check(), Ok(()));
    assert_eq!(
        pii.mentions().map(|m| (m.kind, m.normalized)).collect::<Vec<_>>(),
        [(PiiKind::Email, "desk@example.com")]
    );
    assert_eq!(suppressed, [(PiiKind::Card, 1), (PiiKind::Phone, 2), (PiiKind::UkNhs, 1)]);
}

#[test]
fn allow_list_case_sensitivity_flags() {
    let text = "Contact Support@Example.COM or Billing@Example.COM.";

    // Case-sensitive allow rule for "support@example.com":
    // Email normalizes the domain to lowercase ("Support@example.com"), while
    // covered text is "Support@Example.COM". Neither equals "support@example.com"
    // case-sensitively, so it is kept.
    let sensitive =
        PiiDetector::pack(Pack::Contact).with_allow_list(AllowList::new().with_term("support@example.com").unwrap());
    assert_eq!(
        layer(&sensitive, text).mentions().map(|m| m.normalized).collect::<Vec<_>>(),
        ["Support@example.com", "Billing@example.com"]
    );

    // Exact match on covered text "Support@Example.COM" suppresses it even in
    // case-sensitive mode.
    let exact_covered =
        PiiDetector::pack(Pack::Contact).with_allow_list(AllowList::new().with_term("Support@Example.COM").unwrap());
    assert_eq!(
        layer(&exact_covered, text).mentions().map(|m| m.normalized).collect::<Vec<_>>(),
        ["Billing@example.com"]
    );

    // Case-insensitive allow list suppresses "Support@Example.COM" via
    // "support@example.com", regardless of builder method order.
    let insensitive_after = PiiDetector::pack(Pack::Contact)
        .with_allow_list(AllowList::new().with_term("support@example.com").unwrap().case_insensitive());
    let insensitive_before = PiiDetector::pack(Pack::Contact)
        .with_allow_list(AllowList::new().case_insensitive().with_term("support@example.com").unwrap());
    assert_eq!(insensitive_after, insensitive_before);
    assert_eq!(
        layer(&insensitive_after, text).mentions().map(|m| m.normalized).collect::<Vec<_>>(),
        ["Billing@example.com"]
    );
}

#[test]
fn allow_list_shape_and_glob_patterns() {
    let text = "Cards 4111 1111 1111 1111 and 5500 0000 0000 0004, SSN 123-45-6789, \
                mail qa-bot@internal.example.com and real@customer.org.";

    // Shape pattern `4111 dddd dddd dddd` on covered text and glob
    // `*@internal.example.com` on email, plus per-kind shape `ddd-dd-dddd` on `UsSsn`.
    let detector = wide()
        .allow_shape("4111 dddd dddd dddd")
        .unwrap()
        .allow_glob("*@internal.example.com")
        .unwrap()
        .allow_kind_shape(PiiKind::UsSsn, "ddd-dd-dddd")
        .unwrap();

    let (pii, suppressed) = detector.layer_with_suppressed(text, ViewId::SOURCE);
    assert_eq!(pii.check(), Ok(()));
    assert_eq!(
        pii.mentions().map(|m| (m.kind, m.normalized)).collect::<Vec<_>>(),
        [(PiiKind::Card, "5500000000000004"), (PiiKind::Email, "real@customer.org"),]
    );
    assert_eq!(suppressed, [(PiiKind::Card, 1), (PiiKind::Email, 1), (PiiKind::UsSsn, 1)]);

    // Pattern inspection and escapes.
    let shape = Pattern::shape(r"\d\a\w-d-a-w-?-\*").unwrap();
    assert_eq!(shape.kind(), PatternKind::Shape);
    assert_eq!(shape.as_str(), r"\d\a\w-d-a-w-?-\*");
    assert_eq!(shape.case(), Case::Sensitive);
    assert!(shape.matches("daw-7-Z-9-#-*"));
    assert!(!shape.matches("9aw-7-Z-9-#-*"));

    let glob = Pattern::glob(r"ticket-\?-\*").unwrap();
    assert_eq!(glob.kind(), PatternKind::Glob);
    assert!(glob.matches("ticket-?-*"));
    assert!(!glob.matches("ticket-1-abc"));
}

#[test]
fn file_loading_for_allow_and_deny_lists() {
    let dir = ScratchDir::new("files");
    let allow_path = dir.write(
        "allow.txt",
        "\u{feff}# Global allow list for test fixtures\n\
         \n\
         desk@example.com\n\
         # Support line\n\
         +44 20 7946 0958 \n\
         \t  \n",
    );
    let shape_path = dir.write(
        "shapes.txt",
        "# Test card prefix\n\
         4111 dddd dddd dddd\n",
    );
    let deny_path = dir.write(
        "deny.txt",
        "# Internal codenames\n\
         \n\
         Project Hyperion\n\
         Sunburst\n",
    );

    let codename = PiiKind::custom("project-codename");
    let allow = AllowList::from_file(&allow_path).unwrap().with_shapes_file(&shape_path).unwrap();
    let deny = DenyList::from_file(codename, &deny_path).unwrap();

    let detector = PiiDetector::pack(Pack::AllStructured).with_allow_list(allow).with_deny_list(deny);

    let text = "Write to desk@example.com or jane@example.com, call +44 20 7946 0958, \
                card 4111 1111 1111 1111, regarding Project Hyperion and Sunburst.";
    let (pii, suppressed) = detector.layer_with_suppressed(text, ViewId::SOURCE);
    assert_eq!(pii.check(), Ok(()));
    assert_eq!(
        pii.mentions().map(|m| (m.kind, m.normalized)).collect::<Vec<_>>(),
        [(PiiKind::Email, "jane@example.com"), (codename, "Project Hyperion"), (codename, "Sunburst"),]
    );
    assert_eq!(suppressed, [(PiiKind::Card, 1), (PiiKind::Email, 1), (PiiKind::Phone, 1)]);

    // Missing file returns ListError::Io.
    let missing = dir.path().join("does-not-exist.txt");
    assert!(matches!(AllowList::from_file(&missing), Err(ListError::Io { .. })));
    assert!(matches!(DenyList::from_file(codename, &missing), Err(ListError::Io { .. })));
}

#[test]
fn deny_list_exact_and_fuzzy_edits_0_1_2_and_capitalised_only() {
    let kind = PiiKind::custom("internal-codename");
    // Exact (0 edits), 1 edit ("Palantyr"), 2 edits ("Palxntyr"), 3 edits ("Pxlxntyr"),
    // plus lowercase exact ("palantir") and lowercase 1-edit ("palantyr").
    let text = "Found Palantir, Palantyr, Palxntyr, Pxlxntyr, palantir, and palantyr.";

    // Distance 0, case-sensitive: only exact "Palantir".
    let d0 = PiiDetector::new(vec![])
        .with_deny_list(DenyList::from_terms(kind, ["Palantir"]).unwrap().with_max_edits(0).unwrap());
    assert_eq!(
        layer(&d0, text).mentions().map(|m| (m.kind, m.confidence, m.normalized)).collect::<Vec<_>>(),
        [(kind, PiiConfidence::Format, "Palantir")]
    );

    // Distance 1, case-sensitive: "Palantir" (0 edits), "Palantyr" (1 sub),
    // and "palantir" (1 sub: 'P' -> 'p').
    let d1 = PiiDetector::new(vec![])
        .with_deny_list(DenyList::from_terms(kind, ["Palantir"]).unwrap().with_max_edits(1).unwrap());
    assert_eq!(
        layer(&d1, text).mentions().map(|m| m.normalized).collect::<Vec<_>>(),
        ["Palantir", "Palantyr", "palantir"]
    );

    // Distance 1 with capitalised_only: drops lowercase "palantir".
    let d1_cap = PiiDetector::new(vec![]).with_deny_list(
        DenyList::from_terms(kind, ["Palantir"]).unwrap().with_max_edits(1).unwrap().capitalised_only(),
    );
    assert_eq!(layer(&d1_cap, text).mentions().map(|m| m.normalized).collect::<Vec<_>>(), ["Palantir", "Palantyr"]);

    // Distance 2 with capitalised_only: matches 0, 1, and 2 edits when
    // capitalised ("Palantir", "Palantyr", "Palxntyr"), rejects 3 edits
    // ("Pxlxntyr") and lowercase ("palantir", "palantyr").
    let d2_cap = PiiDetector::new(vec![]).with_deny_list(
        DenyList::from_terms(kind, ["Palantir"]).unwrap().with_max_edits(2).unwrap().capitalized_only(),
    );
    assert_eq!(
        layer(&d2_cap, text).mentions().map(|m| m.normalized).collect::<Vec<_>>(),
        ["Palantir", "Palantyr", "Palxntyr"]
    );

    // Distance 2, case-insensitive without capitalised_only: matches all five
    // within 2 edits ("Palantir", "Palantyr", "Palxntyr", "palantir", "palantyr").
    let d2_ci = PiiDetector::new(vec![]).with_deny_list(
        DenyList::from_terms(kind, ["Palantir"]).unwrap().case_insensitive().with_max_edits(2).unwrap(),
    );
    assert_eq!(
        layer(&d2_ci, text).mentions().map(|m| m.normalized).collect::<Vec<_>>(),
        ["Palantir", "Palantyr", "Palxntyr", "palantir", "palantyr"]
    );

    // Validation errors on DenyList.
    assert_eq!(DenyList::new(PiiKind::Email), Err(ListError::ReservedKind("email")));
    assert_eq!(DenyList::custom(""), Err(ListError::EmptyKind));
    assert_eq!(DenyList::new(kind).unwrap().with_max_edits(3), Err(ListError::InvalidEditDistance(3)));
    assert_eq!(DenyList::new(kind).unwrap().with_term("   "), Err(ListError::EmptyTerm));
}

#[test]
fn config_hash_changes_when_any_allow_or_deny_list_changes() {
    let base = PiiDetector::pack(Pack::Contact);
    let kind_a = PiiKind::custom("codename-a");
    let kind_b = PiiKind::custom("codename-b");

    let variants = [
        base.clone(),
        base.clone().allow_term("desk@example.com").unwrap(),
        base.clone().allow_term("other@example.com").unwrap(),
        base.clone().with_allow_list(AllowList::new().case_insensitive().with_term("desk@example.com").unwrap()),
        base.clone().allow_shape("ddd-dd-dddd").unwrap(),
        base.clone().allow_glob("ddd-dd-dddd").unwrap(),
        base.clone().allow_kind_term(PiiKind::Email, "desk@example.com").unwrap(),
        base.clone().allow_kind_term(PiiKind::Phone, "desk@example.com").unwrap(),
        base.clone().with_deny_list(DenyList::from_terms(kind_a, ["Apollo"]).unwrap()),
        base.clone().with_deny_list(DenyList::from_terms(kind_a, ["Mercury"]).unwrap()),
        base.clone().with_deny_list(DenyList::from_terms(kind_b, ["Apollo"]).unwrap()),
        base.clone().with_deny_list(DenyList::from_terms(kind_a, ["Apollo"]).unwrap().case_insensitive()),
        base.clone().with_deny_list(DenyList::from_terms(kind_a, ["Apollo"]).unwrap().with_max_edits(1).unwrap()),
        base.clone().with_deny_list(DenyList::from_terms(kind_a, ["Apollo"]).unwrap().with_max_edits(2).unwrap()),
        base.clone().with_deny_list(DenyList::from_terms(kind_a, ["Apollo"]).unwrap().capitalised_only()),
    ];

    let hashes: Vec<u64> = variants.iter().map(Annotator::config_hash).collect();
    for i in 0..hashes.len() {
        assert_eq!(hashes[i], variants[i].config_hash());
        for j in (i + 1)..hashes.len() {
            assert_ne!(hashes[i], hashes[j], "variant {i} collided with variant {j}");
        }
    }
}

#[test]
fn audit_report_counts_suppressed_mentions_without_values() {
    let secret_email = "secret-ceo@example.com";
    let public_email = "support@example.com";
    let secret_card = "4111 1111 1111 1111";
    let secret_phone = "(555) 123-4567";
    let text = format!("Email {secret_email} or {public_email}, charge {secret_card}, call {secret_phone}.");

    let detector = PiiDetector::cursor()
        .allow_term(public_email)
        .unwrap()
        .allow_kind_shape(PiiKind::Card, "4111 dddd dddd dddd")
        .unwrap()
        .allow_kind_term(PiiKind::Phone, "+15551234567")
        .unwrap()
        .allow_kind_term(PiiKind::Phone, "5551234567")
        .unwrap();

    let tokenizer = HmacTokenizer::new(KEY).unwrap();
    let (pii, report) = detector.detect_and_audit(&text, ViewId::SOURCE, &tokenizer, PiiAuditReport::DEFAULT_SAMPLES);
    assert_eq!(pii.mentions().count(), 1);
    assert_eq!(report.total, 1);
    assert_eq!(report.total_suppressed, 3);
    assert_eq!(report.suppressed(PiiKind::Email), 1);
    assert_eq!(report.suppressed(PiiKind::Card), 1);
    assert_eq!(report.suppressed(PiiKind::Phone), 1);
    assert_eq!(report.suppressed(PiiKind::Iban), 0);

    let email_report = report.kind(PiiKind::Email).unwrap();
    assert_eq!((email_report.mentions, email_report.distinct, email_report.suppressed), (1, 1, 1));
    let card_report = report.kind(PiiKind::Card).unwrap();
    assert_eq!((card_report.mentions, card_report.distinct, card_report.suppressed), (0, 0, 1));
    assert!(card_report.samples.is_empty());

    let display = report.to_string();
    let debug = format!("{report:?}");
    assert!(display.contains("card: 0 mentions, 0 distinct, 1 suppressed"));
    assert!(display.contains("email: 1 mentions, 1 distinct, 1 suppressed"));
    assert!(display.contains("phone: 0 mentions, 0 distinct, 1 suppressed"));

    // Neither active nor suppressed values nor suppressed tokens may appear.
    let suppressed_email_token = tokenizer.token(PiiKind::Email, public_email).unwrap();
    let suppressed_card_token = tokenizer.token(PiiKind::Card, "4111111111111111").unwrap();
    for forbidden in [
        secret_email,
        public_email,
        "secret-ceo",
        "support",
        secret_card,
        "4111111111111111",
        secret_phone,
        "5551234567",
        &suppressed_email_token,
        &suppressed_card_token,
    ] {
        assert!(!display.contains(forbidden), "leaked {forbidden:?} in Display:\n{display}");
        assert!(!debug.contains(forbidden), "leaked {forbidden:?} in Debug:\n{debug}");
    }
}
