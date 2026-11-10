//! Integration tests for format-string redaction (`FormatRedactor`), fixed
//! static replacement (`StaticReplacer`), unkeyed SHA-256 digests
//! (`Sha256Tokenizer`), caller lookup tables (`MapReplacer`), partial
//! character retention (`Truncate` and `MaskPolicy`), initials abbreviation
//! (`Abbreviator`), and per-kind strategy composition (`RewritePolicy`).

mod support;

use support::*;
use turbo_analysis::Span;
use turbo_pii::rewrite::source_view;
use turbo_pii::rewrite::{
    Abbreviator, FormatError, FormatRedactor, HmacTokenizer, MapError, MapReplacer, MapTarget, MaskPolicy,
    Pseudonymizer, RewriteError, RewritePolicy, RewriteStrategy, Sha256Replacer, Sha256Tokenizer, StaticReplacer,
    TokenizerError, Truncate,
};
use turbo_pii::{Pii, PiiDetector, PiiKind};

const PERSON_KIND: PiiKind = test_kind(0);
const KEY: &[u8] = b"0123456789abcdef0123456789abcdef";

fn cursor() -> PiiDetector {
    PiiDetector::cursor()
}

#[test]
fn format_redactor_expands_templates_reuses_numbers_and_rejects_bad_syntax() {
    let text = "Contact jane@example.com, call (555) 123-4567, or reply to jane@example.com and bob@example.com.";
    let pii = layer(&cursor(), text);
    let sources: Vec<Span> = pii.mentions().map(|m| m.span).collect();

    let kind_only = FormatRedactor::new("{{{REDACTED-%t}}}").unwrap();
    assert_eq!(kind_only.template(), "{{{REDACTED-%t}}}");
    assert!(!kind_only.uses_number());
    let r1 = kind_only.rewrite(&source_view(text), &pii).unwrap();
    assert_eq!(
        r1.text(),
        "Contact {{{REDACTED-EMAIL}}}, call {{{REDACTED-PHONE}}}, or reply to {{{REDACTED-EMAIL}}} and {{{REDACTED-EMAIL}}}."
    );
    assert_offset_and_span_maps(text, &sources, &r1);

    let numbered = FormatRedactor::new("[%t-%v]").unwrap();
    assert!(numbered.uses_number());
    let r2 = numbered.rewrite(&source_view(text), &pii).unwrap();
    assert_eq!(r2.text(), "Contact [EMAIL-1], call [PHONE-1], or reply to [EMAIL-1] and [EMAIL-2].");
    assert_offset_and_span_maps(text, &sources, &r2);

    let lower_and_percent = FormatRedactor::new("<%k#%v:100%%>").unwrap();
    assert_eq!(lower_and_percent.format_label("us-ssn", 3), "<us-ssn#3:100%>");
    assert_eq!(lower_and_percent.format_label("person-name", 2), "<person-name#2:100%>");
    let r3 = lower_and_percent.rewrite(&source_view(text), &pii).unwrap();
    assert_eq!(
        r3.text(),
        "Contact <email#1:100%>, call <phone#1:100%>, or reply to <email#1:100%> and <email#2:100%>."
    );
    assert_offset_and_span_maps(text, &sources, &r3);

    assert_eq!(FormatRedactor::new(""), Err(FormatError::EmptyTemplate));
    assert_eq!(FormatRedactor::new("prefix%"), Err(FormatError::TrailingPercent));
    assert_eq!(FormatRedactor::new("[%q]"), Err(FormatError::UnknownSpecifier('q')));
}

#[test]
fn static_replacer_is_consistent_within_and_across_documents() {
    let text1 = "From jane@example.com to bob@example.com, phone (555) 123-4567.";
    let text2 = "Card 4111 1111 1111 1111 for jane@example.com.";
    let pii1 = layer(&cursor(), text1);
    let pii2 = layer(&cursor(), text2);

    let def = StaticReplacer::default();
    assert_eq!(def.replacement(PiiKind::Email), "[REDACTED]");
    let r1 = def.rewrite(&source_view(text1), &pii1).unwrap();
    assert_eq!(r1.text(), "From [REDACTED] to [REDACTED], phone [REDACTED].");
    let s1: Vec<Span> = pii1.mentions().map(|m| m.span).collect();
    assert_offset_and_span_maps(text1, &s1, &r1);

    let by_kind = StaticReplacer::new("<PII>")
        .with_custom_kinds(test_kinds())
        .with(PiiKind::Email, "<EMAIL>")
        .with(PiiKind::Card, "<CARD>")
        .with(PERSON_KIND, "<NAME>");
    assert_eq!(by_kind.replacement(PiiKind::Email), "<EMAIL>");
    assert_eq!(by_kind.replacement(PiiKind::Phone), "<PII>");
    assert_eq!(by_kind.replacement(PERSON_KIND), "<NAME>");

    let k1 = by_kind.rewrite(&source_view(text1), &pii1).unwrap();
    let k2 = by_kind.rewrite(&source_view(text2), &pii2).unwrap();
    assert_eq!(k1.text(), "From <EMAIL> to <EMAIL>, phone <PII>.");
    assert_eq!(k2.text(), "Card <CARD> for <EMAIL>.");

    let empty = StaticReplacer::new("");
    assert_eq!(
        empty
            .rewrite(&source_view("mail jane@example.com"), &layer(&cursor(), "mail jane@example.com"))
            .map(|r| r.into_text()),
        Err(RewriteError::EmptyValue(Span::new(5, 21)))
    );
}

#[test]
fn sha256_tokenizer_hashes_normalized_values_and_validates_lengths() {
    let sha = Sha256Tokenizer::new();
    assert_eq!(sha.length(), 64);
    assert!(!sha.has_kind_prefix());
    assert_eq!(sha.prefix(), "");

    let expected_jane: String =
        turbo_sha256::sha256(b"jane@example.com").0.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(expected_jane.len(), 64);
    assert_eq!(sha.digest("email", "jane@example.com").unwrap(), expected_jane);

    let text = "Contact JANE@EXAMPLE.COM or jane@example.com.";
    let pii = layer(&cursor(), text);
    let r = sha.rewrite(&source_view(text), &pii).unwrap();
    let expected_upper: String =
        turbo_sha256::sha256(b"JANE@example.com").0.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(r.text(), format!("Contact {expected_upper} or {expected_jane}."));
    let sources: Vec<Span> = pii.mentions().map(|m| m.span).collect();
    assert_offset_and_span_maps(text, &sources, &r);

    let prefixed = Sha256Replacer::with_length(12).unwrap().with_kind_prefix().with_prefix("sha256:");
    assert_eq!(prefixed.length(), 12);
    assert!(prefixed.has_kind_prefix());
    assert_eq!(prefixed.prefix(), "sha256:");
    assert_eq!(prefixed.digest("email", "jane@example.com").unwrap(), format!("sha256:EMAIL-{}", &expected_jane[..12]));

    for bad_len in [0, 1, 3, 65, 100] {
        assert_eq!(Sha256Tokenizer::with_length(bad_len), Err(TokenizerError::Length(bad_len)));
    }
    assert_eq!(sha.digest("email", ""), Err(TokenizerError::EmptyValue));
    let empty_norm = manual(&[(0, 4, PiiKind::Email, "")]);
    assert_eq!(
        sha.rewrite(&source_view("abcd"), &empty_norm).map(|r| r.into_text()),
        Err(RewriteError::EmptyValue(Span::new(0, 4)))
    );
}

#[test]
fn map_replacer_supports_inline_tsv_files_case_sensitivity_targets_and_fallbacks() {
    let inline = MapReplacer::new(
        [("jane@example.com", "USER_JANE"), ("5551234567", "MAIN_DESK"), ("Zoë Krüger", "EXEC_ZOE")],
        false,
    )
    .unwrap()
    .with_fallback(FormatRedactor::new("[UNMAPPED-%t-%v]").unwrap());
    assert!(!inline.is_case_sensitive());
    assert_eq!(inline.len(), 3);
    assert!(!inline.is_empty());
    assert_eq!(inline.get("JANE@EXAMPLE.COM"), Some("USER_JANE"));
    assert_eq!(inline.get("zoë krüger"), Some("EXEC_ZOE"));

    let text = "Mail JANE@EXAMPLE.COM, bob@example.com, call (555) 123-4567, or ask bob@example.com.";
    let pii = layer(&cursor(), text);
    let r = inline.rewrite(&source_view(text), &pii).unwrap();
    assert_eq!(r.text(), "Mail USER_JANE, [UNMAPPED-EMAIL-1], call MAIN_DESK, or ask [UNMAPPED-EMAIL-1].");
    let sources: Vec<Span> = pii.mentions().map(|m| m.span).collect();
    assert_offset_and_span_maps(text, &sources, &r);

    // Case-sensitive mode distinguishes case and surface vs normalized targets.
    let cs = MapReplacer::new([("(555) 123-4567", "SURFACE_PHONE"), ("jane@example.com", "LOWER_JANE")], true)
        .unwrap()
        .with_target(MapTarget::Surface)
        .with_fallback(StaticReplacer::new("MISS"));
    assert!(cs.is_case_sensitive());
    assert_eq!(cs.target(), MapTarget::Surface);
    let r_cs = cs
        .rewrite(
            &source_view("JANE@example.com jane@example.com (555) 123-4567"),
            &layer(&cursor(), "JANE@example.com jane@example.com (555) 123-4567"),
        )
        .unwrap();
    assert_eq!(r_cs.text(), "MISS LOWER_JANE SURFACE_PHONE");

    let norm_only = cs.with_target(MapTarget::Normalized);
    let r_norm =
        norm_only.rewrite(&source_view("call (555) 123-4567"), &layer(&cursor(), "call (555) 123-4567")).unwrap();
    assert_eq!(r_norm.text(), "call MISS");

    // TSV and custom-delimited parsing + file loading.
    let tsv = "# comment line\n\njane@example.com\tJANE_TSV\nbob@example.com\tBOB_TSV\n";
    let from_tsv = MapReplacer::from_tsv(tsv, true).unwrap();
    assert_eq!(from_tsv.get("jane@example.com"), Some("JANE_TSV"));
    assert_eq!(from_tsv.get("bob@example.com"), Some("BOB_TSV"));

    let tmp_path = std::env::temp_dir().join(format!("turbo_pii_map_test_{}.tsv", std::process::id()));
    std::fs::write(&tmp_path, tsv).unwrap();
    let loaded = MapReplacer::from_tsv_file(&tmp_path, true).unwrap();
    let _ = std::fs::remove_file(&tmp_path);
    assert_eq!(loaded, from_tsv);

    assert_eq!(MapReplacer::new([("  ", "VAL")], true), Err(MapError::EmptyKey));
    assert_eq!(MapReplacer::new([("KEY", "  ")], true), Err(MapError::EmptyReplacement));
    // new keeps the later of two keys equal once trimmed and folded;
    // new_unique refuses them, unless case sensitivity keeps them apart.
    assert!(MapReplacer::new([(" k ", "B"), ("K", "A")], false).is_ok());
    assert_eq!(MapReplacer::new_unique([(" k ", "B"), ("K", "A")], false), Err(MapError::DuplicateKey));
    assert_eq!(MapReplacer::new_unique([(" k ", "B"), ("k", "A")], true), Err(MapError::DuplicateKey));
    assert!(MapReplacer::new_unique([("k", "B"), ("K", "A")], true).is_ok());
    assert_eq!(
        MapReplacer::from_tsv("ok\tval\nbad_line_without_tab\n", true),
        Err(MapError::MissingDelimiter { line: 2 })
    );
    assert!(matches!(MapReplacer::from_tsv_file("/nonexistent/turbo_pii_missing.tsv", true), Err(MapError::Io(_))));
}

#[test]
fn rewrite_policy_composes_per_kind_strategies_and_maps_multibyte_offsets() {
    let map = MapReplacer::new([("vip@example.com", "VIP_CONTACT")], false)
        .unwrap()
        .with_fallback(FormatRedactor::new("[EMAIL-%v]").unwrap());
    let policy = RewritePolicy::new(StaticReplacer::new("[REDACTED]"))
        .with_custom_kinds(test_kinds())
        .with(PiiKind::Email, map)
        .with(PiiKind::Phone, Truncate::last_n(4).keeping_format())
        .with(PiiKind::Card, Sha256Tokenizer::with_length(8).unwrap().with_kind_prefix())
        .with(PiiKind::UsSsn, FormatRedactor::new("{{{REDACTED-%t}}}").unwrap())
        .with(PERSON_KIND, Abbreviator::dotted().preserving_hyphens());

    let text = "Für Jean-Luc Picard (😀): mail vip@example.com or jane@example.com, \
                call (555) 123-4567, SSN 123-45-6789, card 4111 1111 1111 1111, \
                and reply to jane@example.com.";
    let mut pii = layer(&wide(), text);
    // Add a custom person-name mention at "Jean-Luc Picard" (bytes 5..20) and re-sort via manual layer.
    let mut rows: Vec<(u32, u32, PiiKind, String)> =
        pii.mentions().map(|m| (m.span.start, m.span.end, m.kind, m.normalized.to_owned())).collect();
    rows.push((5, 20, PERSON_KIND, "Jean-Luc Picard".to_owned()));
    rows.sort_by_key(|r| turbo_analysis_layers::mention_order(r.0, r.1, r.2));
    let borrowed: Vec<(u32, u32, PiiKind, &str)> = rows.iter().map(|r| (r.0, r.1, r.2, r.3.as_str())).collect();
    pii = manual(&borrowed);

    let r = policy.rewrite(&source_view(text), &pii).unwrap();
    let expected_card =
        Sha256Tokenizer::with_length(8).unwrap().with_kind_prefix().digest("card", "4111111111111111").unwrap();
    assert_eq!(
        r.text(),
        format!(
            "Für J.-L.P. (😀): mail VIP_CONTACT or [EMAIL-1], \
             call (***) ***-4567, SSN {{{{{{REDACTED-US-SSN}}}}}}, card {expected_card}, \
             and reply to [EMAIL-1]."
        )
    );
    let sources: Vec<Span> = pii.mentions().map(|m| m.span).collect();
    assert_offset_and_span_maps(text, &sources, &r);

    // Overlap coalescing vs refusing_overlaps across all new strategies.
    let overlap_text = "https://john:pass@example.com/path";
    let overlap_pii = layer(&wide(), overlap_text);
    let expected_err = Err(RewriteError::Overlap(Span::new(8, 17), Span::new(13, 29)));

    assert_eq!(
        FormatRedactor::new("{{{REDACTED-%t}}}")
            .unwrap()
            .rewrite(&source_view(overlap_text), &overlap_pii)
            .unwrap()
            .text(),
        "https://{{{REDACTED-URL-CREDENTIAL}}}/path"
    );
    assert_eq!(
        FormatRedactor::new("{{{REDACTED-%t}}}")
            .unwrap()
            .refusing_overlaps()
            .rewrite(&source_view(overlap_text), &overlap_pii)
            .map(|r| r.into_text()),
        expected_err.clone()
    );
    assert_eq!(
        StaticReplacer::default()
            .refusing_overlaps()
            .rewrite(&source_view(overlap_text), &overlap_pii)
            .map(|r| r.into_text()),
        expected_err.clone()
    );
    assert_eq!(
        Sha256Tokenizer::new()
            .refusing_overlaps()
            .rewrite(&source_view(overlap_text), &overlap_pii)
            .map(|r| r.into_text()),
        expected_err.clone()
    );
    assert_eq!(
        MapReplacer::new([("a", "b")], false)
            .unwrap()
            .refusing_overlaps()
            .rewrite(&source_view(overlap_text), &overlap_pii)
            .map(|r| r.into_text()),
        expected_err.clone()
    );
    assert_eq!(
        Truncate::keep_end(2)
            .refusing_overlaps()
            .rewrite(&source_view(overlap_text), &overlap_pii)
            .map(|r| r.into_text()),
        expected_err.clone()
    );
    assert_eq!(
        Abbreviator::dotted()
            .refusing_overlaps()
            .rewrite(&source_view(overlap_text), &overlap_pii)
            .map(|r| r.into_text()),
        expected_err.clone()
    );
    assert_eq!(
        policy.clone().refusing_overlaps().rewrite(&source_view(overlap_text), &overlap_pii).map(|r| r.into_text()),
        expected_err.clone()
    );
    // Inner strategy refusing_overlaps is honored by RewritePolicy and MapReplacer fallback.
    assert_eq!(
        RewritePolicy::new(Truncate::keep_start(5).refusing_overlaps())
            .rewrite(&source_view(overlap_text), &overlap_pii)
            .map(|r| r.into_text()),
        expected_err.clone()
    );
    let policy_override_refusing =
        RewritePolicy::new(StaticReplacer::default()).with(PiiKind::Email, Truncate::keep_start(5).refusing_overlaps());
    assert!(policy_override_refusing.refuses_overlaps());
    assert_eq!(
        policy_override_refusing.rewrite(&source_view(overlap_text), &overlap_pii).map(|r| r.into_text()),
        expected_err.clone()
    );
    let map_refusing_fallback = MapReplacer::new([("unmatched", "X")], false)
        .unwrap()
        .with_fallback(Truncate::keep_start(5).refusing_overlaps());
    assert!(map_refusing_fallback.refuses_overlaps());
    assert_eq!(
        map_refusing_fallback.rewrite(&source_view(overlap_text), &overlap_pii).map(|r| r.into_text()),
        expected_err.clone()
    );
    assert_eq!(
        RewritePolicy::new(map_refusing_fallback)
            .rewrite(&source_view(overlap_text), &overlap_pii)
            .map(|r| r.into_text()),
        expected_err
    );

    // Partial rewrite on a higher-ranked mention never exposes a stricter overlapping mention.
    let nested_text = "Contact alice@example.com today.";
    let nested_pii = manual(&[(8, 25, PiiKind::Email, "alice@example.com"), (8, 13, PERSON_KIND, "alice")]);
    for email_strat in [
        RewriteStrategy::from(Truncate::keep_start(5)),
        RewriteStrategy::from(MaskPolicy::first_n(5, '*').unwrap()),
        RewriteStrategy::from(Abbreviator::dotted()),
        RewriteStrategy::from(
            MapReplacer::new([("other@example.com", "OTHER")], false).unwrap().with_fallback(Truncate::keep_start(5)),
        ),
    ] {
        let mixed = RewritePolicy::new(StaticReplacer::new("[REDACTED]")).with(PiiKind::Email, email_strat.clone());
        let out = mixed.rewrite(&source_view(nested_text), &nested_pii).unwrap();
        assert!(!out.text().contains("alice") && !out.text().contains("A."), "{email_strat:?} -> {}", out.text());
        assert_offset_and_span_maps(nested_text, &[Span::new(8, 25)], &out);
    }

    // Long overlapping cluster preserves winner tail and masks covered prefix.
    let n = 4_000;
    let s = "a".repeat(n);
    let pii = manual(&[(0, n as u32, PiiKind::Email, "e"), (0, (n / 2) as u32, PERSON_KIND, "p")]);
    let policy = RewritePolicy::new(MaskPolicy::new('*').unwrap()).with(PiiKind::Email, Truncate::keep_end(4));
    let out = policy.rewrite(&source_view(&s), &pii).unwrap();
    assert!(out.text().ends_with("aaaa") && out.text()[..n - 4].chars().all(|c| c == '*'));

    // EmptyValue is propagated by RewritePolicy for HmacTokenizer and Sha256Tokenizer.
    let empty_pii = manual(&[(0, 4, PiiKind::Email, "")]);
    assert_eq!(
        RewritePolicy::new(HmacTokenizer::new(KEY).unwrap())
            .rewrite(&source_view("abcd"), &empty_pii)
            .map(|r| r.into_text()),
        Err(RewriteError::EmptyValue(Span::new(0, 4)))
    );
    assert_eq!(
        RewritePolicy::new(Sha256Tokenizer::new()).rewrite(&source_view("abcd"), &empty_pii).map(|r| r.into_text()),
        Err(RewriteError::EmptyValue(Span::new(0, 4)))
    );

    // Exhaustive multi-byte plan test across all strategies.
    let strategies: Vec<RewriteStrategy> = vec![
        FormatRedactor::new("{{{REDACTED-%t-%v}}}").unwrap().into(),
        StaticReplacer::new("[REDACTED]").into(),
        Sha256Tokenizer::with_length(16).unwrap().into(),
        MapReplacer::new([("value0", "MAPPED_ZERO")], false).unwrap().into(),
        Truncate::keep_start(2).into(),
        Truncate::keep_end(2).into(),
        Abbreviator::dotted().into(),
        HmacTokenizer::new(KEY).unwrap().into(),
        Pseudonymizer::new("[", "]").into(),
    ];
    for sample in ["", "Zoë😀中𝕒!", "é-ß-東京-🎉"] {
        let bounds: Vec<u32> = (0..=sample.len()).filter(|&i| sample.is_char_boundary(i)).map(|i| i as u32).collect();
        for (i, &s) in bounds.iter().enumerate() {
            for &e in &bounds[i + 1..] {
                let span = Span::new(s, e);
                let val = format!("value{s}");
                let pii: Pii = manual(&[(s, e, PiiKind::Email, &val)]);
                for strat in &strategies {
                    let rw = strat.rewrite(&source_view(sample), &pii).unwrap();
                    assert_offset_and_span_maps(sample, &[span], &rw);
                }
            }
        }
    }
}
