use std::collections::BTreeMap;
use turbo_analysis::Span;
use turbo_goldens::sentences::sentence_suite;
use turbo_sentdetect::{AbbreviationInventories, RuleSplitter};
use turbo_unicode::props::white_space;

fn normalized(s: &str) -> String {
    s.split(white_space).filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")
}

#[test]
fn language_inventory_conformance() {
    for suite in ["pragmatic-segmenter", "pysbd"] {
        let mut counts: BTreeMap<String, [usize; 4]> = BTreeMap::new();
        for case in sentence_suite(suite) {
            let want: Vec<_> = case.sentences.iter().map(|s| normalized(s)).filter(|s| !s.is_empty()).collect();
            let result = |splitter: RuleSplitter| {
                let mut spans: Vec<Span> = Vec::new();
                splitter.split(&case.text, &mut spans);
                spans.iter().map(|s| normalized(&case.text[s.start as usize..s.end as usize])).collect::<Vec<_>>()
            };
            let old = result(RuleSplitter::new(&case.lang).without_inventory());
            let new = result(RuleSplitter::new(&case.lang));
            let c = counts.entry(case.lang.clone()).or_default();
            c[0] += 1;
            c[1] += usize::from(old == want);
            c[2] += usize::from(new == want);
            if old == want && new != want {
                c[3] += 1;
                eprintln!("REGRESSION {}: {:?} -> {:?}, text {:?}", case.id, old, new, case.text);
            }
        }
        for (lang, [n, old, new, regressions]) in counts {
            eprintln!("{suite}\t{lang}\t{n}\t{old}\t{new}\t{regressions}");
            assert_eq!(regressions, 0, "{suite}/{lang}");
        }
    }
}

#[test]
fn bundled_languages_are_sorted_valid_and_have_distinct_keys() {
    let mut hashes = Vec::new();
    for lang in ["ar", "bg", "da", "de", "en", "es", "fr", "it", "kk", "nl", "pl", "pt", "ru", "sk"] {
        let inv = AbbreviationInventories::for_language(lang);
        assert_eq!(inv.language(), Some(lang));
        assert!(!inv.is_empty());
        assert_eq!(inv, AbbreviationInventories::for_language(&format!("{}-XX", lang.to_ascii_uppercase())));
        hashes.push(inv.config_hash());
        let mut last = None;
        let mut words = Vec::new();
        for (class, entry) in inv.bundled() {
            if let Some((c, text)) = last {
                assert!(c != class || text < entry.text);
            }
            last = Some((class, entry.text));
            assert!(entry.text.ends_with('.'));
            assert!(entry.text.len() <= 64);
            assert!(entry.text.split(' ').all(|s| !s.is_empty() && !s.chars().any(white_space)));
            words.push(entry.text);
        }
        let n = words.len();
        words.sort_unstable();
        words.dedup();
        assert_eq!(words.len(), n);
    }
    hashes.sort_unstable();
    assert!(hashes.windows(2).all(|w| w[0] != w[1]));
}

#[test]
fn language_classes_and_context_remain_configurable() {
    use turbo_sentdetect::{AbbreviationClass, AbbreviationInventory, ContextRule, Verdict};
    for (lang, text, abbreviation) in [
        ("de", "Frau Prof. Schneider kommt.", "Prof."),
        ("fr", "M. Dupont arrive.", "M."),
        ("es", "La Sra. García llegó.", "Sra."),
        ("it", "Il Dott. Rossi arriva.", "Dott."),
        ("pt", "A Dra. Silva chegou.", "Dra."),
        ("nl", "Dhr. Jansen komt.", "Dhr."),
        ("sk", "Ing. Novák prišiel.", "Ing."),
    ] {
        let inventory = AbbreviationInventories::for_language_with_classes(lang, AbbreviationClass::Titles);
        let at = text.find(abbreviation).unwrap() + abbreviation.len() - 1;
        assert_eq!(inventory.judge(text, at), Verdict::Suppress, "{lang}");
        assert_ne!(inventory.config_hash(), AbbreviationInventories::for_language(lang).config_hash());
    }
    let nl = AbbreviationInventories::for_language("nl");
    assert_eq!(nl.judge("Dhr. Jansen komt.", 3), Verdict::Suppress);
    let excluded = nl.clone().with_exclusion("dhr.");
    assert_eq!(excluded.judge("Dhr. Jansen komt.", 3), Verdict::Neutral);
    assert_ne!(nl.config_hash(), excluded.config_hash());
    assert_eq!(AbbreviationInventory::EMPTY.judge("Dhr. Jansen komt.", 3), Verdict::Neutral);
    let fr = AbbreviationInventories::for_language("fr");
    assert_eq!(fr.judge("etc. Paris", 3), Verdict::Neutral);
    assert_eq!(fr.judge("etc. puis", 3), Verdict::Suppress);
    assert_ne!(fr.config_hash(), fr.clone().with_addition("xyz.", ContextRule::Numeric).unwrap().config_hash());
}
