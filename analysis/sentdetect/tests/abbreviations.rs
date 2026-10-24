//! The classified abbreviation inventory in the maxent and rule
//! detectors, where it is on by default: false sentence ends it removes,
//! true ends it keeps, and the JVM parity mode without it.

use std::path::Path;
use std::sync::Arc;

use turbo_analysis::{Annotator, Document, View};
use turbo_maxent::ModelFile;
use turbo_sentdetect::{
    AbbreviationClass, AbbreviationInventories, AbbreviationInventory, ContextRule, Decision, RuleSentenceDetector,
    RuleSplitter, Rules, Scratch, SentenceDetector, SentenceModel, WhitespaceMode,
};

const MODELS: [&str; 2] = ["en-plain.bin", "en-abbrev.bin"];

fn model(name: &str) -> Arc<SentenceModel> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures").join(name);
    Arc::new(SentenceModel::open(ModelFile::read(path).unwrap()).unwrap())
}

fn me(d: &SentenceDetector, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    d.detect(text, &mut out);
    out.iter().map(|(s, _)| text[s.start as usize..s.end as usize].to_owned()).collect()
}

fn rules(s: &RuleSplitter, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    s.split(text, &mut out);
    out.iter().map(|s| text[s.start as usize..s.end as usize].to_owned()).collect()
}

fn en_rules() -> RuleSplitter {
    RuleSplitter::new("en")
}

/// The detector of a fixture model, by default and in JVM parity mode.
fn detectors(name: &str) -> (SentenceDetector, SentenceDetector) {
    let d = SentenceDetector::new(model(name), WhitespaceMode::Unicode);
    (d.clone(), d.with_jvm_parity())
}

/// FNV-1a over `bytes` from the offset basis.
fn fnv(bytes: &[u8]) -> u64 {
    turbo_analysis::fnv1a(bytes)
}

fn fixture_text() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sentences.txt")).unwrap()
}

#[test]
fn the_issue_examples_segment_out_of_the_box() {
    let a = "Dr. Smith went to Washington. He arrived at 5 p.m. on Tuesday.";
    let b = "See Vol. 3, No. 12 and pp. 14-18 for details.";
    let want_a = ["Dr. Smith went to Washington.", "He arrived at 5 p.m. on Tuesday."];
    assert_eq!(rules(&en_rules(), a), want_a);
    assert_eq!(rules(&en_rules(), b), [b]);
    // Without the inventory the rule splitter breaks both.
    assert_eq!(rules(&RuleSplitter::new("en").without_inventory(), a).len(), 3);
    assert_eq!(rules(&RuleSplitter::new("en").without_inventory(), b).len(), 2);
    for name in MODELS {
        let (d, _) = detectors(name);
        assert_eq!(d.abbreviations(), &AbbreviationInventory::english());
        assert!(!d.is_jvm_parity());
        assert_eq!(me(&d, a), want_a, "{name}");
        assert_eq!(me(&d, b), [b], "{name}");
    }
}

#[test]
fn false_ends_are_suppressed() {
    let text = "See fig. 3 and Sec. 230 of 42 U.S.C. 1983 today.";
    for name in MODELS {
        let (d, base) = detectors(name);
        assert_eq!(me(&base, text), ["See fig.", "3 and Sec. 230 of 42 U.S.C. 1983 today."], "{name}");
        assert_eq!(me(&d, text), [text], "{name}");
        // The model's probabilities are its own; the split is vetoed.
        let (mut seen, mut out) = (Vec::new(), Vec::new());
        d.detect_traced(text, &mut Scratch::new(), &mut out, |c| seen.push((c.at.byte, c.decision)));
        assert!(seen.contains(&(7, Decision::Vetoed)), "{name}: {seen:?}");
    }
    let r = en_rules();
    for text in [
        "The case Roe v. Wade was decided. It mattered.",
        "Mr. Brown met Mrs. Green and Gen. Lee. They talked.",
        "Under Art. IV the court ruled. It held.",
        "Smith Jr. and his son left. They ran.",
    ] {
        assert_eq!(rules(&r, text).len(), 2, "{text:?}: {:?}", rules(&r, text));
    }
}

#[test]
fn true_ends_are_kept() {
    let texts = [
        "Apples, pears, etc. The end came.",
        "Read op. cit. Then stop.",
        "Smith et al. found it. Then Prof. J. Doe left.",
        "He wrote vol. Then he slept.",
        "It was Smith Jr. He left early.",
        "The deal closed with Acme Inc. It was late.",
    ];
    for name in MODELS {
        let (d, base) = detectors(name);
        for text in texts {
            assert_eq!(me(&d, text), me(&base, text), "{name}: {text:?}");
        }
        if name == "en-plain.bin" {
            // en-abbrev's dictionary holds `etc.`, en-plain has none.
            assert_eq!(me(&d, "Apples, pears, etc. The end came."), ["Apples, pears, etc.", "The end came."]);
        }
    }
    let r = en_rules();
    for text in texts {
        assert_eq!(rules(&r, text).len(), 2, "{text:?}: {:?}", rules(&r, text));
    }
    // A word that is never an abbreviation ends the sentence although the
    // CLDR suppressions list `OK.`.
    let text = "Everything is OK. Smith left.";
    assert_eq!(rules(&RuleSplitter::new("en").without_inventory(), text), [text]);
    assert_eq!(rules(&r, text), ["Everything is OK.", "Smith left."]);
}

#[test]
fn jvm_parity_is_the_detector_without_an_inventory() {
    let text = fixture_text();
    let bits = |v: &[(turbo_analysis::Span, f64)]| v.iter().map(|&(s, p)| (s, p.to_bits())).collect::<Vec<_>>();
    for name in MODELS {
        let (d, parity) = detectors(name);
        assert!(parity.is_jvm_parity());
        let empty = d.clone().with_abbreviations(AbbreviationInventory::EMPTY);
        let german = d.clone().with_abbreviations(AbbreviationInventories::for_language("de"));
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        parity.detect(&text, &mut a);
        empty.detect(&text, &mut b);
        german.detect(&text, &mut c);
        assert_eq!(bits(&a), bits(&b), "{name}");
        assert_eq!(bits(&a), bits(&c), "{name}");
        // The parity hash is the hash before inventories, FNV-1a of the
        // whitespace mode's number (2 for Unicode); the default differs.
        let old = (turbo_analysis::FNV1A_OFFSET_BASIS ^ 2).wrapping_mul(turbo_analysis::FNV1A_PRIME);
        assert_eq!(parity.config_hash(), old);
        assert_ne!(d.config_hash(), old);
        assert_eq!(d.config_hash(), d.clone().with_abbreviations(AbbreviationInventory::english()).config_hash());
    }
    // The rule splitter without its inventory: the hash before inventories,
    // over the language code, a zero, every tailoring on and paragraph
    // line breaks. A language with no inventory keeps that hash.
    let old = |code: &str| {
        let mut bytes = code.as_bytes().to_vec();
        bytes.push(0);
        bytes.extend(0x7f_u64.to_le_bytes());
        bytes.push(2);
        fnv(&bytes)
    };
    let parity = RuleSentenceDetector { splitter: RuleSplitter::new("en").without_inventory() };
    assert_eq!(parity.config_hash(), old("en"));
    assert_ne!(RuleSentenceDetector::new("en").config_hash(), old("en"));
    assert_eq!(RuleSentenceDetector::new("de").config_hash(), old("de"));
    assert_eq!(
        rules(&RuleSplitter::new("en").with_abbreviations(AbbreviationInventory::EMPTY), &text),
        rules(&RuleSplitter::new("en").without_inventory(), &text)
    );
}

#[test]
fn the_inventory_only_vetoes_or_allows() {
    let text = fixture_text();
    for name in MODELS {
        let (d, base) = detectors(name);
        let trace = |d: &SentenceDetector| {
            let (mut seen, mut out) = (Vec::new(), Vec::new());
            d.detect_traced(&text, &mut Scratch::new(), &mut out, |c| {
                seen.push((c.at, c.probs.iter().map(|p| p.to_bits()).collect::<Vec<_>>(), c.decision))
            });
            seen
        };
        let (a, b) = (trace(&base), trace(&d));
        let mut changed = 0;
        for (x, y) in a.iter().zip(&b).take_while(|(x, y)| x.0 == y.0) {
            assert_eq!(x.1, y.1, "{name}: features or probabilities changed at {:?}", x.0);
            if x.2 != y.2 {
                changed += 1;
                assert_ne!(x.2, Decision::NoSplit, "{name}: a no-split changed at {:?}", x.0);
            }
        }
        // The fixture text holds a.m., p.m., Dr. and the like, which the
        // fixture models already handle; the inventory changes little.
        assert!(changed <= 5, "{name}: {changed} decisions changed");
    }
}

#[test]
fn the_rule_splitter_honours_its_switches() {
    let text = "Under Art. IV the court ruled. It held.";
    let r = en_rules();
    assert_eq!(rules(&r, text), ["Under Art. IV the court ruled.", "It held."]);
    // The inventory belongs to the abbreviation tailoring.
    let off = r.clone().with_rules(Rules { abbreviations: false, ..Rules::ALL });
    assert_eq!(rules(&off, text), ["Under Art.", "IV the court ruled.", "It held."]);
    // Classes and additions.
    let titles =
        RuleSplitter::new("en").with_abbreviations(AbbreviationInventory::with_classes(AbbreviationClass::Titles));
    assert_eq!(rules(&titles, text).len(), 3);
    let added = RuleSplitter::new("en")
        .with_abbreviations(AbbreviationInventory::EMPTY.with_addition("Art.", ContextRule::Numeric).unwrap());
    assert_eq!(rules(&added, text).len(), 2);
    assert_eq!(rules(&RuleSplitter::new("en").without_inventory(), text).len(), 3);
    // Languages without a bundled inventory get none.
    for tag in ["de", "pt", "qaa", "und"] {
        assert!(RuleSplitter::new(tag).abbreviations().is_empty(), "{tag}");
    }
    assert!(RuleSplitter::uax29().abbreviations().is_empty());
    assert_eq!(RuleSplitter::new("en-GB").abbreviations(), &AbbreviationInventory::english());
}

#[test]
fn annotators_hash_the_inventory() {
    let plain = RuleSentenceDetector { splitter: RuleSplitter::new("en").without_inventory() };
    let with = RuleSentenceDetector::new("en");
    let fewer = RuleSentenceDetector {
        splitter: RuleSplitter::new("en").with_abbreviations(AbbreviationInventory::with_classes(
            AbbreviationClass::General | AbbreviationClass::Titles,
        )),
    };
    let excluded = RuleSentenceDetector {
        splitter: en_rules().with_abbreviations(AbbreviationInventory::english().with_exclusion("Dr.")),
    };
    let hashes = [plain.config_hash(), with.config_hash(), fewer.config_hash(), excluded.config_hash()];
    for (i, a) in hashes.iter().enumerate() {
        for b in &hashes[i + 1..] {
            assert_ne!(a, b);
        }
    }
    let empty =
        RuleSentenceDetector { splitter: RuleSplitter::new("en").with_abbreviations(AbbreviationInventory::EMPTY) };
    assert_eq!(plain.config_hash(), empty.config_hash());
    assert_eq!(with.config_hash(), RuleSentenceDetector { splitter: en_rules() }.config_hash());

    let doc = Document::new("Dr. Smith met Gen. Lee at noon. They talked.");
    let view = doc.get::<View>().unwrap();
    let (sentences,) = with.annotate((view,)).unwrap();
    assert_eq!(sentences.spans.columns(), (&[0, 32][..], &[31, 44][..]));
}

#[test]
fn multi_token_entries_are_found_by_part_without_a_scan() {
    let en = AbbreviationInventory::english();
    fn parts<'a>(inv: &'a AbbreviationInventory, t: &str) -> Vec<(&'a str, usize)> {
        inv.multi_token_entries(t).collect()
    }
    assert_eq!(parts(&en, "cit."), [("loc. cit.", 1), ("op. cit.", 1)]);
    assert_eq!(parts(&en, "et"), [("et al.", 0), ("et seq.", 0)]);
    assert!(parts(&en, "home.").is_empty());
    let n = 500;
    let big =
        (0..n).fold(en.clone(), |inv, i| inv.with_addition(&format!("Xq{i}. Yq{i}."), ContextRule::Trailing).unwrap());
    assert_eq!(big.multi_token_len(), en.multi_token_len() + n);
    assert!(parts(&big, "home.").is_empty());
    assert_eq!(parts(&big, "Yq42."), [("Xq42. Yq42.", 1)]);
    assert_eq!(parts(&big, "Xq42."), [("Xq42. Yq42.", 0)]);
    let fewer = big.clone().with_exclusion("Xq42. Yq42.");
    assert!(parts(&fewer, "Yq42.").is_empty());
    assert_eq!(fewer.multi_token_len(), big.multi_token_len() - 1);
    // The default detector splits a text without abbreviations as the
    // parity detector does.
    let text = "The cat sat on the mat. It was warm. We left home at noon. Then it rained.";
    assert_eq!(rules(&en_rules(), text), rules(&en_rules().without_inventory(), text));
}
