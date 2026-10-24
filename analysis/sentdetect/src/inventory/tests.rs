use super::english::{ENGLISH, PENULTIMATES};
use super::*;

use AbbreviationClass::*;
use Verdict::{Allow, Neutral, Suppress};

/// The verdict on the full stop that ends the first occurrence of
/// `stop` in `text`.
fn at(inv: &AbbreviationInventory, text: &str, stop: &str) -> Verdict {
    let i = text.find(stop).unwrap_or_else(|| panic!("{stop:?} in {text:?}")) + stop.len() - 1;
    assert_eq!(&text[i..=i], ".");
    inv.judge(text, i)
}

#[test]
fn tables_are_sorted_unique_and_well_formed() {
    for (c, table) in AbbreviationClass::ALL.iter().zip(ENGLISH.classes) {
        assert!(table.windows(2).all(|w| w[0].text < w[1].text), "{c:?} is not sorted and unique");
        for a in table {
            assert!(a.text.ends_with('.'), "{a:?}");
            assert!(a.text.len() <= judge::MAX_TOKEN, "{a:?}");
            assert!(a.text.split(' ').all(|t| !t.is_empty() && !t.chars().any(char::is_whitespace)), "{a:?}");
            assert!(!a.text.contains('\u{2014}'));
        }
    }
    assert!(PENULTIMATES.windows(2).all(|w| w[0] < w[1]));
    assert!(PENULTIMATES.iter().all(|w| w.to_ascii_lowercase() == *w && !w.contains('.')));
    // No text is in two classes as written.
    let mut all: Vec<&str> = ENGLISH.classes.iter().flat_map(|t| t.iter().map(|a| a.text)).collect();
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), n);
}

#[test]
fn the_issue_lists_are_bundled_in_their_classes() {
    let has = |class: AbbreviationClass, text: &str| ENGLISH.classes[class as usize].iter().any(|a| a.text == text);
    for t in ["etc.", "vs.", "e.g.", "i.e.", "fig."] {
        assert!(has(General, t), "{t}");
    }
    for t in ["Mr.", "Mrs.", "Dr.", "Prof.", "Sen.", "Rep.", "Gen."] {
        assert!(has(Titles, t), "{t}");
    }
    for t in ["et al.", "op. cit.", "ib.", "cf.", "Ph.D.", "vol.", "no.", "pp."] {
        assert!(has(Academic, t), "{t}");
    }
    for t in ["Inc.", "Corp.", "Ltd.", "Co.", "v.", "Sec.", "Art.", "No.", "U.S.C.", "Fed. Reg."] {
        assert!(has(LegalTechnical, t), "{t}");
    }
}

#[test]
fn classes_compose() {
    let set = General | Titles;
    assert!(set.contains(General) && set.contains(Titles) && !set.contains(Academic));
    assert_eq!(set | Academic | LegalTechnical, AbbreviationClasses::ALL);
    assert_eq!(AbbreviationPreset::Minimal.classes(), General.into());
    assert_eq!(AbbreviationPreset::Standard.classes(), General | Titles);
    assert_eq!(AbbreviationPreset::Scholarly.classes(), General | Titles | Academic);
    assert_eq!(AbbreviationPreset::Full.classes(), AbbreviationClasses::ALL);
    assert!(AbbreviationClasses::NONE.is_empty());
    // Only the active classes answer.
    let titles = AbbreviationInventory::with_classes(Titles);
    assert_eq!(at(&titles, "We met Dr. Smith.", "Dr."), Suppress);
    assert_eq!(at(&titles, "Bring pens, etc. and paper.", "etc."), Neutral);
    assert_eq!(titles.bundled().count(), ENGLISH.classes[Titles as usize].len());
    assert_eq!(AbbreviationInventory::preset(AbbreviationPreset::Full), AbbreviationInventory::english());
}

#[test]
fn empty_judges_nothing() {
    let e = AbbreviationInventory::EMPTY;
    assert!(e.is_empty() && e.is_unset());
    assert_eq!(at(&e, "We met Dr. Smith.", "Dr."), Neutral);
    assert_eq!(AbbreviationInventory::with_classes(AbbreviationClasses::NONE).judge("Dr. X", 2), Neutral);
    assert_eq!(AbbreviationInventories::for_language("und"), AbbreviationInventory::EMPTY);
    assert_eq!(AbbreviationInventories::for_language("en-US"), AbbreviationInventory::english());
    assert_eq!(AbbreviationInventory::default(), AbbreviationInventory::EMPTY);
}

#[test]
fn titles_before_a_name_or_an_initial() {
    let en = AbbreviationInventory::english();
    assert_eq!(at(&en, "Dr. Smith went home.", "Dr."), Suppress);
    assert_eq!(at(&en, "Ask Prof. J. Doe.", "Prof."), Suppress);
    assert_eq!(at(&en, "Ask Sen. \"Kay\" Hagan.", "Sen."), Suppress);
    assert_eq!(at(&en, "(Mrs. Brown) left.", "Mrs."), Suppress);
    assert_eq!(at(&en, "Then MR. Smith left.", "MR."), Neutral);
    // Nothing follows: no opinion.
    assert_eq!(at(&en, "He saw the Gen.", "Gen."), Neutral);
    assert_eq!(at(&en, "He saw the Gen. ", "Gen."), Neutral);
    // `Jr.` follows a name and can end a sentence.
    assert_eq!(at(&en, "It was Smith Jr. He left.", "Jr."), Neutral);
    assert_eq!(at(&en, "Smith Jr. and his son.", "Jr."), Suppress);
}

#[test]
fn numbered_references_before_a_number() {
    let en = AbbreviationInventory::english();
    for (text, stop) in [
        ("See fig. 3 for details.", "fig."),
        ("See Fig. 3 for details.", "Fig."),
        ("See Vol. 3, No. 12 and pp. 14-18.", "Vol."),
        ("See Vol. 3, No. 12 and pp. 14-18.", "No."),
        ("See Vol. 3, No. 12 and pp. 14-18.", "pp."),
        ("Under Sec. 230 it is.", "Sec."),
        ("Under Art. IV of it.", "Art."),
        ("In vol. xii of it.", "vol."),
        ("In Sec. (4) of it.", "Sec."),
        ("Item No. #4 is gone.", "No."),
        ("At 85 Fed. Reg. 1234 it says.", "Reg."),
        ("At 85 Fed. Reg. 1234 it says.", "Fed."),
        ("Under Pub. L. 111-148 it is.", "L."),
    ] {
        assert_eq!(at(&en, text, stop), Suppress, "{stop} in {text:?}");
    }
    // A word after it: no opinion, and the true ends stay.
    assert_eq!(at(&en, "He wrote vol. Then he slept.", "vol."), Neutral);
    assert_eq!(at(&en, "See Art. Museums open late.", "Art."), Neutral);
    // `No.` before a word is the word no.
    assert_eq!(at(&en, "He said No. Then he left.", "No."), Allow);
    // A lone `I` is the pronoun, a lone lower case letter is a word (and
    // `no` before a word is the word no).
    assert_eq!(at(&en, "Read Vol. I think so.", "Vol."), Neutral);
    assert_eq!(at(&en, "Read vol. v of it.", "vol."), Neutral);
    assert_eq!(at(&en, "Read no. v of it.", "no."), Allow);
    // Letters that are no canonical numeral.
    assert_eq!(at(&en, "Read Sec. IIII now.", "Sec."), Neutral);
    assert_eq!(at(&en, "Read Sec. MIX now.", "Sec."), Suppress);
    assert_eq!(at(&en, "Read Sec. Did it.", "Sec."), Neutral);
}

#[test]
fn trailing_abbreviations_keep_true_ends() {
    let en = AbbreviationInventory::english();
    assert_eq!(at(&en, "Apples, pears, etc. and more.", "etc."), Suppress);
    assert_eq!(at(&en, "Apples, pears, etc. The end.", "etc."), Neutral);
    assert_eq!(at(&en, "He works at Acme Inc. The firm grew.", "Inc."), Neutral);
    assert_eq!(at(&en, "Acme Inc. was founded in 1990.", "Inc."), Suppress);
    assert_eq!(at(&en, "He arrived at 5 p.m. on Tuesday.", "p.m."), Suppress);
    assert_eq!(at(&en, "He arrived at 5 p.m. Then he ate.", "p.m."), Neutral);
    assert_eq!(at(&en, "Under 42 U.S.C. 1983 it is.", "U.S.C."), Suppress);
    // Punctuation right after the stop.
    assert_eq!(at(&en, "Apples, etc., and more.", "etc."), Suppress);
    assert_eq!(at(&en, "Acme Inc.; Beta Corp.", "Inc."), Suppress);
    // The end of the text.
    assert_eq!(at(&en, "Apples, pears, etc.", "etc."), Neutral);
}

#[test]
fn stops_inside_an_entry() {
    let en = AbbreviationInventory::english();
    assert_eq!(at(&en, "Some fruit, e.g. Apples.", "e."), Suppress);
    assert_eq!(at(&en, "Under 42 U.S.C. 1983 it is.", "U."), Suppress);
    assert_eq!(at(&en, "Under 42 U.S.C. 1983 it is.", "U.S."), Suppress);
    assert_eq!(at(&en, "As op. cit. shows.", "op."), Suppress);
    assert_eq!(at(&en, "Ask Ph.D. students.", "Ph."), Suppress);
    // Not an entry.
    assert_eq!(at(&en, "Version 1.2. Then.", "1."), Neutral);
    assert_eq!(at(&en, "Ask Mr.Smith now.", "Mr."), Neutral);
}

#[test]
fn multi_token_entries() {
    let en = AbbreviationInventory::english();
    assert_eq!(at(&en, "Smith et al. found it.", "al."), Suppress);
    assert_eq!(at(&en, "Smith (et al. 2020) found it.", "al."), Suppress);
    assert_eq!(at(&en, "It was found by Smith et al. The end.", "al."), Neutral);
    assert_eq!(at(&en, "Et al. found it.", "al."), Suppress);
    // `al.` without `et`, `op.` without `cit.`.
    assert_eq!(at(&en, "He met Al. Then left.", "Al."), Neutral);
    assert_eq!(at(&en, "As op. shows.", "op."), Neutral);
    // Excluding the entry removes it.
    let no_et_al = AbbreviationInventory::english().with_exclusion("et al.");
    assert_eq!(at(&no_et_al, "Smith et al. found it.", "al."), Neutral);
}

#[test]
fn never_abbreviations_allow_the_end() {
    let en = AbbreviationInventory::english();
    assert_eq!(at(&en, "Do it. Now.", "it."), Allow);
    assert_eq!(at(&en, "Yes. We go.", "Yes."), Allow);
    assert_eq!(at(&en, "A pen. More.", "pen."), Neutral);
    let without = AbbreviationInventory::english().with_exclusion("it");
    assert_eq!(at(&without, "Do it. Now.", "it."), Neutral);
    // Additions alone have no never-abbreviation words.
    let added = AbbreviationInventory::EMPTY.with_addition("Xyz.", ContextRule::Prepositive).unwrap();
    assert_eq!(at(&added, "Do it. Now.", "it."), Neutral);
}

#[test]
fn additions_and_exclusions() {
    let inv = AbbreviationInventory::EMPTY.with_addition("approx.", ContextRule::Numeric).unwrap();
    assert!(!inv.is_empty());
    assert_eq!(at(&inv, "It is approx. 5 m.", "approx."), Suppress);
    assert_eq!(at(&inv, "Dr. Who.", "Dr."), Neutral);
    // An addition overrides a bundled rule.
    let en = AbbreviationInventory::english();
    assert_eq!(at(&en, "Acme Inc. The firm.", "Inc."), Neutral);
    let inc = en.clone().with_addition("Inc.", ContextRule::Prepositive).unwrap();
    assert_eq!(at(&inc, "Acme Inc. The firm.", "Inc."), Suppress);
    // A multi-token addition.
    let multi = en.clone().with_addition("Sup. Ct.", ContextRule::Trailing).unwrap();
    assert_eq!(at(&multi, "The Sup. Ct. held it.", "Sup."), Suppress);
    assert_eq!(at(&multi, "The Sup. Ct. held it.", "Ct."), Suppress);
    // Exclusions match as written, before case folding.
    let no_dr = en.clone().with_exclusion("Dr.");
    assert_eq!(at(&no_dr, "We met Dr. Smith.", "Dr."), Neutral);
    assert_eq!(at(&no_dr, "We met Mr. Smith.", "Mr."), Suppress);
    for bad in ["Dr", "", "a  b.", "a\tb.", " a.", &"x".repeat(70)] {
        assert_eq!(
            AbbreviationInventory::EMPTY.with_addition(bad, ContextRule::Trailing),
            Err(AbbreviationError(bad.into()))
        );
    }
    assert!(AbbreviationError("x".into()).to_string().contains("full stop"));
}

#[test]
fn only_full_stops_are_judged() {
    let en = AbbreviationInventory::english();
    assert_eq!(en.judge("Dr! Smith", 2), Neutral);
    assert_eq!(en.judge("Dr. Smith", 99), Neutral);
    // A token longer than any entry, and a text without whitespace.
    let long = format!("{}Dr. Smith", "x".repeat(100));
    assert_eq!(en.judge(&long, long.find('.').unwrap()), Neutral);
    let run = "Dr.".repeat(5000);
    assert_eq!(en.judge(&run, run.len() - 1), Neutral);
}

#[test]
fn the_hash_follows_every_setting() {
    let en = AbbreviationInventory::english();
    let hashes = [
        AbbreviationInventory::EMPTY.config_hash(),
        en.config_hash(),
        AbbreviationInventory::with_classes(General | Titles).config_hash(),
        AbbreviationInventory::with_classes(Titles).config_hash(),
        en.clone().with_addition("Xyz.", ContextRule::Trailing).unwrap().config_hash(),
        en.clone().with_addition("Xyz.", ContextRule::Numeric).unwrap().config_hash(),
        en.clone().with_exclusion("Dr.").config_hash(),
        en.clone().with_exclusion("it").config_hash(),
        AbbreviationInventory::EMPTY.with_exclusion("Dr.").config_hash(),
    ];
    for (i, a) in hashes.iter().enumerate() {
        for b in &hashes[i + 1..] {
            assert_ne!(a, b, "{hashes:?}");
        }
    }
    // Stable, and independent of the order of additions.
    assert_eq!(en.config_hash(), AbbreviationInventory::english().config_hash());
    let ab = en.clone().with_addition("A.", ContextRule::Trailing).unwrap().with_addition("B.", ContextRule::Trailing);
    let ba = en.clone().with_addition("B.", ContextRule::Trailing).unwrap().with_addition("A.", ContextRule::Trailing);
    assert_eq!(ab.unwrap().config_hash(), ba.unwrap().config_hash());
    assert!(!AbbreviationInventory::EMPTY.with_exclusion("x").is_unset());
}

/// Many multi-token entries the user added, `Xq<i>. Yq<i>.`.
fn with_many(inv: AbbreviationInventory, n: usize) -> AbbreviationInventory {
    (0..n).fold(inv, |inv, i| inv.with_addition(&format!("Xq{i}. Yq{i}."), ContextRule::Trailing).unwrap())
}

#[test]
fn ordinary_sentence_ends_scan_no_entries() {
    use super::index::PROBES;
    let text = "The cat sat on the mat. It was warm. We left home at noon. Then it rained. Fine.";
    let probes = |inv: &AbbreviationInventory, text: &str| {
        PROBES.with(|c| c.set((0, 0)));
        for (p, _) in text.match_indices('.') {
            inv.judge(text, p);
        }
        PROBES.with(|c| c.get())
    };
    let big = with_many(AbbreviationInventory::english(), 1000);
    for inv in [AbbreviationInventory::english(), big.clone()] {
        assert_eq!(probes(&inv, text).0, 0, "multi-token entries compared at ordinary ends");
    }
    // Only the entries with the token as a part are compared: `op. cit.`
    // at `op.`; `loc. cit.`, which fails, then `op. cit.` at `cit.`.
    assert_eq!(probes(&big, "See op. cit. for more."), (3, 2));
    assert_eq!(at(&big, "See op. cit. for more.", "cit."), Suppress);
    assert_eq!(at(&big, "Xq7. Yq7. and more.", "Yq7."), Suppress);
    assert_eq!(at(&big, "Xq7. Yq7. and more.", "Xq7."), Suppress);
    assert_eq!(at(&big, "Xq7. Yq8. and more.", "Yq8."), Neutral);
}

#[test]
fn the_index_follows_additions_and_exclusions() {
    let en = AbbreviationInventory::english();
    // An exclusion wins over an addition, made before or after it.
    let ex = en.clone().with_exclusion("Dr.").with_addition("Dr.", ContextRule::Prepositive).unwrap();
    assert_eq!(at(&ex, "Dr. Smith", "Dr."), Neutral);
    let ex = en.clone().with_addition("Dr.", ContextRule::Prepositive).unwrap().with_exclusion("Dr.");
    assert_eq!(at(&ex, "Dr. Smith", "Dr."), Neutral);
    let ex = en.clone().with_exclusion("op. cit.");
    assert_eq!(at(&ex, "See op. cit. for more.", "cit."), Neutral);
    assert_eq!(at(&ex, "See loc. cit. for more.", "cit."), Suppress);
    // An addition takes the place of a bundled entry of the same text.
    let etc = en.clone().with_addition("etc.", ContextRule::Prepositive).unwrap();
    assert_eq!(at(&etc, "Pens, ink etc. Then", "etc."), Suppress);
    assert_eq!(at(&en, "Pens, ink etc. Then", "etc."), Neutral);
    // Equality is on the configuration, not on how the index was built.
    assert_eq!(en.clone().with_exclusion("x"), AbbreviationInventory::english().with_exclusion("x"));
    assert_ne!(ex, en);
}

#[test]
fn long_lower_case_words_need_no_search() {
    use super::index::PROBES;
    let en = AbbreviationInventory::english();
    let text = "It was extraordinary. Then it rained.";
    PROBES.with(|c| c.set((0, 0)));
    assert_eq!(at(&en, text, "extraordinary."), Neutral);
    assert_eq!(PROBES.with(|c| c.get()), (0, 0), "a long lower case word was searched");
    // Long entries ending in a lower case letter, one-token, folded, or a
    // part of a multi-token entry, are still found.
    let long = en
        .clone()
        .with_addition("approximately.", ContextRule::Trailing)
        .unwrap()
        .with_addition("notwithstanding. ibid.", ContextRule::Trailing)
        .unwrap()
        .with_addition("XYZCORP.", ContextRule::Trailing)
        .unwrap();
    assert_eq!(at(&long, "It is approximately. three", "approximately."), Suppress);
    assert_eq!(at(&long, "Approximately. three", "Approximately."), Suppress);
    assert_eq!(at(&long, "See notwithstanding. ibid. here", "notwithstanding."), Suppress);
    assert_eq!(at(&long, "Shares of XYZCORP. fell", "XYZCORP."), Suppress);
    assert_eq!(at(&long, "It was extraordinary. then", "extraordinary."), Neutral);
    // Never-abbreviation words in any case, and an excluded one.
    assert_eq!(at(&en, "The answer is no. Then", "no."), Allow);
    assert_eq!(at(&en, "The answer is NO. Then", "NO."), Allow);
    assert_eq!(at(&en.clone().with_exclusion("no"), "The answer is no. Then", "no."), Neutral);
}

#[test]
fn one_pass_keys_agree_with_the_filters() {
    use super::index::{entry_key, keys, word_key};
    for t in ["etc.", "Etc.", "extraordinary.", "No.", "e.g.", "\u{e9}t\u{e9}.", "a.", "."] {
        assert_eq!(keys(t), (entry_key(t), word_key(&t[..t.len() - 1])), "{t:?}");
    }
}
