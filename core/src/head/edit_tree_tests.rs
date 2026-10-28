//! Edit trees: mining round-trips every pair, a tree generalizes to words
//! inflected the same way, labels parse back to their trees, and only
//! canonical labels parse.

use super::edit_tree::EditTree;

const PAIRS: [(&str, &str); 14] = [
    ("gegooid", "gooien"),
    ("running", "run"),
    ("walked", "walk"),
    ("was", "be"),
    ("Mice", "mouse"),
    ("Bäume", "Baum"),
    ("größte", "groß"),
    ("über", ""),
    ("", "x"),
    ("", ""),
    ("it's", "it\\"),
    ("cats", "cats"),
    ("ünïcödé", "ünïcödé's"),
    ("\u{1F600}s", "\u{1F600}"),
];

#[test]
fn every_mined_tree_rewrites_its_form_into_its_lemma_and_its_label_parses_back() {
    for (form, lemma) in PAIRS {
        let t = EditTree::mine(form, lemma);
        assert_eq!(t.apply(form).as_deref(), Some(lemma), "{form:?} -> {lemma:?}: {t:?}");
        let label = t.label();
        assert_eq!(EditTree::parse(&label).unwrap(), t, "{label}");
    }
}

#[test]
fn trees_read_as_spacy_writes_them_and_generalize() {
    // "gooi" is kept; "ge" and "d" around it rewrite by substitution.
    let t = EditTree::mine("gegooid", "gooien");
    assert_eq!(t.label(), "(m 2 1 (s 'ge' '') (s 'd' 'en'))");
    assert_eq!(t.apply("geloopd").as_deref(), Some("loopen"));
    let walked = EditTree::mine("walked", "walk");
    assert_eq!(walked.label(), "(m 0 2 () (s 'ed' ''))");
    assert_eq!(walked.apply("jumped").as_deref(), Some("jump"));
    assert_eq!(walked.apply("d"), None, "too short for the kept suffix");
    assert_eq!(walked.apply("jumps"), None, "the suffix does not match");
    assert_eq!(EditTree::mine("was", "be").apply("is"), None);
    assert_eq!(EditTree::mine("", "").label(), "(m 0 0 () ())");
}

/// Deterministic strings over a small alphabet with multibyte characters,
/// so trees of every shape come up.
fn strings(seed: u64, n: usize) -> Vec<String> {
    const ALPHA: [char; 7] = ['a', 'b', 'e', 'ä', 'ß', '\'', '\u{1F600}'];
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    (0..n).map(|_| (0..next() % 9).map(|_| ALPHA[(next() % 7) as usize]).collect()).collect()
}

#[test]
fn mining_round_trips_random_pairs() {
    let forms = strings(7, 3000);
    let lemmas = strings(11, 3000);
    for (f, l) in forms.iter().zip(&lemmas) {
        let t = EditTree::mine(f, l);
        assert_eq!(t.apply(f).as_deref(), Some(l.as_str()), "{f:?} -> {l:?}: {}", t.label());
        assert_eq!(EditTree::parse(&t.label()).unwrap(), t);
    }
}

#[test]
fn only_canonical_labels_parse() {
    for bad in [
        "",
        "()",
        "(m 0 0 () ())x",
        "(m 00 0 () ())",
        "(m 0 0 ()  ())",
        "(s 'a' 'b'",
        "(s 'a\\b' 'c')",
        "(s 'a 'b')",
        "(x 1 2 () ())",
        "(m 4294967296 0 () ())",
    ] {
        assert!(EditTree::parse(bad).is_err(), "{bad:?}");
    }
    let deep = format!("{}(s '' ''){}", "(m 0 0 ".repeat(100), " ())".repeat(100));
    assert!(EditTree::parse(&deep).unwrap_err().message.contains("deep"));
    let quoted = EditTree::Subst { from: "it's".into(), to: "a\\b".into() };
    assert_eq!(quoted.label(), r"(s 'it\'s' 'a\\b')");
    assert_eq!(EditTree::parse(&quoted.label()).unwrap(), quoted);
}
