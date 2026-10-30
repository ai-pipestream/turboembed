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
        // A tree mines from `über` -> "" but gives no word an empty lemma.
        let want = (form.is_empty() || !lemma.is_empty()).then_some(lemma);
        assert_eq!(t.apply(form).as_deref(), want, "{form:?} -> {lemma:?}: {t:?}");
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
    assert_eq!(walked.apply("ed"), None, "it would leave no lemma");
    assert_eq!(EditTree::mine("x", "").apply("x"), None, "nor may any tree");
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
        // An empty lemma of a word is never a lemma the tree gives.
        let want = (f.is_empty() || !l.is_empty()).then_some(l.as_str());
        assert_eq!(t.apply(f).as_deref(), want, "{f:?} -> {l:?}: {}", t.label());
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

#[test]
fn a_capitalized_form_takes_the_lowercase_words_tree_under_l() {
    let dogs = EditTree::mine_cased("dogs", "dog");
    let caps = EditTree::mine_cased("Dogs", "dog");
    assert_eq!(dogs.label(), "(m 0 1 () (s 's' ''))");
    assert_eq!(caps.label(), "(l (m 0 1 () (s 's' '')))");
    assert_eq!(caps.apply("Dogs").as_deref(), Some("dog"));
    assert_eq!(caps.apply("CATS").as_deref(), Some("cat"));
    assert_eq!(caps.apply("cats").as_deref(), Some("cat"), "a lowercase word fits the tree too");
    assert_eq!(caps.apply("Cat"), None, "the suffix must match after lowering");
    assert_eq!(caps.apply("S"), None, "nor may the lemma be empty");
    assert_eq!(EditTree::parse(&caps.label()).unwrap(), caps);
    // spaCy's tree of the same pair is one tree per capital letter.
    assert_eq!(EditTree::mine("Dogs", "dog").label(), "(m 1 1 (s 'D' 'd') (s 's' ''))");
    assert_eq!(EditTree::mine_cased("Running", "run").label(), "(l (m 0 4 () (s 'ning' '')))");
    // A lemma that keeps its capital stays plain, as does a tie.
    assert_eq!(EditTree::mine_cased("Paris", "Paris").label(), "(m 0 0 () ())");
    assert_eq!(EditTree::mine_cased("NASA", "NASA").label(), "(m 0 0 () ())");
    assert_eq!(EditTree::mine_cased("Berlins", "Berlin").label(), "(m 0 1 () (s 's' ''))");
    assert_eq!(EditTree::mine_cased("Bäume", "Baum"), EditTree::mine("Bäume", "Baum"));
    assert_eq!(EditTree::mine_cased("I", "I").label(), "(m 0 0 () ())");
    // The mapping is Unicode 18's simple lowercase, no special casing and
    // no final sigma.
    let ueber = EditTree::mine_cased("Über", "über");
    assert_eq!(ueber.label(), "(l (m 0 0 () ()))");
    assert_eq!(ueber.apply("ÜBER").as_deref(), Some("über"));
    let sigma = EditTree::mine_cased("\u{3A3}\u{39F}\u{3A6}\u{38A}\u{391}", "\u{3C3}\u{3BF}\u{3C6}\u{3AF}\u{3B1}");
    assert_eq!(sigma.label(), "(l (m 0 0 () ()))");
    assert_eq!(sigma.apply("\u{39F}\u{3A3}").as_deref(), Some("\u{3BF}\u{3C3}"));
    assert_eq!(EditTree::mine_cased("\u{130}stanbul", "istanbul").label(), "(l (m 0 0 () ()))");
}

#[test]
fn cased_mining_round_trips_random_pairs() {
    const ALPHA: [char; 8] = ['a', 'A', 'e', 'Ä', 'ä', '\u{1E9E}', 'ß', '\u{1F600}'];
    let mut s = 19u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut word = || (0..next() % 9).map(|_| ALPHA[(next() % 8) as usize]).collect::<String>();
    let (mut lowered, mut plain) = (0, 0);
    for _ in 0..3000 {
        let (f, l) = (word(), word());
        let t = EditTree::mine_cased(&f, &l);
        let want = (f.is_empty() || !l.is_empty()).then_some(l.as_str());
        assert_eq!(t.apply(&f).as_deref(), want, "{f:?} -> {l:?}: {}", t.label());
        assert_eq!(EditTree::parse(&t.label()).unwrap(), t);
        match t {
            EditTree::Lower(_) => lowered += 1,
            _ => plain += 1,
        }
    }
    assert!(lowered > 100 && plain > 100, "{lowered} lowered, {plain} plain");
}

#[test]
fn only_canonical_lower_labels_parse() {
    for bad in ["(l ())", "(l (l (m 0 0 () ())))", "(l(m 0 0 () ()))", "(l (m 0 0 () ()) )", "(l)"] {
        assert!(EditTree::parse(bad).is_err(), "{bad:?}");
    }
    let t = EditTree::parse("(l (s 'was' 'be'))").unwrap();
    assert_eq!(t, EditTree::Lower(Box::new(EditTree::Subst { from: "was".into(), to: "be".into() })));
    assert_eq!(t.apply("WAS").as_deref(), Some("be"));
    assert_eq!(t.apply("is"), None);
}
