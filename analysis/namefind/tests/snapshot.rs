//! The finders and the merge as our own snapshots: the checked-in files
//! under `tests/snapshot-data` hold what the finders find on the crate's
//! own corpus (the hand-made sentences and a seeded generated one), the
//! factory's type order, and the spans `dropOverlappingSpans` keeps for
//! seeded random inputs. Refresh with `TURBO_SNAPSHOT_UPDATE=1` and review
//! the diff.

#[path = "snapshot/support.rs"]
mod support;

use turbo_namefind::{BuiltIn, DictionaryNameFinder, Error, Pattern, RegexNameFinder, Tok, drop_overlapping};

use support::{Rng, check, corpus, dictionary, find_all, xml_to_lines};

#[test]
fn dictionaries() {
    check("dictionaries.snap", &find_all(&dictionary("places", "place").unwrap()));
    check("people.snap", &find_all(&dictionary("people", "person").unwrap()));
    check("overlap.snap", &find_all(&dictionary("overlap", "overlap").unwrap()));
    // The empty fixture converts to an empty dictionary, which a finder
    // refuses: there is no type a name could carry.
    let xml = include_str!("snapshot-data/empty.xml");
    let empty = turbo_namefind::Dictionary::read(&xml_to_lines(xml, "x").unwrap()).unwrap();
    assert_eq!(DictionaryNameFinder::new(empty).unwrap_err(), Error::EmptyDictionary);
}

#[test]
fn builtin_patterns() {
    for &b in &BuiltIn::ALL {
        let finder = RegexNameFinder::defaults(&[b]);
        check(&format!("builtin-{}.snap", b.java_name().to_ascii_lowercase()), &find_all(&finder));
    }
    check("builtin-all.snap", &find_all(&RegexNameFinder::defaults(&BuiltIn::ALL)));
}

#[test]
fn factory_type_order() {
    let mut rng = Rng::new(0xFA57);
    let mut cases = String::new();
    for i in 0..12 {
        let n_defaults = rng.below(3);
        let defaults: Vec<BuiltIn> =
            BuiltIn::ALL.iter().copied().filter(|_| rng.below(2) == 0).take(n_defaults + 1).collect();
        let mut config: Vec<(Box<str>, Vec<Pattern>)> = Vec::new();
        for _ in 0..rng.below(3) {
            let key = ["EMAIL", "WORD", "URL", "PHONE_NUM"][rng.below(4)];
            let pattern = Pattern::tokens(vec![Tok::text(["x", "y", "call"][rng.below(3)])]).unwrap();
            if !config.iter().any(|(k, _)| &**k == key) {
                config.push((key.into(), vec![pattern]));
            }
        }
        let finder = RegexNameFinder::with_config(config.clone(), Some(&defaults)).unwrap();
        cases.push_str(&format!(
            "case {i}: defaults {:?} config {:?} -> {:?}\n",
            defaults,
            config,
            finder.types().collect::<Vec<_>>()
        ));
        let finder = RegexNameFinder::with_config(config, None).unwrap();
        cases.push_str(&format!("case {i} nodefaults -> {:?}\n", finder.types().collect::<Vec<_>>()));
    }
    check("factory-order.snap", &cases);
}

#[test]
fn merge_cases() {
    let mut rng = Rng::new(0xD2AF);
    let types = ["a", "b", "C", "\u{FF21}", "\u{1F600}", "place", "person"];
    let mut cases = String::new();
    for i in 0..60 {
        let n = 1 + rng.below(8);
        let mut spans = Vec::new();
        for _ in 0..n {
            let start = rng.below(12) as u32;
            let len = rng.below(5) as u32;
            spans.push((start, start + len, *rng.pick(&types)));
        }
        let kept = drop_overlapping(&spans);
        cases.push_str(&format!("{i} {spans:?} -> {kept:?}\n"));
    }
    check("merge-cases.snap", &cases);
}

#[test]
fn merged_names() {
    let finders: Vec<Box<dyn turbo_namefind::TokenNameFinder>> = vec![
        Box::new(dictionary("places", "place").unwrap()),
        Box::new(dictionary("people", "person").unwrap()),
        Box::new(dictionary("overlap", "overlap").unwrap()),
        Box::new(RegexNameFinder::defaults(&BuiltIn::ALL)),
    ];
    let mut out = String::new();
    for sentence in corpus() {
        let tokens: Vec<&str> = sentence.split(' ').collect();
        let mut all = Vec::new();
        for f in &finders {
            let mut names = Vec::new();
            f.find(&tokens, &mut names).unwrap();
            all.extend(names.iter().map(|n| (n.start, n.end, f.kinds().label(n.kind).to_string())));
        }
        let mut line = String::new();
        for i in drop_overlapping(&all.iter().map(|&(s, e, ref t)| (s, e, t.as_str())).collect::<Vec<_>>()) {
            line.push_str(&format!("{}:{} {} ", all[i].0, all[i].1, all[i].2));
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    check("merged-names.snap", &out);
}
