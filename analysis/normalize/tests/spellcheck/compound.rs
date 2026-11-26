//! Compound mode (`SymSpellCompoundTest`), and parity with OpenNLP over
//! generated terms and phrases.

use super::model::java_model;
use super::*;

// ---------------------------------------------------------------------
// SymSpellCompoundTest.
// ---------------------------------------------------------------------

#[test]
fn compound_merges_run_on_words() {
    let t = tiny();
    assert_eq!(compound(&t, "helloworld"), "hello world");
    assert_eq!(compound(&t, "spellchecker"), "spell checker");
    let mut e = with(&[("hello", 4569100), ("world", 5705975)]);
    e.add_bigram("hello", "world", 5432109);
    assert_eq!(compound(&e, "helloworld"), "hello world");
    assert_eq!(compound(&e, "hel loworld"), "hello world");
}

#[test]
fn compound_corrects_mixed_sentences() {
    let t = tiny();
    assert_eq!(compound(&t, "teh quikc broen fxo"), "the quick brown fox");
    let e = with(&[("the", 23135851162), ("quick", 119476788)]);
    assert_eq!(compound(&e, "thequikc"), "the quick");
    let r = t.lookup_compound("the quick brown fox", 2).unwrap();
    assert_eq!((r.term.as_str(), r.edit_distance), ("the quick brown fox", 0));
}

#[test]
fn compound_edge_cases() {
    let t = tiny();
    assert_eq!(compound(&t, ""), "");
    assert_eq!(compound(&t, "   "), "");
    assert_eq!(compound(&t, "zzzzqqqq"), "zzzzqqqq");
}

#[test]
fn compound_splits_on_unicode_white_space() {
    let t = tiny();
    let r = t.lookup_compound("hello\u{A0}world", 2).unwrap();
    assert_eq!((r.term.as_str(), r.edit_distance), ("hello world", 1));
    for sep in ['\u{A0}', '\u{85}', '\u{2028}'] {
        assert_eq!(compound(&t, &format!("hello{sep}world{sep}hello")), "hello world hello", "{sep:?}");
    }
    let r = t.lookup_compound("\u{A0}", 2).unwrap();
    assert_eq!((r.term.as_str(), r.edit_distance), ("", 0));
    let r = t.lookup_compound("\u{A0}hello world\u{A0}\u{85}", 2).unwrap();
    assert_eq!((r.term.as_str(), r.edit_distance), ("hello world", 0));
}

#[test]
fn compound_parts_say_where_each_word_came_from() {
    let t = tiny();
    let input = "  teh quikc  helloworld spell checker hel lo";
    let r = t.lookup_compound(input, 2).unwrap();
    let parts: Vec<(&str, &str)> =
        r.parts.iter().map(|p| (&input[p.source.clone()], &r.term[p.output.clone()])).collect();
    assert_eq!(r.term, "the quick hello world spell checker hello");
    assert_eq!(
        parts,
        [
            ("teh", "the"),
            ("quikc", "quick"),
            ("helloworld", "hello world"),
            ("spell", "spell"),
            ("checker", "checker"),
            ("hel lo", "hello"),
        ]
    );
    // The outputs are the phrase with single spaces between them.
    let rebuilt: Vec<&str> = r.parts.iter().map(|p| &r.term[p.output.clone()]).collect();
    assert_eq!(rebuilt.join(" "), r.term);
}

// ---------------------------------------------------------------------
// Parity with OpenNLP over generated terms and phrases.
// ---------------------------------------------------------------------

/// Undoes the escaping of expected.txt: `\u{hex}` is a code point.
fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("\\u{") {
        out.push_str(&rest[..i]);
        let end = rest[i..].find('}').unwrap() + i;
        out.push(char::from_u32(u32::from_str_radix(&rest[i + 3..end], 16).unwrap()).unwrap());
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

/// `term,distance,count` as expected.txt writes a suggestion.
fn item(s: &str) -> (String, usize, u64) {
    let f: Vec<&str> = s.split(',').collect();
    (unescape(f[0]), f[1].parse().unwrap(), f[2].parse().unwrap())
}

fn expected(kind: &str) -> Vec<Vec<String>> {
    fs::read_to_string(data("expected.txt"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with(&format!("{kind}\t")))
        .map(|l| l.split('\t').map(str::to_owned).collect())
        .collect()
}

fn engines() -> [(&'static str, SymSpell); 2] {
    [("tiny", java_model("model-tiny.bin").1.build()), ("levenshtein", java_model("model-levenshtein.bin").1.build())]
}

#[test]
fn lookups_match_opennlp() {
    let engines = engines();
    let rows = expected("lookup");
    assert!(rows.len() > 2000);
    for row in rows {
        let engine = &engines.iter().find(|(n, _)| *n == row[1]).unwrap().1;
        let verbosity = match row[2].as_str() {
            "TOP" => Verbosity::Top,
            "CLOSEST" => Verbosity::Closest,
            _ => Verbosity::All,
        };
        let term = unescape(&row[4]);
        let want: Vec<(String, usize, u64)> =
            if row.len() < 6 || row[5].is_empty() { Vec::new() } else { row[5].split('|').map(item).collect() };
        let got: Vec<(String, usize, u64)> = engine
            .lookup(&term, verbosity, row[3].parse().unwrap())
            .unwrap()
            .iter()
            .map(|s| (s.term.to_owned(), s.edit_distance, s.count))
            .collect();
        assert_eq!(got, want, "{row:?}");
    }
}

#[test]
fn compound_lookups_match_opennlp() {
    let engines = engines();
    let rows = expected("compound");
    assert!(rows.len() > 1500);
    for row in rows {
        let engine = &engines.iter().find(|(n, _)| *n == row[1]).unwrap().1;
        let input = unescape(&row[3]);
        let r = engine.lookup_compound(&input, row[2].parse().unwrap()).unwrap();
        assert_eq!((r.term.clone(), r.edit_distance, r.count), item(&row[4]), "{row:?}");
        // Every part points at whole tokens of the input, in order, and
        // the outputs are the phrase with single spaces between them.
        let mut at = 0;
        for p in &r.parts {
            assert!(p.source.start >= at && p.source.end > p.source.start, "{row:?}");
            let src = &input[p.source.clone()];
            assert!(!src.starts_with(char::is_whitespace) && !src.ends_with(char::is_whitespace));
            at = p.source.end;
        }
        let outputs: Vec<&str> = r.parts.iter().map(|p| &r.term[p.output.clone()]).collect();
        assert_eq!(outputs.join(" "), r.term);
    }
}
