//! The finders and the merge against the files `golden/src/NameFindGolden.java`
//! wrote with OpenNLP on a JVM (see the crate README).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use turbo_maxent::Dictionary;
use turbo_namefind::{
    BuiltIn, DictionaryNameFinder, Error, Flags, JavaRegex, Pattern, RegexNameFinder, TokenNameFinder, drop_overlapping,
};

fn golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("golden")
}

fn read(relative: &str) -> String {
    let path = golden().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A string escaped by the generator: `\uXXXX` per UTF-16 unit outside
/// printable ASCII, `\\`, `\t`, `\n` and `\r`.
fn unescape(s: &str) -> String {
    let mut units = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            units.push(c as u16);
            continue;
        }
        match chars.next() {
            Some('\\') => units.push(0x5C),
            Some('t') => units.push(0x09),
            Some('n') => units.push(0x0A),
            Some('r') => units.push(0x0D),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                units.push(u16::from_str_radix(&hex, 16).unwrap());
            }
            other => panic!("bad escape {other:?} in {s}"),
        }
    }
    String::from_utf16(&units).unwrap()
}

fn escape(s: &str) -> String {
    let mut out = String::new();
    for u in s.encode_utf16() {
        match u {
            0x5C => out.push_str("\\\\"),
            0x09 => out.push_str("\\t"),
            0x0A => out.push_str("\\n"),
            0x0D => out.push_str("\\r"),
            0x20..=0x7E => out.push(u as u8 as char),
            _ => out.push_str(&format!("\\u{u:04X}")),
        }
    }
    out
}

fn list(s: &str) -> Vec<String> {
    if s.is_empty() { Vec::new() } else { s.split('\t').map(unescape).collect() }
}

/// The lines after the header, which must name `component`.
fn body<'a>(text: &'a str, component: &str) -> impl Iterator<Item = &'a str> {
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(format!("component: {component}").as_str()));
    assert_eq!(lines.next(), Some("opennlp: cdf33c653d5d61d5af7d37ffee189a83bd8392ea"));
    assert!(lines.next().unwrap().starts_with("jdk: 21"));
    lines
}

/// Every sentence by key: `h` the hand-made ones, `f` and `d` the
/// generated ones.
struct Sentences {
    by_key: HashMap<String, Vec<String>>,
    order: Vec<String>,
}

impl Sentences {
    fn load() -> Sentences {
        let mut s = Sentences { by_key: HashMap::new(), order: Vec::new() };
        for (i, line) in read("inputs/sentences.txt").lines().enumerate() {
            s.add(format!("h{i}"), line.split(' ').map(String::from).collect());
        }
        for line in body(&read("expected/sentences.txt"), "sentences") {
            let (key, tokens) = line.split_once('\t').unwrap();
            s.add(key.into(), list(tokens));
        }
        s
    }

    fn add(&mut self, key: String, tokens: Vec<String>) {
        self.order.push(key.clone());
        self.by_key.insert(key, tokens);
    }

    /// The keys starting with `prefix`, in file order.
    fn keys(&self, prefix: char) -> impl Iterator<Item = &str> {
        self.order.iter().filter(move |k| k.starts_with(prefix)).map(String::as_str)
    }

    fn tokens(&self, key: &str) -> Vec<&str> {
        self.by_key[key].iter().map(String::as_str).collect()
    }
}

/// Names as the generator writes them.
fn format(finder: &dyn TokenNameFinder, tokens: &[&str]) -> Result<String, Error> {
    let mut names = Vec::new();
    finder.find(tokens, &mut names)?;
    let kinds = finder.kinds();
    let parts: Vec<String> =
        names.iter().map(|n| format!("{} {} {}", n.start, n.end, escape(kinds.label(n.kind)))).collect();
    Ok(parts.join("\t"))
}

/// The expected names of one finder: by sentence key, for the keys with
/// names.
struct Section<'a> {
    name: &'a str,
    types: Option<&'a str>,
    lines: HashMap<&'a str, &'a str>,
}

fn sections<'a>(lines: impl Iterator<Item = &'a str>) -> Vec<Section<'a>> {
    let mut out: Vec<Section<'a>> = Vec::new();
    for line in lines {
        let (key, rest) = line.split_once('\t').unwrap();
        match key {
            "finder" => out.push(Section { name: rest, types: None, lines: HashMap::new() }),
            "types" => out.last_mut().unwrap().types = Some(rest),
            _ => {
                out.last_mut().unwrap().lines.insert(key, rest);
            }
        }
    }
    out
}

/// Runs `finder` over the sentences with the given key prefixes and
/// compares with `section`, returning the number of sentences compared.
fn compare(finder: &dyn TokenNameFinder, section: &Section, sentences: &Sentences, prefixes: &[char]) -> usize {
    let mut failures = Vec::new();
    let mut n = 0;
    for &p in prefixes {
        for key in sentences.keys(p) {
            n += 1;
            let tokens = sentences.tokens(key);
            let got = format(finder, &tokens).unwrap();
            let want = section.lines.get(key).copied().unwrap_or("");
            if got != want {
                failures.push(format!("{} {key} {:?}\n  want {want}\n  got  {got}", section.name, tokens));
            }
        }
    }
    assert!(failures.is_empty(), "{} differences:\n{}", failures.len(), failures[..failures.len().min(10)].join("\n"));
    n
}

fn dictionary(name: &str) -> DictionaryNameFinder {
    let xml = std::fs::read(golden().join(format!("inputs/dictionaries/{name}.xml"))).unwrap();
    DictionaryNameFinder::new(Dictionary::read(&xml).unwrap(), name)
}

fn builtin(name: &str) -> BuiltIn {
    BuiltIn::ALL.into_iter().find(|b| b.java_name() == name).unwrap_or_else(|| panic!("{name}"))
}

#[test]
fn dictionaries() {
    let sentences = Sentences::load();
    let text = read("expected/dictionary.txt");
    let sections = sections(body(&text, "dictionary"));
    assert_eq!(sections.len(), 4);
    for section in &sections {
        let n = compare(&dictionary(section.name), section, &sentences, &['h', 'd']);
        assert_eq!(n, 1031);
    }
}

#[test]
fn builtin_patterns() {
    let sentences = Sentences::load();
    let text = read("expected/regex-builtin.txt");
    let sections = sections(body(&text, "regex-builtin"));
    let mut names = Vec::new();
    for section in &sections {
        names.push(section.name);
        let finder = match section.name {
            "ALL" => RegexNameFinder::defaults(&BuiltIn::ALL),
            "CONFIG" => {
                let config = vec![
                    ("EMAIL".into(), vec![Pattern::java("[a-z]+@[a-z]+").unwrap()]),
                    ("WORD".into(), vec![Pattern::java(r"[A-Z]\w+").unwrap(), Pattern::java(r"\d+").unwrap()]),
                ];
                RegexNameFinder::with_config(config, Some(&BuiltIn::ALL)).unwrap()
            }
            name => RegexNameFinder::defaults(&[builtin(name)]),
        };
        if let Some(types) = section.types {
            assert_eq!(finder.types().map(escape).collect::<Vec<_>>().join("\t"), types);
        }
        let prefixes: &[char] = if section.name == "CONFIG" { &['h'] } else { &['h', 'f'] };
        compare(&finder, section, &sentences, prefixes);
    }
    assert_eq!(names, ["USA_PHONE_NUM", "EMAIL", "URL", "MGRS", "DEGREES_MIN_SEC_LAT_LON", "ALL", "CONFIG"]);
    // Every pattern found something somewhere.
    for section in &sections {
        assert!(section.lines.len() > 5, "{}", section.name);
    }
}

#[test]
fn factory_type_order() {
    let text = read("expected/regex-factory.txt");
    let mut n = 0;
    for line in body(&text, "regex-factory") {
        let (left, want) = line.split_once("\t=>\t").unwrap_or_else(|| (line.strip_suffix("\t=>").unwrap(), ""));
        let fields: Vec<&str> = left.split('\t').collect();
        assert_eq!((fields[0], fields[2]), ("defaults", "config"));
        let defaults: Option<Vec<BuiltIn>> = match fields[1] {
            "null" => None,
            "" => Some(Vec::new()),
            names => Some(names.split(',').map(builtin).collect()),
        };
        let keys = list(&fields[3..].join("\t"));
        let config = keys.iter().map(|k| (k.as_str().into(), vec![Pattern::java("x").unwrap()])).collect();
        let finder = RegexNameFinder::with_config(config, defaults.as_deref()).unwrap();
        let got: Vec<String> = finder.types().map(escape).collect();
        assert_eq!(got.join("\t"), want, "{line}");
        n += 1;
    }
    assert_eq!(n, 434);
}

#[test]
fn user_patterns() {
    let sentences = Sentences::load();
    let inputs = read("inputs/patterns.txt");
    let mut expectations = inputs.lines().map(|l| l.split('\t').next().unwrap());
    let text = read("expected/regex-user.txt");
    let mut lines = body(&text, "regex-user").peekable();
    let keys: Vec<&str> = sentences.keys('h').chain(sentences.keys('f').take(400)).collect();
    let (mut compared, mut refused_texts, mut patterns) = (0, 0, 0);
    let mut failures = Vec::new();
    while let Some(line) = lines.next() {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields[0], "pattern");
        let (flags, pattern, java) = (fields[1], unescape(fields[2]), fields[4]);
        let expect = expectations.next().unwrap();
        patterns += 1;
        let flags = Flags { case_insensitive: flags.contains('i'), dotall: flags.contains('s') };
        let mut want: HashMap<&str, &str> = HashMap::new();
        while let Some(l) = lines.next_if(|l| !l.starts_with("pattern\t")) {
            let (key, rest) = l.split_once('\t').unwrap();
            want.insert(key, rest);
        }
        let compiled = JavaRegex::compile_with(&pattern, flags);
        match (java, expect, compiled) {
            ("error", "syntax", Err(Error::Syntax { .. })) => continue,
            ("ok", "unsupported", Err(Error::Unsupported { .. })) => continue,
            ("ok", "ok", Ok(regex)) => {
                let finder = RegexNameFinder::new(vec![Pattern::Java(regex)], "T").unwrap();
                for &key in &keys {
                    let tokens = sentences.tokens(key);
                    match format(&finder, &tokens) {
                        Ok(got) => {
                            compared += 1;
                            let want = want.get(key).copied().unwrap_or("");
                            if got != want {
                                failures.push(format!("{pattern:?} {key} {tokens:?}\n  want {want}\n  got  {got}"));
                            }
                        }
                        Err(Error::Text { .. }) => refused_texts += 1,
                        Err(e) => panic!("{pattern:?} {key}: {e}"),
                    }
                }
            }
            (java, expect, ours) => panic!("{pattern:?}: Java {java}, expected {expect}, got {ours:?}"),
        }
    }
    assert!(failures.is_empty(), "{} differences:\n{}", failures.len(), failures[..failures.len().min(10)].join("\n"));
    assert_eq!(patterns, 114);
    println!("{compared} sentences compared, {refused_texts} refused");
    assert!(refused_texts * 50 < compared, "{refused_texts} refused of {compared}");
}

fn parse_spans(s: &str) -> Vec<(u32, u32, String)> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split('\t')
        .map(|span| {
            let mut parts = span.splitn(3, ' ');
            let start = parts.next().unwrap().parse().unwrap();
            let end = parts.next().unwrap().parse().unwrap();
            (start, end, unescape(parts.next().unwrap()))
        })
        .collect()
}

fn dropped(spans: &[(u32, u32, String)]) -> Vec<(u32, u32, String)> {
    let keys: Vec<(u32, u32, &str)> = spans.iter().map(|s| (s.0, s.1, s.2.as_str())).collect();
    drop_overlapping(&keys).into_iter().map(|i| spans[i].clone()).collect()
}

fn format_spans(spans: &[(u32, u32, String)]) -> String {
    spans.iter().map(|(s, e, t)| format!("{s} {e} {}", escape(t))).collect::<Vec<_>>().join("\t")
}

#[test]
fn merge() {
    let sentences = Sentences::load();
    let finders: Vec<Box<dyn TokenNameFinder>> = vec![
        Box::new(dictionary("places")),
        Box::new(dictionary("people")),
        Box::new(dictionary("overlap")),
        Box::new(dictionary("empty")),
        Box::new(RegexNameFinder::defaults(&BuiltIn::ALL)),
    ];
    let text = read("expected/merge.txt");
    let (mut synthetic, mut real) = (0, 0);
    let mut seen = HashMap::new();
    for line in body(&text, "merge") {
        let (left, want) = line.split_once("\t=>\t").unwrap_or_else(|| (line.strip_suffix("\t=>").unwrap(), ""));
        if let Some(key) = left.strip_prefix("finders\t") {
            let tokens = sentences.tokens(key);
            let mut all = Vec::new();
            for f in &finders {
                let mut names = Vec::new();
                f.find(&tokens, &mut names).unwrap();
                all.extend(names.iter().map(|n| (n.start, n.end, f.kinds().label(n.kind).to_string())));
            }
            assert_eq!(format_spans(&dropped(&all)), want, "{key}");
            seen.insert(key, ());
            real += 1;
        } else {
            assert_eq!(format_spans(&dropped(&parse_spans(left))), want, "{left}");
            synthetic += 1;
        }
    }
    assert_eq!(synthetic, 3000);
    assert!(real > 1000);
    // A sentence missing from the file has no names at all.
    for key in sentences.keys('h').chain(sentences.keys('f')) {
        if !seen.contains_key(key) {
            let tokens = sentences.tokens(key);
            for f in &finders {
                let mut names = Vec::new();
                f.find(&tokens, &mut names).unwrap();
                assert!(names.is_empty(), "{key}");
            }
        }
    }
}
