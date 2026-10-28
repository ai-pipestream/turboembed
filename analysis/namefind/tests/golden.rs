//! The finders and the merge against the goldens of `NameFindGolden.java`
//! in `analysis/goldens-tool`, which ran OpenNLP on a JVM (see the crate
//! README).

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape};
use turbo_maxent::Dictionary;
use turbo_namefind::{
    BuiltIn, DictionaryNameFinder, Error, Flags, JavaRegex, Pattern, RegexNameFinder, TokenNameFinder, drop_overlapping,
};

const fn col(name: &'static str, ty: ColumnType) -> ColumnSpec {
    ColumnSpec::new(name, ty)
}

const STR: ColumnType = ColumnType::Str;
const U32: ColumnType = ColumnType::U32;

const SPAN: &[ColumnSpec] = &[col("case", U32), col("start", U32), col("end", U32), col("type", STR)];

/// The layers of the name finder goldens, all anchored on the document.
const LAYERS: &[(&str, &[ColumnSpec])] = &[
    ("turbo:sentences", &[col("sentence", STR), col("token", STR)]),
    ("turbo:names", &[col("finder", STR), col("sentence", STR), col("start", U32), col("end", U32), col("type", STR)]),
    ("turbo:finder-types", &[col("finder", STR), col("type", STR)]),
    ("turbo:factory-cases", &[col("case", U32), col("has-defaults", U32)]),
    ("turbo:factory-defaults", &[col("case", U32), col("default", STR)]),
    ("turbo:factory-keys", &[col("case", U32), col("key", STR)]),
    ("turbo:factory-types", &[col("case", U32), col("type", STR)]),
    ("turbo:patterns", &[col("pattern", U32), col("flags", STR), col("regex", STR), col("compiled", U32)]),
    (
        "turbo:pattern-names",
        &[col("pattern", U32), col("sentence", STR), col("start", U32), col("end", U32), col("type", STR)],
    ),
    ("turbo:merge-cases", &[col("case", U32), col("spans", U32)]),
    ("turbo:merge-spans", SPAN),
    ("turbo:merge-kept", SPAN),
    ("turbo:merged-names", &[col("sentence", STR), col("start", U32), col("end", U32), col("type", STR)]),
];

/// The golden of a run on the hand-made sentences.
fn read(run: &str) -> Golden {
    let shapes: Vec<Shape> =
        LAYERS.iter().map(|&(id, columns)| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns }).collect();
    let path = turbo_goldens::data_dir().join("namefind").join(run).join("namefind/sentences.golden");
    Golden::read_with(&path, &shapes).unwrap_or_else(|e| panic!("{e}"))
}

fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layer(id).unwrap_or_else(|| panic!("no layer {id}"))
}

fn u32s<'l>(l: &'l GoldenLayer, name: &str) -> &'l [u32] {
    l.u32s(name).unwrap_or_else(|| panic!("{}: no u32 column {name}", l.id))
}

fn strs(l: &GoldenLayer, name: &str) -> Vec<String> {
    l.strs(name).unwrap_or_else(|| panic!("{}: no str column {name}", l.id)).iter().map(str::to_string).collect()
}

/// The values of the `str` column `value` grouped by the `u32` column
/// `key`, which numbers `n` groups from 0.
fn grouped(l: &GoldenLayer, key: &str, value: &str, n: usize) -> Vec<Vec<String>> {
    let mut out = vec![Vec::new(); n];
    for (k, v) in u32s(l, key).iter().zip(strs(l, value)) {
        out[*k as usize].push(v);
    }
    out
}

/// A name as the tests compare it: `start end type`.
fn name(start: u32, end: u32, kind: &str) -> String {
    format!("{start} {end} {kind}")
}

/// The names of a layer with `start`, `end` and `type` columns, by the
/// value of the `str` column `key`, in row order.
fn names_by<'a>(l: &GoldenLayer, key: impl Fn(usize) -> String + 'a) -> BTreeMap<String, Vec<String>> {
    let (start, end, kind) = (u32s(l, "start"), u32s(l, "end"), strs(l, "type"));
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for r in 0..l.len() {
        out.entry(key(r)).or_default().push(name(start[r], end[r], &kind[r]));
    }
    out
}

fn fixture(name: &str) -> PathBuf {
    turbo_goldens::data_dir().join("fixtures/namefind").join(name)
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
        let g = read("generated");
        for (i, line) in g.input_text(turbo_goldens::data_dir()).unwrap().lines().enumerate() {
            s.add(format!("h{i}"), line.split(' ').map(String::from).collect());
        }
        let l = layer(&g, "turbo:sentences");
        for (key, token) in strs(l, "sentence").into_iter().zip(strs(l, "token")) {
            if s.order.last() != Some(&key) {
                s.add(key.clone(), Vec::new());
            }
            s.by_key.get_mut(&key).unwrap().push(token);
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

/// The names a finder finds, as the tests compare them.
fn find(finder: &dyn TokenNameFinder, tokens: &[&str]) -> Result<Vec<String>, Error> {
    let mut names = Vec::new();
    finder.find(tokens, &mut names)?;
    let kinds = finder.kinds();
    Ok(names.iter().map(|n| name(n.start, n.end, kinds.label(n.kind))).collect())
}

/// The names by sentence key.
type NamesByKey = BTreeMap<String, Vec<String>>;

/// The names of each finder of a `turbo:names` layer, by finder and then
/// by sentence key, and the finders in the order they first come.
fn finder_names(g: &Golden) -> (Vec<String>, HashMap<String, NamesByKey>) {
    let l = layer(g, "turbo:names");
    let (finder, sentence) = (strs(l, "finder"), strs(l, "sentence"));
    let mut order: Vec<String> = Vec::new();
    for f in &finder {
        if !order.contains(f) {
            order.push(f.clone());
        }
    }
    let mut out: HashMap<String, NamesByKey> = HashMap::new();
    for (key, names) in names_by(l, |r| format!("{}\t{}", finder[r], sentence[r])) {
        let (f, k) = key.split_once('\t').unwrap();
        out.entry(f.to_string()).or_default().insert(k.to_string(), names);
    }
    (order, out)
}

/// Runs `finder` over the sentences with the given key prefixes and
/// compares with `expected`, the names by sentence key, returning the
/// number of sentences compared.
fn compare(
    finder: &dyn TokenNameFinder,
    what: &str,
    expected: &BTreeMap<String, Vec<String>>,
    sentences: &Sentences,
    prefixes: &[char],
) -> usize {
    let mut failures = Vec::new();
    let mut n = 0;
    for &p in prefixes {
        for key in sentences.keys(p) {
            n += 1;
            let tokens = sentences.tokens(key);
            let got = find(finder, &tokens).unwrap();
            let want = expected.get(key).cloned().unwrap_or_default();
            if got != want {
                failures.push(format!("{what} {key} {tokens:?}\n  want {want:?}\n  got  {got:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{} differences:\n{}", failures.len(), failures[..failures.len().min(10)].join("\n"));
    n
}

fn dictionary(name: &str) -> DictionaryNameFinder {
    let xml = std::fs::read(fixture(&format!("{name}.xml"))).unwrap();
    DictionaryNameFinder::new(Dictionary::read(&xml).unwrap(), name)
}

fn builtin(name: &str) -> BuiltIn {
    BuiltIn::ALL.into_iter().find(|b| b.java_name() == name).unwrap_or_else(|| panic!("{name}"))
}

#[test]
fn dictionaries() {
    let sentences = Sentences::load();
    let (finders, names) = finder_names(&read("dictionary"));
    assert_eq!(finders, ["places", "people", "overlap"]);
    for finder in ["places", "people", "overlap", "empty"] {
        let expected = names.get(finder).cloned().unwrap_or_default();
        let n = compare(&dictionary(finder), finder, &expected, &sentences, &['h', 'd']);
        assert_eq!(n, 1031);
    }
}

#[test]
fn builtin_patterns() {
    let sentences = Sentences::load();
    let g = read("regex-builtin");
    let (finders, names) = finder_names(&g);
    let types = layer(&g, "turbo:finder-types");
    let (type_finder, type_name) = (strs(types, "finder"), strs(types, "type"));
    for finder in &finders {
        let f = match finder.as_str() {
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
        let want: Vec<&str> =
            type_finder.iter().zip(&type_name).filter(|(f, _)| *f == finder).map(|(_, t)| t.as_str()).collect();
        if !want.is_empty() {
            assert_eq!(f.types().collect::<Vec<_>>(), want, "{finder}");
        }
        let prefixes: &[char] = if finder == "CONFIG" { &['h'] } else { &['h', 'f'] };
        compare(&f, finder, &names[finder], &sentences, prefixes);
    }
    assert_eq!(finders, ["USA_PHONE_NUM", "EMAIL", "URL", "MGRS", "DEGREES_MIN_SEC_LAT_LON", "ALL", "CONFIG"]);
    assert_eq!(type_finder.iter().filter(|f| *f == "ALL").count(), 5);
    assert!(type_finder.iter().any(|f| f == "CONFIG"));
    // Every pattern found something somewhere.
    for finder in &finders {
        assert!(names[finder].len() > 5, "{finder}");
    }
}

#[test]
fn factory_type_order() {
    let g = read("regex-factory");
    let cases = layer(&g, "turbo:factory-cases");
    let n = cases.len();
    let defaults = grouped(layer(&g, "turbo:factory-defaults"), "case", "default", n);
    let keys = grouped(layer(&g, "turbo:factory-keys"), "case", "key", n);
    let types = grouped(layer(&g, "turbo:factory-types"), "case", "type", n);
    for (c, &has_defaults) in u32s(cases, "has-defaults").iter().enumerate() {
        let given: Option<Vec<BuiltIn>> = (has_defaults == 1).then(|| defaults[c].iter().map(|d| builtin(d)).collect());
        let config = keys[c].iter().map(|k| (k.as_str().into(), vec![Pattern::java("x").unwrap()])).collect();
        let finder = RegexNameFinder::with_config(config, given.as_deref()).unwrap();
        let got: Vec<&str> = finder.types().collect();
        assert_eq!(got, types[c], "case {c}: defaults {:?}, keys {:?}", given, keys[c]);
    }
    assert_eq!(n, 434);
}

#[test]
fn user_patterns() {
    let sentences = Sentences::load();
    let inputs = std::fs::read_to_string(fixture("patterns.txt")).unwrap();
    let mut expectations = inputs.lines().map(|l| l.split('\t').next().unwrap());
    let g = read("regex-user");
    let patterns = layer(&g, "turbo:patterns");
    let found = layer(&g, "turbo:pattern-names");
    let (number, key) = (u32s(found, "pattern"), strs(found, "sentence"));
    let names = names_by(found, |r| format!("{}\t{}", number[r], key[r]));
    let mut sentences = sentences;
    for (i, row) in std::fs::read_to_string(fixture("regex-boundaries.txt")).unwrap().split_terminator('\n').enumerate()
    {
        sentences.add(format!("b{i}"), row.split('\t').map(|t| t.replace("\\n", "\n").replace("\\r", "\r")).collect());
    }
    let keys: Vec<&str> = sentences.keys('h').chain(sentences.keys('f').take(400)).chain(sentences.keys('b')).collect();
    let (mut compared, mut refused_texts) = (0, 0);
    let mut failures = Vec::new();
    let (flags, regex, compiled) = (strs(patterns, "flags"), strs(patterns, "regex"), u32s(patterns, "compiled"));
    for (p, &number) in u32s(patterns, "pattern").iter().enumerate() {
        let pattern = &regex[p];
        let expect = expectations.next().unwrap();
        let flags = Flags { case_insensitive: flags[p].contains('i'), dotall: flags[p].contains('s') };
        let ours = JavaRegex::compile_with(pattern, flags);
        match (compiled[p], expect, ours) {
            (0, "syntax", Err(Error::Syntax { .. })) => continue,
            (1, "unsupported", Err(Error::Unsupported { .. })) => continue,
            (1, "ok", Ok(regex)) => {
                let finder = RegexNameFinder::new(vec![Pattern::Java(regex)], "T").unwrap();
                for &key in &keys {
                    let tokens = sentences.tokens(key);
                    match find(&finder, &tokens) {
                        Ok(got) => {
                            compared += 1;
                            let want = names.get(&format!("{number}\t{key}")).cloned().unwrap_or_default();
                            if got != want {
                                failures.push(format!("{pattern:?} {key} {tokens:?}\n  want {want:?}\n  got  {got:?}"));
                            }
                        }
                        Err(Error::Text { limit: turbo_namefind::TextLimit::Supplementary, .. }) => {
                            assert!(tokens.iter().any(|t| t.chars().any(|c| c as u32 > 0xFFFF)));
                            refused_texts += 1;
                        }
                        Err(e) => panic!("{pattern:?} {key}: {e}"),
                    }
                }
            }
            (java, expect, ours) => panic!("{pattern:?}: Java compiled {java}, expected {expect}, got {ours:?}"),
        }
    }
    assert!(failures.is_empty(), "{} differences:\n{}", failures.len(), failures[..failures.len().min(10)].join("\n"));
    assert_eq!(patterns.len(), 120);
    println!("{compared} sentences compared, {refused_texts} refused");
    assert!(refused_texts * 50 < compared, "{refused_texts} refused of {compared}");
}

/// Each case's spans, `(start, end, type)`, of a layer numbered by case.
fn case_spans(l: &GoldenLayer, n: usize) -> Vec<Vec<(u32, u32, String)>> {
    let (case, start, end, kind) = (u32s(l, "case"), u32s(l, "start"), u32s(l, "end"), strs(l, "type"));
    let mut out = vec![Vec::new(); n];
    for r in 0..l.len() {
        out[case[r] as usize].push((start[r], end[r], kind[r].clone()));
    }
    out
}

fn dropped(spans: &[(u32, u32, String)]) -> Vec<(u32, u32, String)> {
    let keys: Vec<(u32, u32, &str)> = spans.iter().map(|s| (s.0, s.1, s.2.as_str())).collect();
    drop_overlapping(&keys).into_iter().map(|i| spans[i].clone()).collect()
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
    let g = read("merge");
    let cases = layer(&g, "turbo:merge-cases");
    let spans = case_spans(layer(&g, "turbo:merge-spans"), cases.len());
    let kept = case_spans(layer(&g, "turbo:merge-kept"), cases.len());
    for (c, &count) in u32s(cases, "spans").iter().enumerate() {
        assert_eq!(spans[c].len(), count as usize);
        assert_eq!(dropped(&spans[c]), kept[c], "{:?}", spans[c]);
    }
    assert_eq!(cases.len(), 3000);

    let merged = layer(&g, "turbo:merged-names");
    let key = strs(merged, "sentence");
    let merged = names_by(merged, |r| key[r].clone());
    for (key, want) in &merged {
        let tokens = sentences.tokens(key);
        let mut all = Vec::new();
        for f in &finders {
            let mut names = Vec::new();
            f.find(&tokens, &mut names).unwrap();
            all.extend(names.iter().map(|n| (n.start, n.end, f.kinds().label(n.kind).to_string())));
        }
        let got: Vec<String> = dropped(&all).iter().map(|(s, e, t)| name(*s, *e, t)).collect();
        assert_eq!(&got, want, "{key}");
    }
    assert!(merged.len() > 1000);
    // A sentence missing from the golden has no names at all.
    for key in sentences.keys('h').chain(sentences.keys('f')) {
        if !merged.contains_key(key) {
            let tokens = sentences.tokens(key);
            for f in &finders {
                let mut names = Vec::new();
                f.find(&tokens, &mut names).unwrap();
                assert!(names.is_empty(), "{key}");
            }
        }
    }
}
