//! Reviewed Hunspell snapshots: load, unsupported directives, stem, stemAll, analyze.
//!
//! Files under `tests/snapshots/<run>/<set>/<input>.tsv`. Regenerate with
//! `TURBO_BLESS=1 cargo test -p turbo-hunspell --test snapshots`. Published
//! dictionaries are optional unless `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`.

use std::fs;
use std::path::{Path, PathBuf};

use turbo_hunspell::{Dictionary, LoadMode, Stemmer};

type Unsupported = Vec<(String, u32)>;
type WordRow = (String, String, Vec<String>, Vec<String>);
type WordRows = Vec<WordRow>;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn goldens_data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../goldens/data")
}

fn fixtures() -> PathBuf {
    goldens_data().join("fixtures/hunspell")
}

fn snapshots() -> PathBuf {
    root().join("snapshots")
}

fn pairs() -> Vec<(String, String, String)> {
    fs::read_to_string(root().join("snapshot-pairs.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut p = l.split('\t');
            (p.next().unwrap().into(), p.next().unwrap().into(), p.next().unwrap().into())
        })
        .collect()
}

fn published_model(file: &str) -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("TURBO_GOLDENS_MODELS").map(PathBuf::from),
        std::env::var_os("CARGO_TARGET_DIR").map(|d| PathBuf::from(d).join("turbo-goldens/models")),
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/turbo-goldens/models")),
    ];
    for dir in candidates.into_iter().flatten() {
        let path = dir.join(file);
        if path.is_file() {
            return Some(path);
        }
    }
    if std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1") {
        panic!("hunspell/{file} not found and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1");
    }
    None
}

fn files(run: &str) -> Option<(Vec<u8>, Vec<u8>)> {
    let read = |p: PathBuf| fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    if let Some(fixture) = run.strip_prefix("fixture-") {
        let name = fixture.strip_suffix("+partial").unwrap_or(fixture);
        return Some((read(fixtures().join(format!("{name}.aff"))), read(fixtures().join(format!("{name}.dic")))));
    }
    let affix = published_model(&format!("hunspell/{run}.aff"))?;
    let words = published_model(&format!("hunspell/{run}.dic"))?;
    Some((read(affix), read(words)))
}

fn input_path(set: &str, id: &str) -> PathBuf {
    if set == "hunspell" {
        root().join("inputs").join(format!("{id}.txt"))
    } else {
        goldens_data().join("inputs").join(set).join(format!("{id}.txt"))
    }
}

fn words(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(unescape)
        .collect()
}

fn unescape(line: &str) -> String {
    let mut units = Vec::new();
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            units.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match chars.next() {
            Some('\\') => units.push(u16::from(b'\\')),
            Some('t') => units.push(9),
            Some('n') => units.push(10),
            Some('r') => units.push(13),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                units.push(u16::from_str_radix(&hex, 16).unwrap());
            }
            other => panic!("bad escape {other:?} in {line}"),
        }
    }
    String::from_utf16(&units).unwrap_or_else(|_| panic!("lone surrogate in {line}"))
}

fn escape(s: &str) -> String {
    let mut out = String::new();
    for (i, u) in s.encode_utf16().enumerate() {
        if i == 0 && u == u16::from(b'#') {
            out.push_str("\\u0023");
        } else if u == u16::from(b'\\') {
            out.push_str("\\\\");
        } else if u == 9 {
            out.push_str("\\t");
        } else if u == 10 {
            out.push_str("\\n");
        } else if u == 13 {
            out.push_str("\\r");
        } else if u < 0x20 || (0xD800..0xE000).contains(&u) {
            out.push_str(&format!("\\u{u:04X}"));
        } else if let Some(c) = char::from_u32(u32::from(u)) {
            out.push(c);
        } else {
            out.push_str(&format!("\\u{u:04X}"));
        }
    }
    out
}

fn join_field(parts: &[String]) -> String {
    parts.iter().map(|p| escape(p)).collect::<Vec<_>>().join("\x1f")
}

fn split_field(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split('\x1f').map(unescape).collect()
}

fn snapshot_path(run: &str, set: &str, id: &str) -> PathBuf {
    snapshots().join(run).join(set).join(format!("{id}.tsv"))
}

fn write_snapshot(
    run: &str,
    set: &str,
    id: &str,
    load_err: Option<&str>,
    unsupported: &[(String, u32)],
    rows: &[(String, String, Vec<String>, Vec<String>)],
) {
    let path = snapshot_path(run, set, id);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut body = String::new();
    match load_err {
        Some(msg) => body.push_str(&format!("load_error\t{}\n", escape(msg))),
        None => body.push_str("load_error\t\n"),
    }
    body.push_str(&format!("unsupported\t{}\n", unsupported.len()));
    for (d, line) in unsupported {
        body.push_str(&format!("{}\t{line}\n", escape(d)));
    }
    body.push_str(&format!("words\t{}\n", rows.len()));
    for (word, stem, stems, analyses) in rows {
        body.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            escape(word),
            escape(stem),
            join_field(stems),
            join_field(analyses)
        ));
    }
    fs::write(&path, body).unwrap();
}

fn read_snapshot(run: &str, set: &str, id: &str) -> (Option<String>, Unsupported, WordRows) {
    let text = fs::read_to_string(snapshot_path(run, set, id)).unwrap();
    let mut lines = text.lines();
    let load = lines.next().unwrap();
    assert!(load.starts_with("load_error\t"));
    let err = load["load_error\t".len()..].to_string();
    let load_err = if err.is_empty() { None } else { Some(unescape(&err)) };
    let un_header = lines.next().unwrap();
    let n_un: usize = un_header.strip_prefix("unsupported\t").unwrap().parse().unwrap();
    let mut unsupported = Vec::with_capacity(n_un);
    for _ in 0..n_un {
        let line = lines.next().unwrap();
        let (d, n) = line.split_once('\t').unwrap();
        unsupported.push((unescape(d), n.parse().unwrap()));
    }
    let w_header = lines.next().unwrap();
    let n_w: usize = w_header.strip_prefix("words\t").unwrap().parse().unwrap();
    let mut rows = Vec::with_capacity(n_w);
    for _ in 0..n_w {
        let line = lines.next().unwrap();
        let mut p = line.splitn(4, '\t');
        let word = unescape(p.next().unwrap());
        let stem = unescape(p.next().unwrap());
        let stems = split_field(p.next().unwrap());
        let analyses = split_field(p.next().unwrap());
        rows.push((word, stem, stems, analyses));
    }
    (load_err, unsupported, rows)
}

#[test]
fn hunspell_matches_the_snapshots() {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let pairs = pairs();
    assert_eq!(pairs.len(), 48);
    let mut compared = 0usize;
    let mut failures = Vec::new();
    let mut runs_seen = 0usize;
    for (run, set, id) in &pairs {
        let Some((affix, words_bytes)) = files(run) else {
            eprintln!("{run}: dictionary not downloaded; skipped");
            continue;
        };
        runs_seen += 1;
        let mode = if run.ends_with("+partial") { LoadMode::AllowPartial } else { LoadMode::Strict };
        let loaded = Dictionary::from_bytes(&affix, &words_bytes, mode);
        let input = input_path(set, id);
        let (load_err, unsupported, rows) = match &loaded {
            Err(e) => (Some(e.to_string()), Vec::new(), Vec::new()),
            Ok(dictionary) => {
                let word_list = words(&input);
                let unsupported: Vec<(String, u32)> =
                    dictionary.unsupported_directives().iter().map(|d| (d.directive.clone(), d.line as u32)).collect();
                let stemmer = Stemmer::new(dictionary);
                let mut rows = Vec::with_capacity(word_list.len());
                for text in &word_list {
                    let mut stems = stemmer.stems(text);
                    if stems.is_empty() {
                        stems.push(text.clone());
                    }
                    let stem = stemmer.stem(text).unwrap_or_else(|| text.clone());
                    let analyses = stemmer.analyze(text);
                    rows.push((text.clone(), stem, stems, analyses));
                }
                (None, unsupported, rows)
            }
        };
        if bless {
            write_snapshot(run, set, id, load_err.as_deref(), &unsupported, &rows);
            compared += rows.len().max(1);
            continue;
        }
        let (want_err, want_un, want_rows) = read_snapshot(run, set, id);
        if load_err != want_err {
            failures.push(format!("{run}/{set}/{id}: load_err got {load_err:?} want {want_err:?}"));
        }
        if unsupported != want_un {
            failures.push(format!("{run}/{set}/{id}: unsupported got {unsupported:?} want {want_un:?}"));
        }
        if rows.len() != want_rows.len() {
            failures.push(format!("{run}/{set}/{id}: word count {} vs {}", rows.len(), want_rows.len()));
        } else {
            for (i, (got, want)) in rows.iter().zip(&want_rows).enumerate() {
                compared += 1;
                if got != want {
                    failures.push(format!("{run}/{set}/{id}[{i}]: got {got:?} want {want:?}"));
                }
            }
        }
        if rows.is_empty() {
            compared += 1;
        }
    }
    assert!(
        failures.is_empty(),
        "{} of ~{compared} differ:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
    assert!(runs_seen > 0 || bless, "no hunspell dictionaries available");
}

#[test]
fn escapes_round_trip() {
    for s in ["", "a", "a\\b", "a\tb", "#lead", "café", "a\nb"] {
        assert_eq!(unescape(&escape(s)), s, "{s:?}");
    }
}
