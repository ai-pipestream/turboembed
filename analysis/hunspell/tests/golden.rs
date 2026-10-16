//! Compares the reader and the stemmer with what OpenNLP returned for the
//! same dictionaries and words: the goldens of `HunspellGolden.java` in
//! `analysis/goldens-tool`, under `analysis/goldens/data/hunspell`. A run
//! is a published dictionary (downloaded by `turbo-goldens fetch-models`;
//! the run is skipped when it is missing) or a fixture of
//! `analysis/goldens/data/fixtures/hunspell`, loaded strictly or, for a
//! `+partial` run, with [`LoadMode::AllowPartial`].

use std::fs;
use std::path::{Path, PathBuf};

use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, Shape};
use turbo_hunspell::{Dictionary, LoadMode, Stemmer};

fn goldens() -> PathBuf {
    turbo_goldens::data_dir().join("hunspell")
}

const LOAD: &[ColumnSpec] = &[ColumnSpec::new("thrown", ColumnType::Str), ColumnSpec::new("message", ColumnType::Str)];
const UNSUPPORTED: &[ColumnSpec] =
    &[ColumnSpec::new("directive", ColumnType::Str), ColumnSpec::new("line", ColumnType::U32)];
const WORDS: &[ColumnSpec] = &[
    ColumnSpec::new("word", ColumnType::Str),
    ColumnSpec::new("stem", ColumnType::Str),
    ColumnSpec::new("thrown", ColumnType::Str),
];
const STEMS: &[ColumnSpec] = &[ColumnSpec::new("word", ColumnType::U32), ColumnSpec::new("stem", ColumnType::Str)];
const ANALYSES: &[ColumnSpec] =
    &[ColumnSpec::new("word", ColumnType::U32), ColumnSpec::new("analysis", ColumnType::Str)];

fn shapes() -> [Shape; 5] {
    let shape = |id: &str, columns| Shape { id: id.into(), anchor: GoldenAnchor::Document, columns };
    [
        shape("turbo:hunspell-load", LOAD),
        shape("turbo:hunspell-unsupported", UNSUPPORTED),
        shape("turbo:hunspell-words", WORDS),
        shape("turbo:hunspell-stems", STEMS),
        shape("turbo:hunspell-analyses", ANALYSES),
    ]
}

/// A string `TrainGolden.lossless` wrote: a doubled backslash is one,
/// `\uXXXX` is one UTF-16 unit. The inputs hold no lone surrogates, so
/// every string decodes.
fn lossless(field: &str) -> String {
    let mut units: Vec<u16> = Vec::with_capacity(field.len());
    let mut chars = field.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            units.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match chars.next() {
            Some('\\') => units.push(u16::from(b'\\')),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                units.push(u16::from_str_radix(&hex, 16).unwrap_or_else(|_| panic!("{field:?}: bad escape")));
            }
            other => panic!("{field:?}: unexpected {other:?} after a backslash"),
        }
    }
    String::from_utf16(&units).unwrap_or_else(|_| panic!("{field:?}: a lone surrogate"))
}

/// The two files of a run, or `None` for a published dictionary that has
/// not been downloaded.
fn files(run: &str) -> Option<(Vec<u8>, Vec<u8>)> {
    let read = |p: PathBuf| fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    if let Some(fixture) = run.strip_prefix("fixture-") {
        let dir = turbo_goldens::data_dir().join("fixtures/hunspell");
        let name = fixture.strip_suffix("+partial").unwrap_or(fixture);
        return Some((read(dir.join(format!("{name}.aff"))), read(dir.join(format!("{name}.dic")))));
    }
    let affix = turbo_goldens::published_model(&format!("hunspell/{run}.aff"))?;
    let words = turbo_goldens::published_model(&format!("hunspell/{run}.dic"))?;
    Some((read(affix), read(words)))
}

fn sorted_entries(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    out.sort();
    out
}

/// The rows of a `word:u32` layer grouped by word.
fn by_word(words: &[u32], values: impl Iterator<Item = String>, n: usize) -> Vec<Vec<String>> {
    let mut out = vec![Vec::new(); n];
    for (&w, v) in words.iter().zip(values) {
        out[w as usize].push(v);
    }
    out
}

#[test]
fn stemmer_matches_opennlp() {
    let shapes = shapes();
    let mut failures = Vec::new();
    let mut compared = 0;
    let mut runs = 0;
    for run_dir in sorted_entries(&goldens()) {
        let run = run_dir.file_name().unwrap().to_str().unwrap().to_string();
        let Some((affix, words)) = files(&run) else {
            eprintln!("{run}: the dictionary is not downloaded; skipped");
            continue;
        };
        runs += 1;
        let mode = if run.ends_with("+partial") { LoadMode::AllowPartial } else { LoadMode::Strict };
        let loaded = Dictionary::from_bytes(&affix, &words, mode);
        for set_dir in sorted_entries(&run_dir) {
            for file in sorted_entries(&set_dir) {
                let g = Golden::read_with(&file, &shapes).unwrap_or_else(|e| panic!("{e}"));
                let at = format!("{run} ({})", g.header.input.id);
                let load = g.layer("turbo:hunspell-load").unwrap();
                let (thrown, message) = (load.strs("thrown").unwrap(), load.strs("message").unwrap());
                let dictionary = match &loaded {
                    Err(e) => {
                        if thrown.get(0).is_empty() {
                            failures.push(format!("{at}: failed to load ({e}), OpenNLP loaded it"));
                        } else if e.to_string() != message.get(0) {
                            failures.push(format!("{at}: load error {e:?}, OpenNLP {:?}", message.get(0)));
                        }
                        compared += 1;
                        continue;
                    }
                    Ok(d) => {
                        if !thrown.get(0).is_empty() {
                            failures.push(format!("{at}: loaded, OpenNLP threw {}", message.get(0)));
                            continue;
                        }
                        d
                    }
                };
                let unsupported = g.layer("turbo:hunspell-unsupported").unwrap();
                let want: Vec<(String, u32)> = (0..unsupported.len())
                    .map(|i| {
                        (
                            unsupported.strs("directive").unwrap().get(i).to_string(),
                            unsupported.u32s("line").unwrap()[i],
                        )
                    })
                    .collect();
                let got: Vec<(String, u32)> =
                    dictionary.unsupported_directives().iter().map(|d| (d.directive.clone(), d.line as u32)).collect();
                assert_eq!(got, want, "{at}: unsupported directives");

                let w = g.layer("turbo:hunspell-words").unwrap();
                let (word, stem, thrown) =
                    (w.strs("word").unwrap(), w.strs("stem").unwrap(), w.strs("thrown").unwrap());
                let s = g.layer("turbo:hunspell-stems").unwrap();
                let all = by_word(
                    s.u32s("word").unwrap(),
                    (0..s.len()).map(|i| lossless(s.strs("stem").unwrap().get(i))),
                    w.len(),
                );
                let a = g.layer("turbo:hunspell-analyses").unwrap();
                let analyses = by_word(
                    a.u32s("word").unwrap(),
                    (0..a.len()).map(|i| lossless(a.strs("analysis").unwrap().get(i))),
                    w.len(),
                );
                let stemmer = Stemmer::new(dictionary);
                for i in 0..w.len() {
                    assert_eq!(thrown.get(i), "", "{at}: OpenNLP threw on {:?}", word.get(i));
                    let text = lossless(word.get(i));
                    compared += 1;
                    let mut got = stemmer.stems(&text);
                    if got.is_empty() {
                        got.push(text.clone());
                    }
                    if got != all[i] {
                        failures.push(format!("{at}: stemAll {text:?} -> {got:?}, OpenNLP {:?}", all[i]));
                    }
                    let one = stemmer.stem(&text).unwrap_or_else(|| text.clone());
                    if one != lossless(stem.get(i)) {
                        failures.push(format!("{at}: stem {text:?} -> {one:?}, OpenNLP {:?}", stem.get(i)));
                    }
                    let found = stemmer.analyze(&text);
                    if found != analyses[i] {
                        failures.push(format!("{at}: analyze {text:?} -> {found:?}, OpenNLP {:?}", analyses[i]));
                    }
                }
            }
        }
    }
    assert!(runs > 0, "no goldens under {}", goldens().display());
    assert!(
        failures.is_empty(),
        "{} of {compared} differ:\n{}",
        failures.len(),
        failures[..failures.len().min(60)].join("\n")
    );
    // Every word and every load error, when the published dictionaries
    // are present.
    if runs == 27 {
        assert_eq!(compared, 4263);
    }
}
