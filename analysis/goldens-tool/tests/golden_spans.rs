//! Checks the checked in goldens' spans against their inputs: every span of
//! a text anchored layer must fall on character boundaries of the input's
//! UTF-8 bytes, in order and without overlap, and must not start or end in
//! whitespace (the sentence detectors trim, the tokenizers split on it).
//! A row whose UTF-16 span cuts a surrogate pair (its `turbo:utf16<L>` row
//! has `cut` 1) holds the whole character, so it overlaps its neighbour
//! over that character. Three kinds of run are let off part of it, each for
//! what OpenNLP does there: a tokenizer that keeps line breaks makes them
//! tokens (and the simple tokenizer gives some of them the wrong span, out
//! of order), a run in the legacy whitespace mode does not count every
//! Unicode space as whitespace, and the tokenizer with an abbreviation
//! dictionary emits a listed token whole and then goes on splitting inside
//! it, so its spans overlap. Spans over emoji and CJK Extension B characters
//! are checked by text.
//!
//! This reads only the parts of the format it needs; the full reader is the
//! `turbo-goldens` crate.

use std::fs;
use std::path::{Path, PathBuf};
use turbo_goldens_tool::paths;

/// The rows of a `turbo:utf16<L>` layer: `(start16, end16, cut)`.
type Utf16Rows = Vec<(usize, usize, bool)>;

/// One golden's input id, the spans of each text anchored layer, and the
/// rows of each `turbo:utf16<L>` layer: the layer it is aligned to, then
/// `(start16, end16, cut)` per row.
struct Spans {
    input: String,
    /// The `config` records, key and value.
    config: Vec<(String, String)>,
    layers: Vec<(String, Vec<(usize, usize)>)>,
    utf16: Vec<(String, Utf16Rows)>,
}

impl Spans {
    fn has_config(&self, key: &str, value: &str) -> bool {
        self.config.iter().any(|(k, v)| k == key && v == value)
    }

    /// The UTF-16 rows aligned to `layer`, if the golden has them.
    fn utf16_of(&self, layer: &str) -> Option<&[(usize, usize, bool)]> {
        self.utf16.iter().find(|(l, _)| l == layer).map(|(_, rows)| rows.as_slice())
    }
}

fn read_spans(path: &Path) -> Spans {
    let text = fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let mut input = None;
    let mut config = Vec::new();
    let mut layers = Vec::new();
    let mut utf16 = Vec::new();
    while let Some(line) = lines.next() {
        let fields: Vec<&str> = line.split('\t').collect();
        match fields[0] {
            "input" => input = Some(fields[1].to_owned()),
            "config" => config.push((fields[1].to_owned(), fields[2].to_owned())),
            "layer" if fields[2] == "text" => {
                assert_eq!(&fields[4..6], ["start:u32", "end:u32"], "{}", path.display());
                let rows: usize = fields[3].parse().unwrap();
                let spans = (0..rows)
                    .map(|_| {
                        let row: Vec<&str> = lines.next().unwrap().split('\t').collect();
                        (row[0].parse().unwrap(), row[1].parse().unwrap())
                    })
                    .collect();
                assert_eq!(lines.next().unwrap(), format!("end\t{}", fields[1]));
                layers.push((fields[1].to_owned(), spans));
            }
            "layer" if fields[1].starts_with("turbo:utf16<") => {
                assert_eq!(&fields[4..], ["start16:u32", "end16:u32", "cut:u32"], "{}", path.display());
                let aligned = fields[2].strip_prefix("aligned:").expect("an aligned layer");
                let rows: usize = fields[3].parse().unwrap();
                let spans = (0..rows)
                    .map(|_| {
                        let row: Vec<&str> = lines.next().unwrap().split('\t').collect();
                        (row[0].parse().unwrap(), row[1].parse().unwrap(), row[2] == "1")
                    })
                    .collect();
                assert_eq!(lines.next().unwrap(), format!("end\t{}", fields[1]));
                utf16.push((aligned.to_owned(), spans));
            }
            _ => {}
        }
    }
    Spans { input: input.expect("an input line"), config, layers, utf16 }
}

fn goldens(annotator: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![paths::data_dir().join(annotator)];
    while let Some(dir) = stack.pop() {
        for e in fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == paths::GOLDEN_EXTENSION) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn input_text(id: &str) -> String {
    fs::read_to_string(paths::inputs_dir().join(format!("{id}.txt"))).unwrap()
}

/// The texts of one annotator's layer for one input.
fn texts(annotator: &str, input: &str) -> Vec<String> {
    let path = goldens(annotator).into_iter().find(|p| p.ends_with(format!("{input}.golden"))).unwrap();
    let spans = read_spans(&path);
    let text = input_text(&spans.input);
    spans.layers[0].1.iter().map(|&(s, e)| text[s..e].to_owned()).collect()
}

#[test]
fn every_span_lands_on_characters_of_its_input() {
    let mut checked = 0;
    for annotator in ["sentdetect", "tokenize"] {
        let files = goldens(annotator);
        assert!(files.len() >= 20, "{annotator}: {} goldens", files.len());
        for path in files {
            let spans = read_spans(&path);
            let text = input_text(&spans.input);
            let keeps_new_lines = spans.has_config("keepNewLines", "true");
            let legacy = spans.has_config("whitespace", "legacy");
            let abbreviations = path.iter().any(|p| {
                let dir = p.to_string_lossy();
                dir == "tokenizer-abbreviations" || dir.ends_with("+english-dictionary")
            });
            for (layer, list) in &spans.layers {
                let cut = |i: usize| spans.utf16_of(layer).is_some_and(|rows| rows[i].2);
                let (mut previous_start, mut previous_end) = (0, 0);
                for (i, &(s, e)) in list.iter().enumerate() {
                    let at = format!("{} {layer} item {i} [{s}, {e})", path.display());
                    // Two rows that share a cut character overlap over it.
                    let after = if i > 0 && cut(i) && cut(i - 1) { previous_start } else { previous_end };
                    assert!(s < e && e <= text.len(), "{at}");
                    if !keeps_new_lines {
                        let after = if abbreviations { previous_start } else { after };
                        assert!(after <= s, "{at}");
                    }
                    assert!(text.is_char_boundary(s) && text.is_char_boundary(e), "{at}");
                    let slice = &text[s..e];
                    if !keeps_new_lines && !legacy {
                        assert_eq!(slice.trim(), slice, "{at}");
                    }
                    (previous_start, previous_end) = (s, e);
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 500, "{checked}");
}

#[test]
fn tokens_over_emoji_and_supplementary_characters_hold_them_whole() {
    let tokens = texts("tokenize", "en/emoji");
    for t in ["🎉", "🍕", "👍🏽", "❤️", "🇺🇸"] {
        assert!(tokens.iter().any(|x| x == t), "{t} is not a token of en/emoji: {tokens:?}");
    }
    let tokens = texts("tokenize", "multi/mixed");
    for t in ["東京", "𝐀", "𝔅", "👋🏽"] {
        assert!(tokens.iter().any(|x| x == t), "{t} is not a token of multi/mixed: {tokens:?}");
    }
    assert!(tokens.iter().any(|x| x.contains("𠮷")), "{tokens:?}");
    assert!(tokens.iter().any(|x| x.contains("𠀀")), "{tokens:?}");
}

#[test]
fn sentences_of_a_text_with_emoji_start_and_end_on_its_words() {
    let sentences = texts("sentdetect", "en/emoji");
    assert_eq!(sentences.first().map(String::as_str), Some("We won the match!"));
    assert!(sentences.iter().any(|s| s.ends_with("😀.")), "{sentences:?}");
}

#[test]
fn tokens_that_cut_an_emoji_hold_it_whole_and_keep_their_utf16_spans() {
    let path = goldens("tokenize").into_iter().find(|p| p.ends_with("en/emoji-split.golden")).unwrap();
    let spans = read_spans(&path);
    let text = input_text(&spans.input);
    let (layer, tokens) = &spans.layers[0];
    let utf16 = spans.utf16_of(layer).expect("a turbo:utf16<opennlp:tokens> layer");
    assert_eq!(utf16.len(), tokens.len());
    let units: Vec<u16> = text.encode_utf16().collect();
    let mut cut = 0;
    for (i, (&(s, e), &(s16, e16, c))) in tokens.iter().zip(utf16).enumerate() {
        // Every UTF-16 span is the JVM's own, and it cuts exactly when an end
        // falls on a low surrogate.
        let inside = |at: usize| at < units.len() && (0xdc00..0xe000).contains(&units[at]);
        assert_eq!(c, inside(s16) || inside(e16), "token {i}");
        // The byte span is the UTF-16 span widened to whole characters.
        let (w16, x16) = (s16 - usize::from(inside(s16)), e16 + usize::from(inside(e16)));
        assert_eq!(String::from_utf16(&units[w16..x16]).unwrap(), text[s..e], "token {i}");
        cut += usize::from(c);
    }
    // Three emoji, each split between two tokens.
    assert_eq!(cut, 6);
    let texts: Vec<&str> = tokens.iter().map(|&(s, e)| &text[s..e]).collect();
    assert_eq!(texts.iter().filter(|t| **t == "😀").count(), 5, "{texts:?}");
    assert!(texts.contains(&"&😀"), "{texts:?}");
}
