//! Checks the checked in goldens' spans against their inputs: every span of
//! a text anchored layer must fall on character boundaries of the input's
//! UTF-8 bytes, in order and without overlap, and must not start or end in
//! whitespace (the sentence detector trims, the tokenizer splits on it).
//! Spans over emoji and CJK Extension B characters are checked by text.
//!
//! This reads only the parts of the format it needs; the full reader is the
//! `turbo-goldens` crate.

use std::fs;
use std::path::{Path, PathBuf};
use turbo_goldens_tool::paths;

/// One golden's input id and the spans of each text anchored layer.
struct Spans {
    input: String,
    layers: Vec<(String, Vec<(usize, usize)>)>,
}

fn read_spans(path: &Path) -> Spans {
    let text = fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let mut input = None;
    let mut layers = Vec::new();
    while let Some(line) = lines.next() {
        let fields: Vec<&str> = line.split('\t').collect();
        match fields[0] {
            "input" => input = Some(fields[1].to_owned()),
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
            _ => {}
        }
    }
    Spans { input: input.expect("an input line"), layers }
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
            for (layer, list) in &spans.layers {
                let mut previous_end = 0;
                for (i, &(s, e)) in list.iter().enumerate() {
                    let at = format!("{} {layer} item {i} [{s}, {e})", path.display());
                    assert!(previous_end <= s && s < e && e <= text.len(), "{at}");
                    assert!(text.is_char_boundary(s) && text.is_char_boundary(e), "{at}");
                    let slice = &text[s..e];
                    assert_eq!(slice.trim(), slice, "{at}");
                    previous_end = e;
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
