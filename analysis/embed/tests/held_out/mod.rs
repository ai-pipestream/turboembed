//! The held-out documents the device tests calibrate on.

use std::path::Path;

use turbo_analysis_embed::calibrate::Text;

/// The documents of tests/data/calibration-texts.txt: a sentence a line,
/// a blank line between documents.
pub fn held_out() -> Vec<Text> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/calibration-texts.txt");
    let all = std::fs::read_to_string(path).unwrap();
    let mut out = Vec::new();
    for doc in all.split("\n\n") {
        let (mut text, mut spans) = (String::new(), Vec::new());
        for line in doc.lines().filter(|l| !l.is_empty()) {
            if !text.is_empty() {
                text.push(' ');
            }
            let a = text.len() as u32;
            text.push_str(line);
            spans.push((a, text.len() as u32));
        }
        out.push((text, spans));
    }
    out
}
