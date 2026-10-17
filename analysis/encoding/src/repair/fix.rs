//! ftfy's `fix_encoding_and_explain` and `_fix_encoding_one_step_and_explain`
//! over tracked text, step for step. Ported from ftfy 6.3.1 by Robyn
//! Speer (Apache-2.0; see NOTICE).

use super::badness::{Scan, is_bad_chars};
use super::bytes::{TrackedBytes, decode_utf8, has_altered_utf8, replace_lossy_sequences, restore_byte_a0};
use super::classes::{UTF8_CONTINUATION, UTF8_CONTINUATION_STRICT, UTF8_FIRST_OF_2, UTF8_FIRST_OF_3, UTF8_FIRST_OF_4};
use super::{Action, PlanStep, Tracked};
use crate::charmap::Charmap;
use crate::tables;
use turbo_encoding_layers::Encoding;

/// Fixes `t` until a step changes nothing, appending each step's plan.
pub(crate) fn fix_encoding(mut t: Tracked, plan: &mut Vec<PlanStep>) -> Tracked {
    loop {
        let (next, steps) = one_step(&t);
        plan.extend(steps);
        match next {
            Some(next) if next.chars != t.chars => t = next,
            _ => return t,
        }
    }
}

fn step(action: Action, parameter: &'static str) -> PlanStep {
    PlanStep { action, parameter }
}

/// One step: the text it makes (None for unchanged) and the plan steps.
fn one_step(t: &Tracked) -> (Option<Tracked>, Vec<PlanStep>) {
    if t.chars.is_empty() || t.chars.iter().all(char::is_ascii) || !is_bad_chars(&t.chars) {
        return (None, Vec::new());
    }
    // Text that was UTF-8 decoded with a single-byte code page: encode it
    // back and decode it as UTF-8.
    let mut possible = Vec::new();
    for map in Charmap::ALL {
        let Some(mut bytes) = encode(t, map) else { continue };
        possible.push(map);
        let mut steps = vec![step(Action::Encode, map.name())];
        if map != Charmap::MacRoman && has_altered_utf8(&bytes) {
            let replaced = restore_byte_a0(&bytes);
            if replaced.bytes != bytes.bytes {
                steps.push(step(Action::Transcode, "restore_byte_a0"));
                bytes = replaced;
            }
        }
        if map.is_sloppy() {
            let replaced = replace_lossy_sequences(&bytes);
            if replaced.bytes != bytes.bytes {
                steps.push(step(Action::Transcode, "replace_lossy_sequences"));
                bytes = replaced;
            }
        }
        let variants = bytes.bytes.iter().any(|&b| b == 0xED || b == 0xC0);
        if let Some(fixed) = decode_utf8(&bytes, variants) {
            steps.push(step(Action::Decode, if variants { "utf-8-variants" } else { "utf-8" }));
            return (Some(fixed), steps);
        }
    }
    // UTF-8 mojibake inside text that is otherwise fine.
    if has_utf8_clue(t) {
        let fixed = decode_inconsistent_utf8(t);
        if fixed.chars != t.chars {
            return (Some(fixed), vec![step(Action::Apply, "decode_inconsistent_utf8")]);
        }
    }
    // Latin-1 meant as windows-1252: C1 controls standing for its
    // punctuation. ftfy decodes with Python's strict windows-1252, which
    // has no character for 0x81, 0x8D, 0x8F, 0x90 and 0x9D.
    if possible.contains(&Charmap::Latin1)
        && let Some(fixed) = latin1_as_windows_1252(t)
        && fixed.chars != t.chars
    {
        return (Some(fixed), vec![step(Action::Encode, "latin-1"), step(Action::Decode, "windows-1252")]);
    }
    // ftfy's C1 fallback always records `transcode:fix_c1_controls` when a
    // C1 control is present, even when only the five unassigned windows-1252
    // bytes (0x81, 0x8D, 0x8F, 0x90, 0x9D) appear and the text is unchanged
    // (tested by ftfy_cases.tsv "don't leak our internal use of byte 0x1A").
    if t.chars.iter().any(|&c| is_c1(c)) {
        return (Some(fix_c1_controls(t)), vec![step(Action::Transcode, "fix_c1_controls")]);
    }
    (None, Vec::new())
}

fn is_c1(c: char) -> bool {
    ('\u{80}'..='\u{9F}').contains(&c)
}

fn encode(t: &Tracked, map: Charmap) -> Option<TrackedBytes> {
    let bytes = t.chars.iter().map(|&c| map.encode_char(c)).collect::<Option<Vec<u8>>>()?;
    Some(TrackedBytes { bytes, src: t.src.clone() })
}

fn latin1_as_windows_1252(t: &Tracked) -> Option<Tracked> {
    let table = tables::single_byte_table(Encoding::Windows1252).expect("windows-1252 is single-byte");
    let mut out = Tracked::default();
    for (&c, &src) in t.chars.iter().zip(&t.src) {
        let byte = u8::try_from(u32::from(c)).ok()?;
        let decoded = match byte {
            0x81 | 0x8D | 0x8F | 0x90 | 0x9D => return None,
            0x80..=0xFF => char::from_u32(u32::from(table[usize::from(byte - 0x80)]))?,
            _ => c,
        };
        out.push(decoded, src);
    }
    Some(out)
}

/// ftfy's `fix_c1_controls`: each C1 control as the windows-1252
/// character of its byte.
fn fix_c1_controls(t: &Tracked) -> Tracked {
    let mut out = Tracked::default();
    for (&c, &src) in t.chars.iter().zip(&t.src) {
        let fixed = if is_c1(c) { Charmap::SloppyWindows1252.decode_byte(c as u8) } else { c };
        out.push(fixed, src);
    }
    out
}

/// ftfy's `UTF8_DETECTOR_RE` at `i`, given the scan of the whole text:
/// the end of a run of characters that look like UTF-8 read as a
/// single-byte code page, or `i` for none. The run must not follow a
/// character that is itself a likely continuation byte.
fn utf8_clue_end(scan: &Scan<'_>, i: usize) -> usize {
    if i > 0 && scan.has(i - 1, UTF8_CONTINUATION_STRICT) {
        return i;
    }
    let mut j = i;
    loop {
        let continued = |from: usize, n: usize| (from..from + n).all(|k| scan.has(k, UTF8_CONTINUATION));
        if scan.has(j, UTF8_FIRST_OF_2) && continued(j + 1, 1) {
            j += 2;
        } else if scan.has(j, UTF8_FIRST_OF_3) && continued(j + 1, 2) {
            j += 3;
        } else if scan.has(j, UTF8_FIRST_OF_4) && continued(j + 1, 3) {
            j += 4;
        } else {
            return j;
        }
    }
}

fn has_utf8_clue(t: &Tracked) -> bool {
    let scan = Scan::new(&t.chars);
    (0..t.chars.len()).any(|i| utf8_clue_end(&scan, i) > i)
}

/// ftfy's `decode_inconsistent_utf8`: every run `UTF8_DETECTOR_RE`
/// finds, shorter than the text and bad by the heuristic, is fixed on
/// its own with the whole of `fix_encoding`.
fn decode_inconsistent_utf8(t: &Tracked) -> Tracked {
    let scan = Scan::new(&t.chars);
    let mut out = Tracked::default();
    let mut i = 0;
    while i < t.chars.len() {
        let end = utf8_clue_end(&scan, i);
        if end == i {
            out.push(t.chars[i], t.src[i]);
            i += 1;
            continue;
        }
        let run = t.slice(i, end);
        if end - i < t.chars.len() && is_bad_chars(&run.chars) {
            out.extend(fix_encoding(run, &mut Vec::new()));
        } else {
            out.extend(run);
        }
        i = end;
    }
    out
}
