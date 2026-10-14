//! What the Rust port is expected to return for what the JVM returned,
//! where the two are not the same by design: the rules the crates' golden
//! tests apply before comparing, kept here so that the differential run
//! over generated inputs applies the same ones.
//!
//! - A span that cuts a surrogate pair (`turbo:utf16<L>` with `cut` 1) has
//!   no byte offset. The maxent tokenizer does not take that split and
//!   multiplies on (`joined_me_spans`); the simple, whitespace and UAX #29
//!   tokenizers give the whole character as one span
//!   (`whole_character_spans`).
//! - `SimpleTokenizer` with `keepNewLines` throws when a line break comes
//!   before the first token and gives a line break after whitespace the
//!   span of the whitespace; the port fixes both, so its expectation is
//!   worked out from the plain simple tokens (`simple_newline_spans`).
//! - `WordpieceEncoder` gives a piece the span of a nonspacing mark that
//!   accent stripping removed after it; upstream `tokenizers`, which the
//!   port follows, ends the piece before the mark (`without_trailing_marks`).
//! - A normalizer golden made by a diff of source and result has edits at
//!   the diff's granularity; two edit lists that change the same stretches
//!   into the same text are equivalent (`view_edits_equivalent`).

use turbo_unicode::GeneralCategory;
use turbo_unicode::category::general_category;
use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};

use crate::format::Golden;

/// One span as the JVM returned it, read from a golden's `opennlp:tokens`
/// (or `opennlp:sentences`) block with its `turbo:utf16<L>` block, its
/// confidence and its word type when the golden has them.
#[derive(Clone, Debug, PartialEq)]
pub struct JvmSpan {
    /// The UTF-8 span, widened to whole characters where it cuts one.
    pub utf8: (u32, u32),
    /// The JVM's exact UTF-16 span.
    pub utf16: (u32, u32),
    /// Whether it starts or ends between the two halves of a pair.
    pub cut: bool,
    pub prob: Option<f64>,
    pub kind: Option<String>,
}

/// The JVM's spans of the layer `id` of `golden`, with their UTF-16
/// offsets from `turbo:utf16<id>` where the golden has that block and
/// from the text otherwise, the confidence from `turbo:confidence<id>`
/// and the word type from `turbo:word-types`. Fails when the golden has
/// no such layer or a span is not on a character boundary without a
/// UTF-16 block saying so.
pub fn jvm_spans(golden: &Golden, text: &str, id: &str) -> Result<Vec<JvmSpan>, String> {
    let layer = golden.layer(id).ok_or_else(|| format!("no {id} layer"))?;
    let (starts, ends) = (layer.u32s("start").ok_or("a start column")?, layer.u32s("end").ok_or("an end column")?);
    let units = golden.layer(&format!("turbo:utf16<{id}>"));
    let prob = golden.layer(&format!("turbo:confidence<{id}>")).and_then(|l| l.f64s("value"));
    let kind = golden.layer("turbo:word-types").and_then(|l| l.strs("kind"));
    let unit = |b: u32| byte_to_utf16(text, b as usize).map(|u| u as u32).ok_or(format!("{b} cuts a character"));
    (0..layer.len())
        .map(|i| {
            let (utf16, cut) = match units {
                Some(u) => {
                    let s16 = u.u32s("start16").ok_or("a start16 column")?;
                    let e16 = u.u32s("end16").ok_or("an end16 column")?;
                    let cut = u.u32s("cut").ok_or("a cut column")?;
                    ((s16[i], e16[i]), cut[i] == 1)
                }
                None => ((unit(starts[i])?, unit(ends[i])?), false),
            };
            Ok(JvmSpan {
                utf8: (starts[i], ends[i]),
                utf16,
                cut,
                prob: prob.map(|p| p[i]),
                kind: kind.map(|k| k.get(i).to_string()),
            })
        })
        .collect()
}

/// The JVM's token spans (`opennlp:tokens`).
pub fn jvm_token_spans(golden: &Golden, text: &str) -> Result<Vec<JvmSpan>, String> {
    jvm_spans(golden, text, "opennlp:tokens")
}

/// What the Rust maxent tokenizer produces for the spans the JVM wrote: a
/// span whose end falls between the two halves of a pair is joined with
/// the span after it, whose start is the same point, and their
/// probabilities multiply, since the port does not take that split and
/// goes on multiplying the token's decisions. Gives (UTF-8 start, UTF-8
/// end, probability). Fails on spans that do not pair up.
pub fn joined_me_spans(text: &str, spans: &[JvmSpan]) -> Result<Vec<(u32, u32, f64)>, String> {
    let byte = |unit: u32| utf16_to_byte(text, unit as usize).map(|b| b as u32);
    let mut out = Vec::new();
    let mut open: Option<(u32, f64)> = None;
    for s in spans {
        let p = s.prob.ok_or("a confidence per span")?;
        let joined = open.is_some();
        let (start, prob) = match open.take() {
            Some((start, prob)) => {
                if byte(s.utf16.0).is_some() {
                    return Err("a joined span continues where the last one stopped".into());
                }
                (start, prob * p)
            }
            None => (byte(s.utf16.0).ok_or("a span start between two halves without a split before it")?, p),
        };
        match byte(s.utf16.1) {
            Some(end) => {
                if !s.cut && !joined && s.utf8 != (start, end) {
                    return Err(format!("the UTF-8 span {:?} disagrees with the UTF-16 one {:?}", s.utf8, s.utf16));
                }
                out.push((start, end, prob));
            }
            None => open = Some((start, prob)),
        }
    }
    if open.is_some() {
        return Err("a span ends between two halves at the end of the text".into());
    }
    Ok(out)
}

/// The spans a golden of a tokenizer that classifies UTF-16 units
/// expects of the port: two rows holding the halves of one character
/// outside the BMP become one span covering the character, with the first
/// row's type. Every other span is kept as it is.
pub fn whole_character_spans(text: &str, spans: &[JvmSpan]) -> Result<Vec<JvmSpan>, String> {
    let mut out = Vec::new();
    let mut it = spans.iter();
    while let Some(span) = it.next() {
        if !span.cut {
            out.push(span.clone());
            continue;
        }
        let (high, mid) = span.utf16;
        let other = it.next().filter(|s| s.cut).ok_or("a half span without its other half")?;
        let (mid2, low) = other.utf16;
        if !(mid == mid2 && mid == high + 1 && low == mid + 1) {
            return Err(format!("{:?} and {:?} are not the halves of one character", span.utf16, other.utf16));
        }
        let start = utf16_to_byte(text, high as usize).ok_or("the pair starts a character")? as u32;
        let end = utf16_to_byte(text, low as usize).ok_or("the pair ends a character")? as u32;
        if text[start as usize..end as usize].chars().count() != 1 {
            return Err(format!("{start}..{end} is not one character"));
        }
        if (span.utf8, other.utf8) != ((start, end), (start, end)) {
            return Err(format!("the rows {:?} and {:?} do not both hold the whole character", span.utf8, other.utf8));
        }
        out.push(JvmSpan {
            utf8: (start, end),
            utf16: (high, low),
            cut: false,
            prob: span.prob,
            kind: span.kind.clone(),
        });
    }
    Ok(out)
}

/// What `SimpleTokenizer` with `keepNewLines` should return, worked out
/// from the plain simple tokens of the same text (a line break is
/// whitespace, so never inside a token): those spans plus a one byte span
/// at every line feed and carriage return, in text order. The Java has
/// two defects in this mode (it throws when a line break comes before the
/// first token, and gives a line break after whitespace the span of the
/// whitespace), which the port fixes; where the Java output equals this
/// expectation, the two agree.
pub fn simple_newline_spans(text: &str, plain: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = plain.to_vec();
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' || b == b'\r' {
            out.push((i as u32, i as u32 + 1));
        }
    }
    out.sort_unstable();
    out
}

/// `span` with the nonspacing marks at its end left out: `WordpieceEncoder`
/// runs a piece's span over the marks accent stripping removed after it,
/// and the port does not. A span of marks alone is kept.
pub fn without_trailing_marks(text: &str, [start, end]: [u32; 2]) -> [u32; 2] {
    let piece = &text[start as usize..end as usize];
    let kept = piece.trim_end_matches(|c: char| general_category(c) == GeneralCategory::Mn);
    if kept.is_empty() { [start, end] } else { [start, start + kept.len() as u32] }
}

/// One edit of a normalized view as a golden row holds it: the source byte
/// span it replaces and the text it writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewEdit {
    pub start: usize,
    pub end: usize,
    pub with: String,
}

/// The source with the edits applied. The edits must be in text order
/// and inside the text.
pub fn apply_view_edits(text: &str, edits: &[ViewEdit]) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for e in edits {
        if e.start < at || e.end < e.start || e.end > text.len() {
            return Err(format!("the edit {}..{} is out of order or outside the text", e.start, e.end));
        }
        if !text.is_char_boundary(e.start) || !text.is_char_boundary(e.end) {
            return Err(format!("the edit {}..{} cuts a character", e.start, e.end));
        }
        out.push_str(&text[at..e.start]);
        out.push_str(&e.with);
        at = e.end;
    }
    out.push_str(&text[at..]);
    Ok(out)
}

/// Whether two edit lists change the same stretches of the source into
/// the same text, whatever their granularity. Edits of either list that
/// overlap or touch, directly or through edits of the other list, form a
/// region; the lists are equivalent when, for every region, the source
/// with the one list's edits applied equals the source with the other's.
/// This is the check for goldens whose edits were reconstructed by a
/// diff: the diff may cut one change into several rows or join two
/// changes that are a character apart, and the Rust edits may span an
/// unchanged character that the diff kept. Where a region's results
/// differ, the region is widened over the following ones until they
/// agree: a diff over repeated characters ("xxxẍ", "6/98/2635 4") may
/// delete one occurrence in one region and keep another in the next, so
/// the two lists agree only over both regions together.
pub fn view_edits_equivalent(text: &str, rust: &[ViewEdit], golden: &[ViewEdit]) -> bool {
    let mut all: Vec<(usize, usize)> = rust.iter().chain(golden).map(|r| (r.start, r.end)).collect();
    all.sort_unstable();
    let mut regions: Vec<(usize, usize)> = Vec::new();
    for (s, e) in all {
        match regions.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => regions.push((s, e)),
        }
    }
    let agree = |s: usize, e: usize| {
        let inside = |rows: &[ViewEdit]| -> Vec<ViewEdit> {
            rows.iter()
                .filter(|r| r.start >= s && r.end <= e)
                .map(|r| ViewEdit { start: r.start - s, end: r.end - s, with: r.with.clone() })
                .collect()
        };
        apply_view_edits(&text[s..e], &inside(rust)) == apply_view_edits(&text[s..e], &inside(golden))
    };
    let mut i = 0;
    while i < regions.len() {
        let start = regions[i].0;
        let mut end = regions[i].1;
        while !agree(start, end) {
            i += 1;
            match regions.get(i) {
                Some(&(_, next_end)) => end = next_end,
                None => return false,
            }
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(utf8: (u32, u32), utf16: (u32, u32), cut: bool, prob: f64) -> JvmSpan {
        JvmSpan { utf8, utf16, cut, prob: Some(prob), kind: None }
    }

    #[test]
    fn halves_of_a_pair_join_and_their_probabilities_multiply() {
        // "'😀." as TokenizerME splits it: ' | high | low | .
        let text = "'😀.";
        let spans = [
            span((0, 1), (0, 1), false, 0.9),
            span((1, 5), (1, 2), true, 0.5),
            span((1, 5), (2, 3), true, 0.5),
            span((5, 6), (3, 4), false, 0.8),
        ];
        assert_eq!(joined_me_spans(text, &spans).unwrap(), [(0, 1, 0.9), (1, 5, 0.25), (5, 6, 0.8)]);
        let whole = whole_character_spans(text, &spans).unwrap();
        assert_eq!(whole.iter().map(|s| s.utf8).collect::<Vec<_>>(), [(0, 1), (1, 5), (5, 6)]);
        assert_eq!(whole[1].utf16, (1, 3));
        assert!(!whole[1].cut);
        assert!(joined_me_spans(text, &spans[..2]).is_err());
        assert!(whole_character_spans(text, &spans[..2]).is_err());
    }

    #[test]
    fn newline_tokens_join_the_plain_tokens_in_order() {
        let text = "a\r\n b\n";
        assert_eq!(simple_newline_spans(text, &[(0, 1), (4, 5)]), [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]);
    }

    #[test]
    fn trailing_marks_leave_a_piece() {
        let text = "Cafe\u{301} x\u{301}\u{301}";
        assert_eq!(without_trailing_marks(text, [0, 6]), [0, 4]);
        assert_eq!(without_trailing_marks(text, [7, 12]), [7, 8]);
        assert_eq!(without_trailing_marks(text, [8, 12]), [8, 12]);
    }

    #[test]
    fn edits_of_another_granularity_are_equivalent_when_the_text_agrees() {
        let text = "aXYb";
        let e = |start, end, with: &str| ViewEdit { start, end, with: with.into() };
        let one = [e(1, 3, "xy")];
        let two = [e(1, 2, "x"), e(2, 3, "y")];
        assert!(view_edits_equivalent(text, &one, &two));
        assert!(!view_edits_equivalent(text, &one, &[e(1, 3, "xz")]));
        assert!(!view_edits_equivalent(text, &one, &[]));
        assert_eq!(apply_view_edits(text, &two).unwrap(), "axyb");
        assert!(apply_view_edits(text, &[e(2, 1, "")]).is_err());
        // A diff over repeated characters: one list deletes the first x and
        // inserts one before the last, the other changes only the marks.
        // Region by region they differ; over both regions they agree.
        let text = "\u{301}\u{323}xxxx\u{308}\u{301}\u{323}";
        let rust = [e(0, 4, "\u{323}\u{301}"), e(7, 14, "\u{1e8d}\u{323}\u{301}")];
        let golden = [e(0, 5, "\u{323}\u{301}"), e(8, 10, "\u{1e8d}"), e(10, 12, "\u{323}"), e(12, 14, "\u{301}")];
        assert!(view_edits_equivalent(text, &rust, &golden));
        assert_eq!(apply_view_edits(text, &rust), apply_view_edits(text, &golden));
        // Not when the texts differ after all.
        assert!(!view_edits_equivalent(text, &rust, &[e(0, 5, "\u{323}\u{301}"), e(8, 10, "\u{1e8d}")]));
    }
}
