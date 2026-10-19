//! Mojibake repair: a port of the encoding fix of ftfy 6.3.1 by Robyn
//! Speer (Apache-2.0; see NOTICE), `fix_encoding` and the steps under it,
//! with every fix recorded as an edit so offsets map through the repair.
//!
//! The text is checked with ftfy's badness heuristic; if it looks like
//! mojibake, it is encoded back to bytes with each of ftfy's single-byte
//! code pages in turn (Latin-1, the sloppy windows-125x pages,
//! ISO-8859-2, Mac OS Roman, code page 437) and decoded as UTF-8, with
//! ftfy's byte repairs in between (0xA0 bytes that became spaces, UTF-8
//! sequences that lost a byte to U+FFFD). When no code page works, UTF-8
//! mojibake is fixed piece by piece inside text that is otherwise fine,
//! and C1 controls are read as windows-1252. Steps repeat until the text
//! stops changing, which undoes double encoding.
//!
//! Every character of the result carries the span of the source text it
//! came from: encoding gives each byte the span of its character,
//! decoding gives each character the union of its bytes' spans. The
//! output is then written as one edit per run of characters that came
//! from one source run and differ from it: "Ã©" to "é" is one edit over
//! the two source characters, and unchanged text is kept as it is.
//!
//! What is not ported: the rest of ftfy's `fix_text` (HTML entities,
//! terminal escapes, quotes, width, ligatures, line breaks, surrogates,
//! control characters, normalization). Those are other normalizers here.

mod badness;
mod bytes;
mod classes;
mod fix;

pub use badness::is_bad;

use turbo_analysis::{Alignment, ViewBuilder};

/// A span of the source text, in bytes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Src {
    start: u32,
    end: u32,
}

impl Src {
    /// The span covering both.
    fn join(self, other: Src) -> Src {
        Src { start: self.start.min(other.start), end: self.end.max(other.end) }
    }
}

/// Characters, each with the source span it came from.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub(crate) struct Tracked {
    chars: Vec<char>,
    src: Vec<Src>,
}

impl Tracked {
    /// `text`, whose characters start at byte `base` of the source.
    fn new(text: &str, base: usize) -> Tracked {
        let mut t = Tracked::default();
        for (i, c) in text.char_indices() {
            let start = (base + i) as u32;
            t.push(c, Src { start, end: start + c.len_utf8() as u32 });
        }
        t
    }

    fn push(&mut self, c: char, src: Src) {
        self.chars.push(c);
        self.src.push(src);
    }

    fn extend(&mut self, other: Tracked) {
        self.chars.extend(other.chars);
        self.src.extend(other.src);
    }

    fn slice(&self, start: usize, end: usize) -> Tracked {
        Tracked { chars: self.chars[start..end].to_vec(), src: self.src[start..end].to_vec() }
    }
}

/// What one step of a repair did, in ftfy's terms.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// Encoded the text with a code page.
    Encode,
    /// Decoded the bytes with an encoding.
    Decode,
    /// Changed the bytes, or single characters, by a named fix.
    Transcode,
    /// Ran a named fix over parts of the text.
    Apply,
}

impl Action {
    /// ftfy's name for the action.
    pub fn name(self) -> &'static str {
        match self {
            Action::Encode => "encode",
            Action::Decode => "decode",
            Action::Transcode => "transcode",
            Action::Apply => "apply",
        }
    }
}

/// One step of a repair: the action and the code page, encoding or fix
/// it used, as ftfy explains its fixes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlanStep {
    pub action: Action,
    pub parameter: &'static str,
}

/// A repaired text with the alignment from the text it was made from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repaired {
    pub text: String,
    pub alignment: Alignment,
    /// The steps, in order, as ftfy's `fix_encoding_and_explain` lists
    /// them. Empty when nothing changed, except when a single step of
    /// ftfy's `fix_encoding_and_explain` decodes back to the input text
    /// (such as `replace_lossy_sequences` on `"\u{FFFD}\u{FFFD}"` or
    /// `fix_c1_controls` on the five C1 controls `0x81`, `0x8D`, `0x8F`,
    /// `0x90`, `0x9D` that Windows-1252 leaves unassigned).
    pub plan: Vec<PlanStep>,
}

/// ftfy's `fix_encoding` over the whole of `text`.
pub fn fix_encoding(text: &str) -> Repaired {
    let mut out = ViewBuilder::new(text);
    let mut plan = Vec::new();
    repair_into(text, 0, text.len(), &mut out, &mut plan);
    let (text, alignment) = out.finish();
    Repaired { text, alignment, plan }
}

/// The longest piece repaired at once, in characters: ftfy's default
/// `max_decode_length`.
const MAX_SEGMENT_CHARS: usize = 1_000_000;

/// Repairs `text` one line at a time, as ftfy's `fix_text` splits it (each
/// piece ends after a newline, or after a million characters), writing
/// through `out`, which must stand at offset 0 of `text`.
pub fn repair_lines(text: &str, out: &mut ViewBuilder<'_>) {
    let mut start = 0;
    while start < text.len() {
        let mut end = text[start..].find('\n').map_or(text.len(), |i| start + i + 1);
        if let Some((cut, _)) = text[start..end].char_indices().nth(MAX_SEGMENT_CHARS) {
            end = start + cut;
        }
        repair_into(text, start, end, out, &mut Vec::new());
        start = end;
    }
}

/// Repairs `text[start..end]` and writes it through `out`: unchanged runs
/// kept, each changed run one edit.
fn repair_into(text: &str, start: usize, end: usize, out: &mut ViewBuilder<'_>, plan: &mut Vec<PlanStep>) {
    let segment = &text[start..end];
    // Source spans are u32 offsets, as everywhere in a document; text
    // past that is left as it is.
    if segment.is_ascii() || u32::try_from(end).is_err() {
        return;
    }
    if out.position() < start {
        out.keep(start);
    }
    let fixed = fix::fix_encoding(Tracked::new(segment, start), plan);
    let mut i = 0;
    let mut piece = String::new();
    while i < fixed.chars.len() {
        let src = fixed.src[i];
        piece.clear();
        while i < fixed.chars.len() && fixed.src[i] == src {
            piece.push(fixed.chars[i]);
            i += 1;
        }
        let (s, e) = (src.start as usize, src.end as usize);
        debug_assert!(s >= out.position(), "sources come in order and never overlap");
        // A source run no output came from.
        if s > out.position() {
            out.replace(s, "");
        }
        if text[s..e] == piece {
            out.keep(e);
        } else {
            out.replace(e, &piece);
        }
    }
    if end > out.position() {
        out.replace(end, "");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::Span;

    #[test]
    fn each_fix_is_one_edit_over_its_source_characters() {
        let src = "cafÃ© and naÃ¯ve";
        let r = fix_encoding(src);
        assert_eq!(r.text, "café and naïve");
        let edits = r.alignment.edits();
        assert_eq!(edits.len(), 2);
        assert_eq!(&src[edits[0].parent.range()], "Ã©");
        assert_eq!(&r.text[edits[0].child.range()], "é");
        assert_eq!(&src[edits[1].parent.range()], "Ã¯");
        assert_eq!(
            r.plan,
            vec![
                PlanStep { action: Action::Encode, parameter: "latin-1" },
                PlanStep { action: Action::Decode, parameter: "utf-8" },
            ]
        );
        // "naïve" maps back to "naÃ¯ve".
        let naive = Span::new(10, 16);
        assert_eq!(&r.text[naive.range()], "naïve");
        assert_eq!(&src[r.alignment.span_to_parent(naive).range()], "naÃ¯ve");
    }

    #[test]
    fn clean_text_is_left_alone() {
        for text in ["", "plain", "café", "日本語"] {
            let r = fix_encoding(text);
            assert_eq!(r.text, text);
            assert!(r.alignment.edits().is_empty());
            assert!(r.plan.is_empty());
        }
    }

    #[test]
    fn lines_are_repaired_on_their_own() {
        let src = "naÃ¯ve\nplain â€œquotedâ€\u{9D}\n";
        let mut out = ViewBuilder::new(src);
        repair_lines(src, &mut out);
        let (text, alignment) = out.finish();
        assert_eq!(text, "naïve\nplain \u{201C}quoted\u{201D}\n");
        for e in alignment.edits() {
            assert!(!src[e.parent.range()].contains('\n'));
        }
    }
}
