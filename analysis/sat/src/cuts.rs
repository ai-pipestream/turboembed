//! From a boundary score per token to sentence spans, without the model:
//! the windows the tokens are read in, the stitching of the windows'
//! scores, the cuts wtpsplit makes from them, and the snapping and
//! trimming that keep every sentence on grapheme cluster boundaries.
//!
//! Every offset is a UTF-8 byte offset into the text. wtpsplit counts
//! code points; a cut after code point `i` is the byte offset after that
//! character, so the two agree on every text.

use turbo_analysis::Span;
use turbo_unicode::grapheme::grapheme_boundaries;
// Retain the public path used by callers while sharing one implementation.
pub use turbo_unicode::whitespace::is_python_space;

/// The tokens `[start, end)` one window reads, not counting the
/// template's special tokens around them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub start: u32,
    pub end: u32,
}

/// wtpsplit's windows over `tokens` tokens (`extract`): `block` tokens
/// from every `stride`-th token, the last window moved back so that it
/// ends at the last token, and nothing after it. A text shorter than
/// `block` is one window. `block` and `stride` are at least 1, and
/// `stride` at most `block`, so every token is in a window.
pub fn windows(tokens: u32, block: u32, stride: u32, out: &mut Vec<Window>) {
    if tokens == 0 || block == 0 || stride == 0 {
        return;
    }
    debug_assert!(stride <= block);
    let block = block.min(tokens);
    let mut j = 0;
    while j < tokens {
        let end = j.saturating_add(block);
        if end >= tokens {
            out.push(Window { start: tokens - block, end: tokens });
            return;
        }
        out.push(Window { start: j, end });
        j = j.saturating_add(stride);
    }
}

/// Gives the bare word start piece the tokenizer puts before the first
/// word of a text (`lone`, the SentencePiece U+2581 alone, when no piece
/// starts with the word's first characters) the first character of the
/// text as its span. The bundle's tokenizer gives that piece an empty
/// span, as it stands for no character of the text; the upstream
/// tokenizer gives it the character the prefix was put before, which is
/// where wtpsplit reads its score. A bare piece elsewhere stands for a
/// space and keeps its span.
pub fn place_prefix(text: &str, lone: i32, ids: &[i32], spans: &mut [[u32; 2]]) {
    debug_assert_eq!(ids.len(), spans.len());
    for (&id, span) in ids.iter().zip(spans.iter_mut()) {
        let [start, end] = *span;
        if id == lone && start == end {
            let first = text.get(start as usize..).and_then(|s| s.chars().next()).map_or(0, char::len_utf8) as u32;
            *span = [start, start + first];
        }
    }
}

/// The logistic function in f32, as wtpsplit computes it from its
/// stitched scores (`1 / (1 + exp(-x))`).
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The byte offsets a sentence may end at: for each token whose score's
/// probability is over `threshold`, the end of its span, which is the
/// end of its last character. A token is put at the end of its span, as
/// wtpsplit puts its score on its last character; when two tokens end at
/// one offset, the later one's score is the one read there. A token with
/// an empty span covers no character and is left out. The probability
/// is compared with the threshold in f32, as NumPy 2 compares wtpsplit's
/// float32 probabilities with a Python float. `scores` holds
/// one stitched score per token, `spans` the tokens' spans in text
/// order. The result is in text order, each offset once.
pub fn boundaries(spans: &[[u32; 2]], scores: &[f32], threshold: f64, out: &mut Vec<u32>) {
    debug_assert_eq!(spans.len(), scores.len());
    let threshold = threshold as f32;
    for (k, (&[start, end], &score)) in spans.iter().zip(scores).enumerate() {
        if end <= start {
            continue;
        }
        // A later token ending here overrides this one.
        if spans[k + 1..].iter().take_while(|s| s[0] < end).any(|s| s[1] == end && s[1] > s[0]) {
            continue;
        }
        if sigmoid(score) > threshold {
            out.push(end);
        }
    }
    out.sort_unstable();
    out.dedup();
}

/// The cuts wtpsplit makes at `boundaries` (`indices_to_sentences`): each
/// boundary moved past the whitespace that follows it, so a sentence
/// keeps the whitespace after it. In text order, each once, and none at
/// 0 or at the end of the text.
pub fn cut_points(text: &str, boundaries: &[u32], out: &mut Vec<u32>) {
    let len = text.len() as u32;
    let mut last = 0;
    for &b in boundaries {
        let mut at = b.max(last) as usize;
        if at > b as usize {
            // Inside whitespace an earlier cut already passed over.
            continue;
        }
        let Some(tail) = text.get(at..) else {
            continue;
        };
        at += tail.chars().take_while(|&c| is_python_space(c)).map(char::len_utf8).sum::<usize>();
        let at = at as u32;
        if at > last && at < len {
            out.push(at);
        }
        last = at;
    }
}

/// The text's extended grapheme cluster boundaries (UAX #29), 0 and the
/// end of the text included, in order.
pub fn grapheme_index(text: &str) -> Vec<u32> {
    grapheme_boundaries(text).map(|b| b as u32).collect()
}

/// Moves each cut forward to the next grapheme cluster boundary, so no
/// sentence ends inside a cluster: after a base letter whose combining
/// mark the tokenizer put in the next token, or inside an emoji
/// sequence. `graphemes` is `grapheme_index` of the text. Cuts that meet
/// are kept once.
pub fn snap(graphemes: &[u32], cuts: &mut Vec<u32>) {
    let len = *graphemes.last().unwrap_or(&0);
    for c in cuts.iter_mut() {
        let i = graphemes.partition_point(|&g| g < *c);
        *c = graphemes.get(i).copied().unwrap_or(len);
    }
    cuts.dedup();
    cuts.retain(|&c| c > 0 && c < len);
}

/// The pieces of text between the cuts, as wtpsplit's `split` returns
/// them: each run between two cuts, then, when `split_on_newlines`, each
/// piece but the last without one trailing line feed, and every piece
/// split at each line feed it still holds (the line feeds belong to no
/// piece). Splitting can give empty pieces, which are kept, as wtpsplit
/// keeps them; a text with no cuts is one piece, or none when it is
/// empty.
pub fn pieces(text: &str, cuts: &[u32], split_on_newlines: bool, out: &mut Vec<Span>) {
    let len = text.len() as u32;
    let mut runs = Vec::with_capacity(cuts.len() + 1);
    let mut start = 0;
    for &c in cuts.iter().chain(std::iter::once(&len)) {
        let c = c.min(len);
        if c > start && text.is_char_boundary(c as usize) {
            runs.push(Span::new(start, c));
            start = c;
        }
    }
    if !split_on_newlines {
        out.extend(runs);
        return;
    }
    let bytes = text.as_bytes();
    let n = runs.len();
    for (i, run) in runs.into_iter().enumerate() {
        let mut end = run.end;
        if i + 1 < n && bytes[end as usize - 1] == b'\n' {
            end -= 1;
        }
        let mut from = run.start;
        for (k, &b) in bytes[run.start as usize..end as usize].iter().enumerate() {
            if b == b'\n' {
                let at = run.start + k as u32;
                out.push(Span::new(from, at));
                from = at + 1;
            }
        }
        out.push(Span::new(from, end));
    }
}

/// `span` without the grapheme clusters of whitespace at either end; None
/// when nothing else is left. A cluster is whitespace when every
/// character in it is (`is_python_space`), so a space that a combining
/// mark extends stays, and the span's own ends count as cluster
/// boundaries, so a carriage return cut from its line feed is trimmed.
pub fn trim(text: &str, graphemes: &[u32], span: Span) -> Option<Span> {
    if span.start >= span.end || text.get(span.start as usize..span.end as usize).is_none() {
        return None;
    }
    let blank = |a: u32, b: u32| text.get(a as usize..b as usize).is_some_and(|s| s.chars().all(is_python_space));
    // The cluster boundaries strictly inside the span.
    let lo = graphemes.partition_point(|&g| g <= span.start);
    let hi = graphemes.partition_point(|&g| g < span.end);
    let inner = &graphemes[lo..hi.max(lo)];
    let mut start = span.start;
    let mut k = 0;
    loop {
        let next = inner.get(k).copied().unwrap_or(span.end);
        if !blank(start, next) {
            break;
        }
        if next == span.end {
            return None;
        }
        start = next;
        k += 1;
    }
    let mut end = span.end;
    let mut k = inner.len();
    loop {
        let prev = if k == 0 { span.start } else { inner[k - 1] };
        if prev < start || !blank(prev, end) {
            break;
        }
        end = prev;
        k -= 1;
    }
    Some(Span::new(start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win(tokens: u32, block: u32, stride: u32) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        windows(tokens, block, stride, &mut out);
        out.into_iter().map(|w| (w.start, w.end)).collect()
    }

    #[test]
    fn windows_step_by_the_stride_and_the_last_ends_at_the_end() {
        assert_eq!(win(0, 4, 2), []);
        assert_eq!(win(5, 0, 2), []);
        assert_eq!(win(5, 2, 0), []);
        assert_eq!(win(3, 4, 2), [(0, 3)]);
        assert_eq!(win(4, 4, 2), [(0, 4)]);
        assert_eq!(win(5, 4, 2), [(0, 4), (1, 5)]);
        assert_eq!(win(10, 4, 2), [(0, 4), (2, 6), (4, 8), (6, 10)]);
        assert_eq!(win(10, 4, 4), [(0, 4), (4, 8), (6, 10)]);
        assert_eq!(win(9, 4, 3), [(0, 4), (3, 7), (5, 9)]);
        assert_eq!(win(7, 1, 1), [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]);
    }

    #[test]
    fn every_token_is_in_a_window() {
        for tokens in 0..40 {
            for block in 1..12 {
                for stride in 1..=block {
                    let mut seen = vec![0u32; tokens as usize];
                    for (a, b) in win(tokens, block, stride) {
                        assert!(b - a == block.min(tokens), "{tokens} {block} {stride}");
                        for s in &mut seen[a as usize..b as usize] {
                            *s += 1;
                        }
                    }
                    assert!(seen.iter().all(|&s| s > 0), "{tokens} {block} {stride}");
                }
            }
        }
    }

    #[test]
    fn a_boundary_is_the_end_of_a_token_over_the_threshold() {
        let spans = [[0, 3], [4, 7], [7, 8], [8, 8], [9, 12]];
        let scores = [5.0, -5.0, 0.0, 9.0, 3.0];
        let mut out = Vec::new();
        boundaries(&spans, &scores, 0.5, &mut out);
        // sigmoid(0) is 0.5, not over 0.5; the empty token covers nothing.
        assert_eq!(out, [3, 12]);
        out.clear();
        boundaries(&spans, &scores, 0.4, &mut out);
        assert_eq!(out, [3, 8, 12]);
    }

    #[test]
    fn the_threshold_is_compared_in_f32() {
        // A threshold just under a probability in f64 that rounds to it
        // in f32 is not passed, as in wtpsplit.
        let p = sigmoid(-2.0);
        let threshold = f64::from(p) - 1e-12;
        assert_eq!(threshold as f32, p);
        let mut out = Vec::new();
        boundaries(&[[0, 1]], &[-2.0], threshold, &mut out);
        assert!(out.is_empty());
        boundaries(&[[0, 1]], &[-2.0], f64::from(p) - 1e-6, &mut out);
        assert_eq!(out, [1]);
    }

    #[test]
    fn the_later_of_two_tokens_ending_together_is_read() {
        let spans = [[0, 4], [2, 4], [5, 6]];
        let mut out = Vec::new();
        boundaries(&spans, &[9.0, -9.0, -9.0], 0.5, &mut out);
        assert_eq!(out, Vec::<u32>::new());
        boundaries(&spans, &[-9.0, 9.0, -9.0], 0.5, &mut out);
        assert_eq!(out, [4]);
    }

    #[test]
    fn the_prefix_piece_is_read_on_the_first_character() {
        // "▁", "東京", "▁", "é", "a", "▁": 2 is the bare piece.
        let t = "東京 éa ";
        let ids = [2, 7, 2, 8, 9, 2];
        let mut spans = [[0, 0], [0, 6], [6, 7], [7, 9], [9, 10], [10, 11]];
        place_prefix(t, 2, &ids, &mut spans);
        assert_eq!(spans, [[0, 3], [0, 6], [6, 7], [7, 9], [9, 10], [10, 11]]);
        // An empty span at the end of the text stays empty.
        let mut spans = [[1, 1]];
        place_prefix("a", 2, &[2], &mut spans);
        assert_eq!(spans, [[1, 1]]);
    }

    fn cuts(text: &str, b: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        cut_points(text, b, &mut out);
        out
    }

    #[test]
    fn a_cut_takes_the_whitespace_after_it() {
        let t = "One. Two.  Three.";
        assert_eq!(cuts(t, &[4, 9]), [5, 11]);
        // A boundary inside whitespace already passed over adds nothing.
        assert_eq!(cuts(t, &[4, 5, 9, 10]), [5, 11]);
        // Nothing at the end of the text.
        assert_eq!(cuts(t, &[17]), Vec::<u32>::new());
        assert_eq!(cuts("A.  ", &[2]), Vec::<u32>::new());
        // The information separators are whitespace to Python, as is NEL.
        assert_eq!(cuts("a\u{1f}b\u{85}c", &[1, 3]), [2, 5]);
    }

    fn pcs<'a>(text: &'a str, c: &[u32], nl: bool) -> Vec<&'a str> {
        let mut out = Vec::new();
        pieces(text, c, nl, &mut out);
        out.iter().map(|s| &text[s.start as usize..s.end as usize]).collect()
    }

    #[test]
    fn pieces_are_split_at_line_feeds_as_wtpsplit_splits_them() {
        assert_eq!(pcs("", &[], true), Vec::<&str>::new());
        assert_eq!(pcs("a b", &[], true), ["a b"]);
        assert_eq!(pcs("One. Two.", &[5], true), ["One. ", "Two."]);
        // One trailing line feed of a piece before the last goes; the last
        // piece's stays and leaves an empty piece.
        assert_eq!(pcs("a\nb\n", &[2], true), ["a", "b", ""]);
        assert_eq!(pcs("a\n\nb", &[3], true), ["a", "", "b"]);
        assert_eq!(pcs("x\r\ny", &[], true), ["x\r", "y"]);
        assert_eq!(pcs("a\nb", &[], false), ["a\nb"]);
    }

    #[test]
    fn a_cut_inside_a_cluster_moves_to_its_end() {
        // e and a combining acute, a family emoji of seven code points.
        let t = "Cafe\u{301} ok 👨\u{200d}👩\u{200d}👧 x";
        let g = grapheme_index(t);
        let mut c = vec![4, 14, 21];
        snap(&g, &mut c);
        assert_eq!(c, [6, 28]);
        for &x in &c {
            assert!(t.is_char_boundary(x as usize));
        }
        // A cut that snaps to the end of the text is dropped.
        let mut c = vec![1];
        snap(&grapheme_index("e\u{301}"), &mut c);
        assert!(c.is_empty());
    }

    fn tr(text: &str, a: u32, b: u32) -> Option<&str> {
        let g = grapheme_index(text);
        trim(text, &g, Span { start: a, end: b }).map(|s| &text[s.start as usize..s.end as usize])
    }

    #[test]
    fn trimming_takes_whole_clusters_of_whitespace() {
        assert_eq!(tr("  a b  ", 0, 7), Some("a b"));
        assert_eq!(tr(" \t\n ", 0, 4), None);
        assert_eq!(tr("", 0, 0), None);
        // A space a combining mark extends is one cluster, not blank.
        assert_eq!(tr(" \u{301}a ", 0, 5), Some(" \u{301}a"));
        // A carriage return cut from its line feed is trimmed.
        assert_eq!(tr("ab\r\ncd", 0, 3), Some("ab"));
        assert_eq!(tr("ab\r\ncd", 4, 6), Some("cd"));
        assert_eq!(tr("x\u{2028}", 0, 4), Some("x"));
        // Out-of-bounds, inverted, or mid-character spans return None.
        assert_eq!(tr("é", 0, 1), None);
        assert_eq!(tr("ab", 2, 1), None);
        assert_eq!(tr("ab", 0, 5), None);
    }

    #[test]
    fn snap_and_pieces_and_cut_points_guard_invalid_offsets() {
        let mut c = vec![1, 5];
        snap(&[], &mut c);
        assert!(c.is_empty());

        let t = "é. ok";
        let g = grapheme_index(t);
        let mut c = vec![1, 99];
        snap(&g, &mut c);
        assert_eq!(c, [2]);

        // Mid-codepoint and out-of-bounds boundaries are ignored by cut_points.
        assert_eq!(cuts(t, &[1, 3, 99]), [4]);

        // Unsorted, duplicate, mid-codepoint, or out-of-bounds cuts in pieces do not panic.
        assert_eq!(pcs(t, &[0, 1, 4, 2, 99], true), ["é. ", "ok"]);

        let mut spans = [[99, 99], [1, 1]];
        place_prefix(t, 2, &[2, 2], &mut spans);
        assert_eq!(spans, [[99, 99], [1, 1]]);
    }
}
