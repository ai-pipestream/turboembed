//! Frequency segmentation with selectable Chinese and OpenNLP policies.

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;
use turbo_unicode::chars::is_ascii_alnum;

use super::hmm;
use super::lexicon::UnigramLexicon;
use crate::TokenizerOptions;
use crate::whitespace::whitespace_spans;
use turbo_unicode::script::{Script, script};

/// The buffers one segmentation reuses: the whitespace-free stretches of
/// the text, the characters of the current stretch, and the best score
/// and back pointer at each of its character boundaries.
#[derive(Debug, Default, Clone)]
pub struct Scratch {
    stretches: Vec<Span>,
    chars: Vec<char>,
    /// The byte offset in the stretch of each character, and the
    /// stretch's length after the last.
    offsets: Vec<u32>,
    best: Vec<f64>,
    previous: Vec<u32>,
    raw: Vec<Span>,
    hmm: hmm::Scratch,
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch::default()
    }
}

/// How dictionary segmentation handles runs that are not words.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum UnigramMode {
    /// Group ASCII alphanumerics and use the trained BMES model for
    /// adjacent single Han characters not forming a dictionary word.
    Chinese,
    /// Group ASCII alphanumerics, with no unknown-word model.
    #[default]
    Dictionary,
    /// Preserve OpenNLP's dictionary-only path, including ASCII splitting.
    OpenNlp,
}

/// The unigram frequency segmenter: a lexicon and the whitespace mode the
/// text is split in, plus an explicit unknown-word policy.
#[derive(Debug, Clone)]
pub struct UnigramSegmenter {
    pub(crate) lexicon: UnigramLexicon,
    pub(crate) whitespace: WhitespaceMode,
    pub(crate) mode: UnigramMode,
}

impl UnigramSegmenter {
    /// A segmenter over `lexicon` that splits the text at whitespace in
    /// `whitespace` mode. The lexicon keeps the mode its lines were split
    /// in; OpenNLP reads one process-wide mode for both, so the two are
    /// the same unless a caller wants otherwise.
    pub fn new(lexicon: UnigramLexicon, whitespace: WhitespaceMode) -> UnigramSegmenter {
        UnigramSegmenter { lexicon, whitespace, mode: UnigramMode::Dictionary }
    }

    /// Selects the segmentation policy. The default is [`UnigramMode::Dictionary`].
    pub fn with_mode(mut self, mode: UnigramMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn mode(&self) -> UnigramMode {
        self.mode
    }

    pub fn lexicon(&self) -> &UnigramLexicon {
        &self.lexicon
    }

    pub fn whitespace(&self) -> WhitespaceMode {
        self.whitespace
    }

    /// Appends the words of `text` to `out`, as byte spans of `text`, in
    /// text order. Whitespace is never part of a word.
    pub fn segment(&self, text: &str, out: &mut Vec<Span>) {
        self.segment_with(text, &mut Scratch::new(), out);
    }

    /// [`segment`](Self::segment) with the caller's buffers, so that
    /// repeated calls allocate nothing once the buffers have grown.
    pub fn segment_with(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<Span>) {
        scratch.stretches.clear();
        let options = TokenizerOptions { whitespace: self.whitespace, keep_newlines: false };
        whitespace_spans(text, options, &mut scratch.stretches);
        for i in 0..scratch.stretches.len() {
            let stretch = scratch.stretches[i];
            if self.mode == UnigramMode::OpenNlp {
                self.decode(text, stretch, scratch, out);
            } else {
                self.chinese(text, stretch, scratch, out);
            }
        }
    }

    fn chinese(&self, text: &str, stretch: Span, scratch: &mut Scratch, out: &mut Vec<Span>) {
        // ASCII alphanumerics are atomic even if individual letters or
        // digits occur in the lexicon. Non-ASCII spans retain dictionary DP.
        let part = &text[stretch.range()];
        // Find ASCII while collecting the characters the dictionary DP
        // already needs, without another scan of the common non-ASCII run.
        if !Self::prepare::<true>(part, scratch) {
            self.decode_run(text, stretch, scratch, out, true);
            return;
        }
        let ascii = |b: u8| is_ascii_alnum(char::from(b));
        let bytes = part.as_bytes();
        let mut start = 0;
        while start < bytes.len() {
            let grouped = ascii(bytes[start]);
            let len = bytes[start..].iter().take_while(|&&b| ascii(b) == grouped).count();
            let span = Span::new(stretch.start + start as u32, stretch.start + (start + len) as u32);
            if grouped {
                out.push(span);
            } else {
                self.decode_run(text, span, scratch, out, false);
            }
            start += len;
        }
    }

    fn decode_run(&self, text: &str, span: Span, scratch: &mut Scratch, out: &mut Vec<Span>, prepared: bool) {
        if !prepared {
            Self::prepare::<false>(&text[span.range()], scratch);
        }
        if self.mode != UnigramMode::Chinese {
            self.decode_prepared(span, scratch, out);
            return;
        }
        // Only the HMM needs the intermediate dictionary words. The other
        // modes append directly to the caller's output, without a copy.
        let mut raw = std::mem::take(&mut scratch.raw);
        raw.clear();
        self.decode_prepared(span, scratch, &mut raw);
        scratch.raw = raw;
        self.unknown_words(text, span.start, scratch, out);
    }

    fn unknown_words(&self, text: &str, base: u32, scratch: &mut Scratch, out: &mut Vec<Span>) {
        let mut i = 0;
        while i < scratch.raw.len() {
            let first = scratch.raw[i];
            let eligible = |s: Span| {
                let mut chars = text[s.range()].chars();
                let c = chars.next().unwrap();
                chars.next().is_none() && script(c) == Script::Han && hmm::observes(c)
            };
            if !eligible(first) {
                out.push(first);
                i += 1;
                continue;
            }
            let mut end = i + 1;
            while end < scratch.raw.len() && eligible(scratch.raw[end]) {
                end += 1;
            }
            let last = scratch.raw[end - 1];
            if end - i == 1 || self.lexicon.log_probability(&text[first.start as usize..last.end as usize]).is_some() {
                out.extend_from_slice(&scratch.raw[i..end]);
            } else {
                // Every raw word in this run is one character; locate the
                // corresponding reusable character and byte-offset slices.
                let a = scratch.offsets.binary_search(&(first.start - base)).unwrap();
                let b = a + end - i;
                hmm::cut(&scratch.chars[a..b], &scratch.offsets[a..=b], base, &mut scratch.hmm, out);
            }
            i = end;
        }
    }

    /// The best path through one whitespace-free stretch, by dynamic
    /// programming over its character boundaries, as the Java's `decode`
    /// does over its code unit positions.
    ///
    /// `best[k]` is the highest sum of scores of a segmentation of the
    /// stretch's first `k` characters, and `previous[k]` the character
    /// where its last word starts. From each character in order, the
    /// character alone is tried at the unknown score, then every lexicon
    /// word that starts there, shortest first; a boundary takes a new path
    /// only when its score is strictly higher, so of two equal paths the
    /// one whose last word starts earlier wins, as in the Java. The Java
    /// also visits the positions between the two halves of a surrogate
    /// pair, but never reaches them (the step over an unlisted character
    /// takes the whole pair, and a word is whole characters), so it skips
    /// them, and visiting only character boundaries changes nothing.
    fn decode(&self, text: &str, stretch: Span, scratch: &mut Scratch, out: &mut Vec<Span>) {
        Self::prepare::<false>(&text[stretch.range()], scratch);
        self.decode_prepared(stretch, scratch, out);
    }

    fn prepare<const CHECK_ASCII: bool>(part: &str, scratch: &mut Scratch) -> bool {
        let Scratch { chars, offsets, .. } = scratch;
        chars.clear();
        offsets.clear();
        let mut ascii = false;
        for (i, c) in part.char_indices() {
            chars.push(c);
            offsets.push(i as u32);
            ascii |= CHECK_ASCII && is_ascii_alnum(c);
        }
        offsets.push(part.len() as u32);
        ascii
    }

    fn decode_prepared(&self, stretch: Span, scratch: &mut Scratch, out: &mut Vec<Span>) {
        let Scratch { chars, offsets, best, previous, .. } = scratch;
        let n = chars.len();
        best.clear();
        best.resize(n + 1, f64::NEG_INFINITY);
        best[0] = 0.0;
        previous.clear();
        previous.resize(n + 1, 0);
        let trie = self.lexicon.trie();
        let unknown = self.lexicon.unknown_log_probability();
        for k in 0..n {
            let here = best[k];
            debug_assert!(here != f64::NEG_INFINITY, "every character boundary is reached");
            let fallback = here + unknown;
            if fallback > best[k + 1] {
                best[k + 1] = fallback;
                previous[k + 1] = k as u32;
            }
            let Some(mut step) = trie.first(chars[k]) else { continue };
            let mut end = k + 1;
            loop {
                if step.is_word() {
                    let total = here + trie.score(step);
                    if total > best[end] {
                        best[end] = total;
                        previous[end] = k as u32;
                    }
                }
                if end == n {
                    break;
                }
                let Some(next) = trie.next(step, chars[end]) else { break };
                step = next;
                end += 1;
            }
        }
        let first = out.len();
        let mut end = n;
        while end > 0 {
            let start = previous[end] as usize;
            out.push(Span { start: stretch.start + offsets[start], end: stretch.start + offsets[end] });
            end = start;
        }
        out[first..].reverse();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OpenNLP's miniature lexicon from `UnigramSegmenterTest`.
    const LEXICON: &str = "\u{6211} 5000 r\n\u{6765}\u{5230} 2000 v\n\u{5317}\u{4eac} 3000 ns\n\
        \u{6e05}\u{534e}\u{5927}\u{5b66} 800 nt\n\u{6e05}\u{534e} 400 ns\n\u{534e}\u{5927} 100 ns\n\
        \u{5927}\u{5b66} 1500 n\n\u{7684} 9000 uj\n";

    fn segmenter() -> UnigramSegmenter {
        let lexicon = UnigramLexicon::parse(LEXICON.as_bytes(), WhitespaceMode::Unicode).unwrap();
        UnigramSegmenter::new(lexicon, WhitespaceMode::Unicode).with_mode(UnigramMode::OpenNlp)
    }

    fn words<'a>(s: &UnigramSegmenter, text: &'a str) -> Vec<&'a str> {
        let mut out = Vec::new();
        s.segment(text, &mut out);
        out.iter().map(|span| &text[span.range()]).collect()
    }

    fn spans(s: &UnigramSegmenter, text: &str) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        s.segment(text, &mut out);
        out.iter().map(|span| (span.start, span.end)).collect()
    }

    // The cases of OpenNLP's UnigramSegmenterTest, with UTF-8 offsets
    // where the Java has UTF-16 ones (each of these characters is one
    // UTF-16 unit and three bytes).

    #[test]
    fn prefers_whole_words_over_fragments() {
        let text = "\u{6211}\u{6765}\u{5230}\u{5317}\u{4eac}\u{6e05}\u{534e}\u{5927}\u{5b66}";
        assert_eq!(
            words(&segmenter(), text),
            ["\u{6211}", "\u{6765}\u{5230}", "\u{5317}\u{4eac}", "\u{6e05}\u{534e}\u{5927}\u{5b66}"]
        );
        assert_eq!(spans(&segmenter(), text), [(0, 3), (3, 9), (9, 15), (15, 27)]);
    }

    #[test]
    fn unknown_characters_fall_back_to_single_characters() {
        let s = segmenter();
        assert_eq!(words(&s, "\u{6211}\u{7231}\u{5317}\u{4eac}"), ["\u{6211}", "\u{7231}", "\u{5317}\u{4eac}"]);
        assert_eq!(words(&s, "xyz"), ["x", "y", "z"]);
        assert_eq!(spans(&s, "xyz"), [(0, 1), (1, 2), (2, 3)]);
        let text = "\u{6211}\u{7231}\u{6e05}\u{534e}\u{5927}\u{5b66}";
        assert_eq!(spans(&s, text), [(0, 3), (3, 6), (6, 18)]);
    }

    #[test]
    fn whitespace_separates() {
        let s = segmenter();
        assert_eq!(words(&s, "\u{5317}\u{4eac} \u{5927}\u{5b66}"), ["\u{5317}\u{4eac}", "\u{5927}\u{5b66}"]);
        assert!(spans(&s, "  ").is_empty());
        assert!(spans(&s, "").is_empty());
        let text = "  \u{6211}\u{6765}\u{5230}\u{5317}\u{4eac}";
        assert_eq!(spans(&s, text), [(2, 5), (5, 11), (11, 17)]);
    }

    #[test]
    fn single_characters() {
        let s = segmenter();
        assert_eq!(spans(&s, "\u{6211}"), [(0, 3)]);
        assert_eq!(spans(&s, "\u{7231}"), [(0, 3)]);
    }

    #[test]
    fn an_unknown_supplementary_character_is_one_token() {
        let text = "\u{20bb7}\u{20bb7}";
        assert_eq!(spans(&segmenter(), text), [(0, 4), (4, 8)]);
    }

    #[test]
    fn words_outside_the_basic_multilingual_plane() {
        // U+20BB7 and U+20BBE are CJK Extension B, four bytes each; the
        // Java walks them as two code units each.
        let lexicon = UnigramLexicon::parse(
            "\u{20bb7}\u{20bbe} 5\n\u{20bb7} 1\n\u{20bbe}\u{91ce} 3\n".as_bytes(),
            WhitespaceMode::Unicode,
        )
        .unwrap();
        let s = UnigramSegmenter::new(lexicon, WhitespaceMode::Unicode).with_mode(UnigramMode::OpenNlp);
        // [U+20BB7][U+20BBE U+91CE] sums log 1 + log 3 - 2 log 9, above
        // [U+20BB7 U+20BBE][U+91CE] at log 5 + log 0.5 - 2 log 9.
        let text = "\u{20bb7}\u{20bbe}\u{91ce}\u{20bb7}";
        assert_eq!(spans(&s, text), [(0, 4), (4, 11), (11, 15)]);
        // An unlisted Extension B character beside listed ones is one word,
        // and a listed word is found after it.
        let text = "\u{20000}\u{20bb7}\u{20bbe}\u{2a6d6}";
        assert_eq!(spans(&s, text), [(0, 4), (4, 12), (12, 16)]);
    }

    #[test]
    fn a_long_lexicon_word_is_one_token() {
        // OpenNLP's test uses 20000 letters; the walk from every position
        // runs to the end of this text, so it is quadratic, in Java too.
        let word = "a".repeat(4_000);
        let lexicon = UnigramLexicon::parse(format!("{word} 1\n").as_bytes(), WhitespaceMode::Unicode).unwrap();
        let s = UnigramSegmenter::new(lexicon, WhitespaceMode::Unicode).with_mode(UnigramMode::OpenNlp);
        assert_eq!(spans(&s, &word), [(0, 4_000)]);
    }

    #[test]
    fn equal_scores_keep_the_path_whose_last_word_starts_first() {
        // count(AB) * count(C) = count(A) * count(BC), so [A][BC] and
        // [AB][C] score the same to the bit (the sums add the same two
        // values); [A][BC]'s last word starts at A's end, before C, so it
        // wins.
        let lexicon = UnigramLexicon::parse(b"ab 2\nc 3\na 3\nbc 2\n", WhitespaceMode::Unicode).unwrap();
        let s = UnigramSegmenter::new(lexicon, WhitespaceMode::Unicode).with_mode(UnigramMode::OpenNlp);
        let ab = s.lexicon().log_probability("ab").unwrap();
        let c = s.lexicon().log_probability("c").unwrap();
        let a = s.lexicon().log_probability("a").unwrap();
        let bc = s.lexicon().log_probability("bc").unwrap();
        assert_eq!((0.0 + ab) + c, (0.0 + a) + bc);
        assert_eq!(words(&s, "abc"), ["a", "bc"]);
    }

    #[test]
    fn the_whitespace_mode_decides_the_stretches() {
        let lexicon = UnigramLexicon::parse(LEXICON.as_bytes(), WhitespaceMode::Unicode).unwrap();
        let unicode = UnigramSegmenter::new(lexicon.clone(), WhitespaceMode::Unicode).with_mode(UnigramMode::OpenNlp);
        let legacy = UnigramSegmenter::new(lexicon, WhitespaceMode::Legacy).with_mode(UnigramMode::OpenNlp);
        // U+0085 is whitespace in Unicode mode only, U+001C in legacy mode only.
        assert_eq!(words(&unicode, "\u{6211}\u{85}\u{6211}"), ["\u{6211}", "\u{6211}"]);
        assert_eq!(words(&legacy, "\u{6211}\u{85}\u{6211}"), ["\u{6211}", "\u{85}", "\u{6211}"]);
        assert_eq!(words(&unicode, "\u{6211}\u{1c}\u{6211}"), ["\u{6211}", "\u{1c}", "\u{6211}"]);
        assert_eq!(words(&legacy, "\u{6211}\u{1c}\u{6211}"), ["\u{6211}", "\u{6211}"]);
    }

    #[test]
    fn scratch_reuse_gives_the_same_words() {
        let s = segmenter();
        let mut scratch = Scratch::new();
        let texts = ["\u{6211}\u{6765}\u{5230}\u{5317}\u{4eac}", "x", "\u{6e05}\u{534e}\u{5927}\u{5b66} \u{7684}", ""];
        for text in texts.iter().chain(texts.iter().rev()) {
            let mut fresh = Vec::new();
            s.segment(text, &mut fresh);
            let mut reused = vec![Span::new(7, 9)];
            s.segment_with(text, &mut scratch, &mut reused);
            assert_eq!(reused[0], Span::new(7, 9), "appends, keeping what was there");
            assert_eq!(&reused[1..], &fresh[..], "{text}");
        }
    }

    #[test]
    fn every_byte_is_covered_once_outside_whitespace() {
        let s = segmenter();
        let text = "a\u{e9}\u{6211}\u{1f600}\u{20bb7}\u{6765}\u{5230} \t\u{3000}\u{5927}\u{5b66}\u{7684}x\u{301}";
        let mut out = Vec::new();
        s.segment(text, &mut out);
        let mut at = 0usize;
        for span in &out {
            assert!(text.is_char_boundary(span.start as usize) && text.is_char_boundary(span.end as usize));
            assert!(span.start < span.end);
            assert!(text[at..span.start as usize].chars().all(|c| WhitespaceMode::Unicode.is_whitespace(c as u32)));
            assert!(!text[span.range()].chars().any(|c| WhitespaceMode::Unicode.is_whitespace(c as u32)));
            at = span.end as usize;
        }
        assert_eq!(at, text.len());
    }
}
