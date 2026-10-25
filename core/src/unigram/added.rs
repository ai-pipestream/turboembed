use super::*;
use crate::tokenizer::Special;

impl Unigram {
    /// Extract normalized added tokens before applying the metaspace
    /// pre-tokenizer. Source byte spans survive character-map expansion.
    pub(crate) fn encode_added<S: Sink>(&self, text: &str, base: usize, piece: &str, added: &[Special], out: &mut S) {
        let chars = self.normalized_chars(piece, base);
        let mut normalized = String::new();
        let mut sources = Vec::new();
        for (c, span) in chars {
            normalized.push(c);
            sources.extend(std::iter::repeat_n(span, c.len_utf8()));
        }
        let mut cursor = 0;
        let mut from = 0;
        while cursor < normalized.len() {
            if let Some(s) =
                added.iter().find(|s| !s.content.is_empty() && normalized[cursor..].starts_with(&s.content))
            {
                let before = &normalized[from..cursor];
                let before = if s.lstrip { before.trim_end() } else { before };
                self.added_piece(text, before, &sources[from..from + before.len()], out);
                let end = cursor + s.content.len();
                out.push(s.id, || covering(text, &sources[cursor..end]));
                cursor = end;
                from = end;
            } else {
                cursor += normalized[cursor..].chars().next().unwrap().len_utf8();
            }
        }
        self.added_piece(text, &normalized[from..], &sources[from..], out);
    }

    fn added_piece<S: Sink>(&self, text: &str, piece: &str, original: &[Span], out: &mut S) {
        if piece.is_empty() {
            return;
        }
        if self.whitespace_split {
            let chars: Vec<(char, Span)> = piece.char_indices().map(|(i, c)| (c, original[i])).collect();
            return self.split_words(text, &chars, out);
        }
        let mut s = String::new();
        let mut sources = Vec::new();
        if self.add_prefix_space && !piece.starts_with(' ') && !piece.starts_with(self.metaspace) {
            s.push(self.metaspace);
            sources.extend(std::iter::repeat_n(original[0], self.metaspace.len_utf8()));
        }
        for (i, c) in piece.char_indices() {
            let c = if c == ' ' { self.metaspace } else { c };
            s.push(c);
            sources.extend(std::iter::repeat_n(original[i], c.len_utf8()));
        }
        let span = |from: usize, to: usize| covering(text, &sources[from..to]);
        let mut best = Vec::new();
        if self.whole_text {
            self.segment(&s, span, &mut best, out);
            return;
        }
        let mut start = 0;
        for (i, c) in s.char_indices() {
            if c == self.metaspace && i > start {
                self.segment(&s[start..i], |a, b| span(start + a, start + b), &mut best, out);
                start = i;
            }
        }
        if start < s.len() {
            self.segment(&s[start..], |a, b| span(start + a, start + b), &mut best, out);
        }
    }
}
