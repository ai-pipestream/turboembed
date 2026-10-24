use super::*;
impl Unigram {
    /// The ids of `piece`, the bytes of `text` from `base` on with no
    /// special token in them.
    pub(crate) fn encode<S: Sink>(&self, text: &str, base: usize, piece: &str, out: &mut S) {
        if self.whitespace_split {
            return self.encode_words(text, base, piece, out);
        }
        // The normalized text with the metaspace in place of each space
        // and in front, and when spans are kept, each byte's source.
        let mut s = String::new();
        let mut sources = Vec::new();
        if S::SPANS {
            self.normalize_aligned(piece, base, &mut s, &mut sources);
        } else {
            let normalized = self.normalize(piece);
            s.reserve(normalized.len() + 3);
            // Upstream prepends nothing to an empty text.
            if self.add_prefix_space
                && !normalized.is_empty()
                && !normalized.starts_with(' ')
                && !normalized.starts_with(self.metaspace)
            {
                s.push(self.metaspace);
            }
            for c in normalized.chars() {
                s.push(if c == ' ' { self.metaspace } else { c });
            }
        }
        let span = |from: usize, to: usize| covering(text, &sources[from..to]);
        let mut best = Vec::new();
        if self.whole_text {
            self.segment(&s, span, &mut best, out);
            return;
        }
        // Cut at each metaspace, which starts the word after it.
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

    /// `encode` after a WhitespaceSplit.
    fn encode_words<S: Sink>(&self, text: &str, base: usize, piece: &str, out: &mut S) {
        self.split_words(text, &self.normalized_chars(piece, base), out);
    }

    /// Normalized characters, each with its source in `text`, cut at
    /// each whitespace character, which is dropped; each word, with the
    /// metaspace in front, is cut at each metaspace as `encode` cuts.
    pub(super) fn split_words<S: Sink>(&self, text: &str, chars: &[(char, Span)], out: &mut S) {
        let (mut s, mut sources, mut best) = (String::new(), Vec::new(), Vec::new());
        for word in chars.split(|(c, _)| c.is_whitespace()).filter(|w| !w.is_empty()) {
            s.clear();
            sources.clear();
            let mut put = |c: char, at: Span| {
                s.push(c);
                sources.extend(std::iter::repeat_n(at, c.len_utf8()));
            };
            if self.add_prefix_space && word[0].0 != self.metaspace {
                put(self.metaspace, word[0].1);
            }
            for &(c, at) in word {
                put(c, at);
            }
            let span = |from: usize, to: usize| covering(text, &sources[from..to]);
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

    /// What encode cuts into words, and the source of each of its bytes:
    /// the span in the caller's text of the character, or the grapheme
    /// cluster the precompiled map replaces as a whole, it came from.
    /// A space written around punctuation, and the metaspace in front,
    /// come from the character they are written beside.
    fn normalize_aligned(&self, piece: &str, base: usize, s: &mut String, sources: &mut Vec<Span>) {
        let chars = self.normalized_chars(piece, base);
        let Some(&(first, at)) = chars.first() else { return };
        let mut put = |c: char, at: Span| {
            s.push(c);
            sources.extend(std::iter::repeat_n(at, c.len_utf8()));
        };
        // The metaspace in front comes from the character it is put
        // before.
        if self.add_prefix_space && first != ' ' && first != self.metaspace {
            put(self.metaspace, at);
        }
        for (c, at) in chars {
            put(if c == ' ' { self.metaspace } else { c }, at);
        }
    }

    pub(super) fn normalized_chars(&self, piece: &str, base: usize) -> Vec<(char, Span)> {
        let mut chars: Vec<(char, Span)> = Vec::with_capacity(piece.len());
        match &self.charsmap {
            Some(c) => c.normalize_aligned(piece, base, &mut chars),
            None => chars.extend(piece.char_indices().map(|(i, c)| {
                let at = (base + i) as u32;
                (c, [at, at + c.len_utf8() as u32])
            })),
        }
        // A run collapsed to one space comes from its last character, as
        // upstream's replacement does.
        if self.collapse_spaces {
            let mut kept: Vec<(char, Span)> = Vec::with_capacity(chars.len());
            for (c, at) in chars {
                match kept.last_mut() {
                    Some(last) if c == ' ' && last.0 == ' ' => last.1 = at,
                    _ => kept.push((c, at)),
                }
            }
            chars = kept;
        }
        // The spaces written around punctuation come from it.
        if self.space_punctuation {
            let mut spaced = Vec::with_capacity(chars.len() + chars.len() / 4);
            for (c, at) in chars {
                if c.is_ascii_punctuation() {
                    spaced.extend([(' ', at), (c, at), (' ', at)]);
                } else {
                    spaced.push((c, at));
                }
            }
            chars = spaced;
        }
        if self.collapse_whitespace {
            let mut kept: Vec<(char, Span)> = Vec::with_capacity(chars.len());
            let mut in_run = false;
            for (c, at) in chars {
                let space = c.is_whitespace();
                match kept.last_mut() {
                    Some(last) if space && in_run => last.1 = at,
                    _ => kept.push((if space { ' ' } else { c }, at)),
                }
                in_run = space;
            }
            chars = kept;
        }
        if self.strip {
            let lead = chars.iter().take_while(|(c, _)| c.is_whitespace()).count();
            chars.drain(..lead);
            let trail = chars.iter().rev().take_while(|(c, _)| c.is_whitespace()).count();
            chars.truncate(chars.len() - trail);
        }
        chars
    }

    /// The precompiled map, then runs of spaces collapsed to one.
    pub(crate) fn normalize(&self, text: &str) -> String {
        let mapped = match &self.charsmap {
            Some(c) => c.normalize(text),
            None => text.to_owned(),
        };
        let mut out = if self.collapse_spaces {
            let mut out = String::with_capacity(mapped.len());
            let mut last_space = false;
            for c in mapped.chars() {
                if c == ' ' && last_space {
                    continue;
                }
                last_space = c == ' ';
                out.push(c);
            }
            out
        } else {
            mapped
        };
        if self.space_punctuation {
            let mut spaced = String::with_capacity(out.len() + out.len() / 4);
            for c in out.chars() {
                if c.is_ascii_punctuation() {
                    spaced.push(' ');
                    spaced.push(c);
                    spaced.push(' ');
                } else {
                    spaced.push(c);
                }
            }
            out = spaced;
        }
        if self.collapse_whitespace {
            // Regex \s+ as upstream's Oniguruma reads it in UTF-8: the
            // Unicode White_Space characters, which char::is_whitespace is.
            let mut folded = String::with_capacity(out.len());
            let mut in_run = false;
            for c in out.chars() {
                if c.is_whitespace() {
                    if !in_run {
                        folded.push(' ');
                    }
                    in_run = true;
                } else {
                    folded.push(c);
                    in_run = false;
                }
            }
            out = folded;
        }
        if self.strip {
            let t = out.trim();
            if t.len() != out.len() {
                out = t.to_owned();
            }
        }
        out
    }
}
