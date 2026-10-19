//! Special-token matching and encoding of plain text pieces.

use super::*;

impl Tokenizer {
    /// Special tokens are matched in the raw text first, as upstream does
    /// for its added tokens, one that takes the whitespace before it
    /// taking it; each piece of text between them is encoded on its own.
    pub(super) fn text_ids<S: Sink>(&self, text: &str, out: &mut S) {
        let mut rest = text;
        let mut plain = 0;
        while plain < rest.len() {
            // A byte a special token starts with is never inside a UTF-8
            // sequence, so `plain` is on a character boundary wherever
            // this looks.
            let b = rest.as_bytes()[plain];
            let hit = if self.special_starts[b as usize] {
                self.specials.iter().find(|s| rest[plain..].starts_with(s.content.as_str()))
            } else {
                None
            };
            match hit {
                Some(s) => {
                    let base = text.len() - rest.len();
                    let before = if s.lstrip { rest[..plain].trim_end() } else { &rest[..plain] };
                    self.plain_ids(text, base, before, out);
                    let at = (base + plain) as u32;
                    out.push(s.id, || [at, at + s.content.len() as u32]);
                    rest = &rest[plain + s.content.len()..];
                    plain = 0;
                }
                None if b < 0x80 => plain += 1,
                None => plain += rest[plain..].chars().next().map_or(1, char::len_utf8),
            }
        }
        self.plain_ids(text, text.len() - rest.len(), rest, out);
    }

    /// The ids of `piece`, the bytes of `text` from `base` on with no
    /// special token in them.
    pub(super) fn plain_ids<S: Sink>(&self, text: &str, base: usize, piece: &str, out: &mut S) {
        match &self.kind {
            Kind::WordPiece { normalizer, ascii, .. } => {
                // A buffer per thread: a text's normalized form is gone
                // before the thread's next text, and a batch's threads
                // would otherwise allocate and free one per text.
                let mut normalized = NORMALIZED.take();
                // Each normalized byte's source in `text`, for spans.
                let mut sources = if S::SPANS { SOURCES.take() } else { Vec::new() };
                if S::SPANS {
                    normalize_aligned(normalizer, ascii, piece, base, &mut normalized, &mut sources);
                } else {
                    normalize(normalizer, ascii, piece, &mut normalized);
                }
                let whole = normalized.as_str();
                let pos = |s: &str| s.as_ptr() as usize - whole.as_ptr() as usize;
                let span = |from: usize, to: usize| covering(text, &sources[from..to]);
                let word = |w: &str, out: &mut S| {
                    let w0 = pos(w);
                    self.word_pieces(w, |a, b| span(w0 + a, w0 + b), out);
                };
                let mut rest = whole;
                // Upstream's leftmost-longest matches, each searched for
                // after the last; one that is not a whole word when it must
                // be is passed over, its text left to the words around it.
                let mut from = 0;
                while !self.normalized_specials.is_empty() && from < rest.len() {
                    // On to a byte a special token starts with: the first
                    // byte of a character, never one inside it.
                    match rest.as_bytes()[from..].iter().position(|&b| self.normalized_starts[b as usize]) {
                        Some(k) => from += k,
                        None => break,
                    }
                    let at = &rest[from..];
                    let hit = self.normalized_specials.iter().find(|s| at.starts_with(s.content.as_str()));
                    let Some(s) = hit else {
                        from += at.chars().next().map_or(1, char::len_utf8);
                        continue;
                    };
                    let end = from + s.content.len();
                    let alone = !s.single_word
                        || (!rest[..from].chars().next_back().is_some_and(is_word)
                            && !rest[end..].chars().next().is_some_and(is_word));
                    if alone {
                        pre_tokenize(&rest[..from], |w| word(w, out));
                        let s0 = pos(&rest[from..]);
                        out.push(s.id, || span(s0, s0 + s.content.len()));
                        rest = &rest[end..];
                        from = 0;
                    } else {
                        from = end;
                    }
                }
                pre_tokenize(rest, |w| word(w, out));
                // A long text's buffers are not kept for the thread's life.
                if normalized.capacity() <= KEPT_NORMALIZED {
                    NORMALIZED.set(normalized);
                }
                if S::SPANS && sources.capacity() <= KEPT_NORMALIZED {
                    SOURCES.set(sources);
                }
            }
            Kind::Unigram(u) => {
                if !piece.is_empty() {
                    u.encode(text, base, piece, out);
                }
            }
            Kind::Bpe(b) => {
                // The normalized special tokens, leftmost-longest, in what
                // the raw ones left; one that takes the whitespace before
                // it takes it, as the raw ones do.
                let mut rest = piece;
                let mut from = 0;
                while !self.normalized_specials.is_empty() && from < rest.len() {
                    match rest.as_bytes()[from..].iter().position(|&c| self.normalized_starts[c as usize]) {
                        Some(k) => from += k,
                        None => break,
                    }
                    let Some(s) =
                        self.normalized_specials.iter().find(|s| rest[from..].starts_with(s.content.as_str()))
                    else {
                        from += rest[from..].chars().next().map_or(1, char::len_utf8);
                        continue;
                    };
                    let at = base + (piece.len() - rest.len());
                    let before = if s.lstrip { rest[..from].trim_end() } else { &rest[..from] };
                    if !before.is_empty() {
                        b.encode(text, at, before, out);
                    }
                    let s0 = (at + from) as u32;
                    out.push(s.id, || [s0, s0 + s.content.len() as u32]);
                    rest = &rest[from + s.content.len()..];
                    from = 0;
                }
                if !rest.is_empty() {
                    b.encode(text, base + (piece.len() - rest.len()), rest, out);
                }
            }
            Kind::SentencePiece(sp) => sp.encode_into(text, base, piece, out),
        }
    }

    /// Greedy longest-match-first; a word with any piece missing, or longer
    /// than max_chars_per_word, is one unknown token. A piece is never
    /// looked for longer than the longest entry.
    /// `span(start, end)` is the span of the word's bytes start..end.
    pub(super) fn word_pieces<S: Sink>(&self, word: &str, span: impl Fn(usize, usize) -> Span, out: &mut S) {
        let Kind::WordPiece {
            vocab,
            continuing,
            longest,
            longest_continuing,
            continuing_prefix,
            max_chars_per_word,
            ..
        } = &self.kind
        else {
            unreachable!("WordPiece only");
        };
        // A character is at least one byte.
        if word.len() > *max_chars_per_word && word.chars().count() > *max_chars_per_word {
            out.push(self.unk_id, || span(0, word.len()));
            return;
        }
        let first = out.len();
        let mut start = 0;
        while start < word.len() {
            // With no continuing prefix, a piece inside a word is looked
            // up in the whole vocabulary.
            let (table, most) = if start == 0 || continuing_prefix.is_empty() {
                (vocab, *longest)
            } else {
                (continuing, *longest_continuing)
            };
            let mut end = word.len().min(start + most);
            while !word.is_char_boundary(end) {
                end -= 1;
            }
            let mut found = None;
            while start < end {
                if let Some(&id) = table.get(&word[start..end]) {
                    found = Some(id);
                    break;
                }
                end -= word[..end].chars().next_back().map_or(1, char::len_utf8);
            }
            match found {
                Some(id) => out.push(id as i32, || span(start, end)),
                None => {
                    out.truncate(first);
                    out.push(self.unk_id, || span(0, word.len()));
                    return;
                }
            }
            start = end;
        }
    }
}
