//! Special-token matching and encoding of plain text pieces.

use super::*;

impl Tokenizer {
    /// The raw added tokens are matched in the text first, as upstream
    /// matches them; each piece of text between them is encoded on its
    /// own.
    pub(super) fn text_ids<S: Sink>(&self, text: &str, out: &mut S) {
        if self.raw.is_empty() {
            return self.plain_ids(text, 0, text, out);
        }
        self.raw.split(text, |p| match p {
            Piece::Text(r) => self.plain_ids(text, r.start, &text[r], out),
            Piece::Token(s, at) => out.push(s.id, || [at as u32, (at + s.content.len()) as u32]),
        });
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
                // The normalized added tokens, in the normalized piece.
                if self.normalized.is_empty() {
                    pre_tokenize(whole, |w| word(w, out));
                } else {
                    self.normalized.split(whole, |p| match p {
                        Piece::Text(r) => pre_tokenize(&whole[r], |w| word(w, out)),
                        Piece::Token(s, at) => out.push(s.id, || span(at, at + s.content.len())),
                    });
                }
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
                    if self.normalized.is_empty() {
                        u.encode(text, base, piece, out);
                    } else {
                        u.encode_added(text, base, piece, self.normalized.tokens(), out);
                    }
                }
            }
            Kind::Bpe(b) => {
                if self.normalized.is_empty() {
                    if !piece.is_empty() {
                        b.encode(text, base, piece, out);
                    }
                } else if b.changes(piece) {
                    // Matched in the piece's NFC.
                    b.encode_added(text, base, piece, &self.normalized, out);
                } else {
                    self.normalized.split(piece, |p| match p {
                        Piece::Text(r) => b.encode(text, base + r.start, &piece[r], out),
                        Piece::Token(s, at) => {
                            let at = (base + at) as u32;
                            out.push(s.id, || [at, at + s.content.len() as u32])
                        }
                    });
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
