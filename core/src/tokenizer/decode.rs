//! Piece decoders and vocabulary operations.

use super::*;

/// The tokenizer file's decoder: how a row of pieces is written as text.
pub(super) enum Decoder {
    /// WordPiece's: a piece with the continuing prefix joins the one
    /// before it, any other is put after a space; with `cleanup`, the
    /// space before punctuation and English contractions is taken out.
    WordPiece { prefix: String, cleanup: bool },
    /// Metaspace's: the metaspace is a space, and the one the first piece
    /// starts with, the one put in front of the text, is dropped.
    Metaspace { replacement: char, prepended: bool },
    /// ByteLevel's: each GPT-2 character is its byte, and the bytes are
    /// read as UTF-8, a sequence that is not being U+FFFD. A piece with a
    /// character that stands for no byte, a special token, is its text.
    ByteLevel,
    /// SentencePiece's own, from the model (`SentencePiece::decode`):
    /// it reads the whole row at once, so `Tokenizer::decode` hands it the
    /// row rather than piece by piece.
    SentencePiece,
    /// A decoder the core does not run, by its type.
    Other(String),
}

impl Decoder {
    pub(super) fn parse(d: &Value) -> Decoder {
        match d["type"].as_str() {
            Some("WordPiece") => Decoder::WordPiece {
                prefix: d["prefix"].as_str().unwrap_or("##").to_owned(),
                cleanup: d["cleanup"].as_bool().unwrap_or(true),
            },
            Some("Metaspace") => {
                let prepended = match d.get("prepend_scheme").and_then(Value::as_str) {
                    Some(scheme) => scheme != "never",
                    None => d["add_prefix_space"].as_bool().unwrap_or(true),
                };
                match d["replacement"].as_str().map(|r| (r.chars().next(), r.chars().count())) {
                    Some((Some(replacement), 1)) => Decoder::Metaspace { replacement, prepended },
                    _ => Decoder::Other("Metaspace with no one-character replacement".into()),
                }
            }
            Some("ByteLevel") => Decoder::ByteLevel,
            Some(other) => Decoder::Other(other.to_owned()),
            None => Decoder::Other("none".into()),
        }
    }

    /// One piece, written after those before it in `out`.
    pub(super) fn piece(&self, piece: &str, first: bool, out: &mut Vec<u8>) -> Result<()> {
        match self {
            Decoder::WordPiece { prefix, cleanup } => {
                let mut written = String::with_capacity(piece.len() + 1);
                match piece.strip_prefix(prefix.as_str()) {
                    Some(rest) if !first => written.push_str(rest),
                    _ => {
                        if !first {
                            written.push(' ');
                        }
                        written.push_str(piece);
                    }
                }
                let written = if *cleanup { cleanup_piece(&written) } else { Cow::Borrowed(written.as_str()) };
                out.extend_from_slice(written.as_bytes());
            }
            Decoder::Metaspace { replacement, prepended } => {
                let piece = match piece.strip_prefix(*replacement) {
                    Some(rest) if first && *prepended => rest,
                    _ => piece,
                };
                let mut buf = [0u8; 4];
                for c in piece.chars() {
                    let c = if c == *replacement { ' ' } else { c };
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
            Decoder::ByteLevel => {
                let at = out.len();
                for c in piece.chars() {
                    match crate::bpe::char_byte(c) {
                        Some(b) => out.push(b),
                        None => {
                            out.truncate(at);
                            out.extend_from_slice(piece.as_bytes());
                            break;
                        }
                    }
                }
            }
            Decoder::SentencePiece => unreachable!("decoded by the model, the whole row at once"),
            Decoder::Other(kind) => {
                return Err(Error::new(
                    crate::status::UNSUPPORTED,
                    format!("the tokenizer file's decoder is {kind}, which the core does not run"),
                ));
            }
        }
        Ok(())
    }
}

/// WordPiece's cleanup of one piece as written, its space included: no
/// space before . ? ! , and the English contractions.
pub(super) fn cleanup_piece(p: &str) -> Cow<'_, str> {
    if !p.starts_with(' ') {
        return Cow::Borrowed(p);
    }
    pub(super) const JOINED: [(&str, &str); 10] = [
        (" .", "."),
        (" ?", "?"),
        (" !", "!"),
        (" ,", ","),
        (" ' ", "'"),
        (" n't", "n't"),
        (" 'm", "'m"),
        (" 's", "'s"),
        (" 've", "'ve"),
        (" 're", "'re"),
    ];
    let mut s = Cow::Borrowed(p);
    for (from, to) in JOINED {
        if s.contains(from) {
            s = Cow::Owned(s.replace(from, to));
        }
    }
    s
}

impl Kind {
    /// `f(piece, id)` for every piece of the vocabulary.
    pub(super) fn each_piece(&self, mut f: impl FnMut(&str, u32)) {
        match self {
            Kind::WordPiece { vocab, .. } => vocab.iter().for_each(|(p, &id)| f(p, id)),
            Kind::Unigram(u) => u.pieces().enumerate().for_each(|(id, p)| f(p, id as u32)),
            Kind::Bpe(b) => b.pieces().enumerate().for_each(|(id, p)| f(p, id as u32)),
            Kind::SentencePiece(sp) => sp.pieces().enumerate().for_each(|(id, p)| f(p, id as u32)),
        }
    }

    pub(super) fn vocab_size(&self) -> u32 {
        match self {
            Kind::WordPiece { vocab, .. } => vocab.len() as u32,
            Kind::Unigram(u) => u.vocab_size(),
            Kind::Bpe(b) => b.vocab_size(),
            Kind::SentencePiece(sp) => sp.len() as u32,
        }
    }

    pub(super) fn id(&self, piece: &str) -> Option<u32> {
        match self {
            Kind::WordPiece { vocab, .. } => vocab.get(piece).copied(),
            Kind::Unigram(u) => u.id(piece),
            Kind::Bpe(b) => b.id(piece),
            Kind::SentencePiece(sp) => sp.id(piece),
        }
    }

    /// The median length of the vocabulary's entries in characters,
    /// rounded down: numpy's median of the lengths, then int().
    pub(super) fn median_chars(&self) -> usize {
        let mut lens: Vec<usize> = match self {
            Kind::WordPiece { vocab, .. } => vocab.keys().map(|p| p.chars().count()).collect(),
            Kind::Unigram(u) => u.pieces().map(|p| p.chars().count()).collect(),
            Kind::Bpe(b) => b.pieces().map(|p| p.chars().count()).collect(),
            Kind::SentencePiece(sp) => sp.pieces().map(|p| p.chars().count()).collect(),
        };
        lens.sort_unstable();
        let n = lens.len();
        match n {
            0 => 0,
            _ if n % 2 == 1 => lens[n / 2],
            _ => (lens[n / 2 - 1] + lens[n / 2]) / 2,
        }
    }
}

/// The first `n` characters of `text`, all of it when it has fewer.
pub(super) fn char_prefix(text: &str, n: usize) -> &str {
    if text.len() <= n {
        return text;
    }
    text.char_indices().nth(n).map_or(text, |(at, _)| &text[..at])
}
