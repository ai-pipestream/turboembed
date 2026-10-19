//! Row layout, truncation, prompt handling, and public encoding methods.

use super::*;

impl Tokenizer {
    /// What TURBO_TRUNCATE_MODEL means for this bundle.
    pub fn truncation(&self) -> Truncation {
        self.truncation
    }

    pub fn vocab_size(&self) -> u32 {
        self.kind.vocab_size()
    }

    /// `turbo_tokenizer_info.kind`.
    pub fn kind_name(&self) -> &'static str {
        match self.kind {
            Kind::WordPiece { .. } => "wordpiece",
            Kind::Unigram(_) => "unigram",
            Kind::Bpe(_) => "bpe",
            Kind::SentencePiece(_) => "sentencepiece",
        }
    }

    /// The id under mask 0 in rows the session lays out: the pad token,
    /// or the unk token (which every bundle has) when there is no pad
    /// token, so every id a backend reads is in the vocabulary.
    pub fn fill_id(&self) -> i32 {
        if self.pad_id < 0 { self.unk_id } else { self.pad_id }
    }

    /// The vocabulary's piece for `id`, as the tokenizer file spells it
    /// (a SentencePiece word start as U+2581), or None for an id outside
    /// the vocabulary.
    pub fn piece(&self, id: i32) -> Option<&str> {
        usize::try_from(id).ok().and_then(|i| self.pieces.get(i)).map(|p| &**p)
    }

    pub fn specials_per_sequence(&self) -> u32 {
        self.template.iter().filter(|p| p.is_some()).count() as u32
    }

    /// The special tokens the template puts before the text and those it
    /// puts after it, as every row encoded with special tokens has them:
    /// for a caller that lays out rows from pieces it already has.
    pub fn template_ends(&self) -> (Vec<i32>, Vec<i32>) {
        // The manifest holds the text once in every template.
        let at = self.template.iter().position(Option::is_none).unwrap_or(self.template.len());
        let head = self.template[..at].iter().flatten().copied().collect();
        let tail = self.template.get(at + 1..).unwrap_or_default().iter().flatten().copied().collect();
        (head, tail)
    }

    /// One row of ids. Too long for `max_tokens` is cut as `truncation`
    /// says, or CAPACITY for TRUNCATE_NONE.
    pub fn encode(&self, text: &str, opts: Encode) -> Result<Vec<i32>> {
        let mut row = Vec::new();
        self.encode_into(text, opts, &mut row)?;
        Ok(row)
    }

    /// The same row, appended to `out`, which an error leaves as it was:
    /// a batch's rows can share one buffer.
    pub fn encode_into(&self, text: &str, opts: Encode, out: &mut Vec<i32>) -> Result<()> {
        let start = out.len();
        let r = self.append_row(text, opts, out);
        if r.is_err() {
            out.truncate(start);
        }
        r
    }

    /// The row `encode_into` appends, and beside it each token's span in
    /// `text`: the same ids, and `spans` as long as `ids` after it as
    /// before. An error leaves both as they were.
    pub fn encode_spans(&self, text: &str, opts: Encode, ids: &mut Vec<i32>, spans: &mut Vec<Span>) -> Result<()> {
        if ids.len() != spans.len() {
            return Err(invalid(format!("{} ids and {} spans: a span for each id", ids.len(), spans.len())));
        }
        span_extent(text.len(), self.prefix(opts.prompt).len())?;
        let start = ids.len();
        let r = self.append_row(text, opts, &mut Spanned { ids, spans });
        if r.is_err() {
            ids.truncate(start);
            spans.truncate(start);
            return r;
        }
        // Spans are into the text with the prompt's prefix in front.
        let prefix = self.prefix(opts.prompt).len() as u32;
        if prefix > 0 {
            for s in &mut spans[start..] {
                *s = if s[1] <= prefix { NOWHERE } else { [s[0].saturating_sub(prefix), s[1] - prefix] };
            }
        }
        Ok(())
    }

    pub(super) fn append_row<S: Sink>(&self, text: &str, opts: Encode, out: &mut S) -> Result<()> {
        let specials = if opts.add_special_tokens { self.specials_per_sequence() } else { 0 };
        if opts.max_tokens < specials {
            return Err(Error::field(
                INVALID_ARGUMENT,
                3,
                format!("max_tokens {} leaves no room beside {specials} special tokens", opts.max_tokens),
            ));
        }
        let budget = (opts.max_tokens - specials) as usize;
        let text = self.prompted(text, opts.prompt);
        // A template with the text once has its specials around the body
        // written in place; any other is put together after it.
        let start = out.len();
        let once = opts.add_special_tokens && self.template.iter().filter(|p| p.is_none()).count() == 1;
        let (head, tail) = match self.template.iter().position(Option::is_none) {
            Some(at) if once => (&self.template[..at], &self.template[at + 1..]),
            _ => (&[][..], &[][..]),
        };
        for &id in head.iter().flatten() {
            out.push(id, || NOWHERE);
        }
        let body = out.len();
        match &self.stat {
            None => self.text_ids(&text, out),
            Some(st) => {
                match opts.truncation {
                    Truncation::Right => {
                        self.text_ids(char_prefix(&text, budget.saturating_mul(st.median_chars)), out);
                        out.truncate(body + budget);
                    }
                    Truncation::Left => {
                        self.text_ids(&text, out);
                        out.drain(body..out.len().saturating_sub(budget).max(body));
                    }
                    Truncation::None => self.text_ids(&text, out),
                }
                let unk = self.unk_id;
                out.retain_from(body, |id| id != unk);
            }
        }
        let len = out.len() - body;
        if len > budget {
            match opts.truncation {
                Truncation::None => {
                    return Err(Error::new(
                        CAPACITY,
                        format!(
                            "{} tokens is over max_tokens {} and truncation is TRUNCATE_NONE",
                            len + specials as usize,
                            opts.max_tokens
                        ),
                    ));
                }
                Truncation::Right => out.truncate(body + budget),
                Truncation::Left => {
                    out.drain(body..body + len - budget);
                }
            }
        }
        for &id in tail.iter().flatten() {
            out.push(id, || NOWHERE);
        }
        if opts.add_special_tokens && !once {
            let (ids, spans) = out.split_off(start);
            for piece in &self.template {
                match piece {
                    Some(id) => out.push(*id, || NOWHERE),
                    None => out.extend_with(&ids, &spans),
                }
            }
        }
        Ok(())
    }

    /// The text `ids` stand for, as the tokenizer file's decoder writes
    /// it, the special tokens left out when `skip_special_tokens`. The
    /// way back from ids to text for a model's output; for where a token
    /// of the input came from, encode_spans gives its span.
    pub fn decode(&self, ids: &[i32], skip_special_tokens: bool) -> Result<String> {
        if let Kind::SentencePiece(sp) = &self.kind {
            let mut row = Vec::with_capacity(ids.len());
            for (i, &id) in ids.iter().enumerate() {
                if id < 0 || id as usize >= self.pieces.len() {
                    return Err(Error::new(
                        INVALID_ARGUMENT,
                        format!("ids[{i}]: {id} is not an id under {}", self.pieces.len()),
                    ));
                }
                if !(skip_special_tokens && self.special_ids.contains(&id)) {
                    row.push(id as u32);
                }
            }
            return sp.decode(&row);
        }
        // Bytes, as a byte-level piece can hold part of a character.
        let mut out = Vec::new();
        let mut first = true;
        for (i, &id) in ids.iter().enumerate() {
            let Some(piece) = usize::try_from(id).ok().and_then(|i| self.pieces.get(i)) else {
                return Err(Error::new(
                    INVALID_ARGUMENT,
                    format!("ids[{i}]: {id} is not an id under {}", self.pieces.len()),
                ));
            };
            if skip_special_tokens && self.special_ids.contains(&id) {
                continue;
            }
            self.decoder.piece(piece, first, &mut out)?;
            first = false;
        }
        Ok(match String::from_utf8(out) {
            Ok(s) => s,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        })
    }

    /// Tokens `text` produces, with special tokens and no truncation.
    pub fn count(&self, text: &str, prompt: PromptRole) -> usize {
        let mut ids = Vec::new();
        self.text_ids(&self.prompted(text, prompt), &mut ids);
        if self.stat.is_some() {
            let unk = self.unk_id;
            ids.retain(|&id| id != unk);
        }
        ids.len() + self.specials_per_sequence() as usize
    }

    pub(super) fn prefix(&self, prompt: PromptRole) -> &str {
        match prompt {
            PromptRole::None => "",
            PromptRole::Query => &self.prefix_query,
            PromptRole::Document => &self.prefix_document,
        }
    }

    pub(super) fn prompted<'a>(&self, text: &'a str, prompt: PromptRole) -> Cow<'a, str> {
        let prefix = self.prefix(prompt);
        if prefix.is_empty() { Cow::Borrowed(text) } else { Cow::Owned(format!("{prefix}{text}")) }
    }
}

// Keep span extent arithmetic separate so its integer boundaries need no huge allocation.
fn span_extent(text_bytes: usize, prefix_bytes: usize) -> Result<u32> {
    text_bytes
        .checked_add(prefix_bytes)
        .and_then(|len| u32::try_from(len).ok())
        .ok_or_else(|| Error::new(CAPACITY, "text and prompt prefix exceed the u32 span offset limit"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_extent_must_not_wrap_with_a_prompt_prefix() {
        assert!(span_extent(u32::MAX as usize, 1).is_err());
    }

    #[test]
    fn span_extent_accepts_the_last_representable_offset() {
        assert_eq!(span_extent(u32::MAX as usize - 7, 7).unwrap(), u32::MAX);
        assert_eq!(span_extent(0, 0).unwrap(), 0);
    }
}
