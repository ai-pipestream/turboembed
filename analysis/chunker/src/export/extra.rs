//! Codecs for the clause cuts and chunk antecedents layers,
//! split from `export` to keep both files under the size cap.

use super::*;

impl LayerCodec for ClauseCuts {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.sentences.len())?;
        for s in &self.sentences {
            w.usize(s.tokens.len())?;
            for t in &s.tokens {
                w.span(*t);
            }
            w.usize(s.breaks.len())?;
            for b in &s.breaks {
                match b {
                    ClauseBreak::Clause(c) => {
                        w.u8(0);
                        w.u32(*c as u32);
                    }
                    ClauseBreak::Phrase(c) => {
                        w.u8(1);
                        w.u32(*c as u32);
                    }
                    ClauseBreak::Token(c) => {
                        w.u8(2);
                        w.u32(*c as u32);
                    }
                    ClauseBreak::Held => w.u8(3),
                }
            }
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let mut sentences = Vec::new();
        for _ in 0..r.usize()? {
            let mut tokens: Vec<Span> = Vec::new();
            for _ in 0..r.usize()? {
                tokens.push(r.span()?);
            }
            let mut breaks = Vec::new();
            for _ in 0..r.usize()? {
                breaks.push(match r.u8()? {
                    0 => ClauseBreak::Clause(r.u32()? as i32),
                    1 => ClauseBreak::Phrase(r.u32()? as i32),
                    2 => ClauseBreak::Token(r.u32()? as i32),
                    3 => ClauseBreak::Held,
                    tag => return Err(malformed(format!("a clause break tag of {tag}"))),
                });
            }
            sentences.push(SentenceClauseCuts { tokens, breaks });
        }
        done(&r, "clause cuts")?;
        let want = aligned_len::<Sentences>(cx)?;
        if sentences.len() != want {
            return Err(malformed(format!("{} clause cut rows for {want} sentences", sentences.len())));
        }
        let target = resolved::<Sentences>(cx);
        let key = cx.key::<Sentences>(&target.instance);
        let restored = doc
            .get_key(&key)
            .ok_or_else(|| ExportError::DanglingReference { layer: cx.layer().to_owned(), reference: target })?;
        let text = doc.text(restored.spans.view());
        for (i, row) in sentences.iter().enumerate() {
            let sentence = restored.spans.get(Ix::new(i as u32));
            for token in &row.tokens {
                if !sentence.contains(*token)
                    || !text.is_char_boundary(token.start as usize)
                    || !text.is_char_boundary(token.end as usize)
                {
                    return Err(ExportError::InvalidLayer {
                        layer: cx.layer().to_owned(),
                        detail: format!(
                            "clause token {}..{} is outside sentence {i} ({}..{}) or off UTF-8 boundaries",
                            token.start, token.end, sentence.start, sentence.end
                        ),
                    });
                }
            }
        }
        checked(ClauseCuts { sentences }, cx)
    }
}

impl LayerCodec for ChunkAntecedents {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_u32s(&mut w, &self.row)?;
        w.usize(self.links.len())?;
        for l in &self.links {
            w.span(l.mention_span);
            w.u32(l.mention_sentence.get() as u32);
            w.span(l.antecedent_span);
            w.u32(l.antecedent_sentence.get() as u32);
            w.str(&l.mention_text)?;
            w.str(&l.antecedent_text)?;
        }
        w.usize(self.headers.len())?;
        for h in &self.headers {
            w.str(h)?;
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let row = get_u32s(&mut r)?;
        let sentences = resolved::<Sentences>(cx);
        let mut links = Vec::new();
        for _ in 0..r.usize()? {
            let mention_span = r.span()?;
            let mention_sentence = get_ix::<Sentences>(&mut r, cx, &sentences)?;
            let antecedent_span = r.span()?;
            let antecedent_sentence = get_ix::<Sentences>(&mut r, cx, &sentences)?;
            let mention_text = r.string()?;
            let antecedent_text = r.string()?;
            links.push(AntecedentLink {
                mention_span,
                mention_sentence,
                antecedent_span,
                antecedent_sentence,
                mention_text,
                antecedent_text,
            });
        }
        let mut headers = Vec::new();
        for _ in 0..r.usize()? {
            headers.push(r.string()?);
        }
        done(&r, "chunk antecedents")?;
        let chunks = aligned_len::<Chunks>(cx)?;
        if headers.len() != chunks {
            return Err(malformed(format!("{} antecedent headers for {chunks} chunks", headers.len())));
        }
        checked(ChunkAntecedents { row, links, headers }, cx)
    }
}
