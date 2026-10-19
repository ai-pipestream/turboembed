use super::super::validation;
use super::*;

impl Annotator for IncivilityAnnotator {
    const ID: &'static str = "turbo-doccat:incivility";
    const VERSION: &'static str = "1";
    type Reads = (
        View,
        Tokens,
        Option<Sentences>,
        Option<Chunks>,
        Option<Structure>,
        Option<Vectors<Sentences>>,
        Option<Vectors<Chunks>>,
    );
    type Writes = (Incivility,);

    fn annotate(
        &self,
        (view, tokens, sentences, chunks, structure, sent_vecs, chunk_vecs): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if tokens.spans.view() != view.id() {
            return Err(fail("tokens are in another view".into()));
        }
        if sentences.is_some_and(|s| s.spans.view() != view.id()) {
            return Err(fail("sentences are in another view".into()));
        }
        if chunks.is_some_and(|c| c.spans.view() != view.id()) {
            return Err(fail("chunks are in another view".into()));
        }
        if structure.is_some_and(|st| st.spans.view() != view.id()) {
            return Err(fail("structure is in another view".into()));
        }
        let text = view.text();
        validation::tokens(Self::ID, tokens, sentences, text)?;
        if let Some(layer) = sentences {
            validation::checked(Self::ID, layer)?;
            validation::contiguous(Self::ID, &layer.spans, text)?;
        }
        if let Some(layer) = chunks {
            validation::checked(Self::ID, layer)?;
            validation::contiguous(Self::ID, &layer.spans, text)?;
        }
        if let Some(layer) = structure {
            validation::checked(Self::ID, layer)?;
            validation::contiguous(Self::ID, &layer.spans, text)?;
        }
        let n_tokens = tokens.len();
        let (tok_starts, tok_ends) = tokens.spans.columns();

        let mut token_lower: Vec<String> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let slice = text.get(tok_starts[t] as usize..tok_ends[t] as usize).unwrap_or("");
            token_lower.push(slice.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_ascii_lowercase());
        }

        let scorer_model = self.vector_scorer.as_ref().and_then(|s| s.model());
        let check_encoder = |vectors_model: &ModelRef| -> Result<(), Error> {
            if let Some(m) = &scorer_model
                && *vectors_model != *m
            {
                return Err(fail(format!(
                    "vectors are from model {} (revision {}), the scorer is for {} (revision {})",
                    vectors_model.id, vectors_model.revision, m.id, m.revision
                )));
            }
            Ok(())
        };
        if let Some(v) = sent_vecs
            && self.unit == IncivilityUnit::Sentences
        {
            check_encoder(&v.model)?;
            validation::vectors(Self::ID, v, sentences.map_or(0, Layer::len))?;
        }
        if let Some(v) = chunk_vecs
            && self.unit == IncivilityUnit::Chunks
        {
            check_encoder(&v.model)?;
            validation::vectors(Self::ID, v, chunks.map_or(0, Layer::len))?;
        }

        let mut out = Incivility::new();

        match self.unit {
            IncivilityUnit::Sentences => {
                let sents = sentences.ok_or_else(|| {
                    fail("IncivilityUnit::Sentences requires a Sentences layer on the document".into())
                })?;
                let (s_starts, s_ends) = sents.spans.columns();
                let mut tok_cursor = 0usize;
                for s_idx in 0..sents.len() {
                    let s_start_byte = s_starts[s_idx];
                    let s_end_byte = s_ends[s_idx];
                    while tok_cursor < n_tokens && tok_ends[tok_cursor] <= s_start_byte {
                        tok_cursor += 1;
                    }
                    let tok_start = tok_cursor;
                    while tok_cursor < n_tokens && tok_starts[tok_cursor] < s_end_byte {
                        tok_cursor += 1;
                    }
                    let tok_end = tok_cursor;

                    let parent_range = if self.use_reply_context {
                        structure.and_then(|st| {
                            find_parent_or_quote_token_range(st, s_start_byte, s_end_byte, tok_starts, tok_ends)
                        })
                    } else {
                        None
                    };

                    let sent_ix = Ix::<Sentences>::new(s_idx as u32);
                    let u_vec = sent_vecs.and_then(|v| {
                        if s_idx < v.len() && !v.is_empty(sent_ix) { Some(v.row(sent_ix)) } else { None }
                    });

                    let (scores, evidence) =
                        self.score_unit(&token_lower, tok_start, tok_end, parent_range, u_vec).map_err(fail)?;
                    out.try_push(
                        tokens,
                        Some(sents),
                        chunks,
                        structure,
                        IncivilityItem { target: IncivilityTarget::Sentence(sent_ix), scores, evidence: &evidence },
                    )
                    .map_err(fail)?;
                }
            }
            IncivilityUnit::Chunks => {
                let ch = chunks
                    .ok_or_else(|| fail("IncivilityUnit::Chunks requires a Chunks layer on the document".into()))?;
                let (c_starts, c_ends) = ch.spans.columns();
                for c_idx in 0..ch.len() {
                    let c_start_byte = c_starts[c_idx];
                    let c_end_byte = c_ends[c_idx];
                    // Each nested or crossing chunk owns its own token window.
                    let tok_start = tok_ends.partition_point(|&end| end <= c_start_byte);
                    let tok_end = tok_starts.partition_point(|&start| start < c_end_byte);

                    let parent_range = if self.use_reply_context {
                        structure.and_then(|st| {
                            find_parent_or_quote_token_range(st, c_start_byte, c_end_byte, tok_starts, tok_ends)
                        })
                    } else {
                        None
                    };

                    let chunk_ix = Ix::<Chunks>::new(c_idx as u32);
                    let u_vec = chunk_vecs.and_then(|v| {
                        if c_idx < v.len() && !v.is_empty(chunk_ix) { Some(v.row(chunk_ix)) } else { None }
                    });

                    let (scores, evidence) =
                        self.score_unit(&token_lower, tok_start, tok_end, parent_range, u_vec).map_err(fail)?;
                    out.try_push(
                        tokens,
                        sentences,
                        Some(ch),
                        structure,
                        IncivilityItem { target: IncivilityTarget::Chunk(chunk_ix), scores, evidence: &evidence },
                    )
                    .map_err(fail)?;
                }
            }
            IncivilityUnit::Blocks => {
                let st = structure
                    .ok_or_else(|| fail("IncivilityUnit::Blocks requires a Structure layer on the document".into()))?;
                let (b_starts, b_ends) = st.spans.columns();
                for b_idx in 0..st.len() {
                    let b_start_byte = b_starts[b_idx];
                    let b_end_byte = b_ends[b_idx];
                    let tok_start = tok_ends.partition_point(|&e| e <= b_start_byte);
                    let tok_end = tok_starts.partition_point(|&s| s < b_end_byte);

                    let parent_range = if self.use_reply_context {
                        find_block_parent_or_quote_token_range(st, b_idx, tok_starts, tok_ends)
                    } else {
                        None
                    };

                    let block_ix = Ix::<Structure>::new(b_idx as u32);
                    let (scores, evidence) =
                        self.score_unit(&token_lower, tok_start, tok_end, parent_range, None).map_err(fail)?;
                    out.try_push(
                        tokens,
                        sentences,
                        chunks,
                        Some(st),
                        IncivilityItem { target: IncivilityTarget::Block(block_ix), scores, evidence: &evidence },
                    )
                    .map_err(fail)?;
                }
            }
        }

        Ok((out,))
    }

    fn model(&self) -> Option<ModelRef> {
        let ctx_suffix = if self.use_reply_context { "reply-context" } else { "no-reply-context" };
        if let Some(scorer) = &self.vector_scorer
            && let Some(m) = scorer.model()
        {
            return Some(ModelRef {
                id: format!("{}+lexicon:{ctx_suffix}", m.id),
                revision: m.revision,
                sha256: m.sha256,
            });
        }
        Some(ModelRef {
            id: format!("oasis-incivility:lexicon:{ctx_suffix}"),
            revision: "1".into(),
            sha256: self.lexicon.sha256().to_owned(),
        })
    }

    fn config_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in self.lexicon.sha256().as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= (self.unit as u64).wrapping_mul(0x9e3779b97f4a7c15);
        h ^= self.threshold.to_bits();
        h ^= u64::from(self.use_reply_context);
        if let Some(s) = &self.vector_scorer {
            h ^= s.config_hash();
        }
        h
    }
}
