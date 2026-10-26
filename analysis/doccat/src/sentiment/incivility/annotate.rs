use super::super::validation;
use super::*;

impl IncivilityAnnotator {
    /// The context vector the head reads for a unit whose preceding unit
    /// has the vector `preceding`.
    fn context_of<'a>(&'a self, preceding: Option<&'a [f32]>) -> Option<&'a [f32]> {
        match &self.vector_context {
            IncivilityVectorContext::None => None,
            IncivilityVectorContext::Preceding => preceding,
            IncivilityVectorContext::Parent(p) => Some(p),
        }
    }
}

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
            token_lower.push(incivility_token_key(slice));
        }

        // The lexicon stage reads no vectors, so it checks no encoder.
        let scorer_model =
            self.vector_scorer.as_ref().filter(|_| self.stage != IncivilityStage::Lexicon).and_then(|s| s.model());
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

        // The learned stage reads the head alone: it needs one, and the
        // unit's vectors.
        if self.stage == IncivilityStage::Learned {
            if self.vector_scorer.is_none() {
                return Err(fail("IncivilityStage::Learned needs a vector scorer".into()));
            }
            let has = match self.unit {
                IncivilityUnit::Sentences => sent_vecs.is_some(),
                IncivilityUnit::Chunks => chunk_vecs.is_some(),
                IncivilityUnit::Blocks => false,
            };
            if !has {
                return Err(fail("IncivilityStage::Learned needs the unit layer's vectors".into()));
            }
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
                    let row = |i: usize| {
                        let ix = Ix::<Sentences>::new(i as u32);
                        sent_vecs.and_then(|v| (i < v.len() && !v.is_empty(ix)).then(|| v.row(ix)))
                    };
                    let u_vec = row(s_idx);
                    if self.stage == IncivilityStage::Learned && u_vec.is_none() {
                        return Err(fail(format!("IncivilityStage::Learned: unit {s_idx} has no vector")));
                    }
                    let ctx = self.context_of(s_idx.checked_sub(1).and_then(row));

                    let (scores, evidence) = self
                        .score_unit_in_context(&token_lower, tok_start, tok_end, parent_range, u_vec, ctx)
                        .map_err(fail)?;
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
                    let row = |i: usize| {
                        let ix = Ix::<Chunks>::new(i as u32);
                        chunk_vecs.and_then(|v| (i < v.len() && !v.is_empty(ix)).then(|| v.row(ix)))
                    };
                    let u_vec = row(c_idx);
                    if self.stage == IncivilityStage::Learned && u_vec.is_none() {
                        return Err(fail(format!("IncivilityStage::Learned: unit {c_idx} has no vector")));
                    }
                    let ctx = self.context_of(c_idx.checked_sub(1).and_then(row));

                    let (scores, evidence) = self
                        .score_unit_in_context(&token_lower, tok_start, tok_end, parent_range, u_vec, ctx)
                        .map_err(fail)?;
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
        let vector_ctx = match self.vector_context {
            IncivilityVectorContext::None => "",
            IncivilityVectorContext::Preceding => ":preceding-vector",
            IncivilityVectorContext::Parent(_) => ":parent-vector",
        };
        if let Some(scorer) = self.vector_scorer.as_ref().filter(|_| self.stage != IncivilityStage::Lexicon) {
            // A head with no encoder named is known by its configuration.
            let m = scorer.model().unwrap_or_else(|| ModelRef {
                id: "incivility-vector-head".into(),
                revision: "1".into(),
                sha256: sha256(&scorer.config_hash().to_le_bytes()).to_hex(),
            });
            let id = match self.stage {
                IncivilityStage::Learned => format!("{}:learned{vector_ctx}", m.id),
                _ => format!("{}+lexicon:{ctx_suffix}{vector_ctx}", m.id),
            };
            return Some(ModelRef { id, revision: m.revision, sha256: m.sha256 });
        }
        Some(ModelRef {
            id: format!("oasis-incivility:lexicon:{ctx_suffix}"),
            revision: "1".into(),
            sha256: self.lexicon.sha256().to_owned(),
        })
    }

    fn config_hash(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        h.bytes(self.lexicon.sha256().as_bytes())
            .u64(self.unit as u64)
            .f64(self.threshold)
            .bool(self.use_reply_context);
        if let Some(ts) = &self.thresholds {
            for &t in ts {
                h.f64(t);
            }
        }
        if self.fusion_gate.to_bits() != DEFAULT_INCIVILITY_FUSION_GATE.to_bits() {
            h.f64(self.fusion_gate);
        }
        if !self.use_severe_toxicity_rule {
            h.bool(self.use_severe_toxicity_rule);
        }
        if let Some(s) = self.vector_scorer.as_ref().filter(|_| self.stage != IncivilityStage::Lexicon) {
            h.u64(s.config_hash());
        }
        h.u64(self.stage as u64);
        match &self.vector_context {
            IncivilityVectorContext::None => h.u8(0),
            IncivilityVectorContext::Preceding => h.u8(1),
            IncivilityVectorContext::Parent(p) => {
                h.u8(2);
                p.iter().for_each(|&v| {
                    h.f32(v);
                });
                &mut h
            }
        };
        h.finish()
    }
}
