use super::*;

impl Annotator for RuleAspectSentiment {
    const ID: &'static str = "turbo-doccat:aspect-sentiment-rule";
    type Reads = (View, Sentences, Tokens, Option<Dependencies>, Option<PhraseChunks>, Option<PosTags>);
    type Writes = (Opinions, Aspects);

    fn annotate(
        &self,
        (view, sentences, tokens, deps, phrase_chunks, pos_tags): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if sentences.spans.view() != view.id() {
            return Err(fail("sentences are in another view".into()));
        }
        if tokens.spans.view() != view.id() {
            return Err(fail("tokens are in another view".into()));
        }
        if phrase_chunks.is_some_and(|c| c.spans.view() != view.id()) {
            return Err(fail("phrase chunks are in another view".into()));
        }
        let text = view.text();
        let n_tokens = tokens.len();
        let (tok_starts, tok_ends) = tokens.spans.columns();
        let (sent_starts, sent_ends) = sentences.spans.columns();

        let mut token_lower: Vec<String> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let s = tok_starts[t] as usize;
            let e = tok_ends[t] as usize;
            let slice = text.get(s..e).unwrap_or("");
            token_lower.push(clean_word_lower(slice));
        }

        struct PendingAspect {
            target: AspectTarget,
            category: Option<Tag>,
            polarity: PolarityScores,
            opinions: Vec<Ix<Opinions>>,
        }

        let mut all_opinions: Vec<Opinion> = Vec::new();
        let mut pending_aspects: Vec<PendingAspect> = Vec::new();

        let mut tok_cursor = 0usize;
        for s_idx in 0..sentences.len() {
            let s_start_byte = sent_starts[s_idx];
            let s_end_byte = sent_ends[s_idx];

            while tok_cursor < n_tokens && tok_ends[tok_cursor] <= s_start_byte {
                tok_cursor += 1;
            }
            let sent_tok_start = tok_cursor;
            while tok_cursor < n_tokens && tok_starts[tok_cursor] < s_end_byte {
                tok_cursor += 1;
            }
            let sent_tok_end = tok_cursor;
            if sent_tok_start >= sent_tok_end {
                continue;
            }

            let sent_ix = Ix::<Sentences>::new(s_idx as u32);
            let clause_ids = assign_clauses(&token_lower, text, tok_starts, tok_ends, sent_tok_start, sent_tok_end);

            // Step 1: Seed opinions and explicit aspect candidates in the sentence.
            let (mut sent_opinions, mut sent_aspects) = self.extract_sentence_candidates(
                text,
                tok_starts,
                tok_ends,
                &token_lower,
                sent_tok_start,
                sent_tok_end,
                &clause_ids,
                deps,
                phrase_chunks,
                pos_tags,
            );

            // Step 2: If Dependencies are present, run Double Propagation (Qiu et al. 2011)
            // to expand targets from opinions (R1), opinions from targets (R2),
            // conjoined targets (R3), and conjoined opinions (R4).
            if let Some(d) = deps {
                self.run_double_propagation(
                    d,
                    pos_tags,
                    &token_lower,
                    sent_tok_start,
                    sent_tok_end,
                    &clause_ids,
                    &mut sent_opinions,
                    &mut sent_aspects,
                );
            }

            // Sort and deduplicate opinions by token range in ascending order.
            sent_opinions.sort_by_key(|o| (o.tokens.start.get(), o.tokens.end.get()));
            dedup_opinions(&mut sent_opinions);

            // Sort and deduplicate explicit aspects by token range in ascending order,
            // removing any aspect span that overlaps an opinion span.
            sent_aspects.retain(|a| !sent_opinions.iter().any(|o| ranges_overlap(a.tokens, o.tokens)));
            sent_aspects.sort_by_key(|a| (a.tokens.start.get(), a.tokens.end.get()));
            dedup_aspects(&mut sent_aspects);

            // Record the sentence's opinions in `all_opinions` and remember their global `Ix<Opinions>`.
            let base_op_idx = all_opinions.len();
            for op in &sent_opinions {
                all_opinions.push(Opinion {
                    tokens: op.tokens,
                    polarity: op.polarity,
                    intensity: op.intensity,
                    negated: op.negated,
                });
            }

            let sent_text = text.get(s_start_byte as usize..s_end_byte as usize).unwrap_or("");

            if !sent_aspects.is_empty() {
                // Step 3: Link each explicit aspect to its governing/modifying opinions
                // (via dependency paths first when `deps` is present, then same-clause
                // proximity, then sentence fallback).
                for (a_pos, asp) in sent_aspects.iter().enumerate() {
                    let linked_local = link_opinions_to_aspect(
                        asp,
                        a_pos,
                        &sent_aspects,
                        &sent_opinions,
                        deps,
                        &token_lower,
                        sent_tok_start,
                        sent_tok_end,
                    );
                    let linked_ixs: Vec<Ix<Opinions>> =
                        linked_local.iter().map(|&k| Ix::new((base_op_idx + k) as u32)).collect();
                    let polarity = compute_aspect_polarity(&linked_local, &sent_opinions);
                    let asp_text = span_text(text, tok_starts, tok_ends, asp.tokens);
                    let op_text = linked_local
                        .first()
                        .map(|&k| span_text(text, tok_starts, tok_ends, sent_opinions[k].tokens))
                        .unwrap_or("");
                    let category = self.resolve_category(
                        asp.category_hint,
                        asp_text,
                        op_text,
                        sent_text,
                        &token_lower[asp.tokens.start.get()..asp.tokens.end.get()],
                    );
                    pending_aspects.push(PendingAspect {
                        target: AspectTarget::Term { tokens: asp.tokens, sentence: Some(sent_ix) },
                        category,
                        polarity,
                        opinions: linked_ixs,
                    });
                }
            } else if self.emit_implicit_aspects && !sent_opinions.is_empty() {
                // Step 4: Implicit aspect anchored to the sentence when opinion expressions
                // exist without any explicit aspect term in the sentence.
                let all_local: Vec<usize> = (0..sent_opinions.len()).collect();
                let linked_ixs: Vec<Ix<Opinions>> =
                    all_local.iter().map(|&k| Ix::new((base_op_idx + k) as u32)).collect();
                let polarity = compute_aspect_polarity(&all_local, &sent_opinions);
                let op_text = span_text(text, tok_starts, tok_ends, sent_opinions[0].tokens);
                let category = self.resolve_implicit_category(&sent_opinions, &token_lower, op_text, sent_text);
                pending_aspects.push(PendingAspect {
                    target: AspectTarget::Implicit { sentence: sent_ix },
                    category,
                    polarity,
                    opinions: linked_ixs,
                });
            }
        }

        let opinions_layer = Opinions::from_items(tokens, all_opinions).map_err(fail)?;
        let mut aspects_layer = Aspects::new(Arc::clone(&self.categories));
        for p in &pending_aspects {
            aspects_layer
                .try_push(
                    tokens,
                    Some(sentences),
                    Some(&opinions_layer),
                    Aspect { target: p.target, category: p.category, polarity: p.polarity, opinions: &p.opinions },
                )
                .map_err(fail)?;
        }

        Ok((opinions_layer, aspects_layer))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(ModelRef {
            id: "double-propagation:vader".to_string(),
            revision: "1.0.0".to_string(),
            sha256: format!("{:016x}", self.config_hash()),
        })
    }

    fn config_hash(&self) -> u64 {
        let mut h = crate::hash::FoldHasher::default();
        Self::ID.hash(&mut h);
        self.categories.name.hash(&mut h);
        for i in 0..self.categories.len() {
            self.categories.label(Tag(i as u16)).hash(&mut h);
        }
        self.emit_implicit_aspects.hash(&mut h);
        if let Some(scorer) = &self.category_scorer {
            scorer.config_hash().hash(&mut h);
        }
        h.finish()
    }
}
