use super::*;

impl LearnedAspectSentiment {
    pub(super) fn infer(
        &self,
        (view, sentences, tokens, subwords, subword_vectors, deps, phrase_chunks): Refs<'_, Self>,
        cached: &crate::vectors::CachedWindows,
    ) -> Result<(Opinions, Aspects), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };

        let text = view.text();
        let n_tokens = tokens.len();
        let (tok_starts, tok_ends) = tokens.spans.columns();
        let (sent_starts, sent_ends) = sentences.spans.columns();

        let mut token_lower: Vec<String> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let start = tok_starts[t] as usize;
            let end = tok_ends[t] as usize;
            let slice = text.get(start..end).ok_or_else(|| {
                fail(format!(
                    "token {t} span {start}..{end} is out of bounds or not on a UTF-8 char boundary (text len {})",
                    text.len()
                ))
            })?;
            token_lower
                .push(slice.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-').to_ascii_lowercase());
        }

        // 1. Obtain contextual subword vectors (`Vectors<Subwords>`), either from the document
        // or from the retained raw windows, without another encoder run.
        let owned_sub_vecs;
        let sub_vecs: &Vectors<Subwords> = match subword_vectors {
            Some(v) => {
                v.check().map_err(fail)?;
                if v.model != *self.vectors.model() {
                    return Err(fail(format!(
                        "subword vectors are from model {} (revision {}), expected {} (revision {})",
                        v.model.id,
                        v.model.revision,
                        self.vectors.model().id,
                        self.vectors.model().revision
                    )));
                }
                if v.len() != subwords.len() {
                    return Err(fail(format!(
                        "subword vectors has {} items, subwords has {}",
                        v.len(),
                        subwords.len()
                    )));
                }
                v
            }
            None => {
                owned_sub_vecs = self.vectors.cached_tokens(Self::ID, subwords, cached)?;
                &owned_sub_vecs
            }
        };

        if self.bio_head.tags().len() < 5 {
            return Err(fail(format!(
                "BIO head must have at least 5 tags {:?}, got {}",
                BIO_ASPECT_OPINION_LABELS,
                self.bio_head.tags().len()
            )));
        }

        // 2. Score every subword piece with `self.bio_head`.
        let n_bio = self.bio_head.tags().len();
        let mut piece_probs = vec![[0.0f64; 5]; subwords.len()];
        let mut bio_buf = vec![0.0f64; n_bio];
        for (p, dst) in piece_probs.iter_mut().enumerate() {
            let ix = Ix::<Subwords>::new(p as u32);
            if sub_vecs.is_empty(ix) {
                dst[TAG_O as usize] = 1.0;
            } else {
                let norm_row = unit_l2(sub_vecs.row(ix));
                self.bio_head.score(&norm_row, &mut bio_buf).map_err(fail)?;
                dst.copy_from_slice(&bio_buf[..5]);
            }
        }

        // 3. Map subword BIO predictions back to `Tokens` through `Subwords` alignments/offsets.
        let tok_tags = decode::project_subword_bio_to_tokens(tokens, subwords, &piece_probs);

        struct PendingAspect {
            target: AspectTarget,
            category: Option<Tag>,
            polarity: PolarityScores,
            opinions: Vec<Ix<Opinions>>,
        }

        let mut all_opinions: Vec<Opinion> = Vec::new();
        let mut pending_aspects: Vec<PendingAspect> = Vec::new();
        let mut pol_buf = vec![0.0f64; self.polarity_head.tags().len()];
        let mut cat_buf = vec![0.0f64; self.categories.len().max(1)];

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

            // Decode BIO spans inside `sent_tok_start..sent_tok_end`.
            let (raw_asp_ranges, raw_op_ranges) =
                decode_bio_spans(&tok_tags, &token_lower, &clause_ids, sent_tok_start, sent_tok_end, phrase_chunks);

            // Pool each opinion span via #263 and score with `self.polarity_head`.
            let op_spans: Vec<Span> = raw_op_ranges.iter().map(|&r| token_range_span(r, tokens)).collect();
            let pooled_ops: Option<Vectors<Opinions>> = if op_spans.is_empty() {
                None
            } else {
                Some(self.vectors.cached_spans(Self::ID, subwords, &sentences.spans, cached, &op_spans)?)
            };

            let mut sent_opinions: Vec<DecodedOpinion> = Vec::with_capacity(raw_op_ranges.len());
            for (i, &r) in raw_op_ranges.iter().enumerate() {
                let head_tok = r.end.get() - 1;
                let negated = is_token_negated(head_tok, sent_tok_start, sent_tok_end, &token_lower, &clause_ids, deps);
                let (mut pos_p, mut neg_p, neu_p) = if let Some(vecs) = &pooled_ops
                    && !vecs.is_empty(Ix::new(i as u32))
                {
                    let norm_op = unit_l2(vecs.row(Ix::new(i as u32)));
                    self.polarity_head.score(&norm_op, &mut pol_buf).map_err(fail)?;
                    let ps = self.polarity_head.polarity_scores_from_probs(&pol_buf);
                    (ps.positive, ps.negative, ps.neutral)
                } else {
                    (0.0, 0.0, 1.0)
                };
                let signed_prior = (pos_p - neg_p).clamp(-1.0, 1.0);
                let intensity = (pos_p + neg_p).clamp(0.1, 2.0);
                if negated {
                    std::mem::swap(&mut pos_p, &mut neg_p);
                }
                sent_opinions.push(DecodedOpinion {
                    tokens: r,
                    head_token: head_tok,
                    polarity: signed_prior,
                    intensity,
                    negated,
                    pos_prob: pos_p,
                    neg_prob: neg_p,
                    neu_prob: neu_p,
                    clause_id: clause_ids[head_tok - sent_tok_start],
                });
            }

            let base_op_idx = all_opinions.len();
            for op in &sent_opinions {
                all_opinions.push(Opinion {
                    tokens: op.tokens,
                    polarity: op.polarity,
                    intensity: op.intensity,
                    negated: op.negated,
                });
            }

            let sent_aspects: Vec<DecodedAspect> = raw_asp_ranges
                .iter()
                .map(|&r| {
                    let head = r.end.get() - 1;
                    DecodedAspect { tokens: r, head_token: head, clause_id: clause_ids[head - sent_tok_start] }
                })
                .collect();

            if !sent_aspects.is_empty() {
                // Pool each aspect's exact subword range via #263 for category and polarity scoring.
                let asp_spans: Vec<Span> = sent_aspects.iter().map(|a| token_range_span(a.tokens, tokens)).collect();
                let pooled_asps: Vectors<Aspects> =
                    self.vectors.cached_spans(Self::ID, subwords, &sentences.spans, cached, &asp_spans)?;

                for (a_pos, asp) in sent_aspects.iter().enumerate() {
                    let linked_local =
                        link_decoded_opinions(asp, a_pos, &sent_aspects, &sent_opinions, deps, sent_tok_end);
                    let linked_ixs: Vec<Ix<Opinions>> =
                        linked_local.iter().map(|&k| Ix::new((base_op_idx + k) as u32)).collect();

                    let asp_ix = Ix::<Aspects>::new(a_pos as u32);
                    let asp_pol = if !pooled_asps.is_empty(asp_ix) {
                        let norm_asp = unit_l2(pooled_asps.row(asp_ix));
                        self.polarity_head.score(&norm_asp, &mut pol_buf).map_err(fail)?;
                        self.polarity_head.polarity_scores_from_probs(&pol_buf)
                    } else {
                        PolarityScores::new(0.0, 0.0, 1.0)
                    };

                    // Combine the aspect span head polarity with the linked opinion span head polarity
                    // so two aspects with opposite opinions in the same sentence reflect their linked
                    // opinion span embeddings (#263).
                    let polarity = if !linked_local.is_empty() {
                        let mut p_sum = 0.0;
                        let mut n_sum = 0.0;
                        let mut u_sum = 0.0;
                        for &k in &linked_local {
                            p_sum += sent_opinions[k].pos_prob;
                            n_sum += sent_opinions[k].neg_prob;
                            u_sum += sent_opinions[k].neu_prob;
                        }
                        let inv = 1.0 / (linked_local.len() as f64);
                        let op_pol = PolarityScores::new(p_sum * inv, n_sum * inv, u_sum * inv);
                        PolarityScores::new(
                            0.25 * asp_pol.positive + 0.75 * op_pol.positive,
                            0.25 * asp_pol.negative + 0.75 * op_pol.negative,
                            0.25 * asp_pol.neutral + 0.75 * op_pol.neutral,
                        )
                    } else {
                        asp_pol
                    };

                    let category = if let Some(centroids) = &self.category_centroids
                        && !pooled_asps.is_empty(asp_ix)
                    {
                        let (cat_tag, _) =
                            centroids.score_single_label(pooled_asps.row(asp_ix), &mut cat_buf).map_err(fail)?;
                        Some(cat_tag)
                    } else {
                        self.categories.tag("general")
                    };

                    pending_aspects.push(PendingAspect {
                        target: AspectTarget::Term { tokens: asp.tokens, sentence: Some(sent_ix) },
                        category,
                        polarity,
                        opinions: linked_ixs,
                    });
                }
            } else if self.emit_implicit_aspects && !sent_opinions.is_empty() {
                // Implicit aspect anchored to the sentence when opinion expressions exist
                // without any explicit aspect term.
                let all_local: Vec<usize> = (0..sent_opinions.len()).collect();
                let linked_ixs: Vec<Ix<Opinions>> =
                    all_local.iter().map(|&k| Ix::new((base_op_idx + k) as u32)).collect();

                let sent_span = [sentences.spans.get(sent_ix)];
                let pooled_sent: Vectors<Sentences> =
                    self.vectors.cached_spans(Self::ID, subwords, &sentences.spans, cached, &sent_span)?;
                let s0 = Ix::<Sentences>::new(0);

                let mut p_sum = 0.0;
                let mut n_sum = 0.0;
                let mut u_sum = 0.0;
                for op in &sent_opinions {
                    p_sum += op.pos_prob;
                    n_sum += op.neg_prob;
                    u_sum += op.neu_prob;
                }
                let inv = 1.0 / (sent_opinions.len() as f64);
                let polarity = PolarityScores::new(p_sum * inv, n_sum * inv, u_sum * inv);

                let category = if let Some(centroids) = &self.category_centroids
                    && !pooled_sent.is_empty(s0)
                {
                    let (cat_tag, _) = centroids.score_single_label(pooled_sent.row(s0), &mut cat_buf).map_err(fail)?;
                    Some(cat_tag)
                } else {
                    self.categories.tag("general")
                };

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
}
