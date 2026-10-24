//! Learned aspect and opinion extraction over contextual subword vectors and
//! span-pooled embeddings (`#322`).
//!
//! - [`LearnedAspectSentiment`] (`turbo:aspect-sentiment-learned`) reads
//!   `(View, Sentences, Tokens, Subwords, Option<Vectors<Subwords>>, Option<Dependencies>, Option<PhraseChunks>)`
//!   and writes `(Opinions, Aspects)`, matching the output layer schema of
//!   `RuleAspectSentiment` (`turbo-doccat:aspect-sentiment-rule`) while
//!   recording the learned encoder, heads, and execution backend in
//!   provenance.
//! - **Subword BIO tagging**: scores each piece in [`Subwords`]
//!   (`opennlp:subwords`) over its contextual subword vector
//!   (`Vectors<Subwords>`) using a 5-class BIO [`ScoringHead`]
//!   ([`BIO_ASPECT_OPINION_LABELS`]: `["O", "B-ASP", "I-ASP", "B-OP", "I-OP"]`),
//!   projects subword tag distributions back onto [`Tokens`] (`opennlp:tokens`)
//!   via `Subwords` byte-span and token alignments, and decodes contiguous
//!   aspect term (`B-ASP`/`I-ASP`) and opinion expression (`B-OP`/`I-OP`)
//!   token ranges.
//! - **Span-pooled polarity (`#263`)**: pools the exact subword range of each
//!   extracted opinion and aspect (or linked aspect + opinion span) via
//!   [`VectorAnnotator::embed_spans`] and scores three-way [`PolarityScores`]
//!   with the polarity [`ScoringHead`].
//! - **Aspect categories**: scores each aspect's pooled embedding against
//!   [`LabelCentroids`] over an aspect category [`TagSet`] (for example,
//!   `"battery"`, `"display"`, `"service"`, `"food"`, `"price"`).
//! - **Implicit aspects**: when a sentence contains extracted opinion
//!   expressions (`B-OP`/`I-OP`) without any explicit aspect term, anchors an
//!   [`AspectTarget::Implicit`] aspect to the sentence, pools its span via
//!   `#263`, and scores polarity and category.

mod decode;
mod fit;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Execution, Ix, IxRange, Layer, ModelRef, Refs, Span, Tag, TagSet, View};
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Dependencies, Opinion, Opinions, PhraseChunks, PolarityScores, Sentences, Subwords,
    Tokens, Vectors,
};

use self::decode::{
    DecodedAspect, DecodedOpinion, TAG_O, assign_clauses, decode_bio_spans, is_token_negated, link_decoded_opinions,
    token_range_span, unit_l2,
};
use crate::centroids::{LabelCentroids, fnv1a};
use crate::heads::ScoringHead;
use crate::vectors::VectorAnnotator;
use crate::window::Which;

/// Standard 5-class BIO tag labels for joint aspect term and opinion
/// expression extraction (`O`, `B-ASP`, `I-ASP`, `B-OP`, `I-OP`).
pub const BIO_ASPECT_OPINION_LABELS: &[&str; 5] = &["O", "B-ASP", "I-ASP", "B-OP", "I-OP"];

/// Builds the canonical 5-label BIO [`TagSet`] (`"bio-aspect-opinion"`).
pub fn bio_aspect_opinion_tagset() -> Arc<TagSet> {
    TagSet::new("bio-aspect-opinion", BIO_ASPECT_OPINION_LABELS)
}

/// One labeled training document for fitting [`LearnedAspectSentiment`] heads:
/// its subwords, sentences, tokens, explicit aspect spans with category and
/// polarity label (`"positive"`, `"negative"`, `"neutral"`), and opinion
/// spans with polarity label.
pub struct LabeledAbsaDoc<'a> {
    pub subwords: &'a Subwords,
    pub sentences: &'a Sentences,
    pub tokens: &'a Tokens,
    /// Explicit aspect `(token_range, category_label, polarity_label)`.
    pub aspects: &'a [(IxRange<Tokens>, &'a str, &'a str)],
    /// Opinion `(token_range, polarity_label)`.
    pub opinions: &'a [(IxRange<Tokens>, &'a str)],
}

/// Learned aspect and opinion extractor (`turbo:aspect-sentiment-learned`,
/// `#322`).
#[derive(Clone)]
pub struct LearnedAspectSentiment {
    pub vectors: Arc<VectorAnnotator>,
    pub bio_head: ScoringHead,
    pub polarity_head: ScoringHead,
    pub category_centroids: Option<Arc<LabelCentroids>>,
    pub categories: Arc<TagSet>,
    pub emit_implicit_aspects: bool,
}

impl LearnedAspectSentiment {
    /// Creates a new `LearnedAspectSentiment` annotator.
    pub fn new(
        vectors: Arc<VectorAnnotator>,
        bio_head: impl Into<ScoringHead>,
        polarity_head: impl Into<ScoringHead>,
        category_centroids: Option<Arc<LabelCentroids>>,
    ) -> Self {
        let categories = category_centroids
            .as_ref()
            .map(|c| Arc::clone(c.tags()))
            .unwrap_or_else(|| TagSet::new("aspect-categories", &["general"]));
        Self {
            vectors,
            bio_head: bio_head.into(),
            polarity_head: polarity_head.into(),
            category_centroids,
            categories,
            emit_implicit_aspects: true,
        }
    }

    /// Configures whether sentences with extracted opinion expressions but no
    /// explicit aspect term emit an [`AspectTarget::Implicit`] entry.
    pub fn with_implicit_aspects(mut self, enabled: bool) -> Self {
        self.emit_implicit_aspects = enabled;
        self
    }

    /// Overrides the aspect category [`TagSet`] when `category_centroids` is
    /// `None`.
    pub fn with_categories(mut self, categories: Arc<TagSet>) -> Self {
        if self.category_centroids.is_none() {
            self.categories = categories;
        }
        self
    }
}

impl Annotator for LearnedAspectSentiment {
    const ID: &'static str = "turbo:aspect-sentiment-learned";
    type Reads =
        (View, Sentences, Tokens, Subwords, Option<Vectors<Subwords>>, Option<Dependencies>, Option<PhraseChunks>);
    type Writes = (Opinions, Aspects);

    fn annotate(
        &self,
        (view, sentences, tokens, subwords, subword_vectors, deps, phrase_chunks): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if sentences.spans.view() != view.id() {
            return Err(fail("sentences are not in the given view".into()));
        }
        if tokens.spans.view() != view.id() {
            return Err(fail("tokens are not in the given view".into()));
        }
        if subwords.spans.view() != view.id() {
            return Err(fail("subwords are not in the given view".into()));
        }
        if phrase_chunks.is_some_and(|pc| pc.spans.view() != view.id()) {
            return Err(fail("phrase chunks are not in the given view".into()));
        }

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
            token_lower.push(turbo_doccat::sentiment::embed_token_key(slice));
        }

        // 1. Obtain contextual subword vectors (`Vectors<Subwords>`), either from the document
        // or by running `self.vectors.embed_subwords` over the sentence windows (#263).
        let owned_sub_vecs;
        let sub_vecs: &Vectors<Subwords> = match subword_vectors {
            Some(v) => {
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
                owned_sub_vecs = self.vectors.embed_subwords(Self::ID, subwords, Some(&sentences.spans))?;
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
                Some(self.vectors.embed_spans(
                    Self::ID,
                    subwords,
                    Some(&sentences.spans),
                    tokens.spans.view(),
                    &op_spans,
                )?)
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
                let pooled_asps: Vectors<Aspects> = self.vectors.embed_spans(
                    Self::ID,
                    subwords,
                    Some(&sentences.spans),
                    tokens.spans.view(),
                    &asp_spans,
                )?;

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
                let pooled_sent: Vectors<Sentences> = self.vectors.embed_spans(
                    Self::ID,
                    subwords,
                    Some(&sentences.spans),
                    sentences.spans.view(),
                    &sent_span,
                )?;
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

    fn model(&self) -> Option<ModelRef> {
        Some(self.vectors.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.vectors.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        let mut buf = Vec::with_capacity(33);
        buf.extend_from_slice(&self.vectors.config(Which::Sentences).to_le_bytes());
        buf.extend_from_slice(&self.bio_head.config_hash().to_le_bytes());
        buf.extend_from_slice(&self.polarity_head.config_hash().to_le_bytes());
        if let Some(c) = &self.category_centroids {
            buf.extend_from_slice(&c.config_hash().to_le_bytes());
        }
        buf.push(u8::from(self.emit_implicit_aspects));
        fnv1a(&buf)
    }
}
