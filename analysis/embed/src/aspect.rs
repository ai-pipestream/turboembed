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

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Execution, Ix, IxRange, Layer, ModelRef, Refs, Span, Tag, TagSet, View};
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Dependencies, Opinion, Opinions, PhraseChunks, PolarityScores, Sentences, Subwords,
    Tokens, Vectors,
};

use crate::centroids::{CentroidNorm, LabelCentroids, fnv1a};
use crate::heads::{ClassifierHead, ScoringHead};
use crate::vectors::VectorAnnotator;
use crate::window::{Which, pieces_of};

/// Standard 5-class BIO tag labels for joint aspect term and opinion
/// expression extraction (`O`, `B-ASP`, `I-ASP`, `B-OP`, `I-OP`).
pub const BIO_ASPECT_OPINION_LABELS: &[&str; 5] = &["O", "B-ASP", "I-ASP", "B-OP", "I-OP"];

const TAG_O: u16 = 0;
const TAG_B_ASP: u16 = 1;
const TAG_I_ASP: u16 = 2;
const TAG_B_OP: u16 = 3;
const TAG_I_OP: u16 = 4;

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

    /// Fits a `LearnedAspectSentiment` deterministically from labeled ABSA
    /// documents:
    /// - Extracts contextual subword vectors via [`VectorAnnotator::embed_subwords`]
    ///   and fits a 5-class BIO [`ClassifierHead`] (`O`, `B-ASP`, `I-ASP`,
    ///   `B-OP`, `I-OP`).
    /// - Pools opinion and aspect subword ranges via [`VectorAnnotator::embed_spans`]
    ///   (`#263`) and fits a 3-class polarity [`ClassifierHead`] (`positive`,
    ///   `negative`, `neutral`).
    /// - Computes [`LabelCentroids`] over aspect category labels from the
    ///   pooled aspect vectors.
    pub fn fit(
        vectors: Arc<VectorAnnotator>,
        category_labels: &[&str],
        docs: &[LabeledAbsaDoc<'_>],
        l2_reg: f64,
        epochs: usize,
        lr: f64,
    ) -> Result<Self, String> {
        let dim = vectors.dim();
        let model = vectors.model().clone();
        let bio_tags = bio_aspect_opinion_tagset();
        let pol_tags = TagSet::new("sentiment", &["positive", "negative", "neutral"]);
        let cat_tags = TagSet::new("aspect-categories", category_labels);

        let mut bio_vecs: Vec<(Vec<f32>, Tag)> = Vec::new();
        let mut pol_vecs: Vec<(Vec<f32>, Tag)> = Vec::new();
        let mut cat_buckets: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cat_tags.len()];

        for doc in docs {
            let sub_vecs = vectors
                .embed_subwords("turbo:aspect-sentiment-learned:fit", doc.subwords, Some(&doc.sentences.spans))
                .map_err(|e| e.to_string())?;

            // Label every token with BIO tag, then project token BIO tags to subwords.
            let n_tokens = doc.tokens.len();
            let mut tok_bio = vec![Tag(TAG_O); n_tokens];
            for &(range, _, _) in doc.aspects {
                let s = range.start.get();
                let e = range.end.get().min(n_tokens);
                if s < e {
                    tok_bio[s] = Tag(TAG_B_ASP);
                    for item in &mut tok_bio[(s + 1)..e] {
                        *item = Tag(TAG_I_ASP);
                    }
                }
            }
            for &(range, _) in doc.opinions {
                let s = range.start.get();
                let e = range.end.get().min(n_tokens);
                if s < e {
                    tok_bio[s] = Tag(TAG_B_OP);
                    for item in &mut tok_bio[(s + 1)..e] {
                        *item = Tag(TAG_I_OP);
                    }
                }
            }

            let tok_pieces = token_subword_ranges(doc.tokens, doc.subwords);
            for (t_idx, p_range) in tok_pieces.iter().enumerate() {
                let piece_tag = tok_bio[t_idx];
                for p_idx in p_range.start..p_range.end {
                    let ix = Ix::<Subwords>::new(p_idx);
                    if !sub_vecs.is_empty(ix) {
                        bio_vecs.push((unit_l2(sub_vecs.row(ix)), piece_tag));
                    }
                }
            }

            // Pool opinion spans via #263 for polarity head training.
            if !doc.opinions.is_empty() {
                let op_spans: Vec<Span> = doc.opinions.iter().map(|&(r, _)| token_range_span(r, doc.tokens)).collect();
                let pooled_ops: Vectors<Opinions> = vectors
                    .embed_spans(
                        "turbo:aspect-sentiment-learned:fit",
                        doc.subwords,
                        Some(&doc.sentences.spans),
                        doc.tokens.spans.view(),
                        &op_spans,
                    )
                    .map_err(|e| e.to_string())?;
                for (i, &(_, pol_label)) in doc.opinions.iter().enumerate() {
                    let ix = Ix::<Opinions>::new(i as u32);
                    if !pooled_ops.is_empty(ix)
                        && let Some(ptag) = pol_tags.tag(pol_label)
                    {
                        pol_vecs.push((unit_l2(pooled_ops.row(ix)), ptag));
                    }
                }
            }

            // Pool aspect spans via #263 for category centroids (and polarity head when no opinions).
            if !doc.aspects.is_empty() {
                let asp_spans: Vec<Span> =
                    doc.aspects.iter().map(|&(r, _, _)| token_range_span(r, doc.tokens)).collect();
                let pooled_asps: Vectors<Aspects> = vectors
                    .embed_spans(
                        "turbo:aspect-sentiment-learned:fit",
                        doc.subwords,
                        Some(&doc.sentences.spans),
                        doc.tokens.spans.view(),
                        &asp_spans,
                    )
                    .map_err(|e| e.to_string())?;
                for (i, &(_, cat_label, pol_label)) in doc.aspects.iter().enumerate() {
                    let ix = Ix::<Aspects>::new(i as u32);
                    if !pooled_asps.is_empty(ix) {
                        let row = unit_l2(pooled_asps.row(ix));
                        if let Some(ctag) = cat_tags.tag(cat_label) {
                            cat_buckets[ctag.0 as usize].push(row.clone());
                        }
                        if doc.opinions.is_empty()
                            && let Some(ptag) = pol_tags.tag(pol_label)
                        {
                            pol_vecs.push((row, ptag));
                        }
                    }
                }
            }
        }

        let bio_refs: Vec<(&[f32], Tag)> = bio_vecs.iter().map(|(v, t)| (v.as_slice(), *t)).collect();
        let bio_head = ClassifierHead::fit_single_label(model.clone(), dim, bio_tags, &bio_refs, l2_reg, epochs, lr)?;

        // Ensure `pol_vecs` has at least one example; if neutral was unseen in training,
        // add a zero-vector neutral anchor so 3-way softmax stays calibrated.
        if !pol_vecs.iter().any(|(_, t)| *t == Tag(2)) {
            pol_vecs.push((vec![0.0f32; dim as usize], Tag(2)));
        }
        let pol_refs: Vec<(&[f32], Tag)> = pol_vecs.iter().map(|(v, t)| (v.as_slice(), *t)).collect();
        let polarity_head =
            ClassifierHead::fit_single_label(model.clone(), dim, pol_tags, &pol_refs, l2_reg, epochs, lr)?;

        // Build LabelCentroids for categories that have at least one training vector;
        // any category without examples gets a tiny deterministic orthogonal unit vector.
        for (c_idx, bucket) in cat_buckets.iter_mut().enumerate() {
            if bucket.is_empty() {
                let mut fallback = vec![0.0f32; dim as usize];
                fallback[c_idx % (dim as usize)] = 1.0;
                bucket.push(fallback);
            }
        }
        let cat_pairs: Vec<(&str, &[Vec<f32>])> =
            category_labels.iter().zip(&cat_buckets).map(|(&lbl, bucket)| (lbl, bucket.as_slice())).collect();
        let centroids =
            LabelCentroids::from_vector_examples(model, dim, "aspect-categories", &cat_pairs, CentroidNorm::L2, 0.1)?;

        Ok(Self::new(vectors, bio_head, polarity_head, Some(Arc::new(centroids))))
    }
}

#[derive(Clone, Debug)]
struct DecodedOpinion {
    tokens: IxRange<Tokens>,
    head_token: usize,
    polarity: f64,
    intensity: f64,
    negated: bool,
    pos_prob: f64,
    neg_prob: f64,
    neu_prob: f64,
    clause_id: usize,
}

#[derive(Clone, Debug)]
struct DecodedAspect {
    tokens: IxRange<Tokens>,
    head_token: usize,
    clause_id: usize,
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
        let tok_pieces = token_subword_ranges(tokens, subwords);
        let mut tok_tags = vec![TAG_O; n_tokens];
        for t in 0..n_tokens {
            let r = tok_pieces[t].clone();
            if r.is_empty() {
                tok_tags[t] = TAG_O;
                continue;
            }
            let first = r.start as usize;
            let mut avg = piece_probs[first];
            let count = (r.end - r.start) as f64;
            if count > 1.0 {
                for p in (r.start + 1)..r.end {
                    let pp = &piece_probs[p as usize];
                    for k in 0..5 {
                        avg[k] += pp[k];
                    }
                }
                for val in &mut avg {
                    *val /= count;
                }
            }
            let mut best_k = 0usize;
            for k in 1..5 {
                if avg[k] > avg[best_k] {
                    best_k = k;
                }
            }
            tok_tags[t] = best_k as u16;
        }

        let mut token_lower: Vec<String> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let slice = text.get(tok_starts[t] as usize..tok_ends[t] as usize).unwrap_or("");
            token_lower
                .push(slice.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-').to_ascii_lowercase());
        }

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
                if negated {
                    std::mem::swap(&mut pos_p, &mut neg_p);
                }
                let signed_prior = if negated { -(pos_p - neg_p) } else { pos_p - neg_p };
                let intensity = (pos_p + neg_p).clamp(0.1, 2.0);
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

fn unit_l2(v: &[f32]) -> Vec<f32> {
    let norm_sq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    if norm_sq <= 1e-24 {
        return v.to_vec();
    }
    let inv = (1.0 / norm_sq.sqrt()) as f32;
    v.iter().map(|&x| x * inv).collect()
}

/// Maps each token in `tokens` to its overlapping subword piece range
/// `start..end` in `subwords`.
fn token_subword_ranges(tokens: &Tokens, subwords: &Subwords) -> Vec<std::ops::Range<u32>> {
    let n_tokens = tokens.len();
    if let Some(tok_col) = &subwords.token {
        let mut out = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let target = Ix::<Tokens>::new(t as u32);
            let a = tok_col.partition_point(|&x| x < target);
            let b = a + tok_col[a..].partition_point(|&x| x == target);
            out.push(a as u32..b as u32);
        }
        return out;
    }
    let (sub_starts, sub_ends) = subwords.spans.columns();
    let (tok_starts, tok_ends) = tokens.spans.columns();
    let mut out = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        out.push(pieces_of(sub_starts, sub_ends, tok_starts[t], tok_ends[t]));
    }
    out
}

fn token_range_span(range: IxRange<Tokens>, tokens: &Tokens) -> Span {
    let first = tokens.spans.get(range.start);
    let last = tokens.spans.get(Ix::new(range.end.get() as u32 - 1));
    Span::new(first.start, last.end)
}

fn decode_bio_spans(
    tok_tags: &[u16],
    token_lower: &[String],
    clause_ids: &[usize],
    sent_start: usize,
    sent_end: usize,
    phrase_chunks: Option<&PhraseChunks>,
) -> (Vec<IxRange<Tokens>>, Vec<IxRange<Tokens>>) {
    let mut aspects: Vec<IxRange<Tokens>> = Vec::new();
    let mut opinions: Vec<IxRange<Tokens>> = Vec::new();

    let mut t = sent_start;
    while t < sent_end {
        let tag = tok_tags[t];
        if tag == TAG_B_ASP || tag == TAG_I_ASP {
            let start = t;
            let clause = clause_ids[t - sent_start];
            t += 1;
            while t < sent_end
                && (tok_tags[t] == TAG_I_ASP || tok_tags[t] == TAG_B_ASP)
                && clause_ids[t - sent_start] == clause
                && !token_lower[t].is_empty()
            {
                t += 1;
            }
            // Trim leading determiners or empty punctuation tokens if any were tagged.
            let mut s = start;
            while s < t && (token_lower[s].is_empty() || is_determiner(&token_lower[s])) {
                s += 1;
            }
            let mut e = t;
            while e > s && token_lower[e - 1].is_empty() {
                e -= 1;
            }
            if s < e {
                aspects.push(IxRange::new(s as u32, e as u32));
            }
        } else if tag == TAG_B_OP || tag == TAG_I_OP {
            let start = t;
            let clause = clause_ids[t - sent_start];
            t += 1;
            while t < sent_end
                && (tok_tags[t] == TAG_I_OP || tok_tags[t] == TAG_B_OP)
                && clause_ids[t - sent_start] == clause
                && !token_lower[t].is_empty()
            {
                t += 1;
            }
            let mut s = start;
            while s < t && token_lower[s].is_empty() {
                s += 1;
            }
            let mut e = t;
            while e > s && token_lower[e - 1].is_empty() {
                e -= 1;
            }
            if s < e {
                opinions.push(IxRange::new(s as u32, e as u32));
            }
        } else {
            t += 1;
        }
    }

    // If `PhraseChunks` (`opennlp:chunks`) is present, expand any single-token aspect inside an `NP`
    // chunk to include adjacent noun-compound tokens in that `NP` that are not tagged as opinions.
    if let Some(chunks) = phrase_chunks {
        for asp in &mut aspects {
            for (c_idx, crange) in chunks.tokens.iter().enumerate() {
                if !chunks.kinds.label(chunks.kind[c_idx]).eq_ignore_ascii_case("NP") {
                    continue;
                }
                if crange.start.get() <= asp.start.get() && asp.end.get() <= crange.end.get() {
                    let mut s = crange.start.get();
                    while s < asp.start.get()
                        && (token_lower[s].is_empty()
                            || is_determiner(&token_lower[s])
                            || opinions.iter().any(|o| o.start.get() <= s && s < o.end.get()))
                    {
                        s += 1;
                    }
                    if s < asp.start.get() {
                        asp.start = Ix::new(s as u32);
                    }
                    break;
                }
            }
        }
    }

    (aspects, opinions)
}

fn is_determiner(w: &str) -> bool {
    matches!(
        w,
        "the"
            | "a"
            | "an"
            | "this"
            | "that"
            | "these"
            | "those"
            | "my"
            | "our"
            | "your"
            | "his"
            | "her"
            | "its"
            | "their"
    )
}

fn assign_clauses(
    token_lower: &[String],
    text: &str,
    tok_starts: &[u32],
    tok_ends: &[u32],
    sent_start: usize,
    sent_end: usize,
) -> Vec<usize> {
    let mut clauses = Vec::with_capacity(sent_end - sent_start);
    let mut current = 0usize;
    for tok in sent_start..sent_end {
        let w = token_lower[tok].as_str();
        let raw = text.get(tok_starts[tok] as usize..tok_ends[tok] as usize).unwrap_or("");
        if tok > sent_start
            && (matches!(w, "but" | "however" | "yet" | "although" | "though" | "whereas" | "while" | "except" | "and")
                || raw == ";"
                || raw == ",")
        {
            current += 1;
        }
        clauses.push(current);
    }
    clauses
}

fn is_token_negated(
    tok: usize,
    sent_start: usize,
    sent_end: usize,
    token_lower: &[String],
    clause_ids: &[usize],
    deps: Option<&Dependencies>,
) -> bool {
    if let Some(d) = deps
        && d.len() >= sent_end
    {
        for i in sent_start..sent_end {
            if !is_negation_word(&token_lower[i]) {
                continue;
            }
            let h = d.head[i].map(|ix| ix.get());
            if h == Some(tok) {
                return true;
            }
            if let Some(th) = d.head[tok].map(|ix| ix.get())
                && h == Some(th)
                && i < tok
                && clause_ids[i - sent_start] == clause_ids[tok - sent_start]
            {
                return true;
            }
        }
    }
    let min_i = sent_start.max(tok.saturating_sub(3));
    let c = clause_ids[tok - sent_start];
    for i in min_i..tok {
        if clause_ids[i - sent_start] == c && is_negation_word(&token_lower[i]) {
            return true;
        }
    }
    false
}

fn is_negation_word(w: &str) -> bool {
    matches!(
        w,
        "not"
            | "no"
            | "never"
            | "neither"
            | "nor"
            | "none"
            | "hardly"
            | "barely"
            | "scarcely"
            | "without"
            | "cannot"
            | "cant"
    ) || w.ends_with("n't")
}

fn link_decoded_opinions(
    asp: &DecodedAspect,
    asp_pos: usize,
    sent_aspects: &[DecodedAspect],
    sent_opinions: &[DecodedOpinion],
    deps: Option<&Dependencies>,
    sent_end: usize,
) -> Vec<usize> {
    if sent_opinions.is_empty() {
        return Vec::new();
    }
    let mut linked = Vec::new();

    if let Some(d) = deps
        && d.len() >= sent_end
    {
        let t_tok = asp.head_token;
        let t_head = d.head[t_tok].map(|ix| ix.get());
        for (o_idx, op) in sent_opinions.iter().enumerate() {
            let o_tok = op.head_token;
            let o_head = d.head[o_tok].map(|ix| ix.get());
            if o_head == Some(t_tok) || t_head == Some(o_tok) || (t_head.is_some() && t_head == o_head) {
                linked.push(o_idx);
            }
        }
        if !linked.is_empty() {
            return linked;
        }
    }

    let same_clause: Vec<(usize, &DecodedAspect)> =
        sent_aspects.iter().enumerate().filter(|(_, a)| a.clause_id == asp.clause_id).collect();
    for (o_idx, op) in sent_opinions.iter().enumerate() {
        if op.clause_id != asp.clause_id {
            continue;
        }
        let closest = same_clause.iter().min_by_key(|(_, a)| token_distance(a.tokens, op.tokens)).map(|(idx, _)| *idx);
        if closest == Some(asp_pos) {
            linked.push(o_idx);
        }
    }
    if !linked.is_empty() {
        return linked;
    }

    if let Some((o_idx, _)) = sent_opinions
        .iter()
        .enumerate()
        .filter(|(_, op)| op.clause_id == asp.clause_id)
        .min_by_key(|(_, op)| token_distance(asp.tokens, op.tokens))
    {
        linked.push(o_idx);
        return linked;
    }

    if let Some((o_idx, _)) =
        sent_opinions.iter().enumerate().min_by_key(|(_, op)| token_distance(asp.tokens, op.tokens))
    {
        linked.push(o_idx);
    }
    linked
}

fn token_distance(a: IxRange<Tokens>, b: IxRange<Tokens>) -> usize {
    if a.end.get() <= b.start.get() {
        b.start.get() - a.end.get()
    } else if b.end.get() <= a.start.get() {
        a.start.get() - b.end.get()
    } else {
        0
    }
}
