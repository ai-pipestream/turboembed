//! Training and fitting implementation for [`super::LearnedAspectSentiment`].

use std::sync::Arc;

use turbo_analysis::{Ix, Layer, Span, Tag, TagSet};
use turbo_analysis_layers::{Aspects, Opinions, Subwords, Vectors};

use super::decode::{TAG_B_ASP, TAG_B_OP, TAG_I_ASP, TAG_I_OP, TAG_O, token_range_span, token_subword_ranges, unit_l2};
use super::{LabeledAbsaDoc, LearnedAspectSentiment, bio_aspect_opinion_tagset};
use crate::centroids::{CentroidNorm, LabelCentroids};
use crate::heads::ClassifierHead;
use crate::vectors::VectorAnnotator;

impl LearnedAspectSentiment {
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
        Self::fit_inner(vectors, category_labels, docs, l2_reg, epochs, lr, false)
    }

    /// Fits with deterministic background down-sampling for corpora where
    /// background pieces otherwise overwhelm every BIO class.
    pub fn fit_balanced(
        vectors: Arc<VectorAnnotator>,
        category_labels: &[&str],
        docs: &[LabeledAbsaDoc<'_>],
        l2_reg: f64,
        epochs: usize,
        lr: f64,
    ) -> Result<Self, String> {
        Self::fit_inner(vectors, category_labels, docs, l2_reg, epochs, lr, true)
    }

    fn fit_inner(
        vectors: Arc<VectorAnnotator>,
        category_labels: &[&str],
        docs: &[LabeledAbsaDoc<'_>],
        l2_reg: f64,
        epochs: usize,
        lr: f64,
        balance_bio: bool,
    ) -> Result<Self, String> {
        let dim = vectors.dim();
        if dim == 0 {
            return Err("LearnedAspectSentiment::fit requires vector dimension > 0".into());
        }
        if category_labels.is_empty() {
            return Err("LearnedAspectSentiment::fit requires at least one category label".into());
        }
        if docs.is_empty() {
            return Err("LearnedAspectSentiment::fit requires at least one training document".into());
        }
        let model = vectors.model().clone();
        let bio_tags = bio_aspect_opinion_tagset();
        let pol_tags = TagSet::new("sentiment", &["positive", "negative", "neutral"]);
        let cat_tags = TagSet::new("aspect-categories", category_labels);

        let mut bio_vecs: Vec<(Vec<f32>, Tag)> = Vec::new();
        let mut pol_vecs: Vec<(Vec<f32>, Tag)> = Vec::new();
        let mut cat_buckets: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cat_tags.len()];

        for doc in docs {
            let n_tokens = doc.tokens.len();
            for &(range, _, _) in doc.aspects {
                if range.start.get() >= range.end.get() || range.end.get() > n_tokens {
                    return Err(format!(
                        "aspect token range {}..{} is empty or out of bounds for {} tokens",
                        range.start.get(),
                        range.end.get(),
                        n_tokens
                    ));
                }
            }
            for &(range, _) in doc.opinions {
                if range.start.get() >= range.end.get() || range.end.get() > n_tokens {
                    return Err(format!(
                        "opinion token range {}..{} is empty or out of bounds for {} tokens",
                        range.start.get(),
                        range.end.get(),
                        n_tokens
                    ));
                }
            }

            let sub_vecs = vectors
                .embed_subwords("turbo:aspect-sentiment-learned:fit", doc.subwords, Some(&doc.sentences.spans))
                .map_err(|e| e.to_string())?;

            // Label every token with BIO tag, then project token BIO tags to subwords.
            let mut tok_bio = vec![Tag(TAG_O); n_tokens];
            for &(range, _, _) in doc.aspects {
                let s = range.start.get();
                let e = range.end.get();
                tok_bio[s] = Tag(TAG_B_ASP);
                for item in &mut tok_bio[(s + 1)..e] {
                    *item = Tag(TAG_I_ASP);
                }
            }
            for &(range, _) in doc.opinions {
                let s = range.start.get();
                let e = range.end.get();
                tok_bio[s] = Tag(TAG_B_OP);
                for item in &mut tok_bio[(s + 1)..e] {
                    *item = Tag(TAG_I_OP);
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

        if balance_bio {
            let largest_non_background =
                (1..5).map(|tag| bio_vecs.iter().filter(|(_, t)| t.0 == tag).count()).max().unwrap_or(0).max(1);
            let background_count = bio_vecs.iter().filter(|(_, t)| *t == Tag(TAG_O)).count();
            let background_stride = (background_count / largest_non_background).max(1);
            let mut background_seen = 0usize;
            bio_vecs.retain(|(_, tag)| {
                if *tag != Tag(TAG_O) {
                    return true;
                }
                let keep = background_seen.is_multiple_of(background_stride);
                background_seen += 1;
                keep
            });
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
