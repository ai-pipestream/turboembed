use super::*;
impl LabelCentroids {
    /// Computes one `f64` dot product per label between `vector` and each
    /// label's centroid into `out`.
    pub fn dot_scores(&self, vector: &[f32], out: &mut [f64]) -> Result<(), String> {
        let d = self.dim as usize;
        if vector.len() != d {
            return Err(format!("input vector has length {}, expected {d}", vector.len()));
        }
        if out.len() != self.tags.len() {
            return Err(format!("output slice has length {}, expected {}", out.len(), self.tags.len()));
        }
        for (slot, centroid_row) in out.iter_mut().zip(self.centroids.chunks_exact(d)) {
            let mut sum = 0.0f64;
            for (&a, &b) in vector.iter().zip(centroid_row) {
                sum += f64::from(a) * f64::from(b);
            }
            *slot = sum;
        }
        // The centroids are finite (from_centroids), so a value of `vector`
        // that is not makes every sum non-finite, even against a zero
        // centroid (inf * 0 is NaN), and finite f32 products cannot overflow
        // an f64 sum: the first label's sum answers what a pass over the
        // vector would. A head has at least one label.
        if out.first().is_some_and(|s| !s.is_finite()) {
            return Err("input vector contains a non-finite value".into());
        }
        Ok(())
    }

    /// Scores `vector` across all labels using temperature-scaled softmax over
    /// the `f64` dot products, writing the normalized probabilities to `probs`
    /// and returning the winning `(Tag, probability)`.
    pub fn score_single_label(&self, vector: &[f32], probs: &mut [f64]) -> Result<(Tag, f64), String> {
        self.dot_scores(vector, probs)?;
        let inv_temp = 1.0 / self.temperature;
        if inv_temp.is_finite() && probs.iter().all(|p| (*p * inv_temp).is_finite()) {
            for p in probs.iter_mut() {
                *p *= inv_temp;
            }
        } else {
            // Subtract before dividing so tiny temperatures retain the winner.
            let max = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for p in probs.iter_mut() {
                *p = (*p - max) / self.temperature;
            }
        }
        softmax_in_place(probs);
        Ok(argmax_tag(probs))
    }

    /// Scores `vector` across all labels for multi-label classification by
    /// mapping each dot product `d` through `sigmoid((d - bias) * scale)` in
    /// `f64` (`0.0..=1.0`).
    pub fn score_multi_label(&self, vector: &[f32], bias: f64, scale: f64, scores: &mut [f64]) -> Result<(), String> {
        if !bias.is_finite() || !scale.is_finite() {
            return Err("multi-label bias and scale must be finite".into());
        }
        self.dot_scores(vector, scores)?;
        for s in scores.iter_mut() {
            let z = (*s - bias) * scale;
            *s = sigmoid(z);
        }
        Ok(())
    }

    /// Scores `vector` across all labels for multi-label incivility
    /// classification (`sigmoid((dot - bias) * scale)`) and maps the tag names
    /// into [`IncivilityScores`].
    pub fn score_incivility_vector(&self, vector: &[f32], bias: f64, scale: f64) -> Result<IncivilityScores, String> {
        let mut buf = vec![0.0f64; self.tags.len()];
        self.score_multi_label(vector, bias, scale, &mut buf)?;
        let mut out = IncivilityScores::default();
        for (i, &val) in buf.iter().enumerate() {
            let label = self.tags.label(Tag(i as u16));
            let parsed = IncivilityLabel::from_name(label)
                .ok_or_else(|| format!("centroid label {label:?} is not a valid IncivilityLabel"))?;
            out.set(parsed, val);
        }
        Ok(out)
    }

    /// Classifies every item in `vectors` (`opennlp:embeddings<L>`) into
    /// single-label [`Categories<L>`] (`turbo:categories<L>`).
    ///
    /// Items listed in `vectors.empty` receive uniform probability `1.0 / K`
    /// and the neutral label (if one named `"neutral"` exists in `tags`) or
    /// tag `0`.
    pub fn categorize_vectors<L: CategoryScope>(
        &self,
        annotator: &'static str,
        vectors: &Vectors<L>,
    ) -> Result<Categories<L>, Error> {
        let fail = |detail: String| Error::Annotator { annotator, detail };
        vectors.check().map_err(fail)?;
        if vectors.model != self.model {
            return Err(fail(format!(
                "vectors are from model {} (revision {}), centroids are for {} (revision {})",
                vectors.model.id, vectors.model.revision, self.model.id, self.model.revision
            )));
        }
        if vectors.dim != self.dim {
            return Err(fail(format!("vectors have dim {}, centroids have dim {}", vectors.dim, self.dim)));
        }
        let n = vectors.len();
        let n_labels = self.tags.len();
        let empty_tag = self.tags.tag("neutral").unwrap_or(Tag(0));
        let empty_probs = vec![1.0 / n_labels as f64; n_labels];
        let mut out = Categories::with_capacity(Arc::clone(&self.tags), n);
        let mut probs = vec![0.0f64; n_labels];
        for i in 0..n {
            let ix = Ix::<L>::new(i as u32);
            if vectors.is_empty(ix) {
                out.push(empty_tag, &empty_probs);
            } else {
                let (tag, _) = self.score_single_label(vectors.row(ix), &mut probs).map_err(fail)?;
                out.push(tag, &probs);
            }
        }
        Ok(out)
    }
}
