use super::*;
impl LabelCentroids {
    /// Builds `LabelCentroids` by averaging each label's example vectors via
    /// [`centroid`].
    pub fn from_vector_examples<V: AsRef<[f32]>>(
        model: ModelRef,
        dim: u32,
        tag_set_name: &str,
        examples: &[(&str, &[V])],
        norm: CentroidNorm,
        temperature: f64,
    ) -> Result<LabelCentroids, String> {
        if examples.is_empty() {
            return Err("from_vector_examples requires at least one label".into());
        }
        if dim == 0 {
            return Err("centroid dimension must be at least 1".into());
        }
        let mut labels = Vec::with_capacity(examples.len());
        let mut flat = Vec::new();
        for &(label, vecs) in examples {
            let trimmed = label.trim();
            if trimmed.is_empty() {
                return Err("centroid label must not be blank".into());
            }
            if vecs.is_empty() {
                return Err(format!("label {trimmed:?} has no example vectors"));
            }
            let c = centroid(vecs, norm)?.ok_or_else(|| format!("label {trimmed:?} produced an empty centroid"))?;
            if c.len() != dim as usize {
                return Err(format!("label {trimmed:?} centroid has length {}, expected {dim}", c.len()));
            }
            labels.push(trimmed);
            flat.extend_from_slice(&c);
        }
        let tags = TagSet::new(tag_set_name, &labels);
        Self::from_centroids(model, dim, tags, flat, temperature, norm)
    }

    /// Builds `LabelCentroids` directly from example texts per label by
    /// running `subwords` and `vectors` over each label's example sentences
    /// and averaging the resulting sentence vectors.
    pub fn from_text_examples<S: AsRef<str>>(
        vectors: &VectorAnnotator,
        subwords: &SubwordAnnotator,
        tag_set_name: &str,
        examples: &[(&str, &[S])],
        norm: CentroidNorm,
        temperature: f64,
    ) -> Result<LabelCentroids, Error> {
        let fail = |detail: String| Error::Annotator { annotator: "turbo:label-centroids", detail };
        if examples.is_empty() {
            return Err(fail("from_text_examples requires at least one label".into()));
        }
        let dim = vectors.dim();
        let mut labels = Vec::with_capacity(examples.len());
        let mut flat = Vec::new();
        for &(label, texts) in examples {
            let trimmed = label.trim();
            if trimmed.is_empty() {
                return Err(fail("centroid label must not be blank".into()));
            }
            if texts.is_empty() {
                return Err(fail(format!("label {trimmed:?} has no example texts")));
            }
            // Join the example sentences into one document with one sentence
            // span per example so they are embedded in a single batch.
            let mut joined = String::new();
            let mut spans = Spans::new(ViewId::SOURCE);
            for t in texts {
                let s = t.as_ref().trim();
                if s.is_empty() {
                    return Err(fail(format!("label {trimmed:?} has a blank example sentence")));
                }
                if !joined.is_empty() {
                    joined.push(' ');
                }
                let start = joined.len() as u32;
                joined.push_str(s);
                let end = joined.len() as u32;
                spans.push(Span::new(start, end));
            }
            let doc = Document::new(joined);
            let sub = subwords.pieces(doc.get::<turbo_analysis::View>().expect("source view"))?;
            let mut out = vectors.embed(
                "turbo:label-centroids",
                &[Input { subwords: &sub, sentences: Some(&spans), chunks: None, levels: None }],
            );
            let sent_vecs = out
                .remove(0)?
                .sentences
                .ok_or_else(|| fail("VectorAnnotator did not return sentence vectors".into()))?;
            if !sent_vecs.empty.is_empty() {
                return Err(fail(format!(
                    "label {trimmed:?} example sentence {} produced no subword pieces",
                    sent_vecs.empty[0].get()
                )));
            }
            let rows: Vec<&[f32]> = (0..spans.len()).map(|i| sent_vecs.row(Ix::new(i as u32))).collect();
            let c = centroid(&rows, norm)
                .map_err(fail)?
                .ok_or_else(|| fail(format!("label {trimmed:?} produced an empty centroid")))?;
            labels.push(trimmed);
            flat.extend_from_slice(&c);
        }
        let tags = TagSet::new(tag_set_name, &labels);
        Self::from_centroids(vectors.model().clone(), dim, tags, flat, temperature, norm).map_err(fail)
    }
}

/// Groups a document's sentences by their category label in
/// `Categories<Sentences>`, embeds each group's sentences, and averages
/// their sentence vectors into one centroid vector per present category,
/// generalizing `ChunkEmbedProcessor.buildCategoryGroup` on `opennlp-sandbox`
/// (`OPENNLP-1833-grpc-uber-server`).
///
/// Returns `(tag, sentence_indices, bounding_span, centroid_vector)` for each
/// category present in the document, in first-appearance order (or filtered
/// and ordered by `allowlist` when `Some`).
#[allow(clippy::type_complexity)]
pub fn category_sentence_centroids(
    sentences: &Sentences,
    categories: &Categories<Sentences>,
    sentence_vectors: &Vectors<Sentences>,
    allowlist: Option<&[Tag]>,
    norm: CentroidNorm,
) -> Result<Vec<(Tag, Vec<Ix<Sentences>>, Span, Vec<f32>)>, String> {
    sentences.check()?;
    categories.check()?;
    sentence_vectors.check()?;
    let n = sentences.len();
    if categories.len() != n {
        return Err(format!("categories has {} items, sentences has {n}", categories.len()));
    }
    if sentence_vectors.len() != n {
        return Err(format!("sentence_vectors has {} items, sentences has {n}", sentence_vectors.len()));
    }
    let mut buckets: Vec<(Tag, Vec<Ix<Sentences>>, u32, u32)> = Vec::new();
    for i in 0..n {
        let ix = Ix::<Sentences>::new(i as u32);
        if sentence_vectors.is_empty(ix) {
            continue;
        }
        let tag = categories.category[i];
        if let Some(allowed) = allowlist
            && !allowed.contains(&tag)
        {
            continue;
        }
        let span = sentences.spans.get(ix);
        if let Some(pos) = buckets.iter().position(|(t, _, _, _)| *t == tag) {
            buckets[pos].1.push(ix);
            buckets[pos].2 = buckets[pos].2.min(span.start);
            buckets[pos].3 = buckets[pos].3.max(span.end);
        } else {
            buckets.push((tag, vec![ix], span.start, span.end));
        }
    }
    if let Some(allowed) = allowlist {
        buckets.sort_by_key(|(t, _, _, _)| allowed.iter().position(|a| a == t).unwrap_or(usize::MAX));
    }
    let mut out = Vec::with_capacity(buckets.len());
    for (tag, indices, start, end) in buckets {
        let vecs: Vec<&[f32]> = indices.iter().map(|&ix| sentence_vectors.row(ix)).collect();
        if let Some(c) = centroid(&vecs, norm)? {
            out.push((tag, indices, Span::new(start, end), c));
        }
    }
    Ok(out)
}
