//! Centroid vectors and prototype (label centroid) classification over
//! turboembed embeddings (`#321`).
//!
//! Generalizes `org.apache.opennlp.grpc.chunk.Centroids` and
//! `ChunkEmbedProcessor` from `opennlp-sandbox` (`OPENNLP-1833-grpc-uber-server`):
//!
//! - [`centroid`] computes the element-wise arithmetic mean of a non-empty
//!   slice of equal-length `f32` vectors in `f64`, with optional post-mean
//!   L2 normalization ([`CentroidNorm`]).
//! - [`LabelCentroids`] represents a set of class labels where each label is
//!   defined by a set of example sentences (or vectors) averaged into one
//!   prototype centroid vector. Inference is one `f64` dot product per label
//!   followed by temperature-scaled softmax (single-label [`Categories`]) or
//!   scaled sigmoid (multi-label).
//! - Centroids load from and save to a plain typed binary format (`TCENT001`)
//!   with no pickle and no JSON trickery.

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;

use turbo_analysis::{Document, Error, Ix, Key, Layer, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Categories, CategoryScope, Sentences, Vectors};

use crate::subwords::SubwordAnnotator;
use crate::vectors::{Input, VectorAnnotator};

const CENTROID_MAGIC: &[u8; 8] = b"TCENT001";

/// Post-aggregation normalization applied to an arithmetic mean centroid,
/// matching `org.apache.opennlp.grpc.v1.VectorNormalization` in
/// `Centroids.java`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum CentroidNorm {
    /// Leave the element-wise arithmetic mean unnormalized.
    #[default]
    None = 0,
    /// Scale the arithmetic mean to unit L2 length; fails if the mean vector
    /// has zero norm.
    L2 = 1,
}

impl CentroidNorm {
    fn from_byte(b: u8) -> Result<CentroidNorm, String> {
        match b {
            0 => Ok(CentroidNorm::None),
            1 => Ok(CentroidNorm::L2),
            other => Err(format!("unrecognized centroid normalization code {other}")),
        }
    }
}

/// Computes the element-wise centroid (mean) vector of `vectors` in `f64`
/// and optionally L2-normalizes the mean, matching
/// `org.apache.opennlp.grpc.chunk.Centroids.centroid`.
///
/// Returns `Ok(None)` when `vectors` is empty, or `Err` when vectors do not
/// share a positive dimension or when `CentroidNorm::L2` is requested on a
/// zero-norm mean vector.
pub fn centroid<V: AsRef<[f32]>>(vectors: &[V], norm: CentroidNorm) -> Result<Option<Vec<f32>>, String> {
    if vectors.is_empty() {
        return Ok(None);
    }
    let dim = vectors[0].as_ref().len();
    if dim == 0 {
        return Err("centroid vectors must have a non-zero dimension".into());
    }
    let mut sums = vec![0.0f64; dim];
    for (idx, v) in vectors.iter().enumerate() {
        let slice = v.as_ref();
        if slice.len() != dim {
            return Err(format!("vector {idx} has length {}, expected {dim}", slice.len()));
        }
        for (s, &val) in sums.iter_mut().zip(slice) {
            if !val.is_finite() {
                return Err(format!("vector {idx} contains a non-finite value"));
            }
            *s += f64::from(val);
        }
    }
    let count = vectors.len() as f64;
    for s in &mut sums {
        *s /= count;
    }
    let divisor = match norm {
        CentroidNorm::None => 1.0f64,
        CentroidNorm::L2 => {
            let sq_norm: f64 = sums.iter().map(|&v| v * v).sum();
            let d = sq_norm.sqrt();
            if d == 0.0 || !d.is_finite() {
                return Err("embedding centroid has zero norm and cannot be L2-normalized".into());
            }
            d
        }
    };
    let out: Vec<f32> = sums.into_iter().map(|v| (v / divisor) as f32).collect();
    Ok(Some(out))
}

/// Prototype (label centroid) classifier over turboembed vectors.
///
/// Each label in `tags` is represented by one centroid vector of length
/// `dim`. Inference computes one `f64` dot product per label.
#[derive(Clone, Debug)]
pub struct LabelCentroids {
    model: ModelRef,
    dim: u32,
    tags: Arc<TagSet>,
    /// Row-major `[tags.len(), dim]` centroid vectors.
    centroids: Vec<f32>,
    temperature: f64,
    norm: CentroidNorm,
    hash: u64,
}

impl PartialEq for LabelCentroids {
    fn eq(&self, other: &Self) -> bool {
        self.model == other.model
            && self.dim == other.dim
            && self.tags.name == other.tags.name
            && self.tags.len() == other.tags.len()
            && (0..self.tags.len()).all(|i| self.tags.label(Tag(i as u16)) == other.tags.label(Tag(i as u16)))
            && self.centroids == other.centroids
            && self.temperature == other.temperature
            && self.norm == other.norm
            && self.hash == other.hash
    }
}

impl LabelCentroids {
    /// Builds `LabelCentroids` from pre-computed row-major centroid vectors
    /// (`[tags.len(), dim]`).
    pub fn from_centroids(
        model: ModelRef,
        dim: u32,
        tags: Arc<TagSet>,
        centroids: Vec<f32>,
        temperature: f64,
        norm: CentroidNorm,
    ) -> Result<LabelCentroids, String> {
        if dim == 0 {
            return Err("centroid dimension must be at least 1".into());
        }
        if tags.is_empty() {
            return Err("LabelCentroids requires at least one label in its TagSet".into());
        }
        let expected = tags.len() * dim as usize;
        if centroids.len() != expected {
            return Err(format!(
                "centroids buffer has {} elements, expected {} ({} labels * dim {})",
                centroids.len(),
                expected,
                tags.len(),
                dim
            ));
        }
        if let Some(k) = centroids.iter().position(|v| !v.is_finite()) {
            return Err(format!("centroid element {k} is non-finite"));
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(format!("temperature {temperature} must be finite and positive"));
        }
        let mut out = LabelCentroids { model, dim, tags, centroids, temperature, norm, hash: 0 };
        out.hash = fnv1a(&out.to_bytes());
        Ok(out)
    }

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
        let mut flat = Vec::with_capacity(examples.len() * dim as usize);
        for &(label, vecs) in examples {
            let trimmed = label.trim();
            if trimmed.is_empty() {
                return Err("centroid label must not be blank".into());
            }
            if vecs.is_empty() {
                return Err(format!("label {trimmed:?} has no example vectors"));
            }
            let c = centroid(vecs, norm)?
                .ok_or_else(|| format!("label {trimmed:?} produced an empty centroid"))?;
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
        let mut flat = Vec::with_capacity(examples.len() * dim as usize);
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
            let c = centroid(&rows, norm).map_err(fail)?
                .ok_or_else(|| fail(format!("label {trimmed:?} produced an empty centroid")))?;
            labels.push(trimmed);
            flat.extend_from_slice(&c);
        }
        let tags = TagSet::new(tag_set_name, &labels);
        Self::from_centroids(vectors.model().clone(), dim, tags, flat, temperature, norm).map_err(fail)
    }

    pub fn model(&self) -> &ModelRef {
        &self.model
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn tags(&self) -> &Arc<TagSet> {
        &self.tags
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn norm(&self) -> CentroidNorm {
        self.norm
    }

    pub fn config_hash(&self) -> u64 {
        self.hash
    }

    /// The centroid vector of `tag`.
    pub fn centroid_of(&self, tag: Tag) -> &[f32] {
        let d = self.dim as usize;
        let k = tag.0 as usize;
        &self.centroids[k * d..(k + 1) * d]
    }

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
        Ok(())
    }

    /// Scores `vector` across all labels using temperature-scaled softmax over
    /// the `f64` dot products, writing the normalized probabilities to `probs`
    /// and returning the winning `(Tag, probability)`.
    pub fn score_single_label(&self, vector: &[f32], probs: &mut [f64]) -> Result<(Tag, f64), String> {
        self.dot_scores(vector, probs)?;
        let inv_temp = 1.0 / self.temperature;
        for p in probs.iter_mut() {
            *p *= inv_temp;
        }
        softmax_in_place(probs);
        Ok(argmax_tag(probs))
    }

    /// Scores `vector` across all labels for multi-label classification by
    /// mapping each dot product `d` through `sigmoid((d - bias) * scale)` in
    /// `f64` (`0.0..=1.0`).
    pub fn score_multi_label(
        &self,
        vector: &[f32],
        bias: f64,
        scale: f64,
        scores: &mut [f64],
    ) -> Result<(), String> {
        self.dot_scores(vector, scores)?;
        for s in scores.iter_mut() {
            let z = (*s - bias) * scale;
            *s = sigmoid(z);
        }
        Ok(())
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

    /// Serializes these centroids to the plain typed `TCENT001` binary format.
    pub fn write_to<W: Write>(&self, mut w: W) -> io::Result<()> {
        w.write_all(CENTROID_MAGIC)?;
        w.write_all(&self.dim.to_le_bytes())?;
        w.write_all(&(self.tags.len() as u32).to_le_bytes())?;
        w.write_all(&[self.norm as u8])?;
        w.write_all(&self.temperature.to_le_bytes())?;
        write_str(&mut w, &self.model.id)?;
        write_str(&mut w, &self.model.revision)?;
        write_str(&mut w, &self.model.sha256)?;
        write_str(&mut w, &self.tags.name)?;
        for i in 0..self.tags.len() {
            write_str(&mut w, self.tags.label(Tag(i as u16)))?;
        }
        for &val in &self.centroids {
            w.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Serializes these centroids into an in-memory byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_to(&mut buf).expect("writing to Vec<u8> cannot fail");
        buf
    }

    /// Writes these centroids to `path` in the `TCENT001` binary format.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let bytes = self.to_bytes();
        fs::write(path, bytes)
    }

    /// Loads `LabelCentroids` from a byte slice in the `TCENT001` binary
    /// format.
    pub fn from_bytes(bytes: &[u8]) -> Result<LabelCentroids, String> {
        let mut cur = Cursor::new(bytes);
        let magic = cur.read_array::<8>()?;
        if &magic != CENTROID_MAGIC {
            return Err("invalid LabelCentroids header (expected TCENT001)".into());
        }
        let dim = cur.read_u32()?;
        let n_labels = cur.read_u32()? as usize;
        let norm = CentroidNorm::from_byte(cur.read_u8()?)?;
        let temperature = cur.read_f64()?;
        let model = ModelRef {
            id: cur.read_string()?,
            revision: cur.read_string()?,
            sha256: cur.read_string()?,
        };
        let tag_set_name = cur.read_string()?;
        if n_labels == 0 {
            return Err("TCENT001 file has 0 labels".into());
        }
        if dim == 0 {
            return Err("TCENT001 file has dim 0".into());
        }
        let mut labels = Vec::with_capacity(n_labels);
        for _ in 0..n_labels {
            labels.push(cur.read_string()?);
        }
        let total = n_labels
            .checked_mul(dim as usize)
            .ok_or_else(|| "TCENT001 dimensions overflow usize".to_string())?;
        let mut centroids = Vec::with_capacity(total);
        for _ in 0..total {
            centroids.push(cur.read_f32()?);
        }
        cur.finish()?;
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let tags = TagSet::new(&tag_set_name, &label_refs);
        Self::from_centroids(model, dim, tags, centroids, temperature, norm)
    }

    /// Reads `LabelCentroids` from any reader in the `TCENT001` format.
    pub fn read_from<R: Read>(mut r: R) -> Result<LabelCentroids, String> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        Self::from_bytes(&buf)
    }

    /// Loads `LabelCentroids` from `path`.
    pub fn load(path: &Path) -> Result<LabelCentroids, String> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
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

pub(crate) fn softmax_in_place(logits: &mut [f64]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0f64;
    for z in logits.iter_mut() {
        *z = (*z - max_val).exp();
        sum += *z;
    }
    if sum > 0.0 && sum.is_finite() {
        let inv = 1.0 / sum;
        for z in logits.iter_mut() {
            *z *= inv;
        }
    } else {
        let u = 1.0 / logits.len() as f64;
        logits.fill(u);
    }
}

pub(crate) fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

pub(crate) fn argmax_tag(scores: &[f64]) -> (Tag, f64) {
    let mut best_idx = 0usize;
    let mut best_score = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > best_score {
            best_score = s;
            best_idx = i;
        }
    }
    (Tag(best_idx as u16), best_score)
}

pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

pub(crate) fn write_str<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(bytes)
}

pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, pos: 0 }
    }

    pub(crate) fn read_array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let end = self.pos.checked_add(N).ok_or_else(|| "unexpected EOF".to_string())?;
        if end > self.bytes.len() {
            return Err("unexpected EOF".into());
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..end]);
        self.pos = end;
        Ok(out)
    }

    pub(crate) fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.read_array::<1>()?[0])
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.read_array::<4>()?))
    }

    pub(crate) fn read_f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.read_array::<4>()?))
    }

    pub(crate) fn read_f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.read_array::<8>()?))
    }

    pub(crate) fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u32()? as usize;
        let end = self.pos.checked_add(len).ok_or_else(|| "string length overflow".to_string())?;
        if end > self.bytes.len() {
            return Err("unexpected EOF while reading string".into());
        }
        let s = std::str::from_utf8(&self.bytes[self.pos..end]).map_err(|e| format!("invalid UTF-8: {e}"))?;
        self.pos = end;
        Ok(s.to_owned())
    }

    pub(crate) fn finish(&self) -> Result<(), String> {
        if self.pos != self.bytes.len() {
            return Err(format!("{} trailing byte(s) after model data", self.bytes.len() - self.pos));
        }
        Ok(())
    }
}

/// Helper to attach a `Categories<L>` layer to a document with provenance.
pub fn add_categories<L: CategoryScope>(
    doc: &mut Document,
    key: &Key<Categories<L>>,
    categories: Categories<L>,
    by: &'static str,
) -> Result<(), Error> {
    doc.add(key, categories, Provenance::by(by))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> ModelRef {
        ModelRef {
            id: "tiny-bert".into(),
            revision: "rev1".into(),
            sha256: "0123456789abcdef".into(),
        }
    }

    // Parity tests matching CentroidsTest.java on OPENNLP-1833-grpc-uber-server.

    #[test]
    fn averages_vectors_element_wise() {
        let v1 = [0.0f32, 2.0, 4.0];
        let v2 = [2.0f32, 4.0, 8.0];
        let c = centroid(&[&v1[..], &v2[..]], CentroidNorm::None).unwrap().unwrap();
        assert_eq!(c, vec![1.0f32, 3.0, 6.0]);
    }

    #[test]
    fn l2_normalizes_the_arithmetic_mean() {
        let v1 = [3.0f32, 4.0];
        let v2 = [3.0f32, 4.0];
        let c = centroid(&[&v1[..], &v2[..]], CentroidNorm::L2).unwrap().unwrap();
        assert_eq!(c, vec![0.6f32, 0.8]);
    }

    #[test]
    fn refuses_to_normalize_a_zero_centroid() {
        let v1 = [1.0f32, 0.0];
        let v2 = [-1.0f32, 0.0];
        let err = centroid(&[&v1[..], &v2[..]], CentroidNorm::L2).unwrap_err();
        assert!(err.contains("zero norm"), "{err}");
    }

    #[test]
    fn single_vector_is_its_own_centroid() {
        let v1 = [1.5f32, -2.5];
        let c = centroid(&[&v1[..]], CentroidNorm::None).unwrap().unwrap();
        assert_eq!(c, vec![1.5f32, -2.5]);
    }

    #[test]
    fn empty_input_yields_none() {
        let empty: [&[f32]; 0] = [];
        assert_eq!(centroid(&empty, CentroidNorm::None).unwrap(), None);
    }

    #[test]
    fn label_centroids_round_trip_and_deterministic_scoring() {
        let pos_examples: &[&[f32]] = &[&[0.9, 0.1, 0.0], &[0.8, 0.2, 0.0]];
        let neg_examples: &[&[f32]] = &[&[-0.9, -0.1, 0.0], &[-0.8, -0.2, 0.0]];
        let neu_examples: &[&[f32]] = &[&[0.0, 0.0, 1.0], &[0.0, 0.1, 0.9]];
        let lc = LabelCentroids::from_vector_examples(
            test_model(),
            3,
            "sentiment",
            &[("positive", pos_examples), ("negative", neg_examples), ("neutral", neu_examples)],
            CentroidNorm::L2,
            0.25,
        )
        .unwrap();

        let bytes = lc.to_bytes();
        let loaded = LabelCentroids::from_bytes(&bytes).unwrap();
        assert_eq!(lc, loaded);

        let mut probs1 = [0.0f64; 3];
        let mut probs2 = [0.0f64; 3];
        let (tag1, p1) = lc.score_single_label(&[0.95, 0.05, 0.0], &mut probs1).unwrap();
        let (tag2, p2) = loaded.score_single_label(&[0.95, 0.05, 0.0], &mut probs2).unwrap();
        assert_eq!(tag1, Tag(0));
        assert_eq!(tag1, tag2);
        assert_eq!(p1.to_bits(), p2.to_bits());
        assert_eq!(probs1, probs2);
        assert!((probs1.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
