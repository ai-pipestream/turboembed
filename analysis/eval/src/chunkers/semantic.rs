//! A port of the OpenNLP sandbox's semantic chunker's decision rule
//! (`org.apache.opennlp.grpc.chunk.SemanticChunker` in the
//! ai-pipestream/opennlp-sandbox fork, branch
//! OPENNLP-1833-grpc-expansion), the baseline semantic splitter.
//!
//! Every sentence is embedded alone, and a chunk starts after each pair
//! of neighbouring sentences whose cosine is below the threshold. Chunks
//! of fewer than the minimum sentences are then merged into a neighbour,
//! and chunks of more than the maximum are split into windows of it, in
//! that order, so the maximum always holds and a split's remainder can be
//! under the minimum. The arithmetic is the Java's: cosines in f32 from
//! f32 products summed in f64, and the percentile by nearest rank.

use turbo_analysis::Document;
use turbo_analysis_layers::Sentences;

use super::{Chunker, chunk_span, need, setting};
use crate::Result;
use crate::analysis::sentence_run;
use crate::embedder::{Embedder, Role};
use crate::record::Setting;
use crate::spans::Span;

/// The threshold below which two neighbouring sentences are cut apart.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Threshold {
    /// A fixed cosine.
    Similarity(f32),
    /// This percentile, 1 to 99, of the document's own neighbour cosines.
    Percentile(u32),
}

/// The sandbox's default threshold when neither is set.
pub const DEFAULT_SIMILARITY: f32 = 0.5;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Semantic {
    threshold: Threshold,
    min_sentences: u32,
    max_sentences: Option<u32>,
}

impl Semantic {
    /// The chunker as the sandbox resolves its `SemanticChunkingConfig`:
    /// a `percentile_threshold` above 0 wins (one of 100 or more is
    /// refused), then a `similarity_threshold` above 0, then 0.5; a
    /// minimum of 0 or less is 1, and a maximum of 0 or less is none.
    pub fn from_config(similarity: f32, percentile: i32, min_sentences: i32, max_sentences: i32) -> Result<Semantic> {
        let threshold = if percentile > 0 {
            if percentile >= 100 {
                return Err(format!("a percentile threshold of {percentile}: it must be below 100"));
            }
            Threshold::Percentile(percentile as u32)
        } else if similarity > 0.0 {
            Threshold::Similarity(similarity)
        } else {
            Threshold::Similarity(DEFAULT_SIMILARITY)
        };
        Ok(Semantic {
            threshold,
            min_sentences: if min_sentences > 0 { min_sentences as u32 } else { 1 },
            max_sentences: (max_sentences > 0).then_some(max_sentences as u32),
        })
    }

    /// The threshold for a document whose neighbour cosines are `sims`.
    fn resolve(&self, sims: &[f32]) -> f32 {
        match self.threshold {
            Threshold::Similarity(t) => t,
            Threshold::Percentile(p) => {
                let mut sorted = sims.to_vec();
                sorted.sort_by(f32::total_cmp);
                let rank = (f64::from(p) / 100.0 * sorted.len() as f64).ceil() as i64 - 1;
                sorted[rank.clamp(0, sorted.len() as i64 - 1) as usize]
            }
        }
    }
}

/// `cosineSimilarity` of the sandbox: each product in f32, summed in f64,
/// the quotient rounded to f32; 0 when either vector is zero. The
/// f32 product rounding is pinned for sandbox parity and
/// intentionally not the shared widened dot in
/// `turbo_analysis::math`.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x * y);
        na += f64::from(x * x);
        nb += f64::from(y * y);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// The first sentence of each chunk, for `n` sentences whose neighbour
/// cosines are `sims` (`n - 1` of them): a start after every cosine below
/// `threshold`, then the small chunks merged, then the large ones split.
pub fn starts(sims: &[f32], threshold: f32, min: u32, max: Option<u32>, n: usize) -> Vec<usize> {
    let mut starts = vec![0];
    starts.extend(sims.iter().enumerate().filter(|&(_, &s)| s < threshold).map(|(i, _)| i + 1));
    let end = |starts: &[usize], i: usize| if i + 1 < starts.len() { starts[i + 1] - 1 } else { n - 1 };
    // mergeSmallChunks: a small chunk takes in the one after it; a small
    // last chunk goes into the one before; a single chunk stays.
    let min = min as usize;
    if min > 1 {
        let mut i = 0;
        while i < starts.len() {
            if end(&starts, i) - starts[i] + 1 >= min {
                i += 1;
            } else if i + 1 < starts.len() {
                starts.remove(i + 1);
            } else if i > 0 {
                starts.remove(i);
            } else {
                break;
            }
        }
    }
    // splitLargeChunks: a chunk over the maximum ends at it, and the rest
    // is checked in turn.
    if let Some(max) = max {
        let max = max as usize;
        let mut i = 0;
        while i < starts.len() {
            if end(&starts, i) - starts[i] + 1 > max {
                starts.insert(i + 1, starts[i] + max);
            }
            i += 1;
        }
    }
    starts
}

impl Chunker for Semantic {
    fn name(&self) -> &'static str {
        "opennlp-sandbox-semantic"
    }

    fn config(&self) -> Vec<Setting> {
        vec![
            match self.threshold {
                Threshold::Similarity(t) => setting("similarity_threshold", t),
                Threshold::Percentile(p) => setting("percentile_threshold", p),
            },
            setting("min_chunk_sentences", self.min_sentences),
            setting("max_chunk_sentences", self.max_sentences.map_or_else(|| "none".into(), |m| m.to_string())),
            setting("sentence_vectors", "each sentence embedded alone, document role"),
        ]
    }

    fn chunk(&self, doc: &Document, embedder: &mut dyn Embedder) -> Result<Vec<Span>> {
        let sentences = need::<Sentences>(doc, self.name())?;
        let n = sentences.spans.len();
        if n <= 1 {
            return Ok((n == 1).then(|| chunk_span(sentence_run(sentences, 0, 1))).flatten().into_iter().collect());
        }
        let text = doc.source();
        let texts: Vec<&str> = sentences.spans.iter().map(|(_, s)| &text[s.range()]).collect();
        let mut vectors = Vec::new();
        embedder.embed(&texts, Role::Document, &mut vectors)?;
        let dim = embedder.dim();
        let rows: Vec<&[f32]> = vectors.chunks_exact(dim).collect();
        let sims: Vec<f32> = rows.windows(2).map(|w| cosine(w[0], w[1])).collect();
        let starts = starts(&sims, self.resolve(&sims), self.min_sentences, self.max_sentences, n);
        Ok(starts
            .iter()
            .enumerate()
            .filter_map(|(k, &i)| chunk_span(sentence_run(sentences, i, starts.get(k + 1).copied().unwrap_or(n))))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunkers::tests::{analyzed, texts};
    use crate::embedder::stub::Stub;

    #[test]
    fn the_sandbox_config_resolves_as_the_java_does() {
        let t = |c: Semantic| c.threshold;
        assert_eq!(t(Semantic::from_config(0.0, 0, 0, 0).unwrap()), Threshold::Similarity(0.5));
        assert_eq!(t(Semantic::from_config(0.7, 0, 0, 0).unwrap()), Threshold::Similarity(0.7));
        assert_eq!(t(Semantic::from_config(0.7, 90, 0, 0).unwrap()), Threshold::Percentile(90));
        assert!(Semantic::from_config(0.0, 100, 0, 0).is_err());
        let c = Semantic::from_config(0.0, 0, -3, 0).unwrap();
        assert_eq!((c.min_sentences, c.max_sentences), (1, None));
        assert_eq!(c.config()[2], setting("max_chunk_sentences", "none"));
    }

    #[test]
    fn cosine_is_the_javas() {
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
        assert_eq!(cosine(&[2.0, 0.0], &[3.0, 0.0]), 1.0);
        // 1 / sqrt(2), rounded to f32.
        assert_eq!(cosine(&[1.0, 1.0], &[1.0, 0.0]), std::f32::consts::FRAC_1_SQRT_2);
    }

    #[test]
    fn merge_then_split_by_hand() {
        // Five sentences, every neighbour pair below the threshold: five
        // chunks of one.
        let all = [0.0; 4];
        assert_eq!(starts(&all, 0.5, 1, None, 5), [0, 1, 2, 3, 4]);
        // Minimum 2: chunk 0 takes in chunk 1 ([0, 2, 3, 4]); chunk 1 (2)
        // takes in 3 ([0, 2, 4]); the last (4) is alone and goes into the
        // one before: [0, 2], chunks 0-1 and 2-4.
        assert_eq!(starts(&all, 0.5, 2, None, 5), [0, 2]);
        // Maximum 2 with nothing cut: 0-1, 2-3, 4.
        assert_eq!(starts(&[0.9; 4], 0.5, 1, Some(2), 5), [0, 2, 4]);
        // Both: the merge gives 0-1 and 2-4, the split cuts 2-4 into 2-3
        // and 4, a remainder under the minimum.
        assert_eq!(starts(&all, 0.5, 2, Some(2), 5), [0, 2, 4]);
        // Fewer sentences than the minimum: one chunk.
        assert_eq!(starts(&[0.0], 0.5, 3, None, 2), [0]);
        // The comparison is strict: a cosine at the threshold is no cut.
        assert_eq!(starts(&[0.5, 0.4], 0.5, 1, None, 3), [0, 2]);
    }

    #[test]
    fn cuts_where_the_topic_turns() {
        // Vectors over (cat, rust): the first two lines are (1, 0), the
        // last two (0, 1). Neighbour cosines: 1, 0, 1.
        let text = "cat purrs\ncat naps\nrust compiles\nrust ships\n";
        let mut e = Stub::new(&["cat", "rust"]);
        let doc = analyzed(text, &e, None);
        let run = |c: Semantic, e: &mut Stub| texts(text, &c.chunk(&doc, e).unwrap()).join("|");
        let two = "cat purrs\ncat naps|rust compiles\nrust ships";
        // 0 < 0.5: one cut, before the third line.
        assert_eq!(run(Semantic::from_config(0.0, 0, 0, 0).unwrap(), &mut e), two);
        // Percentile 50 of (0, 1, 1): rank ceil(1.5) = 2, the value 1; the
        // cosine 0 is below it, the two 1s are not.
        assert_eq!(run(Semantic::from_config(0.0, 50, 0, 0).unwrap(), &mut e), two);
        // Percentile 10: rank ceil(0.3) = 1, the value 0; nothing is below.
        assert_eq!(run(Semantic::from_config(0.0, 10, 0, 0).unwrap(), &mut e), text.trim_end());
        // Threshold 1.5: every pair is cut, then the maximum is moot.
        assert_eq!(
            run(Semantic::from_config(1.5, 0, 0, 1).unwrap(), &mut e),
            "cat purrs|cat naps|rust compiles|rust ships"
        );
        // One sentence is one chunk, with nothing embedded.
        let one = analyzed("cat alone", &e, None);
        let s = Semantic::from_config(0.0, 0, 0, 0).unwrap().chunk(&one, &mut e).unwrap();
        assert_eq!(texts("cat alone", &s), ["cat alone"]);
    }

    // Golden parity with the Java reference's own test suite
    // (SemanticChunkerTest in the OpenNLP sandbox fork, #955). Its stub
    // provider maps a sentence starting with "A" to (1, 0, 0) and any
    // other to (0, 1, 0), so neighbour cosines are exactly 1 or 0 and
    // each case reduces to the decision rule in `starts`. The expected
    // groups below are the Java tests' asserted sentence indices.
    #[test]
    fn the_sandbox_java_test_cases_hold() {
        let groups = |sims: &[f32], similarity: f32, min: i32, max: i32| {
            let c = Semantic::from_config(similarity, 0, min, max).unwrap();
            let n = sims.len() + 1;
            let starts = starts(sims, c.resolve(sims), c.min_sentences, c.max_sentences, n);
            starts
                .iter()
                .enumerate()
                .map(|(k, &s)| (s..starts.get(k + 1).copied().unwrap_or(n)).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        // splitsWhenAdjacentSentenceSimilarityIsLow: "Aa.Ab.Bc.", 0.9.
        assert_eq!(groups(&[1.0, 0.0], 0.9, 0, 0), [vec![0, 1], vec![2]]);
        // mergesUndersizedTrailingChunkIntoPrecedingChunk: same, min 2.
        assert_eq!(groups(&[1.0, 0.0], 0.9, 2, 0), [vec![0, 1, 2]]);
        // mergesUndersizedLeadingChunkWithFollowingChunk: "Ba.Ab.Ac.".
        assert_eq!(groups(&[0.0, 1.0], 0.9, 2, 0), [vec![0, 1, 2]]);
        // splitsChunksLargerThanMaxChunkSentences: four A sentences, max 2.
        assert_eq!(groups(&[1.0, 1.0, 1.0], 0.9, 0, 2), [vec![0, 1], vec![2, 3]]);
        // cosineSimilarityIsOneForIdenticalVectors.
        assert_eq!(cosine(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]), 1.0);
    }

    // The Java suite has no percentile case, so these ranks are derived
    // from the reference's own formula (nearest rank over the sorted
    // cosines: index ceil(p / 100 * len) - 1, clamped), not transcribed.
    #[test]
    fn the_sandbox_percentile_is_the_javas_rank() {
        let at = |sims: &[f32], p: i32| Semantic::from_config(0.0, p, 0, 0).unwrap().resolve(sims);
        assert_eq!(at(&[0.1, 0.5, 0.9], 10), 0.1);
        assert_eq!(at(&[0.1, 0.5, 0.9], 50), 0.5);
        assert_eq!(at(&[0.1, 0.5, 0.9], 99), 0.9);
        assert_eq!(at(&[0.2, 0.4, 0.6, 0.8], 25), 0.2);
        assert_eq!(at(&[0.2, 0.4, 0.6, 0.8], 50), 0.4);
        assert_eq!(at(&[0.2, 0.4, 0.6, 0.8], 75), 0.6);
        // Sorting is the implementation's, so input order does not matter.
        assert_eq!(at(&[0.9, 0.1, 0.5], 50), 0.5);
    }

    // The first Java case end to end, spans included: two topic-A
    // sentences and one topic-B sentence at threshold 0.9 give chunks
    // [0, 1] and [2]; with a minimum of 2 the trailing chunk is
    // absorbed, exactly as the Java merge does.
    #[test]
    fn the_first_sandbox_java_case_end_to_end() {
        let text = "alpha one\nalpha two\nbeta three\n";
        let mut e = Stub::new(&["alpha", "beta"]);
        let doc = analyzed(text, &e, None);
        let cut = Semantic::from_config(0.9, 0, 0, 0).unwrap().chunk(&doc, &mut e).unwrap();
        assert_eq!(texts(text, &cut), ["alpha one\nalpha two", "beta three"]);
        let merged = Semantic::from_config(0.9, 0, 2, 0).unwrap().chunk(&doc, &mut e).unwrap();
        assert_eq!(texts(text, &merged), [text.trim_end()]);
    }
}
