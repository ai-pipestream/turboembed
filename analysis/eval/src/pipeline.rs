//! A retrieval pipeline over one evaluation set: analyze each document,
//! chunk it, embed each chunk once, embed each query the same way
//! without chunking, and rank by exact cosine (`retrieve`). The run is
//! scored with `metrics` and `spans`, and the cost of each stage is
//! measured as it runs.

use std::collections::BTreeSet;
use std::time::Instant;

use crate::Result;
use crate::analysis::Analysis;
use crate::chunkers::Chunker;
use crate::cost::{Cost, Distribution, StageTimes};
use crate::embedder::{Counted, Embedder, Role};
use crate::metrics::{Run, evaluate};
use crate::record::{self, EVAL_RECORD_VERSION, EvalRecord, Metrics, Provenance, SetId};
use crate::retrieve::Index;
use crate::set::LoadedSet;
use crate::spans::{ChunkRun, evaluate_spans};

/// Texts given to the embedder in one call, to bound the memory its rows
/// take.
const EMBED_BATCH: usize = 1024;

/// What a pipeline does with a document's title, when the set has one.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Titles {
    /// Left out: only the text is chunked and embedded.
    Ignore,
    /// Put before each chunk's text, with a line break between, in what
    /// is embedded; spans stay in the text.
    Prefix,
}

pub struct Pipeline {
    pub analysis: Analysis,
    pub chunker: Box<dyn Chunker>,
    pub embedder: Box<dyn Embedder>,
    pub titles: Titles,
}

/// What a pipeline did on a set.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub run: Run,
    pub chunks: ChunkRun,
    pub metrics: Metrics,
    pub cost: Cost,
}

/// Where the pipeline was built from, as a record names it.
#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    /// The commit, 40 hex.
    pub commit: String,
    /// The branches of origin that contain it, `origin/<name>`.
    pub pushed_to: Vec<String>,
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

impl Pipeline {
    /// The pipeline as a record describes it.
    pub fn description(&self) -> record::Pipeline {
        let mut analyzer = self.analysis.names();
        if self.titles == Titles::Prefix {
            analyzer.push("title prefixed to each chunk".into());
        }
        record::Pipeline {
            analyzer,
            chunker: record::Chunker { name: self.chunker.name().into(), config: self.chunker.config() },
        }
    }

    /// Runs the pipeline over the set and scores it.
    pub fn evaluate(&mut self, set: &LoadedSet) -> Result<Outcome> {
        set.check()?;
        let embedder = &mut Counted { inner: self.embedder.as_mut(), tokens: 0 };

        let t = Instant::now();
        let docs = set
            .docs
            .iter()
            .map(|d| self.analysis.analyze(&d.text, embedder).map_err(|e| format!("document {}: {e}", d.id)))
            .collect::<Result<Vec<_>>>()?;
        let analysis_ms = ms(t);

        let t = Instant::now();
        let (mut doc_of, mut spans) = (Vec::new(), Vec::new());
        for (i, (d, doc)) in set.docs.iter().zip(&docs).enumerate() {
            let chunks = self.chunker.chunk(doc, embedder).map_err(|e| format!("document {}: {e}", d.id))?;
            let mut seen = BTreeSet::new();
            for s in chunks {
                if s.end > d.text.len() || !seen.insert(s) {
                    return Err(format!(
                        "document {}: the chunker gave the span [{}, {}) twice or past the text",
                        d.id, s.start, s.end
                    ));
                }
                doc_of.push(i as u32);
                spans.push(s);
            }
        }
        let chunking_ms = ms(t);
        drop(docs);
        if spans.is_empty() {
            return Err(format!("set {}: the chunker made no chunks", set.name));
        }

        let t = Instant::now();
        let texts: Vec<String> = doc_of
            .iter()
            .zip(&spans)
            .map(|(&d, s)| {
                let d = &set.docs[d as usize];
                let text = &d.text[s.start..s.end];
                match (self.titles, &d.title) {
                    (Titles::Prefix, Some(title)) => format!("{title}\n{text}"),
                    _ => text.to_owned(),
                }
            })
            .collect();
        let chunk_tokens: Vec<u64> = texts.iter().map(|t| embedder.count(t, Role::Document)).collect();
        let dim = embedder.dim();
        let vectors = embed_all(embedder, &texts, Role::Document)?;
        drop(texts);
        let queries: Vec<&str> = set.queries.iter().map(|q| q.text.as_str()).collect();
        let query_vectors = embed_all(embedder, &queries, Role::Query)?;
        let embedding_ms = ms(t);

        let t = Instant::now();
        let ids: Vec<String> = set.docs.iter().map(|d| d.id.clone()).collect();
        let chunk_bytes: Vec<u64> = spans.iter().map(|s| (s.end - s.start) as u64).collect();
        let count = spans.len() as u64;
        let index = Index { dim, vectors, doc: doc_of, span: spans, docs: &ids };
        let names: Vec<&str> = set.queries.iter().map(|q| q.id.as_str()).collect();
        let (run, chunks) = index.search(&names, &query_vectors)?;
        let retrieval_ms = ms(t);

        let doc_scores = evaluate(&run, &set.qrels)?;
        let span_scores = set.spans.as_ref().map(|r| evaluate_spans(&chunks, r)).transpose()?;
        let metrics = Metrics::new(&doc_scores, span_scores.as_ref())?;
        let cost = Cost {
            wall: StageTimes { analysis_ms, chunking_ms, embedding_ms, retrieval_ms },
            tokens_embedded: embedder.tokens,
            chunks: count,
            chunk_bytes: Distribution::of(&chunk_bytes)?,
            chunk_tokens: Distribution::of(&chunk_tokens)?,
        };
        Ok(Outcome { run, chunks, metrics, cost })
    }

    /// Runs the pipeline over the set and makes its record, checked.
    pub fn record(&mut self, set: &LoadedSet, source: &Source, recorded_at: String) -> Result<EvalRecord> {
        let outcome = self.evaluate(set)?;
        let id = self.embedder.identity();
        let r = EvalRecord {
            record_version: EVAL_RECORD_VERSION,
            recorded_at,
            provenance: Provenance {
                commit: source.commit.clone(),
                pushed_to: source.pushed_to.clone(),
                machine: id.machine.clone(),
                device: id.device.clone(),
                bundle: id.bundle.clone(),
            },
            set: SetId { name: set.name.clone(), sha256: set.sha256.clone() },
            pipeline: self.description(),
            metrics: outcome.metrics,
            cost: outcome.cost,
        };
        r.check()?;
        Ok(r)
    }
}

/// Every text's vector, in order, embedded `EMBED_BATCH` at a time.
fn embed_all<S: AsRef<str>>(embedder: &mut dyn Embedder, texts: &[S], role: Role) -> Result<Vec<f32>> {
    let dim = embedder.dim();
    let mut out = Vec::with_capacity(texts.len() * dim);
    let mut part = Vec::new();
    for group in texts.chunks(EMBED_BATCH) {
        let refs: Vec<&str> = group.iter().map(AsRef::as_ref).collect();
        embedder.embed(&refs, role, &mut part)?;
        out.extend_from_slice(&part);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::analysis::Sentencer;
    use crate::chunkers::{Semantic, SentenceWindows};
    use crate::embedder::stub::Stub;
    use crate::metrics::{Qrel, Qrels};
    use crate::run::{gate_dirs, record_path, run_sets};
    use crate::set::{Doc, Query};
    use crate::spans::{Span, SpanRel, SpanRels};

    const VOCAB: [&str; 4] = ["cat", "dog", "rust", "tea"];

    fn doc(id: &str, title: Option<&str>, text: &str) -> Doc {
        Doc { id: id.into(), title: title.map(String::from), text: text.into() }
    }

    /// Three documents, a sentence per line; over (cat, dog, rust, tea)
    /// the stub's chunk vectors are
    ///
    /// - d1: "cat purrs" (1, 0, 0, 0) and "rust compiles" (0, 0, 1, 0);
    /// - d2: "dog barks" (0, 1, 0, 0) and "cat sleeps with dog"
    ///   (1, 1, 0, 0) / sqrt(2);
    /// - d3: "tea brews" (0, 0, 0, 1).
    pub(crate) fn set(d3_title: Option<&str>) -> LoadedSet {
        let qrel = |q: &str, d: &str| Qrel { query: q.into(), doc: d.into(), grade: 1 };
        LoadedSet {
            name: "toy".into(),
            sha256: "e".repeat(64),
            docs: vec![
                doc("d1", None, "cat purrs\nrust compiles\n"),
                doc("d2", None, "dog barks\ncat sleeps with dog\n"),
                doc("d3", d3_title, "tea brews\n"),
            ],
            queries: vec![
                Query { id: "q1".into(), text: "cat".into() },
                Query { id: "q2".into(), text: "dog".into() },
                Query { id: "q3".into(), text: "rust tea".into() },
            ],
            qrels: Qrels::new([qrel("q1", "d2"), qrel("q2", "d2"), qrel("q3", "d1")]).unwrap(),
            // "cat sleeps", bytes 10 to 20 of d2.
            spans: Some(
                SpanRels::new([SpanRel {
                    query: "q1".into(),
                    doc: "d2".into(),
                    span: Span::new(10, 20).unwrap(),
                    grade: 1,
                }])
                .unwrap(),
            ),
            unplaced: Vec::new(),
            absent: Vec::new(),
        }
    }

    fn pipeline(chunker: Box<dyn Chunker>, titles: Titles) -> Pipeline {
        Pipeline {
            analysis: Analysis { sentences: Sentencer::Newline, structure: None },
            chunker,
            embedder: Box::new(Stub::new(&VOCAB)),
            titles,
        }
    }

    fn windows() -> Box<dyn Chunker> {
        Box::new(SentenceWindows::new(1).unwrap())
    }

    fn ranked(run: &Run, q: &str) -> Vec<(String, f64)> {
        run.hits(q).iter().map(|h| (h.doc.clone(), h.score)).collect()
    }

    fn close(got: f64, want: f64) {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }

    #[test]
    fn documents_rank_by_their_best_chunk() {
        let o = pipeline(windows(), Titles::Ignore).evaluate(&set(None)).unwrap();
        let half = f64::from(std::f32::consts::FRAC_1_SQRT_2);
        // q1 (cat): d1's first chunk is 1, d2's second about 0.707, d3 0.
        let q1 = ranked(&o.run, "q1");
        assert_eq!(q1.iter().map(|h| h.0.as_str()).collect::<Vec<_>>(), ["d1", "d2", "d3"]);
        close(q1[0].1, 1.0);
        close(q1[1].1, half);
        // q2 (dog): d2 scores 1 on its first chunk; d1 and d3 tie at 0,
        // and the tie goes to the higher id.
        let q2: Vec<String> = ranked(&o.run, "q2").into_iter().map(|h| h.0).collect();
        assert_eq!(q2, ["d2", "d3", "d1"]);
        // q3 (rust tea): d1's second chunk and d3 tie at 0.707: d3 first.
        let q3: Vec<String> = ranked(&o.run, "q3").into_iter().map(|h| h.0).collect();
        assert_eq!(q3, ["d3", "d1", "d2"]);

        // nDCG@10: q1 finds d2 at rank 2 (1 / log2(3) = 0.630930), q2 at
        // rank 1 (1), q3 d1 at rank 2 (0.630930). Mean 0.753953.
        let m = &o.metrics;
        assert_eq!(m.queries, 3);
        close(m.per_query[0].ndcg_at_10, 0.630930);
        close(m.per_query[1].ndcg_at_10, 1.0);
        close(m.ndcg_at_10, (2.0 * 0.6309298 + 1.0) / 3.0);
        // Recall@1: only q2 has its document first.
        close(m.recall.at_1, 1.0 / 3.0);
        close(m.recall.at_5, 1.0);
    }

    #[test]
    fn span_hits_follow_the_chunks() {
        let o = pipeline(windows(), Titles::Ignore).evaluate(&set(None)).unwrap();
        // q1's chunks: d1 [0, 9) at 1, then d2 [10, 29) at 0.707, then the
        // three at 0 by document id descending, then span.
        let chunks: Vec<(String, Span)> = o.chunks.hits("q1").iter().map(|h| (h.doc.clone(), h.span)).collect();
        let s = |a, b| Span::new(a, b).unwrap();
        assert_eq!(
            chunks,
            [
                ("d1".into(), s(0, 9)),
                ("d2".into(), s(10, 29)),
                ("d3".into(), s(0, 9)),
                ("d2".into(), s(0, 9)),
                ("d1".into(), s(10, 23)),
            ]
        );
        // The relevant span [10, 20) of d2 is in the second chunk.
        // At 1: nothing of it. At 5: all of it, wholly in one chunk, and
        // 10 of the 9 + 19 + 9 + 9 + 13 = 59 bytes retrieved.
        let span = o.metrics.span.unwrap();
        assert_eq!(span.queries, 1);
        assert_eq!((span.mean.at_1.recall, span.mean.at_1.contained), (0.0, 0.0));
        assert_eq!((span.mean.at_5.recall, span.mean.at_5.contained), (1.0, 1.0));
        close(span.mean.at_5.precision, 10.0 / 59.0);
    }

    #[test]
    fn cost_by_hand() {
        let o = pipeline(windows(), Titles::Ignore).evaluate(&set(None)).unwrap();
        let c = &o.cost;
        assert_eq!(c.chunks, 5);
        // Bytes 9, 13, 9, 19, 9; tokens are words plus the 2 specials:
        // 4, 4, 4, 6, 4.
        assert_eq!(c.chunk_bytes, Distribution { min: 9, p50: 9, p90: 19, max: 19 });
        assert_eq!(c.chunk_tokens, Distribution { min: 4, p50: 4, p90: 6, max: 6 });
        // Chunks 4 + 4 + 4 + 6 + 4 = 22, queries 3 + 3 + 4 = 10.
        assert_eq!(c.tokens_embedded, 32);

        // The semantic chunker embeds the sentences of each document with
        // more than one: d1 4 + 4, d2 4 + 6. d2's two sentences are 0.707
        // alike, above 0.5, so it is one chunk of 6 words (8 tokens):
        // chunks 4 + 4 + 8 + 4 = 20, queries 10, sentences 18.
        let semantic = Box::new(Semantic::from_config(0.0, 0, 0, 0).unwrap());
        let o = pipeline(semantic, Titles::Ignore).evaluate(&set(None)).unwrap();
        assert_eq!(o.cost.chunks, 4);
        assert_eq!(o.cost.tokens_embedded, 48);
    }

    #[test]
    fn a_title_prefix_is_embedded_and_spans_stay_in_the_text() {
        // d3 titled "cat": its chunk "cat\ntea brews" is (1, 0, 0, 1) /
        // sqrt(2), level with d2 for q1, and ahead of it by id.
        let o = pipeline(windows(), Titles::Prefix).evaluate(&set(Some("cat"))).unwrap();
        let q1: Vec<String> = ranked(&o.run, "q1").into_iter().map(|h| h.0).collect();
        assert_eq!(q1, ["d1", "d3", "d2"]);
        assert_eq!(o.chunks.hits("q1")[1].span, Span::new(0, 9).unwrap());
        let o = pipeline(windows(), Titles::Ignore).evaluate(&set(Some("cat"))).unwrap();
        assert_eq!(ranked(&o.run, "q1")[2].0, "d3");
        let p = pipeline(windows(), Titles::Prefix).description();
        assert_eq!(p.analyzer, ["opennlp:sentences newline", "opennlp:subwords", "title prefixed to each chunk"]);
        assert_eq!(p.chunker.name, "sentence-windows");
    }

    fn toy(_: &str, _: &Path) -> Result<LoadedSet> {
        Ok(set(None))
    }

    #[test]
    fn run_writes_a_record_per_set_and_the_gate_reads_them() {
        let dir = std::env::temp_dir().join(format!("turbo-eval-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let source =
            Source { commit: "0123456789abcdef0123456789abcdef01234567".into(), pushed_to: vec!["origin/main".into()] };
        let mut p = pipeline(windows(), Titles::Ignore);
        let written = run_sets(&mut p, &["toy".into()], &dir, toy, &dir, &source).unwrap();
        assert_eq!(written, [record_path(&dir, "toy")]);
        let r = EvalRecord::read(&written[0]).unwrap();
        assert_eq!(r.set.name, "toy");
        assert_eq!(r.provenance.bundle.model_id, "stub");
        assert_eq!(r.pipeline.chunker.config, [crate::chunkers::setting("sentences", 1)]);
        // A second run refuses to replace it.
        assert!(run_sets(&mut p, &["toy".into()], &dir, toy, &dir, &source).unwrap_err().contains("never replaced"));
        // The same records on both sides: no gain, so the gate fails.
        let report = gate_dirs(&dir, &dir).unwrap();
        assert!(!report.pass);
        assert_eq!(report.sets[0].difference.mean, 0.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
