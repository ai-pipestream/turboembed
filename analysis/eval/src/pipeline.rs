//! A retrieval pipeline over one evaluation set: analyze each document,
//! chunk it, embed each chunk once, embed each query the same way
//! without chunking, and rank by exact cosine (`retrieve`); or, for a
//! lexical pipeline, rank the chunks by BM25 over their terms (`lexical`)
//! and embed nothing; or, for a hybrid pipeline, combine BM25 and exact
//! cosine over the chunks (`hybrid`). The run is
//! scored with `metrics` and `spans`, and the cost of each stage is
//! measured as it runs.

use std::collections::BTreeSet;
use std::time::Instant;

use crate::Result;
use crate::analysis::Analysis;
use crate::chunkers::Chunker;
use crate::cost::{Cost, Distribution, StageTimes};
use crate::embedder::{Counted, Embedder, Role};
use crate::hybrid::{Hybrid, HybridSearch};
use crate::lexical::{Lexical, Search};
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
    /// BM25 over the chunks' terms instead of the embedder's vectors.
    pub lexical: Option<Lexical>,
    /// BM25 over the chunks' terms fused with the embedder's vectors.
    pub hybrid: Option<Hybrid>,
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
        let retrieval = if let Some(l) = &self.lexical {
            Some(record::Chunker { name: "bm25".into(), config: l.config() })
        } else {
            self.hybrid.as_ref().map(|h| record::Chunker { name: "hybrid".into(), config: h.config() })
        };
        record::Pipeline {
            analyzer,
            chunker: record::Chunker { name: self.chunker.name().into(), config: self.chunker.config() },
            retrieval,
        }
    }

    /// Runs the pipeline over the set and scores it.
    pub fn evaluate(&mut self, set: &LoadedSet) -> Result<Outcome> {
        if self.lexical.is_some() && self.hybrid.is_some() {
            return Err("a pipeline cannot set both lexical and hybrid retrieval".into());
        }
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
        // Vectors the chunker pooled itself, one per chunk: from every
        // document or from none.
        let mut pooled: Option<Vec<f32>> = None;
        for (i, (d, doc)) in set.docs.iter().zip(&docs).enumerate() {
            let (chunks, vectors) =
                self.chunker.chunk_vectors(doc, embedder).map_err(|e| format!("document {}: {e}", d.id))?;
            match (i, vectors, pooled.as_mut()) {
                (_, None, None) => {}
                (0, Some(v), None) => pooled = Some(v),
                (_, Some(v), Some(p)) => p.extend_from_slice(&v),
                _ => return Err(format!("document {}: the chunker pooled vectors for some documents only", d.id)),
            }
            let mut seen = BTreeSet::new();
            for s in chunks {
                if s.start >= s.end
                    || s.end > d.text.len()
                    || !d.text.is_char_boundary(s.start)
                    || !d.text.is_char_boundary(s.end)
                    || !seen.insert(s)
                {
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
        if let Some(p) = &pooled {
            if p.len() != spans.len() * embedder.dim() {
                return Err(format!("set {}: {} pooled values for {} chunks", set.name, p.len(), spans.len()));
            }
            if self.lexical.is_some() || self.hybrid.is_some() {
                return Err("vectors a chunker pools are retrieved by dense retrieval only".into());
            }
        }

        // Bookkeeping for the record, outside every timed stage.
        let ids: Vec<String> = set.docs.iter().map(|d| d.id.clone()).collect();
        let chunk_bytes: Vec<u64> = spans.iter().map(|s| (s.end - s.start) as u64).collect();
        let count = spans.len() as u64;

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
        let (run, chunks, embedding_ms, retrieval_ms) = if let Some(lexical) = &self.lexical {
            // Nothing is embedded: the index and the search are retrieval.
            let t = Instant::now();
            let index = lexical.index(&texts, &doc_of)?;
            let queries: Vec<(&str, &str)> = set.queries.iter().map(|q| (q.id.as_str(), q.text.as_str())).collect();
            lexical.dump(&set.name, &ids, &texts, &doc_of, &queries)?;
            drop(texts);
            let search = Search { index: &index, doc: &doc_of, span: &spans, docs: &ids };
            let (run, chunks) = lexical.search(&search, &queries)?;
            (run, chunks, 0.0, ms(t))
        } else if let Some(hybrid) = &self.hybrid {
            hybrid_run(hybrid, set, embedder, texts, doc_of, spans, &ids, t)?
        } else {
            dense(set, embedder, texts, pooled, doc_of, spans, &ids, t)?
        };
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

/// The dense ranking: every chunk (unless the chunker pooled its
/// vectors) and query embedded, ranked by exact cosine. Returns the runs and the embedding and retrieval times;
/// `t` started before the chunk texts were made.
#[allow(clippy::too_many_arguments)]
fn dense(
    set: &LoadedSet,
    embedder: &mut Counted<'_>,
    texts: Vec<String>,
    pooled: Option<Vec<f32>>,
    doc_of: Vec<u32>,
    spans: Vec<crate::spans::Span>,
    ids: &[String],
    t: Instant,
) -> Result<(Run, ChunkRun, f64, f64)> {
    let dim = embedder.dim();
    let vectors = match pooled {
        Some(v) => v,
        None => embed_all(embedder, &texts, Role::Document)?,
    };
    drop(texts);
    let queries: Vec<&str> = set.queries.iter().map(|q| q.text.as_str()).collect();
    let query_vectors = embed_all(embedder, &queries, Role::Query)?;
    let embedding_ms = ms(t);

    let t = Instant::now();
    let index = Index { dim, vectors, doc: doc_of, span: spans, docs: ids };
    let names: Vec<&str> = set.queries.iter().map(|q| q.id.as_str()).collect();
    let (run, chunks) = index.search(&names, &query_vectors)?;
    let retrieval_ms = ms(t);
    Ok((run, chunks, embedding_ms, retrieval_ms))
}

/// The hybrid ranking: every chunk and query embedded, then BM25 and exact
/// cosine fused over all chunks.
#[allow(clippy::too_many_arguments)]
fn hybrid_run(
    hybrid: &Hybrid,
    set: &LoadedSet,
    embedder: &mut Counted<'_>,
    texts: Vec<String>,
    doc_of: Vec<u32>,
    spans: Vec<crate::spans::Span>,
    ids: &[String],
    t: Instant,
) -> Result<(Run, ChunkRun, f64, f64)> {
    let dim = embedder.dim();
    let vectors = embed_all(embedder, &texts, Role::Document)?;
    let query_texts: Vec<&str> = set.queries.iter().map(|q| q.text.as_str()).collect();
    let query_vectors = embed_all(embedder, &query_texts, Role::Query)?;
    let embedding_ms = ms(t);

    let t = Instant::now();
    let bm25_index = hybrid.index(&texts, &doc_of)?;
    drop(texts);
    let queries: Vec<(&str, &str)> = set.queries.iter().map(|q| (q.id.as_str(), q.text.as_str())).collect();
    let search = HybridSearch { bm25: &bm25_index, dim, vectors: &vectors, doc: &doc_of, span: &spans, docs: ids };
    let (run, chunks) = hybrid.search(&search, &queries, &query_vectors)?;
    let retrieval_ms = ms(t);
    Ok((run, chunks, embedding_ms, retrieval_ms))
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
#[path = "pipeline_tests.rs"]
pub(crate) mod tests;
