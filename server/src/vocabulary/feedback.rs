//! Corpus DF statistics (`TCIDF001`) and spherical k-means Voronoi centroid
//! (`TCENT001`) feedback handlers for `TurboVocabularyService` (`#2346`).

use std::collections::BTreeMap;

use tonic::{Request, Response, Status, Streaming};
use turbo_analysis::{CorpusIdf, ModelRef};
use turbo_analysis_embed::{VoronoiOptions, train_voronoi_centroids};

use crate::api::TURBO_E_CAPACITY;
use crate::kserve::Reply;
use crate::proto::vocabulary::*;

use super::{MAX_LANGUAGE, MAX_NAME, Vocabulary, artifact_id, blocking, check_text, fail, first, internal, invalid};

/// Maximum term rows accepted in one `ImportCorpusStatistics` stream.
const MAX_CORPUS_STAT_TERMS: usize = 4_000_000;
/// Maximum `f32` values accepted in one `TrainVoronoiCentroids` stream (`256 MiB`).
const MAX_VORONOI_FLOATS: usize = 64 << 20;

pub(crate) fn corpus_stats_meta(c: &CorpusStatisticsArtifactDescriptor) -> Vec<u8> {
    super::store::meta(&[
        ("artifact_id", c.artifact_id.clone()),
        ("language", c.language.clone()),
        ("name", c.name.clone()),
        ("doc_count", c.doc_count.to_string()),
        ("term_count", c.term_count.to_string()),
        ("byte_size", c.byte_size.to_string()),
        ("sha256", c.sha256.clone()),
    ])
}

pub(crate) fn read_corpus_stats(body: &str) -> Result<CorpusStatisticsArtifactDescriptor, String> {
    let m: BTreeMap<&str, &str> = body.lines().filter_map(|l| l.split_once('=')).collect();
    Ok(CorpusStatisticsArtifactDescriptor {
        artifact_id: super::store::text(&m, "artifact_id"),
        language: super::store::text(&m, "language"),
        name: super::store::text(&m, "name"),
        doc_count: super::store::num(&m, "doc_count")?,
        term_count: super::store::num(&m, "term_count")?,
        byte_size: super::store::num(&m, "byte_size")?,
        sha256: super::store::text(&m, "sha256"),
    })
}

pub(crate) fn voronoi_meta(v: &VoronoiCentroidArtifactDescriptor) -> Vec<u8> {
    let counts = v.cluster_counts.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
    super::store::meta(&[
        ("artifact_id", v.artifact_id.clone()),
        ("model_id", v.model_id.clone()),
        ("model_revision", v.model_revision.clone()),
        ("model_sha256", v.model_sha256.clone()),
        ("dim", v.dim.to_string()),
        ("num_centroids", v.num_centroids.to_string()),
        ("sample_count", v.sample_count.to_string()),
        ("mean_cosine_bits", v.mean_cosine.to_bits().to_string()),
        ("byte_size", v.byte_size.to_string()),
        ("sha256", v.sha256.clone()),
        ("cluster_counts", counts),
    ])
}

pub(crate) fn read_voronoi(body: &str) -> Result<VoronoiCentroidArtifactDescriptor, String> {
    let m: BTreeMap<&str, &str> = body.lines().filter_map(|l| l.split_once('=')).collect();
    let bits = super::store::num(&m, "mean_cosine_bits")?;
    let counts_raw = super::store::text(&m, "cluster_counts");
    let mut cluster_counts = Vec::new();
    if !counts_raw.is_empty() {
        for piece in counts_raw.split(',') {
            let n = piece.parse::<u64>().map_err(|_| format!("invalid cluster_counts entry {piece:?}"))?;
            cluster_counts.push(n);
        }
    }
    Ok(VoronoiCentroidArtifactDescriptor {
        artifact_id: super::store::text(&m, "artifact_id"),
        model_id: super::store::text(&m, "model_id"),
        model_revision: super::store::text(&m, "model_revision"),
        model_sha256: super::store::text(&m, "model_sha256"),
        dim: u32::try_from(super::store::num(&m, "dim")?).map_err(|_| "dim overflow".to_string())?,
        num_centroids: u32::try_from(super::store::num(&m, "num_centroids")?)
            .map_err(|_| "num_centroids overflow".to_string())?,
        sample_count: super::store::num(&m, "sample_count")?,
        mean_cosine: f64::from_bits(bits),
        byte_size: super::store::num(&m, "byte_size")?,
        sha256: super::store::text(&m, "sha256"),
        cluster_counts,
    })
}

pub(crate) async fn import_corpus_statistics(
    svc: &Vocabulary,
    req: Request<Streaming<ImportCorpusStatisticsRequest>>,
) -> Reply<CorpusStatisticsArtifactDescriptor> {
    let store = svc.writer("ImportCorpusStatistics")?;
    let mut s = req.into_inner();
    let h = first(&mut s, |m: ImportCorpusStatisticsRequest| match m.frame {
        Some(import_corpus_statistics_request::Frame::Header(h)) => Some(h),
        _ => None,
    })
    .await?;
    check_text("language", &h.language, MAX_LANGUAGE)?;
    check_text("name", &h.name, MAX_NAME)?;
    if h.doc_count == 0 {
        return Err(invalid("doc_count must be at least 1"));
    }

    let mut entries: Vec<(String, u64)> = Vec::new();
    while let Some(m) = s.message().await? {
        let Some(import_corpus_statistics_request::Frame::Batch(batch)) = m.frame else {
            return Err(invalid("every frame after the header must be a batch"));
        };
        if entries.len().saturating_add(batch.entries.len()) > MAX_CORPUS_STAT_TERMS {
            return Err(fail(
                TURBO_E_CAPACITY,
                format!("ImportCorpusStatistics exceeds {MAX_CORPUS_STAT_TERMS} term entries"),
            ));
        }
        for item in batch.entries {
            entries.push((item.term, item.doc_freq));
        }
    }

    let desc = blocking(move || {
        let idf = CorpusIdf::from_entries(h.doc_count, entries).map_err(invalid)?;
        let bytes = idf.to_bytes();
        let sha256 = turbo_maxent::sha256(&bytes).to_hex();
        let desc = CorpusStatisticsArtifactDescriptor {
            artifact_id: artifact_id("cidf", &[&h.language, &h.name, &sha256]),
            language: h.language,
            name: h.name,
            doc_count: idf.doc_count(),
            term_count: idf.len() as u64,
            byte_size: bytes.len() as u64,
            sha256,
        };
        store.put_corpus_stats(desc, &bytes).map_err(internal)
    })
    .await?;
    Ok(Response::new(desc))
}

pub(crate) async fn list_corpus_statistics(
    svc: &Vocabulary,
    _: Request<ListCorpusStatisticsRequest>,
) -> Reply<ListCorpusStatisticsResponse> {
    let statistics = svc.store.as_ref().map(|s| s.corpus_statistics()).unwrap_or_default();
    Ok(Response::new(ListCorpusStatisticsResponse { statistics }))
}

pub(crate) async fn train_voronoi(
    svc: &Vocabulary,
    req: Request<Streaming<TrainVoronoiCentroidsRequest>>,
) -> Reply<VoronoiCentroidArtifactDescriptor> {
    let store = svc.writer("TrainVoronoiCentroids")?;
    let mut s = req.into_inner();
    let h = first(&mut s, |m: TrainVoronoiCentroidsRequest| match m.frame {
        Some(train_voronoi_centroids_request::Frame::Header(h)) => Some(h),
        _ => None,
    })
    .await?;
    check_text("model_id", &h.model_id, MAX_NAME)?;
    check_text("model_revision", &h.model_revision, MAX_NAME)?;
    check_text("model_sha256", &h.model_sha256, MAX_NAME)?;
    check_text("tag_set_name", &h.tag_set_name, MAX_NAME)?;
    if h.dim == 0 {
        return Err(invalid("dim must be at least 1"));
    }
    if h.num_centroids == 0 || h.num_centroids > u32::from(u16::MAX) {
        return Err(invalid("num_centroids must be in 1..=65535"));
    }

    let mut samples: Vec<f32> = Vec::new();
    while let Some(m) = s.message().await? {
        let Some(train_voronoi_centroids_request::Frame::Batch(batch)) = m.frame else {
            return Err(invalid("every frame after the header must be a batch"));
        };
        if samples.len().saturating_add(batch.values.len()) > MAX_VORONOI_FLOATS {
            return Err(fail(
                TURBO_E_CAPACITY,
                format!("TrainVoronoiCentroids exceeds {MAX_VORONOI_FLOATS} float elements"),
            ));
        }
        samples.extend_from_slice(&batch.values);
    }

    let desc = blocking(move || -> Result<VoronoiCentroidArtifactDescriptor, Status> {
        let dim = h.dim as usize;
        if samples.is_empty() || !samples.len().is_multiple_of(dim) {
            return Err(invalid(format!(
                "sample float count {} must be a non-zero multiple of dim {dim}",
                samples.len()
            )));
        }
        let sample_count = (samples.len() / dim) as u64;
        let temperature = if h.temperature == 0.0 { 0.1 } else { h.temperature };
        let model =
            ModelRef { id: h.model_id.clone(), revision: h.model_revision.clone(), sha256: h.model_sha256.clone() };
        let opts = VoronoiOptions {
            num_centroids: h.num_centroids as usize,
            max_iters: h.max_iters as usize,
            seed: h.seed,
            tag_set_name: if h.tag_set_name.is_empty() { "voronoi_shards".into() } else { h.tag_set_name },
            temperature,
        };
        let trained = train_voronoi_centroids(model, dim, &samples, &opts).map_err(invalid)?;
        let bytes = trained.to_bytes();
        let sha256 = turbo_maxent::sha256(&bytes).to_hex();
        let desc = VoronoiCentroidArtifactDescriptor {
            artifact_id: artifact_id("vcent", &[&sha256]),
            model_id: h.model_id,
            model_revision: h.model_revision,
            model_sha256: h.model_sha256,
            dim: h.dim,
            num_centroids: h.num_centroids,
            sample_count,
            mean_cosine: trained.mean_cosine(),
            byte_size: bytes.len() as u64,
            sha256,
            cluster_counts: trained.cluster_counts().to_vec(),
        };
        store.put_voronoi(desc, &bytes).map_err(internal)
    })
    .await?;
    Ok(Response::new(desc))
}

pub(crate) async fn list_voronoi_centroids(
    svc: &Vocabulary,
    _: Request<ListVoronoiCentroidsRequest>,
) -> Reply<ListVoronoiCentroidsResponse> {
    let centroids = svc.store.as_ref().map(|s| s.voronoi_centroids()).unwrap_or_default();
    Ok(Response::new(ListVoronoiCentroidsResponse { centroids }))
}
