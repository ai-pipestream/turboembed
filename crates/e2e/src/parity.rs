//! Cross-architecture embedding parity.
//!
//! Same catalog alias + same texts must produce nearly identical vectors
//! on nvidia / intel / apple. Thresholds are evidence-based:
//!
//! | pair | cosine min | why |
//! |---|---|---|
//! | same arch (live vs golden) | **0.99** | ORT MiniLM vs TEI on Machine A was 0.999998; replay must stay that tight |
//! | nvidia ORT FP ↔ intel GenAI | **0.99** | same MiniLM family, mean pool, L2; Intel IR is often FP16 but MiniLM still lands ≥ 0.99 |
//! | any pair involving apple | **0.97** | FP MiniLM English is ~1.000; min 0.9795 is CJK WordPiece UNK drift on an English-only model |
//!
//! Failures print the worst text id, the two arches, and the score. Optional
//! aliases (`bge-small`, `mpnet`) skip when an arch does not serve them.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::catalog::{family_pooling, CatalogIndex};
use crate::client::{self, Clients};
use crate::corpus::parity_texts;
use crate::golden::cosine;
use crate::suite::{Outcome, Report};
use crate::target::Target;
use crate::{HarnessError, Matrix};

/// Live-vs-golden on one architecture (ORT/TEI MiniLM documented 0.999998).
pub const SAME_ARCH_MIN: f32 = 0.99;
/// nvidia ORT FP32 ↔ intel GenAI (FP16 IR tolerated; MiniLM mean+L2).
pub const CROSS_FP_MIN: f32 = 0.99;
/// Apple FP MLX vs nvidia/intel FP. English MiniLM is ~1.0000; the live
/// min (0.9795) is CJK WordPiece UNK drift on MiniLM-L6 (English-only).
/// 0.97 still fails the old pooler bug (cosine ≈ 0).
pub const CROSS_QUANT_MIN: f32 = 0.97;

/// Default aliases: `minilm` is required everywhere; the others run when served.
pub const DEFAULT_PARITY_ALIASES: &[&str] = &["minilm", "bge-small", "mpnet"];

/// Popular catalog embeds × arches for `make e2e-drift` / `--drift`.
/// Same cosine floors as parity (`pair_threshold`). Soft-skip when an
/// arch does not serve the alias. Keep in lockstep with
/// `testdata/e2e/matrix.json` `embeds[]` — see `docs/turboembed-drift.md`.
pub const DEFAULT_DRIFT_ALIASES: &[&str] = &[
    "minilm",
    "minilm-l12",
    "mpnet",
    "bge-small",
    "bge-base",
    "bge-large",
    "bge-m3",
    "e5-small",
    "e5-base",
    "e5-large",
    "gte-small",
    "gte-base",
    "nomic-embed-text",
];

const EMBED_BATCH: usize = 16;
const DEFAULT_SOAK_LIMIT: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityMode {
    /// Write or compare `testdata/e2e/goldens/<arch>/<alias>.json`.
    Goldens { write: bool },
    /// Pairwise cosine across 2–3 live addrs and/or saved dumps.
    Cross,
}

#[derive(Debug, Clone)]
pub struct ParityConfig {
    pub mode: ParityMode,
    pub goldens_dir: PathBuf,
    pub aliases: Vec<String>,
    pub workspace: PathBuf,
    pub token: Option<String>,
    pub catalog: CatalogIndex,
    pub matrix: Matrix,
    /// Live `arch → addr` peers.
    pub peers: BTreeMap<Target, String>,
    /// Saved dump file or directory `arch → path`.
    pub dumps: BTreeMap<Target, PathBuf>,
    pub soak_limit: usize,
}

impl Default for ParityConfig {
    fn default() -> Self {
        Self {
            mode: ParityMode::Cross,
            goldens_dir: PathBuf::from("testdata/e2e/goldens"),
            aliases: DEFAULT_PARITY_ALIASES
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            workspace: PathBuf::from("."),
            token: Some("change-me".into()),
            catalog: CatalogIndex::builtin().expect("catalog"),
            matrix: Matrix::builtin(),
            peers: BTreeMap::new(),
            dumps: BTreeMap::new(),
            soak_limit: DEFAULT_SOAK_LIMIT,
        }
    }
}

/// Cosine floor for a pair of architectures, plus a short reason.
pub fn pair_threshold(a: Target, b: Target) -> (f32, &'static str) {
    if a == b {
        return (SAME_ARCH_MIN, "same-arch golden replay");
    }
    if a.is_mock() || b.is_mock() {
        return (SAME_ARCH_MIN, "mock self-test");
    }
    if a == Target::Apple || b == Target::Apple {
        return (
            CROSS_QUANT_MIN,
            "apple MLX FP vs ORT/GenAI (English ~1.0; 0.97 floor for CJK UNK drift)",
        );
    }
    if matches!(
        (a, b),
        (Target::Nvidia, Target::Intel) | (Target::Intel, Target::Nvidia)
    ) {
        return (
            CROSS_FP_MIN,
            "nvidia ORT FP32 vs intel GenAI MiniLM (FP16 IR tolerated)",
        );
    }
    (CROSS_FP_MIN, "cross-arch FP family")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityItem {
    pub id: String,
    pub text: String,
    pub vector: Vec<f32>,
}

/// Per-arch dump written under `testdata/e2e/goldens/<arch>/<alias>.json`.
///
/// Keeps `text` / `vector` / `dim` so the regular e2e suite's single-vector
/// golden loader still deserializes the first item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityDump {
    pub schema_version: u32,
    pub arch: String,
    pub alias: String,
    pub pooling: String,
    pub normalize: bool,
    pub dim: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    pub text: String,
    pub vector: Vec<f32>,
    pub items: Vec<ParityItem>,
}

impl ParityDump {
    pub fn from_items(
        arch: Target,
        alias: &str,
        pooling: &str,
        backend: Option<String>,
        items: Vec<ParityItem>,
    ) -> Result<Self, String> {
        let first = items
            .first()
            .ok_or_else(|| format!("{alias}: no embeddings captured"))?;
        let dim = first.vector.len() as u32;
        if items.iter().any(|i| i.vector.len() != dim as usize) {
            return Err(format!("{alias}: mixed embedding dims"));
        }
        Ok(Self {
            schema_version: 1,
            arch: arch.as_str().to_string(),
            alias: alias.to_string(),
            pooling: pooling.to_string(),
            normalize: true,
            dim,
            backend,
            text: first.text.clone(),
            vector: first.vector.clone(),
            items,
        })
    }

    pub fn by_id(&self) -> BTreeMap<&str, &ParityItem> {
        self.items.iter().map(|i| (i.id.as_str(), i)).collect()
    }
}

pub fn dump_path(goldens_dir: &Path, arch: Target, alias: &str) -> PathBuf {
    goldens_dir
        .join(arch.as_str())
        .join(format!("{alias}.json"))
}

pub fn load_dump(path: &Path) -> Result<ParityDump, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn save_dump(path: &Path, dump: &ParityDump) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let mut body = serde_json::to_string_pretty(dump).map_err(|e| e.to_string())?;
    body.push('\n');
    std::fs::write(path, body).map_err(|e| format!("{}: {e}", path.display()))
}

/// Resolve a `--dump arch=path` argument: a file, or a directory containing
/// `<alias>.json`.
pub fn resolve_dump_file(path: &Path, alias: &str) -> Option<PathBuf> {
    if path.is_file() {
        return Some(path.to_path_buf());
    }
    let candidate = path.join(format!("{alias}.json"));
    candidate.is_file().then_some(candidate)
}

pub async fn run_parity(config: ParityConfig) -> Result<Report, HarnessError> {
    match config.mode {
        ParityMode::Goldens { write } => run_goldens(config, write).await,
        ParityMode::Cross => run_cross(config).await,
    }
}

async fn run_goldens(config: ParityConfig, write: bool) -> Result<Report, HarnessError> {
    let mut report = Report::default();
    let (target, addr) = config
        .peers
        .iter()
        .next()
        .ok_or_else(|| HarnessError::Addr("--parity-goldens needs --addr / a --peer".into()))?;
    let target = *target;
    let mut clients = client::connect(addr, config.token.as_deref()).await?;
    let listing = client::list_models(&mut clients.ext).await?;
    let served = client::model_map(&listing);
    let texts = parity_texts(&config.workspace, config.soak_limit).map_err(HarnessError::Parity)?;

    for alias in &config.aliases {
        let name = format!("parity-golden:{alias}");
        if !served.contains_key(alias) {
            let required = alias == "minilm" && !target.is_mock();
            if required {
                report.push(
                    name,
                    Outcome::Fail {
                        reason: format!("{alias} is not in ListModels"),
                    },
                );
            } else {
                report.push(
                    name,
                    Outcome::Skip {
                        reason: format!("{alias} not served on {target}"),
                    },
                );
            }
            continue;
        }
        if !config.catalog.available_on(alias, target) && !target.is_mock() {
            report.push(
                name,
                Outcome::Skip {
                    reason: format!("NotAvailableOnArch: {alias} on {target}"),
                },
            );
            continue;
        }
        let pooling = family_pooling(alias);
        match capture_alias(&mut clients, target, alias, pooling, &texts).await {
            Ok(dump) => {
                let path = dump_path(&config.goldens_dir, target, alias);
                if write {
                    if let Err(e) = save_dump(&path, &dump) {
                        report.push(name, Outcome::Fail { reason: e });
                        continue;
                    }
                    report.push(
                        name,
                        Outcome::Pass {
                            detail: format!(
                                "wrote {} items dim={} → {}",
                                dump.items.len(),
                                dump.dim,
                                path.display()
                            ),
                        },
                    );
                } else {
                    match load_dump(&path) {
                        Ok(golden) => match compare_dumps(&dump, &golden, SAME_ARCH_MIN) {
                            Ok(detail) => report.push(name, Outcome::Pass { detail }),
                            Err(reason) => report.push(name, Outcome::Fail { reason }),
                        },
                        Err(_) => report.push(
                            name,
                            Outcome::Skip {
                                reason: format!(
                                    "no golden at {} (capture with --parity-write)",
                                    path.display()
                                ),
                            },
                        ),
                    }
                }
            }
            Err(e) => report.push(
                name,
                Outcome::Fail {
                    reason: e.to_string(),
                },
            ),
        }
    }
    Ok(report)
}

async fn run_cross(config: ParityConfig) -> Result<Report, HarnessError> {
    let mut report = Report::default();
    let arches = collect_arches(&config);
    if arches.len() < 2 {
        return Err(HarnessError::Addr(
            "--parity-cross needs at least two --peer arch=addr and/or --dump arch=path".into(),
        ));
    }

    let texts = parity_texts(&config.workspace, config.soak_limit).map_err(HarnessError::Parity)?;

    let mut captured: BTreeMap<(Target, String), ParityDump> = BTreeMap::new();
    for arch in &arches {
        if let Some(addr) = config.peers.get(arch) {
            let mut clients = client::connect(addr, config.token.as_deref()).await?;
            let listing = client::list_models(&mut clients.ext).await?;
            let served = client::model_map(&listing);
            for alias in &config.aliases {
                if !served.contains_key(alias) {
                    continue;
                }
                if !config.catalog.available_on(alias, *arch) && !arch.is_mock() {
                    continue;
                }
                let pooling = family_pooling(alias);
                match capture_alias(&mut clients, *arch, alias, pooling, &texts).await {
                    Ok(dump) => {
                        captured.insert((*arch, alias.clone()), dump);
                    }
                    Err(e) => {
                        report.push(
                            format!("parity-capture:{}:{alias}", arch.as_str()),
                            Outcome::Fail {
                                reason: e.to_string(),
                            },
                        );
                    }
                }
            }
        }
        if let Some(path) = config.dumps.get(arch) {
            for alias in &config.aliases {
                if captured.contains_key(&(*arch, alias.clone())) {
                    continue;
                }
                if let Some(file) = resolve_dump_file(path, alias) {
                    match load_dump(&file) {
                        Ok(dump) => {
                            captured.insert((*arch, alias.clone()), dump);
                        }
                        Err(e) => report.push(
                            format!("parity-dump:{}:{alias}", arch.as_str()),
                            Outcome::Fail { reason: e },
                        ),
                    }
                }
            }
        }
    }

    for alias in &config.aliases {
        let present: Vec<Target> = arches
            .iter()
            .copied()
            .filter(|a| captured.contains_key(&(*a, alias.clone())))
            .collect();
        if present.len() < 2 {
            let required = alias == "minilm";
            let reason = format!(
                "{alias}: need ≥2 arches with vectors, have {}",
                present
                    .iter()
                    .map(|t| t.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            if required && present.is_empty() {
                report.push(format!("parity-cross:{alias}"), Outcome::Fail { reason });
            } else {
                report.push(format!("parity-cross:{alias}"), Outcome::Skip { reason });
            }
            continue;
        }
        for i in 0..present.len() {
            for j in (i + 1)..present.len() {
                let left = present[i];
                let right = present[j];
                let (min, why) = pair_threshold(left, right);
                let a = &captured[&(left, alias.clone())];
                let b = &captured[&(right, alias.clone())];
                let name = format!("parity-cross:{alias}:{}-{}", left.as_str(), right.as_str());
                match compare_dumps(a, b, min) {
                    Ok(detail) => report.push(
                        name,
                        Outcome::Pass {
                            detail: format!("{detail}  [{why}]"),
                        },
                    ),
                    Err(reason) => report.push(name, Outcome::Fail { reason }),
                }
            }
        }
    }
    Ok(report)
}

fn collect_arches(config: &ParityConfig) -> Vec<Target> {
    let mut set = HashSet::new();
    for t in config.peers.keys() {
        set.insert(*t);
    }
    for t in config.dumps.keys() {
        set.insert(*t);
    }
    let mut v: Vec<_> = set.into_iter().collect();
    v.sort_by_key(|t| t.as_str());
    v
}

async fn capture_alias(
    clients: &mut Clients,
    arch: Target,
    alias: &str,
    pooling: &str,
    texts: &[(String, String)],
) -> Result<ParityDump, HarnessError> {
    let mut items = Vec::with_capacity(texts.len());
    for batch in texts.chunks(EMBED_BATCH) {
        let batch_texts: Vec<String> = batch.iter().map(|(_, t)| t.clone()).collect();
        let resp = client::embed_with(&mut clients.ext, alias, batch_texts, true, Some(pooling))
            .await
            .map_err(|e| HarnessError::Rpc(format!("Embed {alias}: {e}")))?;
        if resp.embeddings.len() != batch.len() {
            return Err(HarnessError::Parity(format!(
                "{alias}: expected {} vectors, got {}",
                batch.len(),
                resp.embeddings.len()
            )));
        }
        for ((id, text), emb) in batch.iter().zip(resp.embeddings) {
            if emb.values.iter().all(|v| *v == 0.0) {
                return Err(HarnessError::Parity(format!(
                    "{alias} {id}: embedding is all zeros"
                )));
            }
            items.push(ParityItem {
                id: id.clone(),
                text: text.clone(),
                vector: emb.values,
            });
        }
    }
    ParityDump::from_items(arch, alias, pooling, None, items).map_err(HarnessError::Parity)
}

/// Compare two dumps on the intersection of text ids.
pub fn compare_dumps(left: &ParityDump, right: &ParityDump, min: f32) -> Result<String, String> {
    if left.dim != right.dim {
        return Err(format!(
            "dim mismatch: {}={} vs {}={}",
            left.arch, left.dim, right.arch, right.dim
        ));
    }
    if left.pooling != right.pooling {
        return Err(format!(
            "pooling mismatch: {}={} vs {}={}",
            left.arch, left.pooling, right.arch, right.pooling
        ));
    }
    let a = left.by_id();
    let b = right.by_id();
    let mut scores = Vec::new();
    let mut missing = 0usize;
    for (id, item) in &a {
        let Some(other) = b.get(id) else {
            missing += 1;
            continue;
        };
        if item.vector.len() != other.vector.len() {
            return Err(format!(
                "{id}: length {} vs {}",
                item.vector.len(),
                other.vector.len()
            ));
        }
        scores.push((*id, cosine(&item.vector, &other.vector)));
    }
    if scores.is_empty() {
        return Err(format!(
            "no overlapping text ids ({} vs {} items, {missing} missing on the right)",
            left.items.len(),
            right.items.len()
        ));
    }
    scores.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
    let worst = scores[0];
    let best = scores[scores.len() - 1];
    let mean = scores.iter().map(|(_, c)| *c).sum::<f32>() / scores.len() as f32;
    if worst.1 < min {
        return Err(format!(
            "cosine {:.4} < {min} on {} ({} vs {}; mean={:.4} n={} best={:.4})",
            worst.1,
            worst.0,
            left.arch,
            right.arch,
            mean,
            scores.len(),
            best.1
        ));
    }
    Ok(format!(
        "n={} min={:.4} mean={:.4} (≥ {min})",
        scores.len(),
        worst.1,
        mean
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, text: &str, vector: Vec<f32>) -> ParityItem {
        ParityItem {
            id: id.into(),
            text: text.into(),
            vector,
        }
    }

    fn dump(arch: Target, items: Vec<ParityItem>) -> ParityDump {
        ParityDump::from_items(arch, "minilm", "mean", None, items).unwrap()
    }

    #[test]
    fn drift_aliases_cover_matrix_embeds() {
        let matrix = Matrix::builtin();
        let drift: std::collections::HashSet<&str> =
            DEFAULT_DRIFT_ALIASES.iter().copied().collect();
        for embed in &matrix.embeds {
            assert!(
                drift.contains(embed.alias.as_str()),
                "add {} to DEFAULT_DRIFT_ALIASES (docs/turboembed-drift.md)",
                embed.alias
            );
        }
        assert!(DEFAULT_DRIFT_ALIASES.contains(&"minilm"));
        assert!(DEFAULT_PARITY_ALIASES.iter().all(|a| drift.contains(a)));
    }

    #[test]
    fn thresholds_match_documented_defaults() {
        let (n, _) = pair_threshold(Target::Nvidia, Target::Intel);
        assert!((n - 0.99).abs() < f32::EPSILON);
        let (a, why) = pair_threshold(Target::Nvidia, Target::Apple);
        assert!((a - 0.97).abs() < f32::EPSILON);
        assert!(why.contains("CJK") || why.contains("0.97"));
        let (i, _) = pair_threshold(Target::Intel, Target::Apple);
        assert!((i - 0.97).abs() < f32::EPSILON);
        let (s, _) = pair_threshold(Target::Nvidia, Target::Nvidia);
        assert!((s - 0.99).abs() < f32::EPSILON);
    }

    #[test]
    fn compare_identical_passes() {
        let d = dump(
            Target::Nvidia,
            vec![
                item("a", "hello", vec![1.0, 0.0, 0.0]),
                item("b", "world", vec![0.0, 1.0, 0.0]),
            ],
        );
        let msg = compare_dumps(&d, &d, 0.99).unwrap();
        assert!(msg.contains("n=2"));
    }

    #[test]
    fn compare_orthogonal_fails_clearly() {
        let left = dump(
            Target::Nvidia,
            vec![item("parity:short", "hello", vec![1.0, 0.0])],
        );
        let right = dump(
            Target::Intel,
            vec![item("parity:short", "hello", vec![0.0, 1.0])],
        );
        let err = compare_dumps(&left, &right, 0.99).unwrap_err();
        assert!(err.contains("cosine 0.0000 < 0.99"));
        assert!(err.contains("parity:short"));
        assert!(err.contains("nvidia"));
        assert!(err.contains("intel"));
    }

    #[test]
    fn compare_quant_bound_allows_097() {
        // ~0.975 cosine: would fail 0.99, pass 0.97 (apple 4-bit band).
        let left = dump(Target::Nvidia, vec![item("x", "t", vec![1.0, 0.0, 0.0])]);
        let right = dump(
            Target::Apple,
            vec![item("x", "t", vec![0.975, 0.2216, 0.0])],
        );
        let score = cosine(&left.items[0].vector, &right.items[0].vector);
        assert!(score > 0.97 && score < 0.99, "got {score}");
        compare_dumps(&left, &right, CROSS_QUANT_MIN).unwrap();
        assert!(compare_dumps(&left, &right, CROSS_FP_MIN).is_err());
    }

    #[test]
    fn pooling_mismatch_fails() {
        let mut left = dump(Target::Nvidia, vec![item("a", "t", vec![1.0, 0.0])]);
        left.pooling = "cls".into();
        let right = dump(Target::Intel, vec![item("a", "t", vec![1.0, 0.0])]);
        let err = compare_dumps(&left, &right, 0.99).unwrap_err();
        assert!(err.contains("pooling mismatch"));
    }

    #[test]
    fn dump_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let d = dump(
            Target::Mock,
            vec![item("parity:short", "hello world", vec![0.2, 0.4, 0.8])],
        );
        let path = dump_path(tmp.path(), Target::Mock, "minilm");
        save_dump(&path, &d).unwrap();
        let loaded = load_dump(&path).unwrap();
        assert_eq!(loaded.alias, "minilm");
        assert_eq!(loaded.items[0].id, "parity:short");
        assert_eq!(loaded.vector, vec![0.2, 0.4, 0.8]);
        assert_eq!(loaded.text, "hello world");
    }
}
