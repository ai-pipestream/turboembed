//! The shared E2E suite. Same cases for every arch; skip vs fail is data-driven.

use std::collections::HashSet;
use std::path::PathBuf;

use inferstream_protocol::extension::ModelInfo;
use tonic::Code;

use crate::catalog::{CatalogIndex, SkipReason};
use crate::client::{self, Clients};
use crate::golden::{golden_cosine, load_golden, lookup_golden};
use crate::matrix::Matrix;
use crate::target::Target;
use crate::HarnessError;

const EMBED_TEXTS: &[&str] = &["hello world", "query: the quick brown fox"];
const TOKENIZE_TEXT: &str = "hello world";
const GENERATE_PROMPT: &str = "Write one sentence about rivers flowing to the sea.";

/// Which slices of the suite to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SuiteFilter {
    All,
    List,
    Embed,
    Tokenize,
    Generate,
}

impl SuiteFilter {
    fn embed(self) -> bool {
        matches!(self, Self::All | Self::Embed)
    }
    fn tokenize(self) -> bool {
        matches!(self, Self::All | Self::Tokenize | Self::Generate)
    }
    fn generate(self) -> bool {
        matches!(self, Self::All | Self::Generate)
    }
}

#[derive(Debug, Clone)]
pub struct SuiteConfig {
    pub target: Target,
    pub addr: String,
    pub token: Option<String>,
    pub matrix: Matrix,
    pub catalog: CatalogIndex,
    pub goldens_dir: Option<PathBuf>,
    pub only: HashSet<String>,
    pub filter: SuiteFilter,
    pub max_tokens: i64,
    pub cosine_min: f32,
}

impl SuiteConfig {
    fn wanted(&self, alias: &str) -> bool {
        self.only.is_empty() || self.only.contains(alias)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Pass { detail: String },
    Skip { reason: String },
    Fail { reason: String },
}

impl Outcome {
    pub fn is_fail(&self) -> bool {
        matches!(self, Self::Fail { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseResult {
    pub name: String,
    pub outcome: Outcome,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    pub cases: Vec<CaseResult>,
}

impl Report {
    pub fn push(&mut self, name: impl Into<String>, outcome: Outcome) {
        self.cases.push(CaseResult {
            name: name.into(),
            outcome,
        });
    }

    pub fn failed(&self) -> bool {
        self.cases.iter().any(|c| c.outcome.is_fail())
    }

    pub fn counts(&self) -> (usize, usize, usize) {
        let mut pass = 0;
        let mut skip = 0;
        let mut fail = 0;
        for c in &self.cases {
            match c.outcome {
                Outcome::Pass { .. } => pass += 1,
                Outcome::Skip { .. } => skip += 1,
                Outcome::Fail { .. } => fail += 1,
            }
        }
        (pass, skip, fail)
    }

    pub fn outcome(&self, name: &str) -> Option<&Outcome> {
        self.cases
            .iter()
            .find(|c| c.name == name)
            .map(|c| &c.outcome)
    }

    pub fn format(&self, target: Target, addr: &str) -> String {
        let mut out = format!("inferstream-e2e  target={target}  addr={addr}\n");
        for case in &self.cases {
            let (tag, detail) = match &case.outcome {
                Outcome::Pass { detail } => ("PASS", detail.as_str()),
                Outcome::Skip { reason } => ("SKIP", reason.as_str()),
                Outcome::Fail { reason } => ("FAIL", reason.as_str()),
            };
            out.push_str(&format!("  {tag:<4}  {:<28} {detail}\n", case.name));
        }
        let (pass, skip, fail) = self.counts();
        out.push_str(&format!(
            "---\n{pass} passed, {skip} skipped, {fail} failed\n"
        ));
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Run,
    Skip(SkipReason),
    Fail(String),
}

/// Skip vs run vs hard-fail for one alias.
///
/// A *served* alias is always exercised (even if the catalog has no row for
/// this arch — the host is answering). Missing aliases skip with
/// `NotAvailableOnArch` when the catalog says so, or `not served` otherwise.
/// `required` aliases missing from ListModels are a hard fail.
pub fn decide(
    alias: &str,
    required: bool,
    target: Target,
    catalog: &CatalogIndex,
    served: Option<&ModelInfo>,
) -> Decision {
    match served {
        Some(info) if info.ready => Decision::Run,
        Some(_) if required => Decision::Fail(format!(
            "required alias {alias} is in ListModels but not ready"
        )),
        Some(_) => Decision::Skip(SkipReason::NotReady {
            alias: alias.to_string(),
        }),
        None if required => Decision::Fail(format!("required alias {alias} is not in ListModels")),
        None if catalog.known(alias) && !catalog.available_on(alias, target) => {
            Decision::Skip(SkipReason::NotAvailableOnArch {
                alias: alias.to_string(),
                arch: target.as_str(),
                available: {
                    let arches = catalog.available_arches(alias);
                    if arches.is_empty() {
                        "none".into()
                    } else {
                        arches.join(", ")
                    }
                },
            })
        }
        None => Decision::Skip(SkipReason::NotServed {
            alias: alias.to_string(),
        }),
    }
}

pub async fn run_suite(config: SuiteConfig) -> Result<Report, HarnessError> {
    let mut clients = client::connect(&config.addr, config.token.as_deref()).await?;
    let mut report = Report::default();

    let Some(listing) = run_list(&mut clients, &config, &mut report).await? else {
        return Ok(report);
    };
    let served = client::model_map(&listing);

    if config.filter.embed() {
        run_embeds(&mut clients, &config, &served, &mut report).await;
    }
    if config.filter.tokenize() {
        run_tokenize(&mut clients, &config, &served, &mut report).await;
    }
    if config.filter.generate() {
        run_generate(&mut clients, &config, &served, &mut report).await;
    }
    if config.filter.embed() {
        run_rerank(&mut clients, &config, &served, &mut report).await;
    }

    Ok(report)
}

async fn run_list(
    clients: &mut Clients,
    config: &SuiteConfig,
    report: &mut Report,
) -> Result<Option<inferstream_protocol::extension::ListModelsResponse>, HarnessError> {
    match client::server_live(&mut clients.oip).await {
        Ok(live) if live.live => report.push(
            "server-live",
            Outcome::Pass {
                detail: "live=true".into(),
            },
        ),
        Ok(_) => report.push(
            "server-live",
            Outcome::Fail {
                reason: "ServerLive.live is false".into(),
            },
        ),
        Err(e) => report.push(
            "server-live",
            Outcome::Fail {
                reason: e.to_string(),
            },
        ),
    }

    let listing = match client::list_models(&mut clients.ext).await {
        Ok(l) => l,
        Err(e) => {
            report.push(
                "list-models",
                Outcome::Fail {
                    reason: e.to_string(),
                },
            );
            return Ok(None);
        }
    };
    let ready = listing.models.iter().filter(|m| m.ready).count();
    report.push(
        "list-models",
        Outcome::Pass {
            detail: format!("{} models, {ready} ready", listing.models.len()),
        },
    );

    let minilm = listing.models.iter().find(|m| m.name == "minilm");
    match decide("minilm", true, config.target, &config.catalog, minilm) {
        Decision::Run => {
            let info = minilm.expect("decide Run implies present");
            let ready = info.ready;
            let backend = info.backend.clone();
            let embedding_dim = info.embedding_dim;
            let mut detail = format!("ready={ready} backend={backend} dim={embedding_dim}");
            let dim_mismatch = !config.target.is_mock()
                && config.matrix.embed("minilm").is_some_and(|expected| {
                    embedding_dim > 0 && embedding_dim != expected.dim as i64
                });
            // Catalog embeds on real arches go through TurboEmbedBackend.
            // `onnxruntime` / `ort` here means the old server path is still up.
            let backend_ok = config.target.is_mock() || backend.eq_ignore_ascii_case("turboembed");
            if dim_mismatch {
                let expected = config.matrix.embed("minilm").map(|e| e.dim).unwrap_or(0);
                report.push(
                    "list-models:minilm",
                    Outcome::Fail {
                        reason: format!("ListModels dim={embedding_dim} expected {expected}"),
                    },
                );
            } else if !backend_ok {
                report.push(
                    "list-models:minilm",
                    Outcome::Fail {
                        reason: format!(
                            "ListModels backend={backend:?} expected turboembed \
                             (old onnxruntime/ort path is not the C ABI façade)"
                        ),
                    },
                );
            } else {
                if embedding_dim > 0 {
                    detail.push_str(" (logical name present)");
                }
                report.push("list-models:minilm", Outcome::Pass { detail });
            }
        }
        Decision::Skip(reason) => report.push(
            "list-models:minilm",
            Outcome::Skip {
                reason: reason.to_string(),
            },
        ),
        Decision::Fail(reason) => report.push("list-models:minilm", Outcome::Fail { reason }),
    }
    Ok(Some(listing))
}

async fn run_embeds(
    clients: &mut Clients,
    config: &SuiteConfig,
    served: &std::collections::HashMap<String, ModelInfo>,
    report: &mut Report,
) {
    let aliases: Vec<_> = config
        .matrix
        .embeds
        .iter()
        .filter(|e| config.wanted(&e.alias))
        .cloned()
        .collect();

    for spec in aliases {
        let name = format!("embed:{}", spec.alias);
        let info = served.get(&spec.alias);
        match decide(
            &spec.alias,
            spec.required,
            config.target,
            &config.catalog,
            info,
        ) {
            Decision::Skip(reason) => {
                report.push(
                    name,
                    Outcome::Skip {
                        reason: reason.to_string(),
                    },
                );
                continue;
            }
            Decision::Fail(reason) => {
                report.push(name, Outcome::Fail { reason });
                continue;
            }
            Decision::Run => {}
        }

        let texts: Vec<String> = EMBED_TEXTS.iter().map(|s| (*s).to_string()).collect();
        match client::embed(&mut clients.ext, &spec.alias, texts, true).await {
            Ok(resp) => {
                if resp.embeddings.len() != EMBED_TEXTS.len() {
                    report.push(
                        name,
                        Outcome::Fail {
                            reason: format!(
                                "expected {} vectors, got {}",
                                EMBED_TEXTS.len(),
                                resp.embeddings.len()
                            ),
                        },
                    );
                    continue;
                }
                let dim = resp.dim as usize;
                let expected = expected_dim(config, &spec.alias, spec.dim, info);
                if let Some(exp) = expected {
                    if dim != exp as usize
                        || resp
                            .embeddings
                            .iter()
                            .any(|e| e.values.len() != exp as usize)
                    {
                        report.push(
                            name,
                            Outcome::Fail {
                                reason: format!("dim={dim} expected {exp}"),
                            },
                        );
                        continue;
                    }
                } else if dim == 0 {
                    report.push(
                        name,
                        Outcome::Fail {
                            reason: "embedding dim is 0".into(),
                        },
                    );
                    continue;
                }
                let first = &resp.embeddings[0].values;
                if first.iter().all(|v| *v == 0.0) {
                    report.push(
                        name,
                        Outcome::Fail {
                            reason: "embedding is all zeros".into(),
                        },
                    );
                    continue;
                }

                let mut detail = format!("dim={dim} vectors={}", resp.embeddings.len());
                if let Some(err) = maybe_golden(config, &spec.alias, first) {
                    report.push(name, Outcome::Fail { reason: err });
                    continue;
                }
                if let Some(score) = last_golden_score(config, &spec.alias, first) {
                    detail.push_str(&format!(" cosine={score:.4}"));
                }
                report.push(name, Outcome::Pass { detail });
            }
            Err(status) => {
                report.push(name, classify_rpc(&spec.alias, spec.required, status));
            }
        }
    }
}

fn expected_dim(
    config: &SuiteConfig,
    alias: &str,
    matrix_dim: u32,
    info: Option<&ModelInfo>,
) -> Option<u32> {
    if config.target.is_mock() {
        return info
            .map(|m| m.embedding_dim)
            .filter(|d| *d > 0)
            .map(|d| d as u32);
    }
    let _ = alias;
    Some(matrix_dim)
}

fn maybe_golden(config: &SuiteConfig, alias: &str, vector: &[f32]) -> Option<String> {
    match golden_score(config, alias, vector) {
        Some(Ok(score)) if score < config.cosine_min => {
            Some(format!("golden cosine {score:.4} < {}", config.cosine_min))
        }
        Some(Err(e)) => Some(e),
        _ => None,
    }
}

fn last_golden_score(config: &SuiteConfig, alias: &str, vector: &[f32]) -> Option<f32> {
    golden_score(config, alias, vector).and_then(Result::ok)
}

fn golden_score(config: &SuiteConfig, alias: &str, vector: &[f32]) -> Option<Result<f32, String>> {
    let dir = config.goldens_dir.as_ref()?;
    let path = lookup_golden(dir, config.target.as_str(), alias)?;
    Some(load_golden(&path).and_then(|g| golden_cosine(vector, &g)))
}

async fn run_tokenize(
    clients: &mut Clients,
    config: &SuiteConfig,
    served: &std::collections::HashMap<String, ModelInfo>,
    report: &mut Report,
) {
    // One embed model (minilm) + one LLM when present.
    let mut models: Vec<(String, bool)> = Vec::new();
    if config.wanted("minilm") {
        models.push(("minilm".into(), true));
    }
    if let Some(llm) = config
        .matrix
        .llms
        .iter()
        .find(|l| config.wanted(&l.alias) && served.contains_key(&l.alias))
    {
        models.push((llm.alias.clone(), llm.required));
    }

    // Dedup if someone named an embed the same as an llm (won't happen).
    let mut seen = HashSet::new();
    models.retain(|(n, _)| seen.insert(n.clone()));

    for (alias, required) in models {
        let name = format!("tokenize:{alias}");
        let info = served.get(&alias);
        match decide(&alias, required, config.target, &config.catalog, info) {
            Decision::Skip(reason) => {
                report.push(
                    name,
                    Outcome::Skip {
                        reason: reason.to_string(),
                    },
                );
                continue;
            }
            Decision::Fail(reason) => {
                report.push(name, Outcome::Fail { reason });
                continue;
            }
            Decision::Run => {}
        }

        match client::tokenize(&mut clients.ext, &alias, vec![TOKENIZE_TEXT.into()]).await {
            Ok(tok) => {
                let Some(enc) = tok.encodings.first() else {
                    report.push(
                        name,
                        Outcome::Fail {
                            reason: "Tokenize returned no encodings".into(),
                        },
                    );
                    continue;
                };
                if enc.input_ids.is_empty() {
                    report.push(
                        name,
                        Outcome::Fail {
                            reason: "Tokenize returned empty input_ids".into(),
                        },
                    );
                    continue;
                }
                match client::detokenize(
                    &mut clients.ext,
                    &alias,
                    vec![enc.input_ids.clone()],
                    true,
                )
                .await
                {
                    Ok(detok) => {
                        let text = detok.texts.first().cloned().unwrap_or_default();
                        if text.trim().is_empty() {
                            report.push(
                                name,
                                Outcome::Fail {
                                    reason: "Detokenize returned empty text".into(),
                                },
                            );
                            continue;
                        }
                        report.push(
                            name,
                            Outcome::Pass {
                                detail: format!(
                                    "ids={} detok={:?}",
                                    enc.input_ids.len(),
                                    trim_preview(&text)
                                ),
                            },
                        );
                    }
                    Err(status) => {
                        report.push(name, classify_rpc(&alias, required, status));
                    }
                }
            }
            Err(status) => {
                report.push(name, classify_rpc(&alias, required, status));
            }
        }
    }
}

async fn run_generate(
    clients: &mut Clients,
    config: &SuiteConfig,
    served: &std::collections::HashMap<String, ModelInfo>,
    report: &mut Report,
) {
    let aliases: Vec<_> = config
        .matrix
        .llms
        .iter()
        .filter(|l| config.wanted(&l.alias))
        .cloned()
        .collect();

    for spec in aliases {
        let name = format!("generate:{}", spec.alias);
        let info = served.get(&spec.alias);
        match decide(
            &spec.alias,
            spec.required,
            config.target,
            &config.catalog,
            info,
        ) {
            Decision::Skip(reason) => {
                report.push(
                    name,
                    Outcome::Skip {
                        reason: reason.to_string(),
                    },
                );
                continue;
            }
            Decision::Fail(reason) => {
                report.push(name, Outcome::Fail { reason });
                continue;
            }
            Decision::Run => {}
        }

        match client::stream_infer(
            &mut clients.oip,
            &spec.alias,
            GENERATE_PROMPT,
            config.max_tokens,
        )
        .await
        {
            Ok(result) => {
                if result.chunks == 0 || result.tokens.iter().all(|t| t.is_empty()) {
                    report.push(
                        name,
                        Outcome::Fail {
                            reason: "ModelStreamInfer produced no tokens".into(),
                        },
                    );
                    continue;
                }
                if !result.saw_final {
                    report.push(
                        name,
                        Outcome::Fail {
                            reason: format!(
                                "stream ended without final=true ({} token chunks)",
                                result.chunks
                            ),
                        },
                    );
                    continue;
                }
                report.push(
                    name,
                    Outcome::Pass {
                        detail: format!(
                            "chunks={} final=true preview={:?}",
                            result.chunks,
                            trim_preview(&result.tokens.concat())
                        ),
                    },
                );
            }
            Err(status) => {
                report.push(name, classify_rpc(&spec.alias, spec.required, status));
            }
        }
    }
}

const RERANK_ALIAS: &str = "ms-marco-minilm-l6";
const RERANK_QUERY: &str = "How many people live in Berlin?";
const RERANK_DOCS: &[&str] = &[
    "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin is well known for its museums.",
    "New York City is famous for its pizza and bagels.",
];
const RERANK_SIGMOID: [f32; 3] = [0.999_856_04, 0.013_124_33, 0.000_012_70];
const RERANK_ATOL: f32 = 0.002;

async fn run_rerank(
    clients: &mut Clients,
    config: &SuiteConfig,
    served: &std::collections::HashMap<String, ModelInfo>,
    report: &mut Report,
) {
    if !config.wanted(RERANK_ALIAS) {
        return;
    }
    let info = served.get(RERANK_ALIAS);
    match decide(RERANK_ALIAS, false, config.target, &config.catalog, info) {
        Decision::Skip(reason) => {
            report.push(
                "rerank:ms-marco-minilm-l6",
                Outcome::Skip {
                    reason: reason.to_string(),
                },
            );
            return;
        }
        Decision::Fail(reason) => {
            report.push("rerank:ms-marco-minilm-l6", Outcome::Fail { reason });
            return;
        }
        Decision::Run => {}
    }
    let info = info.expect("decide Run implies present");
    if !config.target.is_mock() && !info.backend.eq_ignore_ascii_case("turborerank") {
        report.push(
            "rerank:ms-marco-minilm-l6",
            Outcome::Fail {
                reason: format!(
                    "ListModels backend={} — catalog CE must be turborerank, not word-overlap",
                    info.backend
                ),
            },
        );
        return;
    }
    let docs: Vec<String> = RERANK_DOCS.iter().map(|s| (*s).to_string()).collect();
    match client::rerank(&mut clients.ext, RERANK_ALIAS, RERANK_QUERY, docs, 0).await {
        Ok(response) => {
            if response.results.len() != 3 {
                report.push(
                    "rerank:ms-marco-minilm-l6",
                    Outcome::Fail {
                        reason: format!("expected 3 rows, got {}", response.results.len()),
                    },
                );
                return;
            }
            let mut by_index = [0.0f32; 3];
            for row in &response.results {
                if (row.index as usize) < 3 {
                    by_index[row.index as usize] = row.score;
                }
            }
            let only_unit = by_index.iter().all(|s| {
                (*s - 0.0).abs() < 1e-8 || (*s - 1.0).abs() < 1e-8 || (*s - 0.5).abs() < 1e-8
            });
            if only_unit {
                report.push(
                    "rerank:ms-marco-minilm-l6",
                    Outcome::Fail {
                        reason: format!("FAKE: scores look like word-overlap {by_index:?}"),
                    },
                );
                return;
            }
            let mismatch = by_index
                .iter()
                .zip(RERANK_SIGMOID)
                .any(|(g, w)| (g - w).abs() > RERANK_ATOL);
            if mismatch && !config.target.is_mock() {
                report.push(
                    "rerank:ms-marco-minilm-l6",
                    Outcome::Fail {
                        reason: format!("scores {by_index:?} != golden sigmoid {RERANK_SIGMOID:?}"),
                    },
                );
                return;
            }
            report.push(
                "rerank:ms-marco-minilm-l6",
                Outcome::Pass {
                    detail: format!(
                        "backend={} input-order≈golden sigmoid top={}",
                        info.backend, response.results[0].index
                    ),
                },
            );
        }
        Err(status) => {
            report.push(
                "rerank:ms-marco-minilm-l6",
                classify_rpc(RERANK_ALIAS, false, status),
            );
        }
    }
}

fn classify_rpc(alias: &str, required: bool, status: tonic::Status) -> Outcome {
    let skippable = matches!(
        status.code(),
        Code::NotFound | Code::Unimplemented | Code::Unavailable
    );
    if skippable && !required {
        Outcome::Skip {
            reason: format!("{} ({alias}): {}", status.code(), status.message()),
        }
    } else {
        Outcome::Fail {
            reason: format!("{} ({alias}): {}", status.code(), status.message()),
        }
    }
}

fn trim_preview(text: &str) -> String {
    const MAX: usize = 48;
    let flat: String = text.chars().take(MAX).collect();
    if text.chars().count() > MAX {
        format!("{flat}…")
    } else {
        flat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferstream_protocol::extension::ModelInfo;

    fn info(name: &str, ready: bool) -> ModelInfo {
        ModelInfo {
            name: name.into(),
            ready,
            embedding_dim: 384,
            ..Default::default()
        }
    }

    #[test]
    fn required_missing_fails() {
        let cat = CatalogIndex::builtin().unwrap();
        match decide("minilm", true, Target::Nvidia, &cat, None) {
            Decision::Fail(msg) => assert!(msg.contains("minilm")),
            other => panic!("expected fail, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_arch_skips() {
        let cat = CatalogIndex::builtin().unwrap();
        match decide("mpnet", false, Target::Apple, &cat, None) {
            Decision::Skip(SkipReason::NotAvailableOnArch { alias, arch, .. }) => {
                assert_eq!(alias, "mpnet");
                assert_eq!(arch, "apple");
            }
            other => panic!("expected NotAvailableOnArch, got {other:?}"),
        }
    }

    #[test]
    fn served_runs_even_if_catalog_omits() {
        let cat = CatalogIndex::builtin().unwrap();
        let served = info("mpnet", true);
        match decide("mpnet", false, Target::Apple, &cat, Some(&served)) {
            Decision::Run => {}
            other => panic!("served alias must run, got {other:?}"),
        }
    }

    #[test]
    fn supported_but_not_on_serve_list_skips() {
        let cat = CatalogIndex::builtin().unwrap();
        match decide("qwen-7b", false, Target::Intel, &cat, None) {
            Decision::Skip(SkipReason::NotServed { alias }) => assert_eq!(alias, "qwen-7b"),
            other => panic!("expected not served, got {other:?}"),
        }
    }

    #[test]
    fn intel_qwen_05b_runs_when_served() {
        let cat = CatalogIndex::builtin().unwrap();
        let served = info("qwen-0.5b", true);
        match decide("qwen-0.5b", false, Target::Intel, &cat, Some(&served)) {
            Decision::Run => {}
            other => panic!("{other:?}"),
        }
    }
}
