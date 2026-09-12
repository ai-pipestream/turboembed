//! Ensure SHA-256-pinned artifacts for an e2e / serve target.
//!
//! NVIDIA / Intel weights go through `inferstream-fetch` (ONNX, GGUF,
//! OpenVINO GenAI). Apple MLX weights go through `cargo xtask fetch --mlx`.
//! Already-present files with a matching hash are skipped.

use std::collections::{BTreeMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use inferstream_fetch::{
    corpus_known_aliases, embedding_known_aliases, ensure_aliases, llm_known_aliases,
    mlx_known_aliases, ov_genai_known_aliases, workspace_root, ManifestKind,
};

use crate::matrix::Matrix;
use crate::target::Target;

/// Required embed + smoke-sized LLM aliases the default `--fetch` set pulls.
pub const DEFAULT_E2E_EMBEDS: &[&str] = &["minilm"];
pub const DEFAULT_E2E_LLMS: &[&str] = &["default-llm", "qwen-0.5b"];

/// How widely `--fetch` should look for aliases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchScope {
    /// Required matrix embeds + `default-llm` / `qwen-0.5b`.
    Default,
    /// Every matrix alias available on this target (includes `qwen-7b`).
    All,
    /// `serve` list from `config/<arch>.toml`.
    Serve,
}

impl std::fmt::Display for FetchScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Default => "default",
            Self::All => "all",
            Self::Serve => "serve",
        })
    }
}

/// Which fetcher a batch of aliases should go through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    EmbeddingsOnnx,
    LlmsGguf,
    OvGenai,
    Mlx,
    Corpus,
}

impl ArtifactKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EmbeddingsOnnx => "embeddings",
            Self::LlmsGguf => "llms",
            Self::OvGenai => "ov-genai",
            Self::Mlx => "mlx",
            Self::Corpus => "corpus",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchBatch {
    pub kind: ArtifactKind,
    pub aliases: Vec<String>,
}

/// One planned fetch: batches plus aliases dropped because they are not in
/// the matching manifest (e.g. intel `bge-small` has no public GenAI IR).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FetchPlan {
    pub batches: Vec<FetchBatch>,
    pub skipped: Vec<(String, String)>,
}

impl FetchPlan {
    pub fn is_empty(&self) -> bool {
        self.batches.iter().all(|b| b.aliases.is_empty())
    }

    pub fn format(&self) -> String {
        let mut out = String::new();
        for batch in &self.batches {
            if batch.aliases.is_empty() {
                continue;
            }
            out.push_str(&format!(
                "  {:<12} {}\n",
                batch.kind.as_str(),
                batch.aliases.join(", ")
            ));
        }
        for (alias, why) in &self.skipped {
            out.push_str(&format!("  skip         {alias} ({why})\n"));
        }
        if out.is_empty() {
            out.push_str("  (nothing to fetch)\n");
        }
        out
    }
}

/// Collect the alias names `--fetch` should try to materialize.
pub fn candidate_aliases(
    target: Target,
    scope: FetchScope,
    only: &HashSet<String>,
    serve: &[String],
    matrix: &Matrix,
) -> Vec<String> {
    let mut names: Vec<String> = match scope {
        FetchScope::Default => {
            let mut v = Vec::new();
            for e in matrix.embed_on_target(target) {
                if e.required || DEFAULT_E2E_EMBEDS.contains(&e.alias.as_str()) {
                    v.push(e.alias.clone());
                }
            }
            for l in matrix.llm_on_target(target) {
                if l.required || DEFAULT_E2E_LLMS.contains(&l.alias.as_str()) {
                    v.push(l.alias.clone());
                }
            }
            v
        }
        FetchScope::All => {
            let mut v: Vec<String> = matrix
                .embed_on_target(target)
                .map(|e| e.alias.clone())
                .collect();
            v.extend(matrix.llm_on_target(target).map(|l| l.alias.clone()));
            v
        }
        FetchScope::Serve => serve.to_vec(),
    };
    if !only.is_empty() {
        names.retain(|a| only.contains(a));
    }
    let mut seen = HashSet::new();
    names.retain(|a| seen.insert(a.clone()));
    names
}

/// Split candidate aliases into per-kind batches for `target`.
pub fn plan_fetches(
    target: Target,
    scope: FetchScope,
    only: &HashSet<String>,
    serve: &[String],
    matrix: &Matrix,
) -> FetchPlan {
    if target.is_mock() {
        return FetchPlan::default();
    }
    let aliases = candidate_aliases(target, scope, only, serve, matrix);
    let mut plan = FetchPlan::default();
    match target {
        Target::Nvidia => {
            push_filtered(
                &mut plan,
                ArtifactKind::EmbeddingsOnnx,
                &aliases,
                &embedding_known_aliases(),
                "not in embeddings.json",
            );
            push_filtered(
                &mut plan,
                ArtifactKind::LlmsGguf,
                &aliases,
                &llm_known_aliases(),
                "not in llms.json",
            );
        }
        Target::Intel => {
            push_filtered(
                &mut plan,
                ArtifactKind::OvGenai,
                &aliases,
                &ov_genai_known_aliases(),
                "not in ov-genai-embeddings.json",
            );
            push_filtered(
                &mut plan,
                ArtifactKind::LlmsGguf,
                &aliases,
                &llm_known_aliases(),
                "not in llms.json",
            );
        }
        Target::Apple => {
            push_filtered(
                &mut plan,
                ArtifactKind::Mlx,
                &aliases,
                &mlx_known_aliases(),
                "not in mlx source table",
            );
            // Tokenize for LLM aliases uses models/gguf/<family>/tokenizer.json.
            push_filtered(
                &mut plan,
                ArtifactKind::LlmsGguf,
                &aliases,
                &llm_known_aliases(),
                "not in llms.json (tokenizer)",
            );
        }
        Target::Mock => {}
    }
    for alias in unclaimed_aliases(&aliases, &plan) {
        plan.skipped.push((
            alias,
            format!("no fetchable artifact on {}", target.as_str()),
        ));
    }
    plan
}

/// SHA-pinned soak/STS corpus (`models/manifests/corpus.json`). Independent
/// of the arch target — optional (`FETCH_CORPUS=1` / `--fetch-corpus`).
pub fn plan_corpus() -> FetchPlan {
    let known = corpus_known_aliases();
    FetchPlan {
        batches: vec![FetchBatch {
            kind: ArtifactKind::Corpus,
            aliases: known.keys().cloned().collect(),
        }],
        skipped: Vec::new(),
    }
}

/// Append corpus batches onto an existing plan (idempotent).
pub fn with_corpus(mut plan: FetchPlan) -> FetchPlan {
    let extra = plan_corpus();
    plan.batches.extend(extra.batches);
    plan.skipped.extend(extra.skipped);
    plan
}

fn push_filtered(
    plan: &mut FetchPlan,
    kind: ArtifactKind,
    aliases: &[String],
    known: &BTreeMap<String, String>,
    skip_reason: &str,
) {
    let mut kept = Vec::new();
    for alias in aliases {
        if known.contains_key(alias) {
            kept.push(alias.clone());
        } else if matches!(
            kind,
            ArtifactKind::EmbeddingsOnnx | ArtifactKind::OvGenai | ArtifactKind::Mlx
        ) {
            // An embed alias that is not in this kind's table may still be an
            // LLM (or the other way around). Only record a skip when *no*
            // batch will claim it — deferred to `record_unclaimed`.
            let _ = skip_reason;
        }
    }
    if !kept.is_empty() {
        plan.batches.push(FetchBatch {
            kind,
            aliases: kept,
        });
    }
}

/// Aliases requested but not claimed by any batch.
pub fn unclaimed_aliases(aliases: &[String], plan: &FetchPlan) -> Vec<String> {
    let mut claimed = HashSet::new();
    for batch in &plan.batches {
        for a in &batch.aliases {
            claimed.insert(a.clone());
        }
    }
    aliases
        .iter()
        .filter(|a| !claimed.contains(*a))
        .cloned()
        .collect()
}

/// `serve = [...]` from an arch config (`config/nvidia.toml`, …).
pub fn load_serve_list(path: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    #[derive(serde::Deserialize)]
    struct ServeFile {
        #[serde(default)]
        serve: Vec<String>,
    }
    let parsed: ServeFile =
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(parsed.serve)
}

pub fn default_arch_config(root: &Path, target: Target) -> PathBuf {
    root.join("config")
        .join(format!("{}.toml", target.as_str()))
}

/// Materialize every batch. NVIDIA/Intel use the fetch library; Apple MLX
/// shells to `cargo xtask fetch --mlx` (same SHA-256 manifests).
pub fn ensure_plan(plan: &FetchPlan, root: &Path) -> Result<(), String> {
    let stdout = io::stdout();
    let stderr = io::stderr();
    let mut out = stdout.lock();
    let mut err = stderr.lock();
    ensure_plan_io(plan, root, &mut out, &mut err)
}

pub fn ensure_plan_io(
    plan: &FetchPlan,
    root: &Path,
    out: &mut impl Write,
    err: &mut impl Write,
) -> Result<(), String> {
    for batch in &plan.batches {
        if batch.aliases.is_empty() {
            continue;
        }
        writeln!(
            out,
            "--- fetch {} ({}) ---",
            batch.kind.as_str(),
            batch.aliases.join(", ")
        )
        .map_err(|e| e.to_string())?;
        match batch.kind {
            ArtifactKind::EmbeddingsOnnx => {
                run_ensure(ManifestKind::Embeddings, &batch.aliases, root, out, err)?;
            }
            ArtifactKind::LlmsGguf => {
                run_ensure(ManifestKind::Llms, &batch.aliases, root, out, err)?;
            }
            ArtifactKind::OvGenai => {
                run_ensure(ManifestKind::OvGenai, &batch.aliases, root, out, err)?;
            }
            ArtifactKind::Mlx => {
                fetch_mlx(&batch.aliases, root, out, err)?;
            }
            ArtifactKind::Corpus => {
                run_ensure(ManifestKind::Corpus, &batch.aliases, root, out, err)?;
            }
        }
    }
    Ok(())
}

fn run_ensure(
    kind: ManifestKind,
    aliases: &[String],
    root: &Path,
    out: &mut impl Write,
    err: &mut impl Write,
) -> Result<(), String> {
    let rc = ensure_aliases(kind, aliases, root, out, err).map_err(|e| e.to_string())?;
    if rc != 0 {
        return Err(format!("inferstream-fetch --{} exited {rc}", kind.as_str()));
    }
    Ok(())
}

fn fetch_mlx(
    aliases: &[String],
    root: &Path,
    out: &mut impl Write,
    err: &mut impl Write,
) -> Result<(), String> {
    let mut cmd = Command::new("cargo");
    cmd.args(["xtask", "fetch", "--mlx"])
        .args(aliases)
        .current_dir(root);
    writeln!(out, "  $ cargo xtask fetch --mlx {}", aliases.join(" "))
        .map_err(|e| e.to_string())?;
    let _ = out.flush();
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn cargo xtask fetch --mlx: {e}"))?;
    if !status.success() {
        let code = status.code().unwrap_or(1);
        writeln!(err, "error: cargo xtask fetch --mlx exited {code}").map_err(|e| e.to_string())?;
        return Err(format!("cargo xtask fetch --mlx exited {code}"));
    }
    Ok(())
}

/// Resolve the workspace root the same way `inferstream-fetch` does.
pub fn e2e_workspace_root() -> PathBuf {
    workspace_root()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_only() -> HashSet<String> {
        HashSet::new()
    }

    #[test]
    fn mock_plan_is_empty() {
        let plan = plan_fetches(
            Target::Mock,
            FetchScope::All,
            &empty_only(),
            &[],
            &Matrix::builtin(),
        );
        assert!(plan.is_empty());
    }

    #[test]
    fn nvidia_default_is_minilm_and_smoke_llms() {
        let plan = plan_fetches(
            Target::Nvidia,
            FetchScope::Default,
            &empty_only(),
            &[],
            &Matrix::builtin(),
        );
        let kinds: Vec<_> = plan.batches.iter().map(|b| b.kind).collect();
        assert!(kinds.contains(&ArtifactKind::EmbeddingsOnnx));
        assert!(kinds.contains(&ArtifactKind::LlmsGguf));
        let onnx = plan
            .batches
            .iter()
            .find(|b| b.kind == ArtifactKind::EmbeddingsOnnx)
            .unwrap();
        assert_eq!(onnx.aliases, vec!["minilm".to_string()]);
        let llms = plan
            .batches
            .iter()
            .find(|b| b.kind == ArtifactKind::LlmsGguf)
            .unwrap();
        assert!(llms.aliases.contains(&"default-llm".to_string()));
        assert!(llms.aliases.contains(&"qwen-0.5b".to_string()));
        assert!(!llms.aliases.contains(&"qwen-7b".to_string()));
    }

    #[test]
    fn intel_default_uses_ov_genai_not_onnx() {
        let plan = plan_fetches(
            Target::Intel,
            FetchScope::Default,
            &empty_only(),
            &[],
            &Matrix::builtin(),
        );
        assert!(plan
            .batches
            .iter()
            .any(|b| b.kind == ArtifactKind::OvGenai && b.aliases == ["minilm"]));
        assert!(plan
            .batches
            .iter()
            .any(|b| b.kind == ArtifactKind::LlmsGguf));
        assert!(!plan
            .batches
            .iter()
            .any(|b| b.kind == ArtifactKind::EmbeddingsOnnx));
    }

    #[test]
    fn apple_default_is_mlx_plus_llm_tokenizers() {
        let plan = plan_fetches(
            Target::Apple,
            FetchScope::Default,
            &empty_only(),
            &[],
            &Matrix::builtin(),
        );
        let mlx = plan
            .batches
            .iter()
            .find(|b| b.kind == ArtifactKind::Mlx)
            .expect("mlx batch");
        assert!(mlx.aliases.contains(&"minilm".to_string()));
        assert!(mlx.aliases.contains(&"default-llm".to_string()));
        assert!(mlx.aliases.contains(&"qwen-0.5b".to_string()));
        let llms = plan
            .batches
            .iter()
            .find(|b| b.kind == ArtifactKind::LlmsGguf)
            .expect("tokenizer batch");
        assert!(llms.aliases.contains(&"qwen-0.5b".to_string()));
    }

    #[test]
    fn only_restricts_default_set() {
        let only = ["mpnet"].iter().map(|s| (*s).to_string()).collect();
        let nvidia = plan_fetches(
            Target::Nvidia,
            FetchScope::Default,
            &only,
            &[],
            &Matrix::builtin(),
        );
        assert!(nvidia.is_empty(), "mpnet is not in the default set");

        let nvidia_all = plan_fetches(
            Target::Nvidia,
            FetchScope::All,
            &only,
            &[],
            &Matrix::builtin(),
        );
        let onnx = nvidia_all
            .batches
            .iter()
            .find(|b| b.kind == ArtifactKind::EmbeddingsOnnx)
            .unwrap();
        assert_eq!(onnx.aliases, vec!["mpnet".to_string()]);
        assert!(!nvidia_all
            .batches
            .iter()
            .any(|b| b.kind == ArtifactKind::LlmsGguf));
    }

    #[test]
    fn serve_scope_uses_config_list() {
        let serve = vec!["minilm".into(), "qwen-7b".into(), "bge-small".into()];
        let intel = plan_fetches(
            Target::Intel,
            FetchScope::Serve,
            &empty_only(),
            &serve,
            &Matrix::builtin(),
        );
        let ov = intel
            .batches
            .iter()
            .find(|b| b.kind == ArtifactKind::OvGenai)
            .unwrap();
        assert_eq!(ov.aliases, vec!["minilm".to_string()]);
        let llms = intel
            .batches
            .iter()
            .find(|b| b.kind == ArtifactKind::LlmsGguf)
            .unwrap();
        assert_eq!(llms.aliases, vec!["qwen-7b".to_string()]);
        let requested = candidate_aliases(
            Target::Intel,
            FetchScope::Serve,
            &empty_only(),
            &serve,
            &Matrix::builtin(),
        );
        let leftover = unclaimed_aliases(&requested, &intel);
        assert!(leftover.contains(&"bge-small".to_string()));
    }

    #[test]
    fn nvidia_toml_serve_parses() {
        let root = e2e_workspace_root();
        let list = load_serve_list(&default_arch_config(&root, Target::Nvidia)).unwrap();
        assert!(list.contains(&"minilm".to_string()));
        assert!(list.contains(&"default-llm".to_string()));
    }

    #[test]
    fn ensure_empty_plan_is_ok() {
        let tmp = std::env::temp_dir();
        let mut out = Vec::new();
        let mut err = Vec::new();
        ensure_plan_io(&FetchPlan::default(), &tmp, &mut out, &mut err).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn corpus_plan_lists_shakespeare_and_sts() {
        let plan = plan_corpus();
        assert_eq!(plan.batches.len(), 1);
        assert_eq!(plan.batches[0].kind, ArtifactKind::Corpus);
        assert!(plan.batches[0]
            .aliases
            .contains(&"tiny-shakespeare".to_string()));
        assert!(plan.batches[0].aliases.contains(&"sts-pairs".to_string()));
    }
}
