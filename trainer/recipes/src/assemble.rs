//! Assembling verified `turbo-trainer` receipt directories into a single-head
//! or multi-task bundle (#1726), verifying receipts, checkpoint metadata,
//! tensor names and shapes, and the assembled bundle via [`Bundle::open`].

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use turbo::bundle::{Bundle, sha256_hex};
use turbo::head::{
    BiaffineParser, EditTreeLemmatizer, OpenSpanMatcher, RelationClassifier, SentenceBoundary, SpanClassifier,
    TokenSplitter, TokenTagger, boundary, lemmatizer, open_span, parser, relation, span, split, tagger,
};
use turbo::manifest::{Format, HeadKind, HeadProvenance, HeadSpec, LicenseClass, Manifest};
use turbo::safetensors;

use crate::corpus::CorpusRecord;
use crate::receipt::{Receipt, SCHEMA};
use crate::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH};
use crate::{Error, Result};

/// A verified training run ready to be merged into an encoder bundle.
#[derive(Debug, Clone)]
pub struct VerifiedRun {
    pub dir: PathBuf,
    pub receipt: Receipt,
    pub spec: HeadSpec,
    pub license_class: LicenseClass,
    pub weights_bytes: Vec<u8>,
}

/// The result of assembling one or more [`VerifiedRun`]s into a bundle directory.
#[derive(Debug)]
pub struct AssembledBundle {
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub manifest_sha256: String,
    pub license_tier: LicenseClass,
    pub runs: Vec<VerifiedRun>,
}

/// Verifies a single run directory against its `receipt.json`, `head.json`, and
/// `head.safetensors`.
pub fn verify_run(run_dir: &Path) -> Result<VerifiedRun> {
    let at = |msg: String| Error(format!("{}: {msg}", run_dir.display()));
    let receipt_bytes = fs::read(run_dir.join("receipt.json")).map_err(|e| at(format!("receipt.json: {e}")))?;
    let receipt: Receipt = serde_json::from_slice(&receipt_bytes).map_err(|e| at(format!("receipt.json: {e}")))?;
    if receipt.schema != SCHEMA {
        return Err(at(format!("receipt.json: schema {:?} is not {SCHEMA:?}", receipt.schema)));
    }
    if receipt.commit.trim().is_empty() {
        return Err(at("receipt.json: commit is empty".into()));
    }
    let enc = receipt.encoder.as_ref().ok_or_else(|| at("receipt.json: missing encoder record".into()))?;
    if enc.hidden == 0 || !is_hex64(&enc.manifest_sha256) {
        return Err(at("receipt.json: invalid encoder hidden size or manifest_sha256".into()));
    }
    verify_corpus(&receipt.corpus).map_err(at)?;
    let license_class = receipt.corpus.class()?;

    let spec = receipt.head.as_ref().ok_or_else(|| at("receipt.json: missing head spec".into()))?.clone();
    spec.validate("head").map_err(|e| at(e.message))?;
    if spec.license_class != Some(license_class) {
        return Err(at(format!(
            "head {:?}: license_class {:?} does not match corpus class {:?}",
            spec.name, spec.license_class, license_class
        )));
    }

    let training = receipt.training.as_ref().ok_or_else(|| at("receipt.json: missing training record".into()))?;
    if training.dev.is_empty()
        || training.train_loss.is_empty()
        || training.best_epoch >= training.dev.len()
        || training.dev.iter().chain(&training.train_loss).chain(&training.epoch_seconds).any(|v| !v.is_finite())
    {
        return Err(at("receipt.json: invalid best_epoch or non-finite training curves".into()));
    }
    let model_eval =
        receipt.eval.model.as_ref().ok_or_else(|| at("receipt.json: missing held-out model evaluation".into()))?;
    if !model_eval.score().is_finite() || !receipt.eval.baseline.score().is_finite() {
        return Err(at("receipt.json: non-finite evaluation score".into()));
    }

    let head_json_bytes = verify_output(run_dir, &receipt, "head.json")?;
    let disk_spec: HeadSpec = serde_json::from_slice(&head_json_bytes).map_err(|e| at(format!("head.json: {e}")))?;
    if disk_spec != spec {
        return Err(at("head.json does not match receipt.head".into()));
    }
    let weights_bytes = verify_output(run_dir, &receipt, "head.safetensors")?;
    verify_head_tensors(&spec, enc.hidden, training.best_epoch, training.dev[training.best_epoch], &weights_bytes)
        .map_err(at)?;

    Ok(VerifiedRun { dir: run_dir.to_path_buf(), receipt, spec, license_class, weights_bytes })
}

fn verify_corpus(c: &CorpusRecord) -> std::result::Result<(), String> {
    for (field, val) in [
        ("id", c.id.as_str()),
        ("name", c.name.as_str()),
        ("licence", c.licence.as_str()),
        ("spdx", c.spdx.as_str()),
        ("terms", c.terms.as_str()),
        ("attribution", c.attribution.as_str()),
        ("split", c.split.as_str()),
    ] {
        if val.trim().is_empty() {
            return Err(format!("receipt.json: corpus.{field} is empty"));
        }
    }
    let has_digest = is_hex64(&c.corpus_sha256)
        || (!c.files.is_empty() && c.files.iter().all(|f| !f.path.is_empty() && is_hex64(&f.sha256)));
    if !has_digest {
        return Err("receipt.json: corpus has neither corpus_sha256 nor pinned files".into());
    }
    Ok(())
}

fn verify_output(run_dir: &Path, receipt: &Receipt, name: &str) -> Result<Vec<u8>> {
    let at = |msg: String| Error(format!("{}: {msg}", run_dir.display()));
    let out = receipt
        .outputs
        .iter()
        .find(|o| o.path == name)
        .ok_or_else(|| at(format!("receipt.json: outputs does not list {name:?}")))?;
    let bytes = fs::read(run_dir.join(name)).map_err(|e| at(format!("{name}: {e}")))?;
    if bytes.len() as u64 != out.size {
        return Err(at(format!("{name}: {} bytes on disk, receipt says {}", bytes.len(), out.size)));
    }
    let digest = sha256_hex(&bytes);
    if digest != out.sha256 {
        return Err(at(format!("{name}: sha256 {digest} on disk, receipt says {}", out.sha256)));
    }
    Ok(bytes)
}

fn verify_head_tensors(
    spec: &HeadSpec,
    hidden: usize,
    best_epoch: usize,
    best_dev: f64,
    weights_bytes: &[u8],
) -> std::result::Result<(), String> {
    let _ = safetensors::File::parse("head.safetensors", weights_bytes).map_err(|e| e.message)?;
    let (meta, header) = parse_safetensors_header(weights_bytes)?;
    verify_checkpoint_metadata(&meta, best_epoch, best_dev)?;
    let n_hdr = u64::from_le_bytes(weights_bytes[..8].try_into().unwrap()) as usize;
    let data = &weights_bytes[8 + n_hdr..];
    let prefix = format!("{}.", spec.tensor_prefix);
    if header.is_empty() {
        return Err("head.safetensors: contains no tensors".into());
    }
    let expected = match spec.kind {
        HeadKind::TokenTagger => tagger::tensors(spec, hidden),
        HeadKind::BiaffineParser => parser::tensors(spec, hidden),
        HeadKind::EditTreeLemmatizer => lemmatizer::tensors(spec, hidden),
        HeadKind::SentenceBoundary => boundary::tensors(spec, hidden),
        HeadKind::TokenSplit => split::tensors(spec, hidden),
        HeadKind::SpanClassifier => span::tensors(spec, hidden),
        HeadKind::OpenSpan => open_span::tensors(spec, hidden),
        HeadKind::RelationClassifier => relation::tensors(spec, hidden),
    };
    if header.len() != expected.len() {
        return Err(format!("head.safetensors: {} tensors on disk, head expects {}", header.len(), expected.len()));
    }
    for (suffix, exp_shape) in &expected {
        let full = format!("{prefix}{suffix}");
        let entry = header.get(&full).ok_or_else(|| format!("head.safetensors: missing tensor {full:?}"))?;
        if entry.dtype != "F32" {
            return Err(format!("head.safetensors: tensor {full:?} has dtype {:?}, expected \"F32\"", entry.dtype));
        }
        let actual_shape: Vec<u64> = entry.shape.iter().map(|&d| d as u64).collect();
        if &actual_shape != exp_shape {
            return Err(format!(
                "head.safetensors: tensor {full:?} has shape {actual_shape:?}, expected {exp_shape:?}"
            ));
        }
    }
    let get = |suffix: &str| -> Option<Vec<f32>> {
        let full = format!("{prefix}{suffix}");
        let e = header.get(&full)?;
        let slice = data.get(e.data_offsets[0]..e.data_offsets[1])?;
        Some(slice.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    };
    let checked = match spec.kind {
        HeadKind::TokenTagger => TokenTagger::from_tensors(spec, hidden, &get).map(|t| t.hidden()),
        HeadKind::BiaffineParser => BiaffineParser::from_tensors(spec, hidden, &get).map(|p| p.hidden()),
        HeadKind::EditTreeLemmatizer => EditTreeLemmatizer::from_tensors(spec, hidden, &get).map(|l| l.hidden()),
        HeadKind::SentenceBoundary => SentenceBoundary::from_tensors(spec, hidden, &get).map(|b| b.hidden()),
        HeadKind::TokenSplit => TokenSplitter::from_tensors(spec, hidden, &get).map(|s| s.hidden()),
        HeadKind::SpanClassifier => SpanClassifier::from_tensors(spec, hidden, &get).map(|s| s.hidden()),
        HeadKind::OpenSpan => OpenSpanMatcher::from_tensors(spec, hidden, &get).map(|s| s.hidden()),
        HeadKind::RelationClassifier => RelationClassifier::from_tensors(spec, hidden, &get).map(|r| r.hidden()),
    }
    .map_err(|e| format!("head.safetensors: {}", e.message))?;
    if checked != hidden {
        return Err(format!(
            "head {:?} tensors take {checked} features a word, encoder receipt says {hidden}",
            spec.name
        ));
    }
    Ok(())
}

/// Verifies every run in `run_dirs` against `encoder_bundle`, writes the
/// assembled single-head or multi-task bundle to `out_dir`, and verifies it
/// with [`Bundle::open`].
pub fn assemble_bundle(encoder_bundle: &Path, run_dirs: &[&Path], out_dir: &Path) -> Result<AssembledBundle> {
    if run_dirs.is_empty() {
        return Err(Error("assemble: at least one run directory is required".into()));
    }
    let enc_canon =
        fs::canonicalize(encoder_bundle).map_err(|e| Error(format!("{}: {e}", encoder_bundle.display())))?;
    if out_dir == encoder_bundle || fs::canonicalize(out_dir).is_ok_and(|out_canon| out_canon == enc_canon) {
        return Err(Error(format!(
            "assemble: out_dir {} must not equal encoder_bundle {}",
            out_dir.display(),
            encoder_bundle.display()
        )));
    }
    let enc_bundle =
        Bundle::open(encoder_bundle).map_err(|e| Error(format!("{}: {}", encoder_bundle.display(), e.message)))?;
    let enc_manifest_bytes = fs::read(encoder_bundle.join("manifest.json"))
        .map_err(|e| Error(format!("{}: {e}", encoder_bundle.display())))?;
    let mut manifest = Manifest::parse(&enc_manifest_bytes).map_err(|e| Error(e.message))?;
    let enc_sha256 = sha256_hex(&enc_manifest_bytes);
    let enc_hidden = manifest
        .architecture
        .as_ref()
        .ok_or_else(|| Error(format!("{}: bundle has no encoder architecture", encoder_bundle.display())))?
        .hidden as usize;

    if manifest.model.license_class.is_none() {
        let lic: turbo_zoo::Licence = manifest.model.license.parse().map_err(|e: String| {
            Error(format!("{}: model.license {:?}: {e}", encoder_bundle.display(), manifest.model.license))
        })?;
        manifest.model.license_class = Some(to_core_class(lic.class())?);
    }

    let (weights_name, weights_path) = manifest
        .artifacts
        .iter()
        .find(|a| a.format == Format::Safetensors)
        .and_then(|a| a.files.first().map(|p| (a.name.clone(), p.clone())))
        .ok_or_else(|| Error(format!("{}: no FORMAT_SAFETENSORS artifact in bundle", encoder_bundle.display())))?;

    let mut runs = Vec::with_capacity(run_dirs.len());
    for &dir in run_dirs {
        let run = verify_run(dir)?;
        let rec_enc = run.receipt.encoder.as_ref().expect("checked in verify_run");
        if rec_enc.model != manifest.model.id
            || rec_enc.revision != manifest.model.revision
            || rec_enc.manifest_sha256 != enc_sha256
            || rec_enc.hidden != enc_hidden
        {
            return Err(Error(format!(
                "{}: trained on encoder {}@{} (sha256 {}, hidden {}), bundle is {}@{} (sha256 {}, hidden {})",
                dir.display(),
                rec_enc.model,
                rec_enc.revision,
                rec_enc.manifest_sha256,
                rec_enc.hidden,
                manifest.model.id,
                manifest.model.revision,
                enc_sha256,
                enc_hidden
            )));
        }
        if manifest.heads.iter().any(|h| h.name == run.spec.name)
            || runs.iter().any(|o: &VerifiedRun| o.spec.name == run.spec.name)
        {
            return Err(Error(format!("{}: duplicate head name {:?}", dir.display(), run.spec.name)));
        }
        if manifest.heads.iter().any(|h| prefixes_collide(&h.tensor_prefix, &run.spec.tensor_prefix))
            || runs.iter().any(|o: &VerifiedRun| prefixes_collide(&o.spec.tensor_prefix, &run.spec.tensor_prefix))
        {
            return Err(Error(format!("{}: colliding tensor_prefix {:?}", dir.display(), run.spec.tensor_prefix)));
        }
        runs.push(run);
    }

    let base_weights = fs::read(encoder_bundle.join(&weights_path))
        .map_err(|e| Error(format!("{}: {e}", encoder_bundle.join(&weights_path).display())))?;
    let merged_weights = merge_safetensors(&base_weights, &runs)?;

    for f in &manifest.files {
        if f.path == weights_path {
            continue;
        }
        let src = encoder_bundle.join(&f.path);
        let dst = out_dir.join(&f.path);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).map_err(|e| Error(format!("{}: {e}", parent.display())))?;
        }
        fs::copy(&src, &dst).map_err(|e| Error(format!("{}: {e}", dst.display())))?;
    }

    let weights_dst = out_dir.join(&weights_path);
    if let Some(parent) = weights_dst.parent() {
        fs::create_dir_all(parent).map_err(|e| Error(format!("{}: {e}", parent.display())))?;
    }
    fs::write(&weights_dst, &merged_weights).map_err(|e| Error(format!("{}: {e}", weights_dst.display())))?;

    for f in &mut manifest.files {
        if f.path == weights_path {
            f.size = merged_weights.len() as u64;
            f.sha256 = sha256_hex(&merged_weights);
        }
    }

    for run in &runs {
        let mut spec = run.spec.clone();
        spec.artifact = weights_name.clone();
        manifest.heads.push(spec);
        let tr = run.receipt.training.as_ref().expect("checked in verify_run");
        let ev = run.receipt.eval.model.as_ref().expect("checked in verify_run");
        let corpus_sha = if is_hex64(&run.receipt.corpus.corpus_sha256) {
            run.receipt.corpus.corpus_sha256.clone()
        } else {
            sha256_hex(
                run.receipt
                    .corpus
                    .files
                    .iter()
                    .map(|f| format!("{}:{}\n", f.path, f.sha256))
                    .collect::<String>()
                    .as_bytes(),
            )
        };
        manifest.head_provenance.push(HeadProvenance {
            head: run.spec.name.clone(),
            commit: run.receipt.commit.clone(),
            encoder_manifest_sha256: enc_bundle.manifest_sha256.clone(),
            corpus_id: run.receipt.corpus.id.clone(),
            corpus_name: run.receipt.corpus.name.clone(),
            license: run.receipt.corpus.licence.clone(),
            spdx: run.receipt.corpus.spdx.clone(),
            terms: run.receipt.corpus.terms.clone(),
            attribution: run.receipt.corpus.attribution.clone(),
            corpus_sha256: corpus_sha,
            recorded: run.receipt.unix_time.to_string(),
            recipe_only: run.receipt.corpus.recipe_only,
            best_epoch: tr.best_epoch,
            metric: ev.metric_name().into(),
            dev_score: tr.dev[tr.best_epoch],
            test_score: ev.score(),
            baseline_score: run.receipt.eval.baseline.score(),
        });
    }

    let manifest_json = manifest.to_json();
    fs::write(out_dir.join("manifest.json"), &manifest_json)
        .map_err(|e| Error(format!("{}: {e}", out_dir.display())))?;

    let opened = Bundle::open(out_dir).map_err(|e| Error(format!("{}: {}", out_dir.display(), e.message)))?;
    let license_tier =
        opened.manifest.license_tier().ok_or_else(|| Error("assembled bundle has no license_tier".into()))?;

    Ok(AssembledBundle {
        dir: out_dir.to_path_buf(),
        manifest,
        manifest_sha256: opened.manifest_sha256,
        license_tier,
        runs,
    })
}

fn prefixes_collide(a: &str, b: &str) -> bool {
    a == b || a.starts_with(&format!("{b}.")) || b.starts_with(&format!("{a}."))
}

fn to_core_class(class: turbo_zoo::LicenceClass) -> Result<LicenseClass> {
    use turbo_zoo::LicenceClass;
    match class {
        LicenceClass::Permissive => Ok(LicenseClass::Permissive),
        LicenceClass::Lgpl => Ok(LicenseClass::Lgpl),
        LicenceClass::ShareAlike => Ok(LicenseClass::ShareAlike),
        LicenceClass::Gpl => Ok(LicenseClass::Gpl),
        LicenceClass::LocalOnly => Ok(LicenseClass::LocalOnly),
        LicenceClass::Refused => Err(Error("the zoo refuses the model licence".into())),
    }
}

pub(crate) fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn verify_checkpoint_metadata(
    meta: &BTreeMap<String, String>,
    best_epoch: usize,
    best_dev: f64,
) -> std::result::Result<(), String> {
    let epoch_str = meta
        .get(CHECKPOINT_EPOCH)
        .ok_or_else(|| format!("head.safetensors: missing __metadata__.{CHECKPOINT_EPOCH}"))?;
    let epoch: usize = epoch_str
        .parse()
        .map_err(|e| format!("head.safetensors: invalid __metadata__.{CHECKPOINT_EPOCH} {epoch_str:?}: {e}"))?;
    if epoch != best_epoch {
        return Err(format!(
            "head.safetensors: __metadata__.{CHECKPOINT_EPOCH} is {epoch}, receipt.training.best_epoch is {best_epoch}"
        ));
    }
    let dev_str =
        meta.get(CHECKPOINT_DEV).ok_or_else(|| format!("head.safetensors: missing __metadata__.{CHECKPOINT_DEV}"))?;
    let dev: f64 = dev_str
        .parse()
        .map_err(|e| format!("head.safetensors: invalid __metadata__.{CHECKPOINT_DEV} {dev_str:?}: {e}"))?;
    if !dev.is_finite() || (dev - best_dev).abs() > 1e-12 {
        return Err(format!(
            "head.safetensors: __metadata__.{CHECKPOINT_DEV} is {dev}, receipt.training.dev[{best_epoch}] is {best_dev}"
        ));
    }
    Ok(())
}

type SafetensorsHeader = (BTreeMap<String, String>, BTreeMap<String, RawEntry>);

fn parse_safetensors_header(bytes: &[u8]) -> std::result::Result<SafetensorsHeader, String> {
    if bytes.len() < 8 {
        return Err("safetensors file shorter than 8 bytes".into());
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let end = 8usize.checked_add(n).filter(|&e| e <= bytes.len()).ok_or("bad safetensors header length")?;
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&bytes[8..end]).map_err(|e| e.to_string())?;
    let mut meta = BTreeMap::new();
    let mut out = BTreeMap::new();
    for (k, v) in raw {
        if k == "__metadata__" {
            meta = serde_json::from_value(v).map_err(|e| format!("__metadata__: {e}"))?;
            continue;
        }
        let entry: RawEntry = serde_json::from_value(v).map_err(|e| format!("{k}: {e}"))?;
        out.insert(k, entry);
    }
    Ok((meta, out))
}

/// Appends every verified head's tensors onto `base`'s safetensors table.
/// Any `__metadata__` map on `base` or the head checkpoints is omitted in the
/// merged file; per-head checkpoint and corpus provenance live in
/// `manifest.json` (`head_provenance`).
fn merge_safetensors(base: &[u8], runs: &[VerifiedRun]) -> Result<Vec<u8>> {
    let base_n = u64::from_le_bytes(base[..8].try_into().unwrap()) as usize;
    let (_, base_header) = parse_safetensors_header(base).map_err(Error)?;
    let mut data = base[8 + base_n..].to_vec();
    let mut entries: Vec<(String, String, Vec<usize>, [usize; 2])> =
        base_header.into_iter().map(|(k, e)| (k, e.dtype, e.shape, e.data_offsets)).collect();

    for run in runs {
        let n = u64::from_le_bytes(run.weights_bytes[..8].try_into().unwrap()) as usize;
        let run_data = &run.weights_bytes[8 + n..];
        let (_, run_header) = parse_safetensors_header(&run.weights_bytes).map_err(Error)?;
        for (k, e) in run_header {
            if entries.iter().any(|(existing, ..)| existing == &k) {
                return Err(Error(format!("safetensors collision on tensor {k:?}")));
            }
            while !data.len().is_multiple_of(8) {
                data.push(0);
            }
            let start = data.len();
            data.extend_from_slice(&run_data[e.data_offsets[0]..e.data_offsets[1]]);
            let end = data.len();
            entries.push((k, e.dtype, e.shape, [start, end]));
        }
    }

    let mut header = String::from("{");
    for (i, (name, dtype, shape, [s, e])) in entries.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let shape_s = shape.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
        let key = serde_json::to_string(name).expect("string serializes");
        header.push_str(&format!("{key}:{{\"dtype\":{dtype:?},\"shape\":[{shape_s}],\"data_offsets\":[{s},{e}]}}"));
    }
    header.push('}');
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    Ok(out)
}
