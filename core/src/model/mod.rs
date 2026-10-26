//! A bundle's model for one device: loader rules 6 to 8 of docs/bundle.md,
//! after Bundle::open and Tokenizer::load have applied 1 to 5. Which
//! artifact the device loads, its files read once and verified, and every
//! tensor a BERT encoder needs checked against the architecture.

mod heads;
mod place;
mod roles;

pub use heads::{classifier_tensors, mlm_tensors, projection_tensors};
pub use roles::{
    BERT_EMBEDDING_ROLES, BERT_LAYER_ROLES, DEBERTA_EMBEDDING_ROLES, Expected, bert_tensors, embedding_tensors,
};

use std::ffi::c_void;

use serde::Serialize;

use crate::backend::{
    TURBO_FAMILY_BERT, TURBO_FAMILY_DEBERTA_V2, TURBO_FAMILY_ROBERTA, TURBO_FAMILY_STATIC, TURBO_STATIC_TENSORS,
    format_bit, turbo_backend_model, turbo_backend_tensor,
};
use crate::bundle::{AlignedBytes, Bundle, sha256_hex};
use crate::manifest::{
    Activation, Architecture, Artifact, Family, Format, GraphInput, Manifest, PositionEmbedding, TensorRole,
};
use crate::status::{BUNDLE_NO_ARTIFACT, Error, INTERNAL, INVALID_ARGUMENT, Result, UNSUPPORTED};

use place::place_static;
pub(crate) use place::{Placed, place};

/// The manifest's name for an enum value, as it is written there.
fn enum_name(v: impl Serialize) -> String {
    serde_json::to_value(v).ok().and_then(|v| v.as_str().map(str::to_owned)).unwrap_or_default()
}

/// Rule 6: the first artifact, in manifest order, that lists `backend`,
/// was built for `arch` or for any device, is in one of `formats` (the
/// backend's TURBO_FORMAT_BIT set, turbo_backend::formats), and is
/// something this build hands a backend: raw weights, a HEF, or an
/// OpenVINO IR. None is BUNDLE_NO_ARTIFACT, saying why each was skipped.
pub fn choose(m: &Manifest, backend: &str, formats: u32, arch: &str) -> Result<usize> {
    let mut skipped = Vec::new();
    for (i, a) in m.artifacts.iter().enumerate() {
        let why = if !a.backends.iter().any(|b| b == backend) {
            format!("backends {:?} has no {backend}", a.backends)
        } else if !a.target.is_empty() && a.target != arch {
            format!("target {} is not this device's {arch}", a.target)
        } else if formats & format_bit(a.format.value()) == 0 {
            format!("{} is not a format the {backend} backend loads", enum_name(a.format))
        } else if !matches!(a.format, Format::Safetensors | Format::Hef | Format::OpenvinoIr) {
            format!("{} is not a format this build hands a backend", enum_name(a.format))
        } else {
            return Ok(i);
        };
        skipped.push(format!("{}: {why}", a.name));
    }
    Err(Error::new(
        BUNDLE_NO_ARTIFACT,
        format!("no artifact the {backend} backend on {arch} loads: {}", skipped.join("; ")),
    ))
}

/// The hash reported for an artifact: its file's, or for several files
/// the SHA-256 of their hex hashes, as the manifest writes them,
/// concatenated in listed order.
pub fn artifact_sha256(m: &Manifest, a: &Artifact) -> String {
    match a.files.as_slice() {
        [one] => m.file(one).sha256.clone(),
        files => sha256_hex(files.iter().map(|f| m.file(f).sha256.as_str()).collect::<String>().as_bytes()),
    }
}

/// A static model's tensors, in TURBO_STATIC_* order: its table, its
/// weight per token and its token mapping, the last two optional.
pub const STATIC_ROLES: [TensorRole; TURBO_STATIC_TENSORS as usize] =
    [TensorRole::StaticEmbeddings, TensorRole::StaticWeights, TensorRole::StaticMapping];

/// Whether the backend named `backend` runs a model of limited lookahead
/// `lookahead` (0 for none): only the CPU's attention takes a band, and any
/// other would run the whole row and give wrong vectors.
pub fn runs_band(backend: &str, lookahead: u32) -> Result<()> {
    if lookahead == 0 || backend == "cpu" {
        return Ok(());
    }
    Err(Error::new(
        UNSUPPORTED,
        format!(
            "architecture.lookahead {lookahead}: the {backend} backend has no limited-lookahead attention; load the model on the cpu backend"
        ),
    ))
}

/// Whether the backend named `backend` holds a model of TURBO_FAMILY_*
/// `family`: only the CPU has a DeBERTa-v2 encoder so far, and any other
/// would refuse the family at model_load.
pub fn runs_family(backend: &str, family: u32) -> Result<()> {
    if family != TURBO_FAMILY_DEBERTA_V2 || backend == "cpu" {
        return Ok(());
    }
    Err(Error::new(
        UNSUPPORTED,
        format!(
            "FAMILY_DEBERTA_V2: the {backend} backend has no DeBERTa-v2 encoder yet; load the model on the cpu backend"
        ),
    ))
}

/// The family model_load is told. Every combination the manifest can name
/// is listed, so a new family, activation or position embedding does not
/// compile until it has a place here.
fn family(a: &Architecture) -> u32 {
    match (a.family, a.activation, a.position_embedding) {
        (Family::Bert, Activation::GeluErf, PositionEmbedding::Absolute) => TURBO_FAMILY_BERT,
        (Family::Roberta, Activation::GeluErf, PositionEmbedding::Absolute) => TURBO_FAMILY_ROBERTA,
        (Family::DebertaV2, Activation::GeluErf, PositionEmbedding::Relative) => TURBO_FAMILY_DEBERTA_V2,
        (Family::Bert | Family::Roberta, _, PositionEmbedding::Relative)
        | (Family::DebertaV2, _, PositionEmbedding::Absolute) => {
            unreachable!("validate_positions pairs each family with its position embedding")
        }
    }
}

/// The chosen artifact as the core holds it: the verified bytes of each
/// file its tensors are in, read once, and where each tensor is in them;
/// for a HEF, its file's verified bytes too. These bytes are the only host
/// copy; what model_load is handed points into them.
pub struct Weights {
    /// The files the tensors are in: the raw-weights artifact's, or for a
    /// HEF that starts at embeddings, its host_weights artifact's.
    files: Vec<AlignedBytes>,
    /// In the family's TURBO_* order; None for a static model's optional
    /// tensor its artifact does not have.
    tensors: Vec<Option<Placed>>,
    /// TURBO_DTYPE_* every tensor is stored in, a static model's table's;
    /// 0 when there are none.
    pub dtype: u32,
    family: u32,
    arch: [u32; 7],
    position_offset: u32,
    /// architecture.lookahead, 0 for none.
    pub lookahead: u32,
    /// architecture.relative's position_buckets and max_relative_positions,
    /// 0 for a family without them.
    relative: [u32; 2],
    layer_norm_eps: f64,
    /// A compiled artifact's first file, as hashed: the whole HEF, or an
    /// OpenVINO IR's xml.
    artifact: Option<AlignedBytes>,
    /// An OpenVINO IR's second file, its weights (.bin), as hashed.
    artifact2: Option<AlignedBytes>,
    /// TURBO_FORMAT_*, TURBO_INPUT_*, TURBO_OUTPUT_*, TURBO_DTYPE_* or 0,
    /// fixed_seq and fixed_batch, as the manifest gives the artifact.
    shape: [u32; 6],
}

impl Weights {
    /// Rules 7 and 8 for artifact `index`: its files, and its host_weights
    /// artifact's, verified before any byte is used; then every tensor it
    /// hands the backend, present with the shape the architecture implies
    /// and one dtype. Raw weights hand every tensor of the encoder; a HEF
    /// that starts at embeddings hands its host_weights artifact's
    /// embedding tensors, and one that starts at token ids none.
    pub fn load(bundle: &Bundle, index: usize) -> Result<Weights> {
        let m = &bundle.manifest;
        let art = m.artifacts.get(index).ok_or_else(|| {
            Error::new(INVALID_ARGUMENT, format!("artifact {index}: bundle has {} artifacts", m.artifacts.len()))
        })?;
        // A static model's artifacts are raw weights alone (validate()).
        if let Some(st) = &m.static_embedding {
            let dim = m.encoder().dim;
            // The table is read in place, never converted: mapped, not copied.
            let files = art.files.iter().map(|f| bundle.map_verified(f)).collect::<Result<Vec<_>>>()?;
            let tensors = place_static(index, art, &files, st, dim)?;
            let dtype = tensors[0].as_ref().expect("validate() requires static_embeddings").dtype;
            return Ok(Weights {
                files,
                tensors,
                dtype,
                family: TURBO_FAMILY_STATIC,
                arch: [0, dim, 0, 0, st.vocab_size, 0, 1],
                position_offset: 0,
                lookahead: 0,
                relative: [0, 0],
                layer_norm_eps: 0.0,
                artifact: None,
                artifact2: None,
                shape: [art.format.value(), art.graph_input.value(), art.graph_output.value(), 0, 0, 0],
            });
        }
        let a = m.architecture.as_ref().expect("validate() requires architecture for raw weights and HEFs");
        let host = (!art.host_weights.is_empty())
            .then(|| m.artifacts.iter().position(|h| h.name == art.host_weights).expect("validate() checks the name"));

        let (artifact, artifact2, tensor_art) = match art.format {
            Format::Safetensors => (None, None, Some(index)),
            Format::Hef => {
                let hef = bundle.read_verified_aligned(&art.files[0])?;
                (Some(hef), None, if art.graph_input == GraphInput::Embeddings { host } else { None })
            }
            Format::OpenvinoIr => {
                let xml = bundle.read_verified_aligned(&art.files[0])?;
                let bin = bundle.read_verified_aligned(&art.files[1])?;
                (Some(xml), Some(bin), if art.graph_input == GraphInput::Embeddings { host } else { None })
            }
            f => return Err(Error::new(INTERNAL, format!("artifact {:?}: the core hands on no {f:?}", art.name))),
        };
        let (files, tensors, dtype) = match tensor_art {
            Some(t) => {
                let ta = &m.artifacts[t];
                let mut expected = if t == index { bert_tensors(t, ta, a)? } else { embedding_tensors(t, ta, a)? };
                let encoder_len = expected.len();
                if t == index
                    && let Some(c) = &m.classifier
                {
                    expected.extend(classifier_tensors(t, ta, a, c)?);
                }
                if t == index
                    && let Some(p) = &m.projection
                {
                    expected.extend(projection_tensors(t, ta, a, p)?);
                }
                // Rule 8 covers the MLM head too: its tensors share the
                // artifact's dtype, though MlmHead reads them, not the backend.
                if t == index && m.embed.as_ref().is_some_and(|e| e.sparse.is_some()) {
                    expected.extend(mlm_tensors(t, ta, a)?);
                }
                let files = ta.files.iter().map(|f| bundle.read_verified_aligned(f)).collect::<Result<Vec<_>>>()?;
                let (mut tensors, dtype) = place(ta, &files, &expected)?;
                tensors.truncate(encoder_len);
                (files, tensors, dtype)
            }
            None => (Vec::new(), Vec::new(), 0),
        };
        // Rule 7: the host_weights artifact's files are verified whether or
        // not their tensors are handed on.
        if let Some(h) = host.filter(|&h| Some(h) != tensor_art) {
            let verified: Vec<&String> =
                art.files.iter().chain(tensor_art.iter().flat_map(|&t| &m.artifacts[t].files)).collect();
            for f in m.artifacts[h].files.iter().filter(|f| !verified.contains(f)) {
                bundle.read_verified(f)?;
            }
        }
        Ok(Weights {
            files,
            tensors: tensors.into_iter().map(Some).collect(),
            dtype,
            family: family(a),
            arch: [a.layers, a.hidden, a.heads, a.intermediate, a.vocab_size, a.max_positions, a.token_types],
            position_offset: a.position_offset,
            lookahead: a.lookahead.unwrap_or(0),
            relative: a.relative.map_or([0, 0], |r| [r.position_buckets, r.max_relative_positions]),
            layer_norm_eps: a.layer_norm_eps,
            artifact,
            artifact2,
            shape: [
                art.format.value(),
                art.graph_input.value(),
                art.graph_output.value(),
                art.compute_dtype.map_or(0, |d| d.value()),
                art.fixed_seq,
                art.fixed_batch,
            ],
        })
    }

    /// The tensors as model_load takes them, pointing into the files the
    /// core holds. Valid while `self` is.
    pub fn tensors(&self) -> Vec<turbo_backend_tensor> {
        self.tensors
            .iter()
            .map(|t| match t {
                Some(t) => {
                    let mut shape = [0u64; 2];
                    shape[..t.shape.len()].copy_from_slice(&t.shape);
                    turbo_backend_tensor {
                        name: t.name.as_ptr(),
                        data: self.files[t.file][t.range.clone()].as_ptr() as *const c_void,
                        shape,
                        ndim: t.shape.len() as u32,
                        dtype: t.dtype,
                        bytes: t.range.len() as u64,
                    }
                }
                None => turbo_backend_tensor {
                    name: std::ptr::null(),
                    data: std::ptr::null(),
                    shape: [0; 2],
                    ndim: 0,
                    dtype: 0,
                    bytes: 0,
                },
            })
            .collect()
    }

    /// The description model_load takes, over `tensors`.
    pub fn desc(&self, tensors: &[turbo_backend_tensor]) -> turbo_backend_model {
        let [layers, hidden, heads, intermediate, vocab_size, max_positions, token_types] = self.arch;
        let [format, graph_input, graph_output, compute_dtype, fixed_seq, fixed_batch] = self.shape;
        let (artifact, artifact_bytes) = match &self.artifact {
            Some(a) => (a.as_ptr() as *const c_void, a.len() as u64),
            None => (std::ptr::null(), 0),
        };
        let (artifact2, artifact2_bytes) = match &self.artifact2 {
            Some(a) => (a.as_ptr() as *const c_void, a.len() as u64),
            None => (std::ptr::null(), 0),
        };
        turbo_backend_model {
            struct_size: size_of::<turbo_backend_model>() as u32,
            family: self.family,
            dtype: self.dtype,
            layers,
            hidden,
            heads,
            intermediate,
            vocab_size,
            max_positions,
            token_types,
            layer_norm_eps: self.layer_norm_eps,
            tensor_count: tensors.len() as u32,
            position_offset: self.position_offset,
            tensors: if tensors.is_empty() { std::ptr::null() } else { tensors.as_ptr() },
            format,
            graph_input,
            graph_output,
            compute_dtype,
            fixed_seq,
            fixed_batch,
            artifact,
            artifact_bytes,
            artifact2,
            artifact2_bytes,
            head: std::ptr::null(),
            head_out: 0,
            lookahead: self.lookahead,
            position_buckets: self.relative[0],
            max_relative_positions: self.relative[1],
        }
    }

    /// The dtype turbo_model_info reports: the compute dtype the
    /// compilation fixed, else the one the weights are stored in.
    pub fn info_dtype(&self) -> u32 {
        match self.shape[3] {
            0 => self.dtype,
            d => d,
        }
    }

    /// The TURBO_FAMILY_* model_load is told.
    pub fn family(&self) -> u32 {
        self.family
    }

    /// Whether the model is a static one (TURBO_FAMILY_STATIC).
    pub fn is_static(&self) -> bool {
        self.family == TURBO_FAMILY_STATIC
    }

    /// Ids a row may hold: the rows of the word embedding table.
    pub fn vocab_size(&self) -> u32 {
        self.arch[4]
    }

    /// Token types a row may hold: the rows of the type embedding table.
    pub fn token_types(&self) -> u32 {
        self.arch[6]
    }

    /// The verified bytes of each file the tensors are in, in the
    /// artifact's order.
    pub fn files(&self) -> Vec<&[u8]> {
        self.files.iter().map(|f| &f[..]).collect()
    }

    /// A compiled artifact's verified bytes; None for raw weights.
    pub fn artifact(&self) -> Option<&[u8]> {
        self.artifact.as_ref().map(|a| &a[..])
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn weights_load_rejects_out_of_bounds_artifact_index() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
        let bundle = Bundle::open(&dir).expect("tiny-bert-bundle opens");
        let err = match Weights::load(&bundle, bundle.manifest.artifacts.len()) {
            Ok(_) => panic!("out-of-bounds artifact index must fail"),
            Err(e) => e,
        };
        assert_eq!(err.code, INVALID_ARGUMENT);
    }

    #[test]
    fn only_the_cpu_runs_a_banded_model() {
        assert!(runs_band("cpu", 16).is_ok());
        for b in ["cuda", "metal", "levelzero", "npu", "hailo"] {
            assert!(runs_band(b, 0).is_ok());
            let e = runs_band(b, 16).err().unwrap();
            assert_eq!(e.code, UNSUPPORTED);
            assert!(
                e.message.contains(&format!("the {b} backend has no limited-lookahead attention")),
                "{}",
                e.message
            );
        }
    }

    #[test]
    fn only_the_cpu_runs_a_deberta_model() {
        assert!(runs_family("cpu", TURBO_FAMILY_DEBERTA_V2).is_ok());
        for b in ["cuda", "metal", "levelzero", "npu", "hailo"] {
            assert!(runs_family(b, crate::backend::TURBO_FAMILY_BERT).is_ok());
            let e = runs_family(b, TURBO_FAMILY_DEBERTA_V2).err().unwrap();
            assert_eq!(e.code, UNSUPPORTED);
            assert!(e.message.contains(&format!("the {b} backend has no DeBERTa-v2 encoder")), "{}", e.message);
        }
    }
}
