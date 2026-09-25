//! A bundle's model for one device: loader rules 6 to 8 of docs/bundle.md,
//! after Bundle::open and Tokenizer::load have applied 1 to 5. Which
//! artifact the device loads, its files read once and verified, and every
//! tensor a BERT encoder needs checked against the architecture.

use std::ffi::{CString, c_void};

use serde::Serialize;

use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS, TURBO_FAMILY_BERT, format_bit, turbo_backend_model,
    turbo_backend_tensor,
};
use crate::bundle::{AlignedBytes, Bundle, sha256_hex};
use crate::manifest::{
    Activation, Architecture, Artifact, Family, Format, GraphInput, Manifest, PositionEmbedding, TensorRole, role_name,
};
use crate::safetensors;
use crate::status::{BUNDLE_NO_ARTIFACT, Error, Result, invalid};
use crate::{TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32};

/// The manifest's name for an enum value, as it is written there.
fn enum_name(v: impl Serialize) -> String {
    serde_json::to_value(v).ok().and_then(|v| v.as_str().map(str::to_owned)).unwrap_or_default()
}

/// Rule 6: the first artifact, in manifest order, that lists `backend`,
/// was built for `arch` or for any device, is in one of `formats` (the
/// backend's TURBO_FORMAT_BIT set, turbo_backend::formats), and is
/// something this build hands a backend: raw weights, or a HEF. None is
/// BUNDLE_NO_ARTIFACT, saying why each was skipped.
pub fn choose(m: &Manifest, backend: &str, formats: u32, arch: &str) -> Result<usize> {
    let mut skipped = Vec::new();
    for (i, a) in m.artifacts.iter().enumerate() {
        let why = if !a.backends.iter().any(|b| b == backend) {
            format!("backends {:?} has no {backend}", a.backends)
        } else if !a.target.is_empty() && a.target != arch {
            format!("target {} is not this device's {arch}", a.target)
        } else if formats & format_bit(a.format.value()) == 0 {
            format!("{} is not a format the {backend} backend loads", enum_name(a.format))
        } else if !matches!(a.format, Format::Safetensors | Format::Hef) {
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

/// The shape the architecture implies for a BERT tensor: `[out, in]` for a
/// linear layer's weight, `[out]` for its bias and for LayerNorm.
fn bert_shape(role: TensorRole, a: &Architecture) -> Vec<u64> {
    let (h, i) = (a.hidden as u64, a.intermediate as u64);
    use TensorRole::*;
    match role {
        WordEmbeddings => vec![a.vocab_size as u64, h],
        PositionEmbeddings => vec![a.max_positions as u64, h],
        TokenTypeEmbeddings => vec![a.token_types as u64, h],
        QWeight | KWeight | VWeight | AttnOutWeight => vec![h, h],
        FfnInWeight => vec![i, h],
        FfnInBias => vec![i],
        FfnOutWeight => vec![h, i],
        EmbeddingsLnWeight | EmbeddingsLnBias | QBias | KBias | VBias | AttnOutBias | AttnLnWeight | AttnLnBias
        | FfnOutBias | FfnLnWeight | FfnLnBias => vec![h],
    }
}

/// The embedding tensors, at TURBO_BERT_* in turbo_backend.h.
pub const BERT_EMBEDDING_ROLES: [TensorRole; TURBO_BERT_EMBEDDING_TENSORS as usize] = {
    use TensorRole::*;
    [WordEmbeddings, PositionEmbeddings, TokenTypeEmbeddings, EmbeddingsLnWeight, EmbeddingsLnBias]
};

/// Each layer's tensors, at TURBO_BERT_* in turbo_backend.h.
pub const BERT_LAYER_ROLES: [TensorRole; TURBO_BERT_LAYER_TENSORS as usize] = {
    use TensorRole::*;
    [
        QWeight,
        QBias,
        KWeight,
        KBias,
        VWeight,
        VBias,
        AttnOutWeight,
        AttnOutBias,
        AttnLnWeight,
        AttnLnBias,
        FfnInWeight,
        FfnInBias,
        FfnOutWeight,
        FfnOutBias,
        FfnLnWeight,
        FfnLnBias,
    ]
};

/// One tensor a BERT encoder needs: its name in the weights file and the
/// shape the architecture implies.
pub struct Expected {
    pub name: String,
    pub shape: Vec<u64>,
    /// The role and layer, for messages.
    pub what: String,
}

/// Every tensor a BERT encoder needs, in the order turbo_backend.h's
/// TURBO_BERT_* constants give. Every role must be in tensor_names.
pub fn bert_tensors(index: usize, art: &Artifact, a: &Architecture) -> Result<Vec<Expected>> {
    tensors_of(index, art, a, a.layers)
}

/// The embedding tensors alone, for a host lookup: only their roles need
/// be in tensor_names.
pub fn embedding_tensors(index: usize, art: &Artifact, a: &Architecture) -> Result<Vec<Expected>> {
    tensors_of(index, art, a, 0)
}

fn tensors_of(index: usize, art: &Artifact, a: &Architecture, layers: u32) -> Result<Vec<Expected>> {
    let name = |role: TensorRole| {
        art.tensor_names.get(&role).ok_or_else(|| {
            invalid(format!(
                "manifest.json: artifacts[{index}].tensor_names: no {}, which a BERT encoder needs",
                role_name(role)
            ))
        })
    };
    let mut out = Vec::with_capacity((TURBO_BERT_EMBEDDING_TENSORS + layers * TURBO_BERT_LAYER_TENSORS) as usize);
    for role in BERT_EMBEDDING_ROLES {
        out.push(Expected { name: name(role)?.clone(), shape: bert_shape(role, a), what: role_name(role) });
    }
    for layer in 0..layers {
        for role in BERT_LAYER_ROLES {
            out.push(Expected {
                name: name(role)?.replace("{layer}", &layer.to_string()),
                shape: bert_shape(role, a),
                what: format!("{} of layer {layer}", role_name(role)),
            });
        }
    }
    Ok(out)
}

/// The family model_load is told. Every combination the manifest can name
/// is listed, so a new family, activation or position embedding does not
/// compile until it has a place here.
fn family(a: &Architecture) -> u32 {
    match (a.family, a.activation, a.position_embedding) {
        (Family::Bert, Activation::GeluErf, PositionEmbedding::Absolute) => TURBO_FAMILY_BERT,
    }
}

/// A tensor found in one of the artifact's files.
struct Placed {
    name: CString,
    file: usize,
    range: std::ops::Range<usize>,
    shape: Vec<u64>,
}

/// The chosen artifact as the core holds it: the verified bytes of each
/// file its tensors are in, read once, and where each tensor is in them;
/// for a HEF, its file's verified bytes too. These bytes are the only host
/// copy; what model_load is handed points into them.
pub struct Weights {
    /// The files the tensors are in: the raw-weights artifact's, or for a
    /// HEF that starts at embeddings, its host_weights artifact's.
    files: Vec<AlignedBytes>,
    tensors: Vec<Placed>,
    /// TURBO_DTYPE_* every tensor is stored in; 0 when there are none.
    pub dtype: u32,
    family: u32,
    arch: [u32; 7],
    layer_norm_eps: f64,
    /// A compiled artifact's one file, as hashed.
    artifact: Option<AlignedBytes>,
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
        let art = &m.artifacts[index];
        let a = m.architecture.as_ref().expect("validate() requires architecture for raw weights and HEFs");
        let host = (!art.host_weights.is_empty())
            .then(|| m.artifacts.iter().position(|h| h.name == art.host_weights).expect("validate() checks the name"));

        let (artifact, tensor_art) = match art.format {
            Format::Safetensors => (None, Some(index)),
            Format::Hef => {
                let hef = bundle.read_verified_aligned(&art.files[0])?;
                (Some(hef), if art.graph_input == GraphInput::Embeddings { host } else { None })
            }
            f => unreachable!("choose() hands on no {f:?}"),
        };
        let (files, tensors, dtype) = match tensor_art {
            Some(t) => {
                let ta = &m.artifacts[t];
                let expected = if t == index { bert_tensors(t, ta, a)? } else { embedding_tensors(t, ta, a)? };
                let files = ta.files.iter().map(|f| bundle.read_verified_aligned(f)).collect::<Result<Vec<_>>>()?;
                let (tensors, dtype) = place(ta, &files, &expected)?;
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
            tensors,
            dtype,
            family: family(a),
            arch: [a.layers, a.hidden, a.heads, a.intermediate, a.vocab_size, a.max_positions, a.token_types],
            layer_norm_eps: a.layer_norm_eps,
            artifact,
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
            .map(|t| {
                let mut shape = [0u64; 2];
                shape[..t.shape.len()].copy_from_slice(&t.shape);
                turbo_backend_tensor {
                    name: t.name.as_ptr(),
                    data: self.files[t.file][t.range.clone()].as_ptr() as *const c_void,
                    shape,
                    ndim: t.shape.len() as u32,
                    dtype: self.dtype,
                    bytes: t.range.len() as u64,
                }
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
            reserved: 0,
            tensors: if tensors.is_empty() { std::ptr::null() } else { tensors.as_ptr() },
            format,
            graph_input,
            graph_output,
            compute_dtype,
            fixed_seq,
            fixed_batch,
            artifact,
            artifact_bytes,
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

/// Rule 8: find each of `expected` in `files`, the verified bytes of
/// `art`'s files, with its shape, one dtype for all, and an aligned start.
fn place(art: &Artifact, files: &[AlignedBytes], expected: &[Expected]) -> Result<(Vec<Placed>, u32)> {
    let parsed = art
        .files
        .iter()
        .zip(files)
        .map(|(name, bytes)| safetensors::File::parse(name, &bytes[..]))
        .collect::<Result<Vec<_>>>()?;

    let mut dtype = None;
    let mut tensors = Vec::with_capacity(expected.len());
    for e in expected {
        let mut found = parsed.iter().enumerate().filter_map(|(i, f)| f.tensor(&e.name).map(|t| (i, t)));
        let Some((file, t)) = found.next() else {
            return Err(invalid(format!("{}: no tensor {:?} ({})", art.files.join(", "), e.name, e.what)));
        };
        if let Some((other, _)) = found.next() {
            return Err(invalid(format!(
                "{:?} ({}) is in both {} and {}",
                e.name, e.what, art.files[file], art.files[other]
            )));
        }
        let at = &art.files[file];
        let d = match t.dtype {
            safetensors::Dtype::F32 => TURBO_DTYPE_F32,
            safetensors::Dtype::F16 => TURBO_DTYPE_F16,
            safetensors::Dtype::Bf16 => TURBO_DTYPE_BF16,
            _ => {
                return Err(invalid(format!(
                    "{at}: {} ({}) is {}; weights are F32, F16 or BF16",
                    e.name, e.what, t.dtype_name
                )));
            }
        };
        let want = *dtype.get_or_insert(d);
        if d != want {
            return Err(invalid(format!(
                "{at}: {} ({}) is {}; the other weights are {}",
                e.name,
                e.what,
                t.dtype_name,
                dtype_name(want)
            )));
        }
        if t.shape != e.shape {
            return Err(invalid(format!(
                "{at}: {} ({}) has shape {:?}; the architecture implies {:?}",
                e.name, e.what, t.shape, e.shape
            )));
        }
        // The file starts on a 64-byte boundary, so an offset that is a
        // multiple of the element size is an aligned address.
        let begin = t.data.as_ptr() as usize - files[file].as_ptr() as usize;
        let size = t.dtype.size().expect("a weights dtype has a size");
        if !begin.is_multiple_of(size) {
            return Err(invalid(format!(
                "{at}: {} ({}) starts at byte {begin}, not a multiple of its {size}-byte elements",
                e.name, e.what
            )));
        }
        tensors.push(Placed {
            name: CString::new(e.name.as_str()).map_err(|_| invalid(format!("{:?}: a NUL in the name", e.name)))?,
            file,
            range: begin..begin + t.data.len(),
            shape: t.shape.clone(),
        });
    }
    Ok((tensors, dtype.expect("a BERT encoder has tensors")))
}

fn dtype_name(d: u32) -> &'static str {
    match d {
        TURBO_DTYPE_F16 => "F16",
        TURBO_DTYPE_BF16 => "BF16",
        _ => "F32",
    }
}
