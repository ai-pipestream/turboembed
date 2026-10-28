//! A head run end to end: the head's spec for a corpus, encoder features
//! for its sentences, training, prediction through the core's inference
//! path, and the outputs a run writes (`head.safetensors`, `head.json`,
//! `receipt.json`).

use std::collections::BTreeMap;
use std::path::Path;

use turbo::head::TokenTagger;
use turbo::head::tagger::{Work, tensors};
use turbo::manifest::{Adapter, Bilstm, Decode, HashEmbed, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_tagger::WordEncoder;
use turbo_trainer::Example;

use crate::corpus::{Corpus, FileRecord, Sentence, Task};
use crate::{Error, Result};

/// Sentences read by the encoder at a time while predicting, so a large
/// test split never holds all its features at once.
pub const PREDICT_CHUNK: usize = 512;

/// The shape of the head to train.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadOptions {
    pub pooling: WordPooling,
    pub crf: bool,
    /// BiLSTM adapter (hidden, layers), or none.
    pub bilstm: Option<(u32, u32)>,
    /// Hashed word-identity embeddings, or none.
    pub hash: Option<HashEmbed>,
}

/// spaCy's tok2vec rows for NORM, PREFIX, SUFFIX and SHAPE.
pub const HASH_ROWS: [u32; 4] = [5000, 1000, 2500, 2500];

impl Default for HeadOptions {
    fn default() -> Self {
        HeadOptions { pooling: WordPooling::First, crf: false, bilstm: None, hash: None }
    }
}

/// The head a corpus trains: `pos` on `turboembed:pos` for tags,
/// `ner` on `turboembed:entities` for BIO, over the corpus's labels.
///
/// # Errors
///
/// A head the core would refuse.
pub fn spec(corpus: &Corpus, o: &HeadOptions) -> Result<HeadSpec> {
    let (name, layer, scheme) = match corpus.record.task {
        Task::Tags => ("pos", "turboembed:pos", Scheme::Tags),
        Task::Bio => ("ner", "turboembed:entities", Scheme::Bio),
    };
    let spec = HeadSpec {
        name: name.into(),
        layer: layer.into(),
        kind: HeadKind::TokenTagger,
        labels: corpus.labels(),
        scheme,
        word_pooling: o.pooling,
        adapter: o.bilstm.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode: if o.crf { Decode::Crf } else { Decode::Argmax },
        biaffine: None,
        pos_input: None,
        hash_embed: o.hash,
        license_class: None,
        artifact: "weights-f32".into(),
        tensor_prefix: format!("heads.{name}"),
    };
    spec.validate("head").map_err(|e| Error(e.message))?;
    Ok(spec)
}

/// Training examples of `sentences`: each sentence's word features,
/// label indices and words. A sentence with a label `labels` lacks is left out, and
/// counted.
///
/// # Errors
///
/// What the encoder fails on.
pub fn examples(
    enc: &mut WordEncoder,
    sentences: &[Sentence],
    labels: &[String],
    pooling: WordPooling,
) -> Result<(Vec<Example>, usize)> {
    let index: BTreeMap<&str, u32> = labels.iter().enumerate().map(|(i, l)| (l.as_str(), i as u32)).collect();
    let mut out = Vec::with_capacity(sentences.len());
    let mut left_out = 0;
    for chunk in sentences.chunks(PREDICT_CHUNK) {
        let kept: Vec<(&Sentence, Vec<u32>)> = chunk
            .iter()
            .filter_map(|s| {
                s.labels
                    .iter()
                    .map(|l| index.get(l.as_str()).copied())
                    .collect::<Option<Vec<u32>>>()
                    .map(|ids| (s, ids))
            })
            .collect();
        left_out += chunk.len() - kept.len();
        let words: Vec<Vec<&str>> = kept.iter().map(|(s, _)| s.words.iter().map(String::as_str).collect()).collect();
        let features = enc.features(&words, pooling).map_err(|e| Error(e.message))?;
        out.extend(features.into_iter().zip(kept).map(|(features, (s, labels))| Example {
            features,
            labels,
            forms: s.words.clone(),
        }));
    }
    Ok((out, left_out))
}

/// The labels `tagger` gives each sentence, read and decoded exactly as
/// an annotator reads and decodes them.
///
/// # Errors
///
/// What the encoder or the head fails on.
pub fn predict(enc: &mut WordEncoder, tagger: &TokenTagger, sentences: &[Sentence]) -> Result<Vec<Vec<String>>> {
    let pooling = tagger.spec().word_pooling;
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), Work::default());
    let mut out = Vec::with_capacity(sentences.len());
    for chunk in sentences.chunks(PREDICT_CHUNK) {
        let words: Vec<Vec<&str>> = chunk.iter().map(|s| s.words.iter().map(String::as_str).collect()).collect();
        for (f, forms) in enc.features(&words, pooling).map_err(|e| Error(e.message))?.iter().zip(&words) {
            let ids: Vec<u32> = (0..forms.len() as u32).collect();
            tagger.scores_forms(f, &ids, Some(forms), &mut scores, &mut work).map_err(|e| Error(e.message))?;
            tagger.decode(&scores, forms.len(), &mut tags).map_err(|e| Error(e.message))?;
            out.push(tags.iter().map(|&t| tagger.labels()[t as usize].clone()).collect());
        }
    }
    Ok(out)
}

/// The head's tensors under their bundle names, with their shapes, as
/// `crate::safetensors::write` takes them.
pub fn named(spec: &HeadSpec, hidden: usize, values: Vec<(String, Vec<f32>)>) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let shapes: BTreeMap<String, Vec<u64>> = tensors(spec, hidden).into_iter().collect();
    values
        .into_iter()
        .map(|(suffix, v)| {
            let shape = shapes.get(&suffix).cloned().unwrap_or_else(|| vec![v.len() as u64]);
            (format!("{}.{suffix}", spec.tensor_prefix), shape, v)
        })
        .collect()
}

/// Writes `bytes` to `dir/name` and returns its record.
///
/// # Errors
///
/// A file that cannot be written.
pub fn write(dir: &Path, name: &str, bytes: &[u8]) -> Result<FileRecord> {
    let path = dir.join(name);
    std::fs::write(&path, bytes).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    Ok(FileRecord { path: name.into(), size: bytes.len() as u64, sha256: turbo::bundle::sha256_hex(bytes) })
}
