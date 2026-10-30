//! A cache of a corpus's word features at an encoder's last layers
//! (#1616): one read of the corpus, after which heads on any one layer or
//! on a mix of them train without the encoder.
//!
//! A cache directory holds `cache.json` (a [`CacheRecord`]) and, for each
//! split and layer, `<split>.layer<NN>.f32`: the features of every word of
//! the split's sentences in corpus order, [words, hidden] little-endian
//! f32, `NN` the layer counted from 1 as the model counts them. The files
//! are an intermediate a run makes and reads; each is pinned by its size,
//! which the word count fixes, not by a digest.

use std::io::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo::manifest::WordPooling;
use turbo_tagger::WordEncoder;

use crate::corpus::{CorpusRecord, Sentence};
use crate::receipt::EncoderRecord;
use crate::run::PREDICT_CHUNK;
use crate::{Error, Result};

pub const SCHEMA: &str = "turboembed.layer-cache.v1";

/// What a cache holds and how it was made.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheRecord {
    pub schema: String,
    pub command: Vec<String>,
    pub commit: String,
    pub host: String,
    pub unix_time: u64,
    pub corpus: CorpusRecord,
    pub encoder: EncoderRecord,
    pub pooling: WordPooling,
    /// The layers cached, counted from 1, the last the model's last.
    pub layers: Vec<u32>,
    pub splits: Vec<SplitRecord>,
    /// Seconds spent reading features.
    pub features_s: f64,
}

/// One split's sentences and words, and its file per layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitRecord {
    pub name: String,
    pub sentences: u64,
    pub words: u64,
    /// `files[k]` holds `layers[k]`.
    pub files: Vec<String>,
    pub bytes_per_file: u64,
}

/// The file of `split` at `layer`.
pub fn layer_file(split: &str, layer: u32) -> String {
    format!("{split}.layer{layer:02}.f32")
}

/// Writes the features of each split's sentences at the model's last
/// `taps` layers (of `model_layers`) into `dir`, which exists and is
/// empty, and returns the layers and a record of each split.
///
/// # Errors
///
/// What the encoder fails on, `taps` outside 1..=`model_layers`, and a
/// file that cannot be written.
pub fn write(
    enc: &mut WordEncoder,
    splits: &[(&str, &[Sentence])],
    pooling: WordPooling,
    taps: u32,
    model_layers: u32,
    dir: &Path,
) -> Result<(Vec<u32>, Vec<SplitRecord>)> {
    if taps == 0 || taps > model_layers {
        return Err(Error(format!("--layers {taps}: 1 to the model's {model_layers}")));
    }
    let layers: Vec<u32> = (model_layers - taps + 1..=model_layers).collect();
    let mut records = Vec::with_capacity(splits.len());
    for (name, sentences) in splits {
        let files: Vec<String> = layers.iter().map(|&l| layer_file(name, l)).collect();
        let mut out = files
            .iter()
            .map(|f| {
                let p = dir.join(f);
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&p)
                    .map(std::io::BufWriter::new)
                    .map_err(|e| Error(format!("{}: {e}", p.display())))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut words = 0u64;
        for chunk in sentences.chunks(PREDICT_CHUNK) {
            let seqs: Vec<Vec<&str>> = chunk.iter().map(|s| s.words.iter().map(String::as_str).collect()).collect();
            let feats = enc.layer_features(&seqs, pooling, taps).map_err(|e| Error(e.message))?;
            for (f, w) in feats.iter().zip(&mut out) {
                for s in f {
                    let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
                    w.write_all(&bytes).map_err(|e| Error(format!("{name}: {e}")))?;
                }
            }
            words += seqs.iter().map(|s| s.len() as u64).sum::<u64>();
        }
        for (w, f) in out.into_iter().zip(&files) {
            w.into_inner()
                .map_err(|e| Error(format!("{f}: {e}")))?
                .sync_all()
                .map_err(|e| Error(format!("{f}: {e}")))?;
        }
        let hidden = enc.hidden() as u64;
        records.push(SplitRecord {
            name: (*name).to_owned(),
            sentences: sentences.len() as u64,
            words,
            files,
            bytes_per_file: words * hidden * 4,
        });
    }
    Ok((layers, records))
}

/// Each sentence's features [words, hidden] at `layer` from the cache in
/// `dir`, the sentences those `split` was cached from.
///
/// # Errors
///
/// A layer or split the cache does not hold, sentences other than the
/// cached ones (by count), and a file of another size.
pub fn read_layer(
    dir: &Path,
    record: &CacheRecord,
    split: &str,
    layer: u32,
    sentences: &[Sentence],
) -> Result<Vec<Vec<f32>>> {
    let s = record
        .splits
        .iter()
        .find(|s| s.name == split)
        .ok_or_else(|| Error(format!("{}: no split {split:?}", dir.display())))?;
    let k = record
        .layers
        .iter()
        .position(|&l| l == layer)
        .ok_or_else(|| Error(format!("{}: no layer {layer}; it holds {:?}", dir.display(), record.layers)))?;
    let words: u64 = sentences.iter().map(|s| s.words.len() as u64).sum();
    if (sentences.len() as u64, words) != (s.sentences, s.words) {
        return Err(Error(format!(
            "{split}: {} sentences of {words} words, the cache holds {} of {}",
            sentences.len(),
            s.sentences,
            s.words
        )));
    }
    let p = dir.join(&s.files[k]);
    let bytes = std::fs::read(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    if bytes.len() as u64 != s.bytes_per_file {
        return Err(Error(format!("{}: {} bytes, expected {}", p.display(), bytes.len(), s.bytes_per_file)));
    }
    let hidden = record.encoder.hidden;
    let values: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    let mut at = 0;
    Ok(sentences
        .iter()
        .map(|s| {
            let n = s.words.len() * hidden;
            let v = values[at..at + n].to_vec();
            at += n;
            v
        })
        .collect())
}
