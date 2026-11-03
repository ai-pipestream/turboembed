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
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use turbo::manifest::{LayerMix, WordPooling};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;

use crate::cli::{Args, OutDir, commit, encoder_args};
use crate::corpus::{Corpus, CorpusRecord, Sentence, Splits};
use crate::receipt::{EncoderRecord, host, unix_time};
use crate::run::{PREDICT_CHUNK, write as write_file};
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

fn ud_word_bank(id: &str) -> Option<&'static str> {
    for bank in &crate::ud::TREEBANKS {
        if id == bank.id {
            return Some(bank.id);
        }
        if let Some(suffix) = id.strip_prefix(bank.id).and_then(|s| s.strip_prefix('-'))
            && matches!(suffix, "upos" | "xpos" | "feats" | "lemma" | "deps" | "parse")
        {
            return Some(bank.id);
        }
    }
    None
}

/// Whether two corpus records describe the same release and word tokenization:
/// either identical records, or two syntactic-word tasks (`upos`, `xpos`,
/// `feats`, `lemma`, `deps`) over the same UD treebank release.
pub fn same_corpus_release(a: &CorpusRecord, b: &CorpusRecord) -> bool {
    if a == b {
        return true;
    }
    let same_ud = matches!((ud_word_bank(&a.id), ud_word_bank(&b.id)), (Some(x), Some(y)) if x == y);
    same_ud
        && a.files == b.files
        && a.recorded == b.recorded
        && a.train == b.train
        && a.dev == b.dev
        && a.test == b.test
}

/// Reads a corpus for `cache`: accepts any standard corpus id, a bare UD
/// treebank id (`ud-en-ewt`), or a token-split id (`ud-en-ewt-split` or
/// `--task split`).
pub fn read_cache_corpus(id: &str, task: Option<&str>, dir: &Path, unrecorded: bool) -> Result<Corpus> {
    if task == Some("split") || id.ends_with("-split") {
        let bank_id = id.strip_suffix("-split").unwrap_or(id);
        let bank = crate::ud::TREEBANKS
            .iter()
            .find(|b| b.id == bank_id)
            .ok_or_else(|| Error(format!("unknown treebank {bank_id:?}")))?;
        let (splits, files, recorded) = crate::ud_split::read_tokens(bank, dir, unrecorded)?;
        let record = crate::ud_split::record(bank, &splits, files, recorded);
        return Ok(Corpus {
            record,
            splits: Splits {
                train: crate::ud_split::sentences(&splits.train),
                dev: crate::ud_split::sentences(&splits.dev),
                test: crate::ud_split::sentences(&splits.test),
            },
        });
    }
    if let Some(bank) = crate::ud::TREEBANKS.iter().find(|b| b.id == id) {
        return crate::ud::read(bank, crate::ud::Column::Upos, dir, unrecorded);
    }
    crate::read_corpus(id, dir, unrecorded)
}

/// The `layer_mix` field a head carries when trained on `layers` from a
/// model with `model_layers` total layers: `None` on the final layer alone,
/// `Some(LayerMix { layers: L - k + 1, layer: Some(k) })` on an earlier
/// single layer `k`, or `Some(LayerMix { layers: K, layer: None })` on a mix.
pub fn layer_mix_for(layers: &[u32], mixed: bool, model_layers: Option<u32>) -> Option<LayerMix> {
    if mixed {
        Some(LayerMix { layers: layers.len() as u32, layer: None })
    } else if let ([k], Some(total)) = (layers, model_layers)
        && *k < total
    {
        Some(LayerMix { layers: total - *k + 1, layer: Some(*k) })
    } else {
        None
    }
}

/// Stacks per-layer `[words, hidden]` features into `[layers, words, hidden]`
/// per sentence, as the core's `WordReader` reads multi-tap features.
pub fn stack_layers(per_layer: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
    let Some(first) = per_layer.first() else { return Vec::new() };
    (0..first.len()).map(|i| per_layer.iter().flat_map(|l| l[i].iter().copied()).collect()).collect()
}

/// `turbo-train-head cache` command implementation.
pub fn command(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let bundle = PathBuf::from(a.required("--bundle")?);
    let mut out = OutDir::claim(&a.required("--out")?)?;
    let (device, precision, pooling) = encoder_args(&mut a)?;
    let taps: u32 = a.parsed("--layers", 0)?;
    let max_batch: u32 = a.parsed("--max-batch", 32)?;
    let wanted = a.value("--splits")?.unwrap_or_else(|| "train,dev,test".into());
    let task = a.value("--task")?;
    let commit = commit(a.value("--commit")?);
    a.done()?;
    let wanted: Vec<&str> = wanted.split(',').collect();
    if wanted.is_empty() || wanted.iter().any(|w| !["train", "dev", "test"].contains(w)) {
        return Err(Error(format!("--splits {wanted:?}: train, dev and test, comma separated")));
    }
    let corpus = read_cache_corpus(corpus_id, task.as_deref(), dir, unrecorded)?;
    let o = EncoderOptions { device, precision, max_batch: max_batch.max(taps), ..EncoderOptions::default() };
    let mut enc = WordEncoder::open(&bundle, &o).map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let model_layers = encoder.layers.ok_or_else(|| Error(format!("{}: no architecture.layers", bundle.display())))?;
    let t = Instant::now();
    let s = &corpus.splits;
    let splits: Vec<(&str, &[_])> = [("train", &s.train[..]), ("dev", &s.dev[..]), ("test", &s.test[..])]
        .into_iter()
        .filter(|(n, _)| wanted.contains(n))
        .collect();
    let (layers, splits) = write(&mut enc, &splits, pooling, taps, model_layers, &out.path)?;
    let record = CacheRecord {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        host: host(),
        unix_time: unix_time(),
        corpus: corpus.record.clone(),
        encoder,
        pooling,
        layers,
        splits,
        features_s: t.elapsed().as_secs_f64(),
    };
    write_file(&out.path, "cache.json", serde_json::to_string_pretty(&record).expect("serializes").as_bytes())?;
    out.keep = true;
    let words: u64 = record.splits.iter().map(|s| s.words).sum();
    println!(
        "{}: layers {:?} of {words} words in {:.1} s, {:.0} words/s",
        out.path.display(),
        record.layers,
        record.features_s,
        words as f64 / record.features_s.max(1e-9)
    );
    Ok(())
}

pub fn encoder_record(bundle: &Path, enc: &WordEncoder) -> Result<EncoderRecord> {
    let manifest =
        std::fs::read(bundle.join("manifest.json")).map_err(|e| Error(format!("{}: {e}", bundle.display())))?;
    let m: turbo::manifest::Manifest =
        serde_json::from_slice(&manifest).map_err(|e| Error(format!("{}: {e}", bundle.display())))?;
    let x = enc.execution();
    Ok(EncoderRecord {
        model: m.model.id,
        revision: m.model.revision,
        manifest_sha256: turbo::bundle::sha256_hex(&manifest),
        hidden: enc.hidden(),
        layers: m.architecture.as_ref().map(|a| a.layers),
        width: enc.width(),
        backend: x.backend.clone(),
        device: x.device.clone(),
        precision: x.precision.clone(),
    })
}
