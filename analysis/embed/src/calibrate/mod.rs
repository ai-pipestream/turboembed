//! Where the sentence vector step should run, measured once per bundle and
//! device and kept: for each window and rows per run asked, the lowest
//! cosine of the device's vectors to the same weights on the CPU, and the
//! fastest time of the step on each. The device runs a configuration when
//! its vectors clear a cosine floor there and it is faster than the CPU;
//! the CPU runs every other.
//!
//! A compiled graph can lose accuracy as its rows fill (a quantized HEF
//! calibrated on short texts, docs/hailo.md), and a device is not faster
//! than the host for every model, window or batch: a Hailo-8 has no memory
//! of its own and loads its weights from the host for every burst of
//! frames, so a run of a few rows pays for all of them. [`Calibration::measure`]
//! runs the step over the caller's texts at each window and batch in turn,
//! on the device at the precision asked and on the CPU at
//! `Precision::Exact`, which is also the reference the cosine is taken
//! against. The batch is the session's cap on rows a run: the texts are
//! the caller's own, so the runs a cap leaves short, at the end of a call
//! or for a call of few texts, are in what is timed.
//! [`Calibration::load_or_measure`] keeps the result in a directory, under
//! a key made of the bundle's model and manifest hash, the device's
//! backend, name, runtime and driver versions, the precision, the
//! library's build (its version and kernels), and what was asked: the
//! windows, the batches, the floor and a hash of the texts. It is measured
//! again when any of those changes. [`Calibration::options`] gives the
//! options to open the step with for one configuration, and
//! [`Calibration::annotator`] opens it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use turbo::bundle::{Bundle, sha256_hex};
use turbo::status::{Error as CoreError, INVALID_ARGUMENT};
use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Sentences, Vectors};

use crate::encoder::{Device, Encoder, EncoderOptions, Precision};
use crate::{SentenceVectors, SubwordAnnotator, VectorAnnotator, VectorOptions};

mod dispatch;
mod kept;

pub use dispatch::{Cost, Dispatch, shape_of};

/// The cosine floor of an I8 device, the tier's (docs/hailo.md); the core
/// gives I8 none of its own.
pub const I8_FLOOR: f64 = 0.93;

/// The variable that overrides a calibration's choice: `device` or `cpu`.
pub const FORCE_VAR: &str = "TURBO_CALIBRATION_FORCE";

/// A text and its sentences, as byte spans.
pub type Text = (String, Vec<(u32, u32)>);

/// What to measure.
#[derive(Clone, Debug)]
pub struct CalibrationOptions {
    pub device: Device,
    pub precision: Precision,
    /// The windows (tokens per row) to try, each once and at most the
    /// model's.
    pub windows: Vec<u32>,
    /// The rows per run to try at each window, each once and at least 1.
    pub batches: Vec<u32>,
    /// The lowest cosine to accept; None is the core's tolerance for the
    /// device's dtype, or [`I8_FLOOR`] for I8.
    pub floor: Option<f64>,
    /// Timed runs per measurement, of which the fastest is kept: another
    /// process on the machine can only slow a run.
    pub reps: usize,
}

/// One configuration's measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub max_seq: u32,
    pub max_batch: u32,
    pub min_cosine: f64,
    pub device_ms: f64,
    pub cpu_ms: f64,
    /// The fastest times of one sentence sent alone (the first of the
    /// texts), which `Dispatch` fits each side's cost a call to with the
    /// whole set's.
    pub short_device_ms: f64,
    pub short_cpu_ms: f64,
    /// The fastest times of the first text sent alone.
    pub doc_device_ms: f64,
    pub doc_cpu_ms: f64,
}

impl Point {
    /// Whether the device runs this configuration: its vectors clear
    /// `floor` and it is faster than the CPU.
    pub fn device_wins(&self, floor: f64) -> bool {
        self.min_cosine >= floor && self.device_ms < self.cpu_ms
    }
}

/// Which side runs a configuration.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    Device,
    Cpu,
}

/// A measurement: the floor it was decided under, and a point for each
/// window and batch asked, windows outer, in the order asked.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub key: String,
    pub floor: f64,
    pub points: Vec<Point>,
}

impl Calibration {
    /// The key a measurement is kept under (see the module's comment).
    pub fn key(bundle: &Path, o: &CalibrationOptions, texts: &[Text]) -> Result<String, CoreError> {
        Ok(probe(bundle, o, texts)?.0)
    }

    /// Where `load_or_measure` keeps the measurement for `key` in `dir`.
    pub fn file(dir: &Path, key: &str) -> PathBuf {
        dir.join(format!("{}.calibration", &sha256_hex(key.as_bytes())[..32]))
    }

    /// Runs the step over `texts` at each window and batch of `o`, on the
    /// device and on the CPU.
    pub fn measure(bundle: &Path, o: &CalibrationOptions, texts: &[Text]) -> Result<Calibration, CoreError> {
        check_texts(texts)?;
        let (key, model_max_seq) = probe(bundle, o, texts)?;
        once_each("windows", &o.windows, model_max_seq)?;
        once_each("batches", &o.batches, u32::MAX)?;
        let mut floor = o.floor;
        let mut points = Vec::new();
        for &max_seq in &o.windows {
            let open = |device: Device, precision: Precision, max_batch: u32| {
                let enc = EncoderOptions { device, precision, max_seq, max_batch, ..EncoderOptions::default() };
                VectorAnnotator::open(bundle, &VectorOptions { encoder: enc, ..VectorOptions::default() }).map(Arc::new)
            };
            let want = run(&pipeline(bundle, open(Device::Cpu, Precision::Exact, 0)?)?, texts)?;
            let short = first_sentence(texts)?;
            for &max_batch in &o.batches {
                let vectors = open(o.device.clone(), o.precision, max_batch)?;
                floor.get_or_insert_with(|| default_floor(vectors.info().compute_dtype));
                let dim = vectors.dim() as usize;
                let dev = pipeline(bundle, vectors)?;
                let on_cpu = pipeline(bundle, open(Device::Cpu, Precision::Exact, max_batch)?)?;
                let got = run(&dev, texts)?;
                if got.is_empty() || got.len() != want.len() || got.len() % dim != 0 {
                    return Err(CoreError::new(
                        INVALID_ARGUMENT,
                        format!("window {max_seq}: {} device values, {} reference", got.len(), want.len()),
                    ));
                }
                let min_cosine = lowest_cosine(&got, &want, dim);
                let (device_ms, cpu_ms) =
                    fastest(o.reps, || run(&dev, texts).map(drop), || run(&on_cpu, texts).map(drop))?;
                let (short_device_ms, short_cpu_ms) =
                    fastest(4 * o.reps, || run(&dev, &short).map(drop), || run(&on_cpu, &short).map(drop))?;
                let doc = &texts[..1];
                let (doc_device_ms, doc_cpu_ms) =
                    fastest(2 * o.reps, || run(&dev, doc).map(drop), || run(&on_cpu, doc).map(drop))?;
                points.push(Point {
                    max_seq,
                    max_batch,
                    min_cosine,
                    device_ms,
                    cpu_ms,
                    short_device_ms,
                    short_cpu_ms,
                    doc_device_ms,
                    doc_cpu_ms,
                });
            }
        }
        Ok(Calibration { key, floor: floor.unwrap_or(I8_FLOOR), points })
    }

    /// The calibration kept in `dir` for this bundle, device and request,
    /// measured and kept first when there is none, or when the kept one
    /// lacks a point for a window and batch asked, in order.
    pub fn load_or_measure(
        dir: &Path,
        bundle: &Path,
        o: &CalibrationOptions,
        texts: &[Text],
    ) -> Result<Calibration, CoreError> {
        let key = Calibration::key(bundle, o, texts)?;
        let file = Calibration::file(dir, &key);
        let asked = o.windows.iter().flat_map(|&w| o.batches.iter().map(move |&b| (w, b)));
        if let Ok(text) = std::fs::read_to_string(&file)
            && let Some(c) = Calibration::parse(&text)
            && c.key == key
            && c.points.iter().map(|p| (p.max_seq, p.max_batch)).eq(asked)
        {
            return Ok(c);
        }
        let c = Calibration::measure(bundle, o, texts)?;
        let io = |e: std::io::Error| CoreError::new(INVALID_ARGUMENT, format!("{}: {e}", file.display()));
        std::fs::create_dir_all(dir).map_err(io)?;
        let part = file.with_extension("part");
        std::fs::write(&part, c.to_text()).map_err(io)?;
        std::fs::rename(&part, &file).map_err(io)?;
        Ok(c)
    }

    /// The side that runs `max_seq` tokens a row and at most `max_batch`
    /// rows a run: the device where its point says it wins, else the CPU;
    /// `force` (the [`FORCE_VAR`] variable's value) overrides it. An error
    /// when no point was measured for the configuration, or `force` is
    /// neither `device` nor `cpu`.
    pub fn side(&self, max_seq: u32, max_batch: u32, force: Option<&str>) -> Result<Side, CoreError> {
        let forced = forced(force)?;
        let Some(p) = self.points.iter().find(|p| p.max_seq == max_seq && p.max_batch == max_batch) else {
            let measured: Vec<(u32, u32)> = self.points.iter().map(|p| (p.max_seq, p.max_batch)).collect();
            return Err(unmeasured(max_seq, max_batch, &measured));
        };
        Ok(forced.unwrap_or(if p.device_wins(self.floor) { Side::Device } else { Side::Cpu }))
    }

    /// The options to open the step with at `base`'s window and rows per
    /// run, which must both be given (not 0): `base` on the device at the
    /// precision of `o`, or on the CPU at `Precision::Exact`, the
    /// precision the CPU was timed at, as [`Calibration::side`] chooses
    /// with the [`FORCE_VAR`] variable.
    pub fn options(&self, o: &CalibrationOptions, base: &VectorOptions) -> Result<VectorOptions, CoreError> {
        let force = std::env::var(FORCE_VAR).ok();
        let mut v = base.clone();
        match self.side(base.encoder.max_seq, base.encoder.max_batch, force.as_deref())? {
            Side::Device => {
                v.encoder.device = o.device.clone();
                v.encoder.precision = o.precision;
            }
            Side::Cpu => {
                v.encoder.device = Device::Cpu;
                v.encoder.precision = Precision::Exact;
            }
        }
        Ok(v)
    }

    /// The step at `base`'s window and rows per run, on the side the
    /// calibration kept in `dir` chooses, measured first when there is
    /// none (`load_or_measure`). A window and batch that `o` does not ask
    /// for, or a [`FORCE_VAR`] that names neither side, is an error before
    /// anything is measured.
    pub fn annotator(
        dir: &Path,
        bundle: &Path,
        o: &CalibrationOptions,
        texts: &[Text],
        base: &VectorOptions,
    ) -> Result<VectorAnnotator, CoreError> {
        let (max_seq, max_batch) = (base.encoder.max_seq, base.encoder.max_batch);
        forced(std::env::var(FORCE_VAR).ok().as_deref())?;
        if !o.windows.contains(&max_seq) || !o.batches.contains(&max_batch) {
            let asked: Vec<(u32, u32)> =
                o.windows.iter().flat_map(|&w| o.batches.iter().map(move |&b| (w, b))).collect();
            return Err(unmeasured(max_seq, max_batch, &asked));
        }
        let c = Calibration::load_or_measure(dir, bundle, o, texts)?;
        VectorAnnotator::open(bundle, &c.options(o, base)?)
    }
}

/// The side `force` names, None when it is unset; an error for any
/// value but `device` and `cpu`.
fn forced(force: Option<&str>) -> Result<Option<Side>, CoreError> {
    match force {
        None => Ok(None),
        Some("device") => Ok(Some(Side::Device)),
        Some("cpu") => Ok(Some(Side::Cpu)),
        Some(v) => Err(CoreError::new(INVALID_ARGUMENT, format!("{FORCE_VAR} is {v:?}, not device or cpu"))),
    }
}

/// The error for a configuration that is not among `measured`.
fn unmeasured(max_seq: u32, max_batch: u32, measured: &[(u32, u32)]) -> CoreError {
    let list: Vec<String> = measured.iter().map(|(w, b)| format!("{w}x{b}")).collect();
    CoreError::new(
        INVALID_ARGUMENT,
        format!(
            "no calibration point for {max_seq} tokens a row and {max_batch} rows a run; measured: {}",
            list.join(", ")
        ),
    )
}

/// The median time in ms of `reps` runs of the sentence vector step on
/// `vectors` over `texts`, after one untimed run: the measure a chosen
/// configuration is checked with.
pub fn time(bundle: &Path, vectors: &Arc<VectorAnnotator>, texts: &[Text], reps: usize) -> Result<f64, CoreError> {
    check_texts(texts)?;
    let a = pipeline(bundle, vectors.clone())?;
    Ok(times(reps, || run(&a, texts).map(drop), || Ok(()))?.0[reps / 2])
}

/// The sentence vectors the step on `vectors` gives `texts`, in order.
pub fn vectors(bundle: &Path, vectors: &Arc<VectorAnnotator>, texts: &[Text]) -> Result<Vec<f32>, CoreError> {
    check_texts(texts)?;
    run(&pipeline(bundle, vectors.clone())?, texts)
}

/// The lowest cosine between the vectors of `got` and `want`, `dim` values
/// each; a cosine that is not a number counts as the lowest there is.
pub fn lowest_cosine(got: &[f32], want: &[f32], dim: usize) -> f64 {
    got.chunks_exact(dim)
        .zip(want.chunks_exact(dim))
        .map(|(g, w)| cosine(g, w))
        .fold(1f64, |m, c| if c.is_nan() { f64::NEG_INFINITY } else { m.min(c) })
}

/// The first sentence of the first text that has one, as a text of its
/// own.
fn first_sentence(texts: &[Text]) -> Result<Vec<Text>, CoreError> {
    let Some((text, &(a, b))) = texts.iter().find_map(|(t, s)| s.first().map(|f| (t, f))) else {
        return Err(CoreError::new(INVALID_ARGUMENT, "calibration needs a text with a sentence"));
    };
    Ok(vec![(text[a as usize..b as usize].to_owned(), vec![(0, b - a)])])
}

/// Fails unless `values` is not empty and holds each value once, each
/// from 1 to `max`.
fn once_each(what: &str, values: &[u32], max: u32) -> Result<(), CoreError> {
    let mut seen = values.to_vec();
    seen.sort_unstable();
    seen.dedup();
    if seen.is_empty() || seen.len() != values.len() || values.iter().any(|&v| v == 0 || v > max) {
        let most = if max == u32::MAX { String::new() } else { format!(" to the model's {max}") };
        return Err(CoreError::new(
            INVALID_ARGUMENT,
            format!("calibration {what} {values:?}: each once, from 1{most}"),
        ));
    }
    Ok(())
}

/// Fails unless each sentence of each text is a span of it: start at most
/// end, end at most the text's length, both on UTF-8 boundaries.
fn check_texts(texts: &[Text]) -> Result<(), CoreError> {
    for (i, (text, spans)) in texts.iter().enumerate() {
        for (j, &(a, b)) in spans.iter().enumerate() {
            let (a, b) = (a as usize, b as usize);
            if a > b || b > text.len() || !text.is_char_boundary(a) || !text.is_char_boundary(b) {
                return Err(CoreError::new(
                    INVALID_ARGUMENT,
                    format!("text {i}, sentence {j}: {a}..{b} is not a span of the text's {} bytes", text.len()),
                ));
            }
        }
    }
    Ok(())
}

fn device_options(o: &CalibrationOptions, max_seq: u32, max_batch: u32) -> EncoderOptions {
    EncoderOptions { device: o.device.clone(), precision: o.precision, max_seq, max_batch, ..EncoderOptions::default() }
}

/// The key, and the model's own window.
fn probe(bundle: &Path, o: &CalibrationOptions, texts: &[Text]) -> Result<(String, u32), CoreError> {
    if o.floor.is_some_and(|f| !f.is_finite()) {
        return Err(CoreError::new(INVALID_ARGUMENT, "calibration floor must be finite"));
    }
    let model = crate::model_ref(&Bundle::open(bundle)?);
    let enc = Encoder::open(bundle, &device_options(o, 0, 1))?;
    let d = enc.device_info();
    let f = crate::encoder::field;
    let mut t = String::new();
    for (text, spans) in texts {
        t.push_str(text);
        t.push('\u{1f}');
        t.push_str(&format!("{spans:?}"));
        t.push('\u{1e}');
    }
    let list = |v: &[u32]| v.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    let key = [
        model.id,
        model.sha256,
        f(&d.backend),
        f(&d.name),
        f(&d.runtime_version),
        f(&d.driver_version),
        o.precision.name().to_owned(),
        turbo::tuning::LIBRARY_BUILD.to_owned(),
        list(&o.windows),
        list(&o.batches),
        o.floor.map_or("none".to_owned(), |f| f.to_string()),
        sha256_hex(t.as_bytes()),
    ]
    .join(" | ");
    Ok((key, enc.info().max_seq))
}

/// The core's tolerance for `dtype`, or [`I8_FLOOR`] for a dtype it gives
/// none (I8).
fn default_floor(dtype: u32) -> f64 {
    turbo::record::tolerance(dtype).map_or(I8_FLOOR, |t| t.min_cosine)
}

/// The bundle's subwords, then the sentence vector step on `vectors`.
fn pipeline(bundle: &Path, vectors: Arc<VectorAnnotator>) -> Result<Analyzer, CoreError> {
    Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SubwordAnnotator::open(bundle)?)
        .add(SentenceVectors(vectors))
        .build()
        .map_err(|e| CoreError::new(INVALID_ARGUMENT, e.to_string()))
}

/// A document for each text, with its sentences.
fn documents(texts: &[Text]) -> Result<Vec<Document>, CoreError> {
    texts
        .iter()
        .map(|(text, spans)| {
            let mut d = Document::new(text.as_str());
            let mut s = Spans::new(ViewId::SOURCE);
            for &(a, b) in spans {
                s.push(Span::new(a, b));
            }
            d.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("calibration")).map(|_| d)
        })
        .collect::<Result<_, _>>()
        .map_err(|e| CoreError::new(INVALID_ARGUMENT, e.to_string()))
}

/// The sentence vectors of every text, in order; an error when a text
/// with sentences gets none.
fn run(a: &Analyzer, texts: &[Text]) -> Result<Vec<f32>, CoreError> {
    let err = |e: turbo_analysis::Error| CoreError::new(INVALID_ARGUMENT, e.to_string());
    let mut docs = documents(texts)?;
    for r in a.run(&mut docs) {
        r.map_err(err)?;
    }
    let mut out = Vec::new();
    for (i, (d, (_, spans))) in docs.iter().zip(texts).enumerate() {
        match d.get::<Vectors<Sentences>>() {
            Some(v) => out.extend_from_slice(&v.data),
            None if spans.is_empty() => {}
            None => return Err(CoreError::new(INVALID_ARGUMENT, format!("text {i}: no sentence vectors"))),
        }
    }
    Ok(out)
}

fn cosine(g: &[f32], w: &[f32]) -> f64 {
    let dot = turbo_analysis::math::dot_f64(g, w);
    let n = |v: &[f32]| turbo_analysis::math::sum_sq_f64(v).sqrt();
    let d = n(g) * n(w);
    if d == 0.0 { if n(g) == n(w) { 1.0 } else { 0.0 } } else { dot / d }
}

/// The fastest times of `reps` runs each of `a` and `b`, in ms: what
/// another process on the machine cannot add to, since it only ever slows
/// a run.
fn fastest(
    reps: usize,
    a: impl FnMut() -> Result<(), CoreError>,
    b: impl FnMut() -> Result<(), CoreError>,
) -> Result<(f64, f64), CoreError> {
    let (ta, tb) = times(reps, a, b)?;
    Ok((ta[0], tb[0]))
}

/// The times of `reps` runs each of `a` and `b`, in ms, sorted, the two
/// taking turns after one untimed run each, so drift falls on both.
fn times(
    reps: usize,
    mut a: impl FnMut() -> Result<(), CoreError>,
    mut b: impl FnMut() -> Result<(), CoreError>,
) -> Result<(Vec<f64>, Vec<f64>), CoreError> {
    if reps == 0 {
        return Err(CoreError::new(INVALID_ARGUMENT, "calibration needs at least one timed run"));
    }
    a()?;
    b()?;
    let (mut ta, mut tb) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
    let time = |f: &mut dyn FnMut() -> Result<(), CoreError>| -> Result<f64, CoreError> {
        let s = Instant::now();
        f()?;
        Ok(s.elapsed().as_secs_f64() * 1e3)
    };
    for i in 0..reps {
        if i % 2 == 0 {
            ta.push(time(&mut a)?);
            tb.push(time(&mut b)?);
        } else {
            tb.push(time(&mut b)?);
            ta.push(time(&mut a)?);
        }
    }
    ta.sort_by(f64::total_cmp);
    tb.sort_by(f64::total_cmp);
    Ok((ta, tb))
}
