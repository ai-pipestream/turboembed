//! How the sentence vector step should run on a device, measured once per
//! bundle and device and kept: the widest window whose vectors stay within
//! a cosine floor of the same weights on the CPU, and whether the device
//! is faster than the CPU at that window.
//!
//! A compiled graph can lose accuracy as its rows fill (a quantized HEF
//! calibrated on short texts, docs/hailo.md), and a device is not faster
//! than the host for every model. [`Calibration::measure`] runs the step
//! over the caller's texts at each window in turn, on the device and as
//! the reference on the CPU at `Precision::Exact`, and keeps the lowest
//! cosine and the median times. [`Calibration::load_or_measure`] keeps the
//! result in a directory, under a key made of the bundle's model and
//! manifest hash, the device's backend, name, runtime and driver versions,
//! the precision, the library's build (its version and kernels), and what
//! was asked: the windows, the floor and a hash of the texts. It is measured
//! again when any of those changes. [`Calibration::options`] gives the
//! options to open the step with.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use turbo::bundle::{Bundle, sha256_hex};
use turbo::status::{Error as CoreError, INVALID_ARGUMENT};
use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Sentences, Vectors};

use crate::encoder::{Device, Encoder, EncoderOptions, Precision};
use crate::{SentenceVectors, SubwordAnnotator, VectorAnnotator, VectorOptions};

/// The cosine floor of an I8 device, the tier's (docs/hailo.md); the core
/// gives I8 none of its own.
pub const I8_FLOOR: f64 = 0.93;

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
    /// The lowest cosine to accept; None is the core's tolerance for the
    /// device's dtype, or [`I8_FLOOR`] for I8.
    pub floor: Option<f64>,
    /// Timed runs per measurement, of which the median is kept.
    pub reps: usize,
}

/// One window's measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub max_seq: u32,
    pub min_cosine: f64,
    pub device_ms: f64,
    pub cpu_ms: f64,
}

/// A measured decision.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub key: String,
    pub floor: f64,
    /// The widest window that clears the floor; 0 when none does.
    pub max_seq: u32,
    /// Whether the device runs the step: a window clears the floor and the
    /// device is faster than the CPU there.
    pub use_device: bool,
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

    /// Runs the step over `texts` at each window of `o`, on the device and
    /// on the CPU, and decides.
    pub fn measure(bundle: &Path, o: &CalibrationOptions, texts: &[Text]) -> Result<Calibration, CoreError> {
        check_texts(texts)?;
        let (key, model_max_seq) = probe(bundle, o, texts)?;
        let mut seen = o.windows.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.is_empty() || seen.len() != o.windows.len() || o.windows.iter().any(|&w| w == 0 || w > model_max_seq) {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("calibration windows {:?}: each once, from 1 to the model's {model_max_seq}", o.windows),
            ));
        }
        let mut floor = o.floor;
        let mut points = Vec::new();
        for &max_seq in &o.windows {
            let open = |device: Device, precision: Precision| {
                let enc = EncoderOptions { device, precision, max_seq, ..EncoderOptions::default() };
                VectorAnnotator::open(bundle, &VectorOptions { encoder: enc, ..VectorOptions::default() }).map(Arc::new)
            };
            let dev = open(o.device.clone(), o.precision)?;
            floor.get_or_insert_with(|| default_floor(dev.info().compute_dtype));
            let dim = dev.dim() as usize;
            let (on_dev, reference, on_cpu) = (
                pipeline(bundle, dev)?,
                pipeline(bundle, open(Device::Cpu, Precision::Exact)?)?,
                pipeline(bundle, open(Device::Cpu, Precision::Model)?)?,
            );
            let got = run(&on_dev, texts)?;
            let want = run(&reference, texts)?;
            if got.is_empty() || got.len() != want.len() || got.len() % dim != 0 {
                return Err(CoreError::new(
                    INVALID_ARGUMENT,
                    format!("window {max_seq}: {} device values, {} reference", got.len(), want.len()),
                ));
            }
            let min_cosine = lowest_cosine(&got, &want, dim);
            let (device_ms, cpu_ms) =
                medians(o.reps, || run(&on_dev, texts).map(drop), || run(&on_cpu, texts).map(drop))?;
            points.push(Point { max_seq, min_cosine, device_ms, cpu_ms });
        }
        let floor = floor.unwrap_or(I8_FLOOR);
        let (max_seq, use_device) = decide(floor, &points);
        Ok(Calibration { key, floor, max_seq, use_device, points })
    }

    /// The calibration kept in `dir` for this bundle, device and request,
    /// measured and kept first when there is none, or when the kept one is
    /// incomplete (a point for each window asked, in order) or its
    /// decision does not follow from its points.
    pub fn load_or_measure(
        dir: &Path,
        bundle: &Path,
        o: &CalibrationOptions,
        texts: &[Text],
    ) -> Result<Calibration, CoreError> {
        let key = Calibration::key(bundle, o, texts)?;
        let file = Calibration::file(dir, &key);
        if let Ok(text) = std::fs::read_to_string(&file)
            && let Some(c) = Calibration::parse(&text)
            && c.key == key
            && c.points.iter().map(|p| p.max_seq).eq(o.windows.iter().copied())
            && decide(c.floor, &c.points) == (c.max_seq, c.use_device)
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

    /// The options to open the step with: the device at the chosen window,
    /// or the CPU at the model's own.
    pub fn options(&self, o: &CalibrationOptions, base: &VectorOptions) -> VectorOptions {
        let mut v = base.clone();
        if self.use_device {
            v.encoder.device = o.device.clone();
            v.encoder.precision = o.precision;
            v.encoder.max_seq = self.max_seq;
        } else {
            v.encoder.device = Device::Cpu;
        }
        v
    }

    /// The kept form: one record per line, tab separated.
    pub fn to_text(&self) -> String {
        let mut s = String::from("calibration 1\n");
        let _ = writeln!(s, "key\t{}", self.key);
        let _ = writeln!(s, "floor\t{}", self.floor);
        let _ = writeln!(s, "max_seq\t{}", self.max_seq);
        let _ = writeln!(s, "use_device\t{}", self.use_device);
        for p in &self.points {
            let _ = writeln!(s, "point\t{}\t{}\t{}\t{}", p.max_seq, p.min_cosine, p.device_ms, p.cpu_ms);
        }
        s
    }

    /// Reads [`Calibration::to_text`]'s form, with the key, the floor, the
    /// window and the choice each exactly once and at least one point;
    /// None for anything else.
    pub fn parse(text: &str) -> Option<Calibration> {
        let mut lines = text.lines();
        if lines.next()? != "calibration 1" {
            return None;
        }
        let mut key: Option<String> = None;
        let mut floor: Option<f64> = None;
        let mut max_seq: Option<u32> = None;
        let mut use_device: Option<bool> = None;
        let mut points = Vec::new();
        for line in lines {
            let (name, rest) = line.split_once('\t')?;
            let again = match name {
                "key" => key.replace(rest.to_owned()).is_some(),
                "floor" => floor.replace(rest.parse().ok()?).is_some(),
                "max_seq" => max_seq.replace(rest.parse().ok()?).is_some(),
                "use_device" => use_device.replace(rest.parse().ok()?).is_some(),
                "point" => {
                    let f: Vec<&str> = rest.split('\t').collect();
                    let [a, b, d, e] = f[..] else { return None };
                    points.push(Point {
                        max_seq: a.parse().ok()?,
                        min_cosine: b.parse().ok()?,
                        device_ms: d.parse().ok()?,
                        cpu_ms: e.parse().ok()?,
                    });
                    false
                }
                _ => return None,
            };
            if again {
                return None;
            }
        }
        if points.is_empty() {
            return None;
        }
        Some(Calibration { key: key?, floor: floor?, max_seq: max_seq?, use_device: use_device?, points })
    }
}

/// The widest window whose lowest cosine clears `floor` (0 when none), and
/// whether the device is faster than the CPU there.
pub fn decide(floor: f64, points: &[Point]) -> (u32, bool) {
    let best = points.iter().filter(|p| p.min_cosine >= floor).max_by_key(|p| p.max_seq);
    (best.map_or(0, |p| p.max_seq), best.is_some_and(|p| p.device_ms < p.cpu_ms))
}

/// The lowest cosine between the vectors of `got` and `want`, `dim` values
/// each; a cosine that is not a number counts as the lowest there is.
pub fn lowest_cosine(got: &[f32], want: &[f32], dim: usize) -> f64 {
    got.chunks_exact(dim)
        .zip(want.chunks_exact(dim))
        .map(|(g, w)| cosine(g, w))
        .fold(1f64, |m, c| if c.is_nan() { f64::NEG_INFINITY } else { m.min(c) })
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

/// The key, and the model's own window.
fn probe(bundle: &Path, o: &CalibrationOptions, texts: &[Text]) -> Result<(String, u32), CoreError> {
    let model = crate::model_ref(&Bundle::open(bundle)?);
    let enc = Encoder::open(
        bundle,
        &EncoderOptions { device: o.device.clone(), precision: o.precision, max_batch: 1, ..EncoderOptions::default() },
    )?;
    let d = enc.device_info();
    let f = crate::encoder::field;
    let mut t = String::new();
    for (text, spans) in texts {
        let _ = write!(t, "{text}\u{1f}{spans:?}\u{1e}");
    }
    let windows: Vec<String> = o.windows.iter().map(u32::to_string).collect();
    let key = [
        model.id,
        model.sha256,
        f(&d.backend),
        f(&d.name),
        f(&d.runtime_version),
        f(&d.driver_version),
        o.precision.name().to_owned(),
        turbo::tuning::LIBRARY_BUILD.to_owned(),
        windows.join(","),
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

/// The sentence vectors of every text, in order; an error when a text
/// with sentences gets none.
fn run(a: &Analyzer, texts: &[Text]) -> Result<Vec<f32>, CoreError> {
    let err = |e: turbo_analysis::Error| CoreError::new(INVALID_ARGUMENT, e.to_string());
    let mut docs: Vec<Document> = texts
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
        .map_err(err)?;
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
    let dot: f64 = g.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum();
    let n = |v: &[f32]| v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    let d = n(g) * n(w);
    if d == 0.0 { if n(g) == n(w) { 1.0 } else { 0.0 } } else { dot / d }
}

/// The median times of `reps` runs each of `a` and `b`, in ms, the two
/// taking turns after one untimed run each, so drift falls on both.
fn medians(
    reps: usize,
    mut a: impl FnMut() -> Result<(), CoreError>,
    mut b: impl FnMut() -> Result<(), CoreError>,
) -> Result<(f64, f64), CoreError> {
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
    Ok((ta[reps / 2], tb[reps / 2]))
}
