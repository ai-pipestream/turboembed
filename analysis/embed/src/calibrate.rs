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
//! result in a directory, under a key made of the bundle's manifest hash,
//! the device's backend, name, runtime and driver versions, the precision
//! and the library's version, so it is measured again only when one of
//! those changes. [`Calibration::options`] gives the options to open the
//! step with.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use turbo::bundle::Bundle;
use turbo::bundle::sha256_hex;
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
    /// The windows (tokens per row) to try; each at most the model's.
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
    /// The key a bundle and device are kept under: the bundle's model and
    /// manifest hash, the device's backend, name, runtime and driver
    /// versions, the precision and the library's version.
    pub fn key(bundle: &Path, o: &CalibrationOptions) -> Result<String, CoreError> {
        let model = crate::model_ref(&Bundle::open(bundle)?);
        let enc = Encoder::open(
            bundle,
            &EncoderOptions {
                device: o.device.clone(),
                precision: o.precision,
                max_batch: 1,
                ..EncoderOptions::default()
            },
        )?;
        let d = enc.device_info();
        let f = crate::encoder::field;
        Ok([
            model.id,
            model.sha256,
            f(&d.backend),
            f(&d.name),
            f(&d.runtime_version),
            f(&d.driver_version),
            o.precision.name().to_owned(),
            turbo::record::library_version().to_owned(),
        ]
        .join(" | "))
    }

    /// Runs the step over `texts` at each window of `o`, on the device and
    /// on the CPU, and decides.
    pub fn measure(bundle: &Path, o: &CalibrationOptions, texts: &[Text]) -> Result<Calibration, CoreError> {
        if o.windows.is_empty() || texts.is_empty() || o.reps == 0 {
            return Err(CoreError::new(INVALID_ARGUMENT, "calibration needs windows, texts and a run"));
        }
        let key = Calibration::key(bundle, o)?;
        let mut floor = o.floor;
        let mut points = Vec::new();
        for &max_seq in &o.windows {
            let open = |device: Device, precision: Precision| {
                let enc = EncoderOptions { device, precision, max_seq, ..EncoderOptions::default() };
                VectorAnnotator::open(bundle, &VectorOptions { encoder: enc, ..VectorOptions::default() }).map(Arc::new)
            };
            let (dev, reference, cpu) = (
                open(o.device.clone(), o.precision)?,
                open(Device::Cpu, Precision::Exact)?,
                open(Device::Cpu, Precision::Model)?,
            );
            floor.get_or_insert_with(|| default_floor(dev.info().compute_dtype));
            let got = run(bundle, &dev, texts)?;
            let want = run(bundle, &reference, texts)?;
            let dim = dev.dim() as usize;
            let min_cosine =
                got.chunks_exact(dim).zip(want.chunks_exact(dim)).map(|(g, w)| cosine(g, w)).fold(1f64, f64::min);
            let device_ms = median(o.reps, || run(bundle, &dev, texts).map(drop))?;
            let cpu_ms = median(o.reps, || run(bundle, &cpu, texts).map(drop))?;
            points.push(Point { max_seq, min_cosine, device_ms, cpu_ms });
        }
        let floor = floor.unwrap_or(I8_FLOOR);
        let best = points.iter().filter(|p| p.min_cosine >= floor).max_by_key(|p| p.max_seq);
        Ok(Calibration {
            key,
            floor,
            max_seq: best.map_or(0, |p| p.max_seq),
            use_device: best.is_some_and(|p| p.device_ms < p.cpu_ms),
            points,
        })
    }

    /// The calibration kept in `dir` for this bundle and device, measured
    /// and kept first when there is none.
    pub fn load_or_measure(
        dir: &Path,
        bundle: &Path,
        o: &CalibrationOptions,
        texts: &[Text],
    ) -> Result<Calibration, CoreError> {
        let key = Calibration::key(bundle, o)?;
        let file = dir.join(format!("{}.calibration", &sha256_hex(key.as_bytes())[..32]));
        if let Ok(text) = std::fs::read_to_string(&file)
            && let Some(c) = Calibration::parse(&text)
            && c.key == key
        {
            return Ok(c);
        }
        let c = Calibration::measure(bundle, o, texts)?;
        let io = |e: std::io::Error| CoreError::new(INVALID_ARGUMENT, format!("{}: {e}", file.display()));
        std::fs::create_dir_all(dir).map_err(io)?;
        std::fs::write(&file, c.to_text()).map_err(io)?;
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

    /// Reads [`Calibration::to_text`]'s form; None for anything else.
    pub fn parse(text: &str) -> Option<Calibration> {
        let mut lines = text.lines();
        if lines.next()? != "calibration 1" {
            return None;
        }
        let mut c = Calibration { key: String::new(), floor: 0.0, max_seq: 0, use_device: false, points: Vec::new() };
        for line in lines {
            let (name, rest) = line.split_once('\t')?;
            match name {
                "key" => c.key = rest.to_owned(),
                "floor" => c.floor = rest.parse().ok()?,
                "max_seq" => c.max_seq = rest.parse().ok()?,
                "use_device" => c.use_device = rest.parse().ok()?,
                "point" => {
                    let f: Vec<&str> = rest.split('\t').collect();
                    let [a, b, d, e] = f[..] else { return None };
                    c.points.push(Point {
                        max_seq: a.parse().ok()?,
                        min_cosine: b.parse().ok()?,
                        device_ms: d.parse().ok()?,
                        cpu_ms: e.parse().ok()?,
                    });
                }
                _ => return None,
            }
        }
        Some(c)
    }
}

/// The core's tolerance for `dtype`, or [`I8_FLOOR`] for a dtype it gives
/// none (I8).
fn default_floor(dtype: u32) -> f64 {
    turbo::record::tolerance(dtype).map_or(I8_FLOOR, |t| t.min_cosine)
}

/// The sentence vectors of every text, in order.
fn run(bundle: &Path, vectors: &Arc<VectorAnnotator>, texts: &[Text]) -> Result<Vec<f32>, CoreError> {
    let err = |e: turbo_analysis::Error| CoreError::new(INVALID_ARGUMENT, e.to_string());
    let a = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SubwordAnnotator::open(bundle)?)
        .add(SentenceVectors(vectors.clone()))
        .build()
        .map_err(err)?;
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
    Ok(docs.iter().flat_map(|d| d.get::<Vectors<Sentences>>().map(|v| v.data.clone()).unwrap_or_default()).collect())
}

fn cosine(g: &[f32], w: &[f32]) -> f64 {
    let dot: f64 = g.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum();
    let n = |v: &[f32]| v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    let d = n(g) * n(w);
    if d == 0.0 { if n(g) == n(w) { 1.0 } else { 0.0 } } else { dot / d }
}

/// The median time of `reps` runs of `f` after one untimed run, in ms.
fn median(reps: usize, mut f: impl FnMut() -> Result<(), CoreError>) -> Result<f64, CoreError> {
    f()?;
    let mut t = Vec::with_capacity(reps);
    for _ in 0..reps {
        let s = Instant::now();
        f()?;
        t.push(s.elapsed().as_secs_f64() * 1e3);
    }
    t.sort_by(f64::total_cmp);
    Ok(t[reps / 2])
}
