//! The kept form of a [`Calibration`]: one record per line, tab separated.

use std::fmt::Write as _;

use super::{Calibration, Point};

impl Calibration {
    /// The kept form: the key and the floor, then a line per point.
    pub fn to_text(&self) -> String {
        let mut s = String::from("calibration 5\n");
        let _ = writeln!(s, "key\t{}", self.key);
        let _ = writeln!(s, "floor\t{}", self.floor);
        for p in &self.points {
            let _ = writeln!(
                s,
                "point\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                p.max_seq,
                p.max_batch,
                p.min_cosine,
                p.device_ms,
                p.cpu_ms,
                p.short_device_ms,
                p.short_cpu_ms,
                p.doc_device_ms,
                p.doc_cpu_ms
            );
        }
        s
    }

    /// Reads [`Calibration::to_text`]'s form, with the key and the floor
    /// each exactly once and at least one unique, nonzero shape. Timings
    /// must be finite and positive; the floor finite. A cosine may be
    /// negative infinity, the failed-vector sentinel, but not NaN or over
    /// one. Invalid records return None so they can be measured again.
    pub fn parse(text: &str) -> Option<Calibration> {
        let mut lines = text.lines();
        if lines.next()? != "calibration 5" {
            return None;
        }
        let mut key: Option<String> = None;
        let mut floor: Option<f64> = None;
        let mut points: Vec<Point> = Vec::new();
        for line in lines {
            let (name, rest) = line.split_once('\t')?;
            let again = match name {
                "key" => key.replace(rest.to_owned()).is_some(),
                "floor" => floor.replace(rest.parse().ok()?).is_some(),
                "point" => {
                    let f: Vec<&str> = rest.split('\t').collect();
                    let [a, b, c, d, e, g, h, i, j] = f[..] else { return None };
                    let p = Point {
                        max_seq: a.parse().ok()?,
                        max_batch: b.parse().ok()?,
                        min_cosine: c.parse().ok()?,
                        device_ms: d.parse().ok()?,
                        cpu_ms: e.parse().ok()?,
                        short_device_ms: g.parse().ok()?,
                        short_cpu_ms: h.parse().ok()?,
                        doc_device_ms: i.parse().ok()?,
                        doc_cpu_ms: j.parse().ok()?,
                    };
                    let times =
                        [p.device_ms, p.cpu_ms, p.short_device_ms, p.short_cpu_ms, p.doc_device_ms, p.doc_cpu_ms];
                    if p.max_seq == 0
                        || p.max_batch == 0
                        || p.min_cosine.is_nan()
                        || p.min_cosine > 1.0
                        || times.iter().any(|t| !t.is_finite() || *t <= 0.0)
                        || points.iter().any(|q| (q.max_seq, q.max_batch) == (p.max_seq, p.max_batch))
                    {
                        return None;
                    }
                    points.push(p);
                    false
                }
                _ => return None,
            };
            if again {
                return None;
            }
        }
        let floor = floor?;
        if points.is_empty() || !floor.is_finite() {
            return None;
        }
        Some(Calibration { key: key?, floor, points })
    }
}
