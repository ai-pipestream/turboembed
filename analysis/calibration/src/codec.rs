//! Canonical little-endian TCAL0001 format. Lengths are u32 and every
//! decoded count is bounded by remaining input before allocating.
use crate::{Calibration, DataIdentity, Identity, Result, Task, Transform};
fn put_u32(b: &mut Vec<u8>, n: usize) {
    b.extend(u32::try_from(n).expect("validated calibration size").to_le_bytes());
}
fn put_text(b: &mut Vec<u8>, s: &str) {
    put_u32(b, s.len());
    b.extend(s.as_bytes());
}
fn put_values(b: &mut Vec<u8>, v: &[f64]) {
    put_u32(b, v.len());
    for x in v {
        b.extend(x.to_le_bytes());
    }
}
impl Calibration {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = b"TCAL0001".to_vec();
        let i = self.identity();
        for s in [&i.model, &i.revision, &i.sha256, &i.domain] {
            put_text(&mut b, s);
        }
        b.push(match i.task {
            Task::Categorization => 0,
            Task::TokenTagging => 1,
        });
        b.extend(i.inference_hash.to_le_bytes());
        put_u32(&mut b, i.labels.len());
        for s in &i.labels {
            put_text(&mut b, s);
        }
        for d in [self.fitted_on(), self.conformal_on()] {
            put_text(&mut b, &d.split);
            put_text(&mut b, &d.sha256);
        }
        match self.transform() {
            Transform::Temperature(t) => {
                b.push(0);
                b.extend(t.to_le_bytes());
            }
            Transform::Platt { slope, intercept } => {
                b.push(1);
                b.extend(slope.to_le_bytes());
                b.extend(intercept.to_le_bytes());
            }
            Transform::Isotonic { upper, value } => {
                b.push(2);
                put_values(&mut b, upper);
                put_values(&mut b, value);
            }
        }
        put_values(&mut b, self.scores());
        b
    }
    /// Loading always requires the intended model/task/domain/label order.
    pub fn from_bytes(bytes: &[u8], expected: &Identity) -> Result<Self> {
        let mut r = Read(bytes);
        if r.take(8)? != b"TCAL0001" {
            return Err("unknown calibration format".into());
        }
        let (model, revision, sha256, domain) = (r.text()?, r.text()?, r.text()?, r.text()?);
        let task = match r.byte()? {
            0 => Task::Categorization,
            1 => Task::TokenTagging,
            _ => return Err("unknown calibration task".into()),
        };
        let inference_hash = u64::from_le_bytes(r.take(8)?.try_into().unwrap());
        let n = r.count(4)?;
        if n > 65536 {
            return Err("too many calibration labels".into());
        }
        let mut labels = Vec::new();
        for _ in 0..n {
            labels.push(r.text()?);
        }
        let identity = Identity { model, revision, sha256, domain, task, labels, inference_hash };
        let fitted_on = DataIdentity { split: r.text()?, sha256: r.text()? };
        let conformal_on = DataIdentity { split: r.text()?, sha256: r.text()? };
        let transform = match r.byte()? {
            0 => Transform::Temperature(r.f64()?),
            1 => Transform::Platt { slope: r.f64()?, intercept: r.f64()? },
            2 => Transform::Isotonic { upper: r.values()?, value: r.values()? },
            _ => return Err("unknown calibration transform".into()),
        };
        let scores = r.values()?;
        if !r.0.is_empty() {
            return Err("trailing calibration bytes".into());
        }
        let result = Self::new(identity, transform, fitted_on, conformal_on, scores)?;
        result.bind(expected)?;
        Ok(result)
    }
}
struct Read<'a>(&'a [u8]);
impl<'a> Read<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.0.len() {
            return Err("truncated calibration artifact".into());
        }
        let (v, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(v)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn count(&mut self, size: usize) -> Result<usize> {
        let n = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize;
        if n > self.0.len() / size {
            return Err("calibration count exceeds remaining bytes".into());
        }
        Ok(n)
    }
    fn text(&mut self) -> Result<String> {
        let n = self.count(1)?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "calibration string is not UTF-8".into())
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn values(&mut self) -> Result<Vec<f64>> {
        let n = self.count(8)?;
        (0..n).map(|_| self.f64()).collect()
    }
}
