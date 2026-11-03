use crate::status::{Result, invalid};

pub(super) fn finite(values: &[f32]) -> Result<()> {
    if values.iter().all(|x| x.is_finite()) { Ok(()) } else { Err(invalid("deberta non-finite computation")) }
}

pub(super) struct Norm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}
impl Norm {
    pub fn apply(&self, values: &mut [f32], eps: f64) -> Result<()> {
        finite(values)?;
        let width = self.weight.len();
        for row in values.chunks_exact_mut(width) {
            let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / width as f64;
            let variance = row.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / width as f64;
            let scale = (variance + eps).sqrt().recip();
            for ((x, &w), &b) in row.iter_mut().zip(&self.weight).zip(&self.bias) {
                *x = ((f64::from(*x) - mean) * scale * f64::from(w) + f64::from(b)) as f32;
            }
        }
        finite(values)
    }
}

/// HF make_log_bucket_position, including its signed distance convention.
pub(crate) fn bucket(distance: isize, buckets: usize, maximum: usize) -> usize {
    let mid = buckets / 2;
    let abs = distance.unsigned_abs();
    let magnitude = if abs <= mid {
        abs
    } else {
        // HF uses float32 logarithms here. Keep bucket boundaries identical.
        (((abs as f32 / mid as f32).ln() / ((maximum - 1) as f32 / mid as f32).ln() * (mid - 1) as f32).ceil() as usize)
            + mid
    };
    let signed = if distance < 0 { -(magnitude as isize) } else { magnitude as isize };
    (signed + buckets as isize).clamp(0, (2 * buckets - 1) as isize) as usize
}
