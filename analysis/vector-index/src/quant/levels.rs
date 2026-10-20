//! Lloyd-Max levels for a standard normal coordinate: for each bit
//! width, the levels that minimize the expected squared error, each the
//! mean of the normal density over its cell. They are constants rather
//! than computed at run time, so no `exp` from a platform's math library
//! can change a code. The test regenerates them.

/// The positive half of each grid, ascending; the grid is symmetric.
const HALVES: [&[f32]; 4] = [
    &[0.797_884_6],
    &[0.452_780_03, 1.510_417_6],
    &[0.245_094_18, 0.756_005_3, 1.343_909_3, 2.151_945_7],
    &[0.128_395_03, 0.388_048_3, 0.656_759_1, 0.942_340_46, 1.256_231_2, 1.618_046_4, 2.069_017_2, 2.732_589_5],
];

/// The grid of one bit width, with the cell boundaries between its levels.
#[derive(Clone, Debug)]
pub(crate) struct Grid {
    levels: Vec<f32>,
    /// `thresholds[i]` is the midpoint of levels `i` and `i + 1`, in `f64`:
    /// a value is coded `i` when it lies in `(thresholds[i - 1], thresholds[i]]`.
    thresholds: Vec<f64>,
}

impl Grid {
    /// The grid of `bits` (1 to 4) bits.
    pub(crate) fn new(bits: u8) -> Grid {
        let half = HALVES[usize::from(bits) - 1];
        let levels: Vec<f32> = half.iter().rev().map(|&l| -l).chain(half.iter().copied()).collect();
        let thresholds = levels.windows(2).map(|w| (f64::from(w[0]) + f64::from(w[1])) / 2.0).collect();
        Grid { levels, thresholds }
    }

    pub(crate) fn levels(&self) -> &[f32] {
        &self.levels
    }

    /// The code of `value`: the index of the nearest level, the lower one
    /// on a tie.
    pub(crate) fn encode(&self, value: f64) -> u32 {
        self.thresholds.partition_point(|&t| value > t) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mean of the standard normal density over `[a, b]`, by Simpson's
    /// rule on a fine grid; the tails beyond 12 carry nothing an `f32`
    /// level can show.
    fn centroid(a: f64, b: f64) -> f64 {
        let (a, b) = (a.max(-12.0), b.min(12.0));
        let n = 200_000;
        let h = (b - a) / f64::from(n);
        let (mut mass, mut moment) = (0.0, 0.0);
        for i in 0..=n {
            let x = a + h * f64::from(i);
            let w = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let d = (-x * x / 2.0).exp();
            mass += w * d;
            moment += w * d * x;
        }
        moment / mass
    }

    #[test]
    fn every_level_is_the_mean_of_its_cell() {
        for bits in 1..=4u8 {
            let g = Grid::new(bits);
            let l = g.levels();
            assert_eq!(l.len(), 1 << bits);
            for i in 0..l.len() {
                let lo = if i == 0 { f64::NEG_INFINITY } else { g.thresholds[i - 1] };
                let hi = if i + 1 == l.len() { f64::INFINITY } else { g.thresholds[i] };
                let c = centroid(lo, hi);
                assert!((c - f64::from(l[i])).abs() < 2e-6, "bits {bits} level {i}: {} vs centroid {c}", l[i]);
            }
        }
    }

    #[test]
    fn codes_follow_the_cells() {
        let g = Grid::new(2);
        let t = &g.thresholds;
        assert_eq!(g.encode(f64::NEG_INFINITY), 0);
        assert_eq!(g.encode(t[0]), 0);
        assert_eq!(g.encode(t[0].next_up()), 1);
        assert_eq!(g.encode(0.0), 1);
        assert_eq!(g.encode(0.0f64.next_up()), 2);
        assert_eq!(g.encode(f64::INFINITY), 3);
    }
}
