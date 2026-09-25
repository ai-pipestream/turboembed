//! The CUDA backend's GELU for an F16 output (gelu_f16 in
//! core/cuda/kernels.cu), computed here on the host as the device does,
//! operation for operation, with the constants read out of the kernel
//! source: every F16 value in [-8, 8] against GELU in f64. Needs no
//! device and no CUDA toolkit, so it runs without the `cuda` feature.
//!
//! The device's exponential and reciprocal (__expf, __fdividef) are
//! within a few ulps of the host's, far under F16's rounding; a tie below
//! may round the other way on a device.

const KERNELS: &str = include_str!("../cuda/kernels.cu");

/// The F16 values the fit rounds other than GELU does, each within 2e-3
/// of an ulp of the midpoint between two F16 values (checked below).
const TIES: [f32; 10] = [
    -3.779_296_9,
    -0.239_868_16,
    -0.058_197_02,
    -0.001_338_958_7,
    -0.000_546_455_4,
    -0.000_386_476_52,
    0.000_386_476_52,
    0.000_546_455_4,
    0.001_338_958_7,
    0.058_197_02,
];

/// `constexpr float <name> = <value>f;` from kernels.cu.
fn constant(name: &str) -> f32 {
    let head = format!("constexpr float {name} = ");
    let at = KERNELS.find(&head).unwrap_or_else(|| panic!("kernels.cu has no {name}")) + head.len();
    let value = &KERNELS[at..at + KERNELS[at..].find(';').unwrap()];
    value.strip_suffix('f').unwrap().parse().unwrap_or_else(|e| panic!("{name} is {value}: {e}"))
}

struct Fit {
    p: f32,
    a: [f32; 6],
}

impl Fit {
    fn read() -> Fit {
        let a = std::array::from_fn(|i| constant(&format!("GELU_F16_A{}", i + 1)));
        Fit { p: constant("GELU_F16_P"), a }
    }

    /// gelu_f16, with __expf as the device computes it, 2 to the power of
    /// x log2(e), and __fdividef as a division.
    fn gelu(&self, v: f32) -> f32 {
        let z = v.abs() * std::f32::consts::FRAC_1_SQRT_2;
        let tau = 1.0 / self.p.mul_add(z, 1.0);
        let mut p = self.a[5].mul_add(tau, self.a[4]);
        for &a in self.a[..4].iter().rev() {
            p = p.mul_add(tau, a);
        }
        let r = (p * tau) * ((-(z * z)) * std::f32::consts::LOG2_E).exp2();
        (0.5 * v.min(16.0)).mul_add(r.copysign(-v), v.max(0.0))
    }
}

/// erfc for a >= 0: 1 - erf's Taylor series below 2, where it loses under
/// three digits, and the continued fraction above, in f64.
fn erfc(a: f64) -> f64 {
    if a < 2.0 {
        let z = a * a;
        let (mut term, mut sum, mut n) = (a, a, 0.0);
        loop {
            n += 1.0;
            term *= -z / n;
            let t = term / (2.0 * n + 1.0);
            sum += t;
            if t.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        return 1.0 - sum * std::f64::consts::FRAC_2_SQRT_PI;
    }
    let mut k = a;
    for n in (1..=200).rev() {
        k = a + f64::from(n) / 2.0 / k;
    }
    (-a * a).exp() / (std::f64::consts::PI.sqrt() * k)
}

/// GELU with the error function, in f64, with no cancellation for x < 0.
fn gelu(x: f64) -> f64 {
    let r = erfc(x.abs() * std::f64::consts::FRAC_1_SQRT_2);
    if x < 0.0 { 0.5 * x * r } else { 0.5 * x * (2.0 - r) }
}

/// The F16 value of an F16's bits, in f64.
fn from_f16(bits: u16) -> f64 {
    let (e, m) = (i32::from(bits >> 10 & 0x1f), f64::from(bits & 0x3ff));
    let v = match e {
        0 => m * 2f64.powi(-24),
        31 => f64::INFINITY,
        _ => (1024.0 + m) * 2f64.powi(e - 25),
    };
    if bits & 0x8000 != 0 { -v } else { v }
}

/// x rounded to F16, to nearest, ties to even, as __floats2half2_rn does:
/// the value and its ulp.
fn to_f16(x: f64) -> (f64, f64) {
    let a = x.abs();
    let e = ((a.to_bits() >> 52) as i32 - 1023).max(-14);
    let ulp = 2f64.powi(e - 10);
    let r = (a / ulp).round_ties_even() * ulp;
    (if r >= 65520.0 { f64::INFINITY } else { r }.copysign(x), ulp)
}

#[test]
fn gelu_f16_rounds_as_gelu_does() {
    let fit = Fit::read();
    let (mut worst, mut worst_at, mut ties) = (0.0f64, 0.0f64, Vec::new());
    let mut count = 0;
    for bits in 0..=u16::MAX {
        let x = from_f16(bits);
        if x.abs() > 8.0 {
            continue;
        }
        count += 1;
        let (got, want) = (f64::from(fit.gelu(x as f32)), gelu(x));
        let diff = (got - want).abs();
        if diff > worst {
            (worst, worst_at) = (diff, x);
        }
        let (rounded, ulp) = to_f16(want);
        if to_f16(got).0 != rounded {
            // How near want is to the midpoint either side of its F16 value.
            let off = ((want - rounded).abs() - ulp / 2.0).abs() / ulp;
            println!("x = {x}: GELU {want:e}, the fit {got:e}, {off:.1e} of an ulp from a tie");
            assert!(off < 2e-3, "x = {x}: the fit rounds to another F16 than GELU, {off:e} of an ulp from a tie");
            ties.push(x as f32);
        }
    }
    println!("{count} F16 values in [-8, 8]: largest difference {worst:.3e} at {worst_at}, {} ties", ties.len());
    assert!(count > 30_000);
    assert!(worst <= 2e-6, "the fit is {worst:e} from GELU at {worst_at}");
    for x in &ties {
        assert!(TIES.contains(x), "x = {x} rounds to another F16 than GELU and is not a listed tie");
    }
}

/// Past 8 the fit is exactly v, and 0 at the negative side, as F16 holds
/// GELU there; +inf stays +inf, and NaN stays NaN.
#[test]
fn gelu_f16_holds_the_tails() {
    let fit = Fit::read();
    for bits in 0..=0x7bffu16 {
        let x = from_f16(bits);
        if x <= 8.0 {
            continue;
        }
        assert_eq!(to_f16(f64::from(fit.gelu(x as f32))).0, x, "gelu({x})");
        assert_eq!(to_f16(f64::from(fit.gelu(-x as f32))).0, 0.0, "gelu(-{x})");
    }
    assert_eq!(fit.gelu(f32::INFINITY), f32::INFINITY);
    assert!(fit.gelu(f32::NAN).is_nan());
    assert_eq!(fit.gelu(0.0), 0.0);
}
