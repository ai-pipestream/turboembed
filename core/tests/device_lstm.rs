//! `turbo::device::lstm::Lstm` against `turbo::cpu::lstm::Lstm` (#860):
//! random weights, single sequences of 1, 2 and 40 steps, mixed-length
//! batches, both directions, `run` and `run_gathered`, and the same
//! refusals.
//!
//! The device is TURBO_TEST_DEVICE (cpu, cuda or cuda:N), else cuda when
//! TURBO_TEST_REQUIRE_CUDA=1, else the CPU, where the harness checks
//! itself bit for bit. A named device that cannot make the LSTM fails the
//! test; it never falls back to the CPU.

use turbo::cpu::lstm as cpu;
use turbo::device::Device;
use turbo::device::lstm::{Lstm, Weights, Work};
use turbo::status::{INVALID_ARGUMENT, INVALID_SHAPE};

/// The bound a device's states are held to, against the CPU's.
const BOUND: f32 = 1e-5;

fn device() -> Device {
    let required = std::env::var("TURBO_TEST_REQUIRE_CUDA").is_ok_and(|v| v == "1");
    match std::env::var("TURBO_TEST_DEVICE") {
        Ok(name) => Device::parse(&name).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={name}: {}", e.message)),
        Err(_) if required => Device::Cuda(0),
        Err(_) => Device::Cpu,
    }
}

/// A fixed stream of values in [-scale, scale), from a 64-bit LCG.
struct Rng(u64);

impl Rng {
    fn values(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }
}

struct Model {
    input: usize,
    hidden: usize,
    w: [Vec<f32>; 2],
    u: [Vec<f32>; 2],
    b: [Vec<f32>; 2],
}

impl Model {
    fn new(input: usize, hidden: usize, seed: u64) -> Model {
        let mut rng = Rng(seed);
        let g = 4 * hidden;
        let mut one = || (rng.values(g * input, 0.4), rng.values(g * hidden, 0.4), rng.values(g, 0.5));
        let ((w0, u0, b0), (w1, u1, b1)) = (one(), one());
        Model { input, hidden, w: [w0, w1], u: [u0, u1], b: [b0, b1] }
    }

    fn dirs(&self) -> [Weights<'_>; 2] {
        let d = |k: usize| Weights { w: &self.w[k], u: &self.u[k], b: &self.b[k], reverse: k == 1 };
        [d(0), d(1)]
    }

    fn cpu(&self) -> cpu::Lstm {
        cpu::Lstm::new(self.input, self.hidden, &self.dirs()).unwrap()
    }

    fn on(&self, device: Device) -> Lstm {
        Lstm::new(device, self.input, self.hidden, &self.dirs())
            .unwrap_or_else(|e| panic!("the LSTM on {device:?}: {}", e.message))
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// `lengths` sequences of random rows through the device and the CPU,
/// twice on the device with one workspace: the worst difference.
fn check_run(m: &Model, lengths: &[usize], seed: u64) -> f32 {
    let rows: usize = lengths.iter().sum();
    let x = Rng(seed).values(rows * m.input, 1.0);
    let mut want = vec![f32::NAN; rows * 2 * m.hidden];
    m.cpu().run(&x, lengths, &mut want, &mut cpu::Work::default()).unwrap();
    let (lstm, mut w) = (m.on(device()), Work::default());
    let mut worst = 0f32;
    for _ in 0..2 {
        let mut got = vec![f32::NAN; want.len()];
        lstm.run(&x, lengths, &mut got, &mut w).unwrap();
        worst = worst.max(max_diff(&got, &want));
    }
    worst
}

/// As [`check_run`], the rows read from a table of `distinct` rows.
fn check_gathered(m: &Model, lengths: &[usize], distinct: usize, seed: u64) -> f32 {
    let mut rng = Rng(seed);
    let pw = 8 * m.hidden;
    let table = rng.values(distinct * pw, 2.0);
    let rows: usize = lengths.iter().sum();
    let index: Vec<usize> =
        rng.values(rows, 1.0).iter().map(|v| ((v + 1.0) / 2.0 * distinct as f32) as usize % distinct).collect();
    let mut want = vec![f32::NAN; rows * 2 * m.hidden];
    m.cpu().run_gathered(&table, &index, lengths, &mut want, &mut cpu::Work::default()).unwrap();
    let lstm = m.on(device());
    let on_device = lstm.table(&table).unwrap();
    assert_eq!(on_device.rows(), distinct);
    let mut got = vec![f32::NAN; want.len()];
    lstm.run_gathered(&on_device, &index, lengths, &mut got, &mut Work::default()).unwrap();
    max_diff(&got, &want)
}

const SHAPES: [(usize, usize); 3] = [(3, 2), (50, 25), (100, 128)];

const BATCHES: [&[usize]; 7] =
    [&[1], &[2], &[40], &[40, 40, 2, 1], &[7, 5, 5, 3, 1, 1], &[40, 39, 20, 20, 20, 9, 2, 2, 1, 1, 1], &[3; 64]];

#[test]
fn runs_match_the_cpu() {
    for (i, &(input, hidden)) in SHAPES.iter().enumerate() {
        let m = Model::new(input, hidden, 11 + i as u64);
        for (j, lengths) in BATCHES.iter().enumerate() {
            let d = check_run(&m, lengths, 100 + j as u64);
            assert!(d <= BOUND, "{:?} input {input} hidden {hidden} lengths {lengths:?}: {d}", device());
            if device() == Device::Cpu {
                assert_eq!(d, 0.0, "the CPU through the device surface changed a bit");
            }
        }
    }
}

#[test]
fn gathered_runs_match_the_cpu() {
    for (i, &(input, hidden)) in SHAPES.iter().enumerate() {
        let m = Model::new(input, hidden, 21 + i as u64);
        for (j, lengths) in BATCHES.iter().enumerate() {
            for distinct in [1, 17, 300] {
                let d = check_gathered(&m, lengths, distinct, 200 + j as u64);
                assert!(d <= BOUND, "{:?} hidden {hidden} lengths {lengths:?} table {distinct}: {d}", device());
            }
        }
    }
}

#[test]
fn refusals_match_the_cpu() {
    let m = Model::new(3, 2, 5);
    let (lstm, reference) = (m.on(device()), m.cpu());
    let x = Rng(9).values(5 * 3, 1.0);
    let mut nan = x.clone();
    nan[4] = f32::NAN;
    let cases: [(&str, &[f32], &[usize], usize); 6] = [
        ("a zero length", &x, &[5, 0], 5),
        ("lengths out of order", &x, &[2, 3], 5),
        ("an input too short", &x[..14], &[5], 5),
        ("an output too short", &x, &[5], 4),
        ("a non-finite input", &nan, &[3, 2], 5),
        ("an empty batch", &x[..0], &[], 0),
    ];
    for (what, x, lengths, rows) in cases {
        let mut a = vec![0.0; rows * 4];
        let mut b = a.clone();
        let got = lstm.run(x, lengths, &mut a, &mut Work::default()).map_err(|e| e.code);
        let want = reference.run(x, lengths, &mut b, &mut cpu::Work::default()).map_err(|e| e.code);
        assert_eq!(got, want, "{what}");
    }
    assert_eq!(lstm.table(&[0.0; 15]).map(|_| ()).map_err(|e| e.code), Err(INVALID_SHAPE));
    let table = lstm.table(&Rng(3).values(2 * 16, 1.0)).unwrap();
    let mut out = vec![0.0; 3 * 4];
    let past = lstm.run_gathered(&table, &[0, 1, 2], &[3], &mut out, &mut Work::default());
    assert_eq!(past.map_err(|e| e.code), Err(INVALID_SHAPE));
    let mut bad = Rng(3).values(2 * 16, 1.0);
    bad[20] = f32::INFINITY;
    let table = lstm.table(&bad).unwrap();
    let inf = lstm.run_gathered(&table, &[0, 1, 1], &[3], &mut out, &mut Work::default());
    assert_eq!(inf.map_err(|e| e.code), Err(INVALID_ARGUMENT));
}

/// A device the library has no LSTM for is refused when the LSTM is made.
#[cfg(not(feature = "cuda"))]
#[test]
fn a_device_not_built_in_is_refused() {
    let m = Model::new(3, 2, 1);
    let e = Lstm::new(Device::Cuda(0), 3, 2, &m.dirs()).err().expect("an LSTM on a CUDA device in a build without it");
    assert_eq!(e.code, turbo::status::UNSUPPORTED);
}
