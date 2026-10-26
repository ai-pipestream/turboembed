//! `turbo::device::lstm::Lstm` against `turbo::cpu::lstm::Lstm` (#860):
//! random weights, single sequences of 1, 2 and 40 steps, mixed-length
//! batches, both directions, `run` and `run_gathered`, and the same
//! refusals.
//!
//! The device is TURBO_TEST_DEVICE (cpu, cuda, levelzero, or either GPU
//! with :N), else cuda with TURBO_TEST_REQUIRE_CUDA=1 or levelzero with
//! TURBO_TEST_REQUIRE_LEVELZERO=1, else the CPU, where the harness checks
//! itself bit for bit. A named device that cannot make the LSTM fails the
//! test; it never falls back to the CPU.

use turbo::cpu::lstm as cpu;
use turbo::device::Device;
use turbo::device::lstm::{Lstm, Weights, Work};
use turbo::status::{INVALID_ARGUMENT, INVALID_SHAPE, UNSUPPORTED};

/// The bound a device's states are held to, against the CPU's.
const BOUND: f32 = 1e-5;

fn device() -> Device {
    let required = |v: &str| std::env::var(v).is_ok_and(|v| v == "1");
    match std::env::var("TURBO_TEST_DEVICE") {
        Ok(name) => Device::parse(&name).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={name}: {}", e.message)),
        Err(_) if required("TURBO_TEST_REQUIRE_CUDA") => Device::Cuda(0),
        Err(_) if required("TURBO_TEST_REQUIRE_LEVELZERO") => Device::LevelZero(0),
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
    let mut worst = 0f32;
    for (i, &(input, hidden)) in SHAPES.iter().enumerate() {
        let m = Model::new(input, hidden, 11 + i as u64);
        for (j, lengths) in BATCHES.iter().enumerate() {
            let d = check_run(&m, lengths, 100 + j as u64);
            worst = worst.max(d);
            assert!(d <= BOUND, "{:?} input {input} hidden {hidden} lengths {lengths:?}: {d}", device());
            if device() == Device::Cpu {
                assert_eq!(d, 0.0, "the CPU through the device surface changed a bit");
            }
        }
    }
    println!("{:?}: the worst difference from the CPU is {worst:e}", device());
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
    // Finite weights and inputs whose projections pass f32's range: the
    // gates are refused, on a device as on the CPU.
    let mut wide = Model::new(3, 2, 6);
    wide.w[1].fill(3e38);
    let ones = vec![1.0f32; 5 * 3];
    let (mut a, mut b) = (vec![0.0; 5 * 4], vec![0.0; 5 * 4]);
    let got = wide.on(device()).run(&ones, &[5], &mut a, &mut Work::default()).map_err(|e| e.code);
    let want = wide.cpu().run(&ones, &[5], &mut b, &mut cpu::Work::default()).map_err(|e| e.code);
    assert_eq!((got, want), (Err(INVALID_ARGUMENT), Err(INVALID_ARGUMENT)), "an overflowing gate");
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
#[test]
fn a_device_not_built_in_is_refused() {
    let m = Model::new(3, 2, 1);
    let absent = [(Device::Cuda(0), cfg!(feature = "cuda")), (Device::LevelZero(0), cfg!(feature = "levelzero"))];
    for (device, _) in absent.into_iter().filter(|&(_, built)| !built) {
        let e = Lstm::new(device, 3, 2, &m.dirs()).err().expect("an LSTM on a device not built in");
        assert_eq!(e.code, turbo::status::UNSUPPORTED, "{device:?}");
    }
}

/// A table made by an LSTM of another width is read as the CPU form reads
/// the same values: refused where they are not whole rows of the running
/// LSTM's width or an index passes them, and never read past its values.
#[test]
fn a_table_of_another_width_is_refused_as_the_cpu_refuses_it() {
    let (narrow, wide) = (Model::new(3, 2, 7), Model::new(3, 4, 8));
    let (maker, reader, reference) = (narrow.on(device()), wide.on(device()), wide.cpu());
    // 4 rows of 16 for hidden 2: 2 rows of 32 for hidden 4.
    let values = Rng(4).values(4 * 16, 1.0);
    let table = maker.table(&values).unwrap();
    for index in [[0, 1, 1], [0, 1, 3]] {
        let (mut a, mut b) = (vec![0.0; 3 * 8], vec![0.0; 3 * 8]);
        let got = reader.run_gathered(&table, &index, &[3], &mut a, &mut Work::default()).map_err(|e| e.code);
        let want = reference.run_gathered(&values, &index, &[3], &mut b, &mut cpu::Work::default()).map_err(|e| e.code);
        assert_eq!(got, want, "index {index:?}");
        if got.is_ok() {
            assert!(max_diff(&a, &b) <= BOUND, "index {index:?}");
        }
    }
    assert_eq!(
        reader.run_gathered(&table, &[0, 1, 3], &[3], &mut [0.0; 24], &mut Work::default()).map_err(|e| e.code),
        Err(INVALID_SHAPE)
    );
    // 3 rows of 16 are not whole rows of 32.
    let odd = maker.table(&values[..48]).unwrap();
    let got = reader.run_gathered(&odd, &[0, 0, 0], &[3], &mut [0.0; 24], &mut Work::default());
    assert_eq!(got.map_err(|e| e.code), Err(INVALID_SHAPE));
}

/// A table read by the first run after it is made, many times over: its
/// values must have arrived on the device before that run reads them
/// (CUDA's copy from pageable memory can return before its transfer
/// lands).
#[test]
fn a_new_table_is_whole_for_the_first_run() {
    let m = Model::new(100, 128, 31);
    let lengths = [40, 20, 3, 1];
    let rows: usize = lengths.iter().sum();
    let reference = m.cpu();
    let lstm = m.on(device());
    for seed in 0..40 {
        let mut rng = Rng(500 + seed);
        let values = rng.values(300 * 8 * 128, 2.0);
        let index: Vec<usize> = (0..rows).map(|r| (r * 37 + seed as usize) % 300).collect();
        let mut want = vec![f32::NAN; rows * 256];
        reference.run_gathered(&values, &index, &lengths, &mut want, &mut cpu::Work::default()).unwrap();
        let table = lstm.table(&values).unwrap();
        let mut got = vec![f32::NAN; want.len()];
        lstm.run_gathered(&table, &index, &lengths, &mut got, &mut Work::default()).unwrap();
        let d = max_diff(&got, &want);
        assert!(d <= BOUND, "{:?} table {seed}: {d}", device());
    }
}

/// The bound FASTEST's states are held to, against the CPU form's: the
/// F16 tier's 1e-2 (its weights rounded to F16, a relative 2^-11; every
/// sum and state F32).
const FASTEST_BOUND: f32 = 1e-2;

/// FASTEST on a device runs the same sequences within FASTEST_BOUND of the
/// CPU form at MODEL; on the CPU and on Level Zero it is refused, not run
/// at MODEL.
#[test]
fn fastest_holds_f16_weights_and_stays_near_the_cpu() {
    use turbo::device::Precision;
    let m = Model::new(100, 128, 41);
    if device() == Device::Cpu {
        let e = Lstm::new_at(Device::Cpu, Precision::Fastest, 100, 128, &m.dirs()).err().expect("FASTEST on the CPU");
        assert_eq!(e.code, turbo::status::UNSUPPORTED);
        return;
    }
    if let Device::LevelZero(_) = device() {
        // Level Zero has no F16 weights yet: refused, not run at MODEL.
        let e = Lstm::new_at(device(), Precision::Fastest, 100, 128, &m.dirs()).err().expect("FASTEST on Level Zero");
        assert_eq!(e.code, UNSUPPORTED, "{}", e.message);
        return;
    }
    let lstm = Lstm::new_at(device(), Precision::Fastest, 100, 128, &m.dirs())
        .unwrap_or_else(|e| panic!("FASTEST on {:?}: {}", device(), e.message));
    assert_eq!(lstm.precision(), Precision::Fastest);
    let mut worst = 0f32;
    for (j, lengths) in BATCHES.iter().enumerate() {
        let rows: usize = lengths.iter().sum();
        let x = Rng(300 + j as u64).values(rows * 100, 1.0);
        let mut want = vec![0.0; rows * 256];
        m.cpu().run(&x, lengths, &mut want, &mut cpu::Work::default()).unwrap();
        let mut got = vec![0.0; want.len()];
        lstm.run(&x, lengths, &mut got, &mut Work::default()).unwrap();
        let d = max_diff(&got, &want);
        assert!(d <= FASTEST_BOUND, "lengths {lengths:?}: {d}");
        worst = worst.max(d);
    }
    println!("{:?} at FASTEST: the worst difference from the CPU at MODEL is {worst:e}", device());
}

/// A weight past F16's range is refused at FASTEST when the LSTM is made,
/// on every device, naming the weight, rather than rounded to infinity and
/// every run then refused for its gates; at MODEL the same weights build.
#[test]
fn fastest_refuses_a_weight_past_f16() {
    use turbo::device::Precision;
    let mut m = Model::new(3, 2, 51);
    m.u[1][5] = 70000.0;
    for device in [Device::Cpu, device()] {
        let e = Lstm::new_at(device, Precision::Fastest, 3, 2, &m.dirs()).err().expect("a weight past F16 at FASTEST");
        assert_eq!(e.code, INVALID_ARGUMENT, "{device:?}: {}", e.message);
        assert!(e.message.contains("70000") && e.message.contains("F16"), "{}", e.message);
    }
    m.u[1][5] = 65504.0;
    if let Device::Cuda(_) = device() {
        assert!(Lstm::new_at(device(), Precision::Fastest, 3, 2, &m.dirs()).is_ok(), "F16's largest value is held");
    }
    assert!(Lstm::new_at(device(), Precision::Model, 3, 2, &m.dirs()).is_ok(), "MODEL holds any finite weight");
}

/// The F16 range check's edges, on the CPU alone so a build without a GPU
/// runs them: at FASTEST a largest weight of 65504 (F16's largest value)
/// passes the check and meets the CPU's own refusal, UNSUPPORTED, while
/// one past it either way, or far past it below zero, is refused for its
/// range, INVALID_ARGUMENT.
#[test]
fn the_f16_range_check_is_exact_at_its_edges() {
    use turbo::device::Precision;
    for (v, want) in [
        (65504.0, UNSUPPORTED),
        (-65504.0, UNSUPPORTED),
        (65505.0, INVALID_ARGUMENT),
        (-65505.0, INVALID_ARGUMENT),
        (-70000.0, INVALID_ARGUMENT),
    ] {
        let mut m = Model::new(3, 2, 52);
        m.w[0][4] = v;
        let e = Lstm::new_at(Device::Cpu, Precision::Fastest, 3, 2, &m.dirs()).err().expect("FASTEST on the CPU");
        assert_eq!(e.code, want, "a weight of {v}: {}", e.message);
    }
}
