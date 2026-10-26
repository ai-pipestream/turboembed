//! turbo::device::lstm on Level Zero, where a run fails partway (#860):
//! an append refused after others were queued leaves nothing running on
//! memory that is then freed or written, so the next run on the same
//! workspace is the CPU form's again and every allocation is freed when
//! the LSTMs and workspaces go. Without a listed device it says it was
//! skipped; with TURBO_TEST_REQUIRE_LEVELZERO=1 it fails instead.

#![cfg(feature = "levelzero")]

use turbo::cpu::lstm as cpu;
use turbo::device::Device;
use turbo::device::lstm::{Lstm, Weights, Work};

fn values(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

#[test]
fn a_run_failed_partway_leaves_the_workspace_and_memory_whole() {
    let (input, hidden) = (30, 24);
    let g = 4 * hidden;
    let mut seed = 77;
    let (w0, u0, b0) =
        (values(&mut seed, g * input, 0.4), values(&mut seed, g * hidden, 0.4), values(&mut seed, g, 0.5));
    let (w1, u1, b1) =
        (values(&mut seed, g * input, 0.4), values(&mut seed, g * hidden, 0.4), values(&mut seed, g, 0.5));
    let dirs = [Weights { w: &w0, u: &u0, b: &b0, reverse: false }, Weights { w: &w1, u: &u1, b: &b1, reverse: true }];
    let before = turbo::levelzero::live_allocations();
    let lstm = match Lstm::new(Device::LevelZero(0), input, hidden, &dirs) {
        Ok(l) => l,
        Err(e) if std::env::var("TURBO_TEST_REQUIRE_LEVELZERO").as_deref() != Ok("1") => {
            println!("skipped: no LSTM on levelzero device 0 ({})", e.message);
            return;
        }
        Err(e) => panic!("TURBO_TEST_REQUIRE_LEVELZERO=1: {}", e.message),
    };
    let lengths = [40, 33, 12, 1];
    let rows: usize = lengths.iter().sum();
    let x = values(&mut seed, rows * input, 1.0);
    let mut want = vec![0.0; rows * 2 * hidden];
    cpu::Lstm::new(input, hidden, &dirs).unwrap().run(&x, &lengths, &mut want, &mut cpu::Work::default()).unwrap();
    let mut w = Work::default();
    for fail_at in [1, 2, 20, 40] {
        turbo::levelzero::fail_lstm_append(Some(fail_at));
        let mut out = vec![0.0; want.len()];
        let e = lstm.run(&x, &lengths, &mut out, &mut w).expect_err("an injected failure");
        assert_eq!(e.code, turbo::status::RUNTIME, "{}", e.message);
        turbo::levelzero::fail_lstm_append(None);
        // The same workspace again, and a table made after the failure.
        lstm.run(&x, &lengths, &mut out, &mut w).unwrap();
        let worst = out.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(worst <= 1e-5, "after a failure at append {fail_at}: {worst}");
        let table = lstm.table(&values(&mut seed, 5 * 8 * hidden, 1.0)).unwrap();
        lstm.run_gathered(&table, &vec![3; rows], &lengths, &mut out, &mut w).unwrap();
    }
    drop((lstm, w));
    assert_eq!(turbo::levelzero::live_allocations(), before, "every LSTM allocation is freed");
}
