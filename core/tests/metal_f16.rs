//! The Metal backend's linear layers in F16 at FASTEST: what a session
//! reports of its compute dtype and kernel choice, each choice held to
//! FASTEST's floor against the CPU at the few-token, tiled and largest
//! shapes, and the choices line forced, refused and tuned.
//!
//! Built with the `metal` feature only, on macOS. A test that needs a
//! device that runs the kernels says it was skipped, and passes, when the
//! backend lists none. With TURBO_TEST_REQUIRE_METAL=1 it fails instead.
//! docs/metal.md says how to run them.

#![cfg(feature = "metal")]

mod common;
mod metal_support;

use common::*;
use metal_support::*;
use turbo::status::{INVALID_ARGUMENT, UNSUPPORTED_OPTION};
use turbo::*;

/// FASTEST's floor, as CUDA and Level Zero hold it.
const FASTEST_FLOOR: f64 = 0.999;

/// A session on `m` with `choices` read in place of TURBO_METAL_CHOICES.
fn make(m: *mut turbo_model, choices: Option<&str>, precision: u32, tuning: u32) -> Result<Session, Failure> {
    turbo::metal::use_choices(choices);
    let mut d = session_desc(0, 0, precision);
    d.tuning = tuning;
    let s = Session::create(m, Some(&d));
    turbo::metal::use_choices(None);
    s
}

/// Rows of the given lengths, with ids spread over the vocabulary.
fn rows_of(dir: &std::path::Path, lens: &[usize]) -> Tokens {
    let vocab = Tok::create(dir).unwrap().info().vocab_size as usize;
    let rows: Vec<Vec<i32>> = lens
        .iter()
        .enumerate()
        .map(|(r, &len)| (0..len).map(|p| (1000 + (r * 131 + p * 17) % (vocab - 1000)) as i32).collect())
        .collect();
    Tokens::new(&rows, 0)
}

/// The session's vectors for `t` against the CPU's, each row's cosine at
/// least `floor`; returns the lowest.
fn against_cpu(s: &Session, cpu: &Session, t: &Tokens, floor: f64, what: &str) -> f64 {
    s.write_tokens(&t.batch(), None).unwrap();
    let got = s.run().unwrap().rows();
    cpu.write_tokens(&t.batch(), None).unwrap();
    let want = cpu.run().unwrap().rows();
    let mut lowest = 1f64;
    for (r, (a, b)) in got.iter().zip(&want).enumerate() {
        let c = cosine(a, b);
        lowest = lowest.min(c);
        assert!(c >= floor, "{what}: row {r}: cosine {c} with the cpu");
    }
    lowest
}

/// The shapes each choice runs: few tokens (the small kernel), packed
/// tokens that are an odd multiple of 32 (a 64-row tile half past m), and
/// the session's largest, with rows that end early.
fn shapes(dir: &std::path::Path, max_batch: usize, max_seq: usize) -> Vec<(&'static str, Tokens)> {
    let largest: Vec<usize> = (0..max_batch).map(|r| max_seq - (r * 7) % (max_seq / 2)).collect();
    let mut odd = vec![max_seq; 160 / max_seq];
    odd.push(160 - odd.len() * max_seq - 3);
    odd.retain(|&l| l > 0);
    vec![
        ("few tokens", rows_of(dir, &[9, 1, 30])),
        ("160 tokens", rows_of(dir, &odd)),
        ("largest", rows_of(dir, &largest)),
    ]
}

#[test]
fn fastest_computes_in_f16_and_says_so() {
    let _t = turn();
    let Some(_) = metal_device("fastest_computes_in_f16_and_says_so") else { return };
    let l = on_metal(&tiny_bundle());
    let rt = std::mem::ManuallyDrop::new(Rt(l.rt));
    let i = metal(l.rt);
    for (p, dtype) in [
        (TURBO_PRECISION_MODEL, TURBO_DTYPE_F32),
        (TURBO_PRECISION_EXACT, TURBO_DTYPE_F32),
        (TURBO_PRECISION_FASTEST, TURBO_DTYPE_F16),
    ] {
        assert_eq!(rt.capability(i, p).dtype, dtype, "precision {p}");
        let info = make(l.m, None, p, TURBO_AUTOTUNE_OFF).unwrap().info();
        assert_eq!(info.compute_dtype, dtype, "precision {p}");
        // FASTEST has three tilings to choose from; the others one path.
        let want = if p == TURBO_PRECISION_FASTEST { "linear=h64;forced=" } else { "" };
        assert_eq!((field(&info.choices), info.tuned), (want.to_owned(), TURBO_TUNED_DEFAULT), "precision {p}");
    }
}

/// Each FASTEST choice, forced, against the CPU at every shape, and EXACT
/// still at its own floor.
#[test]
fn each_f16_choice_matches_the_cpu_within_fastest_s_floor() {
    let _t = turn();
    let Some(_) = metal_device("each_f16_choice_matches_the_cpu_within_fastest_s_floor") else { return };
    let dir = tiny_bundle();
    let (g, c) = (on_metal(&dir), Loaded::load(&dir).unwrap());
    let cpu = Session::create(c.m, None).unwrap();
    for (choice, precision, floor) in [
        ("h32", TURBO_PRECISION_FASTEST, FASTEST_FLOOR),
        ("h64", TURBO_PRECISION_FASTEST, FASTEST_FLOOR),
        ("h128", TURBO_PRECISION_FASTEST, FASTEST_FLOOR),
        ("f32", TURBO_PRECISION_EXACT, 0.9999),
    ] {
        let line = format!("linear={choice}");
        let s = make(g.m, Some(&line), precision, TURBO_AUTOTUNE_OFF).unwrap();
        let i = s.info();
        assert_eq!((field(&i.choices), i.tuned), (format!("{line};forced=linear"), TURBO_TUNED_FORCED));
        for (what, t) in shapes(&dir, i.max_batch as usize, i.max_seq as usize) {
            let lowest = against_cpu(&s, &cpu, &t, floor, &format!("{line} {what}"));
            println!("{line}, {what}: 1 - lowest cosine with the cpu {:.3e}", 1.0 - lowest);
        }
    }
}

/// A reported line forces the same kernels back; a kernel outside the
/// precision's numerics is refused naming the precision; any other item
/// is refused as an invalid argument.
#[test]
fn the_choices_line_forces_back_and_refuses_what_it_cannot_run() {
    let _t = turn();
    let Some(_) = metal_device("the_choices_line_forces_back_and_refuses_what_it_cannot_run") else { return };
    let l = on_metal(&tiny_bundle());
    let fastest = TURBO_PRECISION_FASTEST;
    let i = make(l.m, Some("linear=h32;forced=linear"), fastest, TURBO_AUTOTUNE_OFF).unwrap().info();
    assert_eq!(field(&i.choices), "linear=h32;forced=linear");
    for (line, precision) in [("linear=f32", fastest), ("linear=h64", TURBO_PRECISION_EXACT)] {
        let e = make(l.m, Some(line), precision, TURBO_AUTOTUNE_OFF).err().expect(line);
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{line}: {e:?}");
    }
    let e = make(l.m, Some("linear=fast"), fastest, TURBO_AUTOTUNE_OFF).err().unwrap();
    assert_eq!(e.code, INVALID_ARGUMENT, "{e:?}");
}

/// A tuned FASTEST session times each tiling and runs one of them, or
/// keeps the default when the GPU's times stay apart; EXACT has nothing
/// to time.
#[test]
fn a_tuned_session_runs_one_of_the_tilings() {
    let _t = turn();
    let Some(_) = metal_device("a_tuned_session_runs_one_of_the_tilings") else { return };
    let dir = tiny_bundle();
    let (g, c) = (on_metal(&dir), Loaded::load(&dir).unwrap());
    let cpu = Session::create(c.m, None).unwrap();
    let s = make(g.m, None, TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_RETUNE).unwrap();
    let i = s.info();
    assert!([TURBO_TUNED_MEASURED, TURBO_TUNED_DEFAULT].contains(&i.tuned), "{}", i.tuned);
    let lines = ["linear=h32;forced=", "linear=h64;forced=", "linear=h128;forced="];
    assert!(lines.contains(&field(&i.choices).as_str()), "{}", field(&i.choices));
    // The tuning runs wrote the session's memory: its first run is whole.
    let (_, t) = shapes(&dir, i.max_batch as usize, i.max_seq as usize).pop().unwrap();
    against_cpu(&s, &cpu, &t, FASTEST_FLOOR, "tuned");
    let i = make(g.m, None, TURBO_PRECISION_EXACT, TURBO_AUTOTUNE_RETUNE).unwrap().info();
    assert_eq!((field(&i.choices), i.tuned), (String::new(), TURBO_TUNED_DEFAULT));
}

/// A model with a finite linear weight past F16's range (65504): its
/// FASTEST session, whose numerics allow only F16 kernels, is refused
/// naming the precision, forced or not, since narrowing that weight would
/// make it infinite; EXACT runs it in F32 with finite vectors.
#[test]
fn a_weight_past_f16_s_range_refuses_fastest() {
    let _t = turn();
    let Some(_) = metal_device("a_weight_past_f16_s_range_refuses_fastest") else { return };
    let mut weights = tiny_weights(0);
    let q = weights.iter_mut().find(|t| t.name.ends_with("layer.0.attention.self.query.weight")).unwrap();
    q.data[..4].copy_from_slice(&100000f32.to_le_bytes());
    let mut f = Fixture::model("metal-f16-past-range");
    f.weights("weights/model.safetensors", &weights);
    let l = f.load_on(metal).unwrap();
    for choices in [None, Some("linear=h64")] {
        let e = make(l.m, choices, TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_OFF).err().expect("FASTEST refused");
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{choices:?}: {e:?}");
    }
    let exact = make(l.m, None, TURBO_PRECISION_EXACT, TURBO_AUTOTUNE_OFF).unwrap();
    assert_eq!(exact.info().compute_dtype, TURBO_DTYPE_F32);
    let v = exact.embed(&TEXTS, None).unwrap();
    assert!(v.iter().flatten().all(|x| x.is_finite()), "EXACT's vectors are finite");
}

/// F16's largest finite value: a FASTEST tensor that reaches it would be
/// infinite.
const F16_MAX: f32 = 65504.0;

/// A real bundle at FASTEST at its longest rows, 8 of up to max_seq
/// tokens (text, rows that end early, and ids spread over the
/// vocabulary), against the CPU at EXACT: every row's cosine reaches
/// FASTEST's floor, and the largest magnitude of each tensor FASTEST stores
/// in F16 stays inside F16's range. Prints both, for docs/metal.md.
#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn a_real_bundle_at_its_longest_rows_stays_inside_f16() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = metal_device("a_real_bundle_at_its_longest_rows_stays_inside_f16") else { return };
    let (g, c) = (on_metal(&dir), Loaded::load(&dir).unwrap());
    let seq = Session::create(g.m, None).unwrap().info().max_seq as usize;
    let tok = Tok::create(&dir).unwrap();
    let vocab = tok.info().vocab_size as usize;
    let text = tok.row(&[long_text(), long_text().to_uppercase(), long_text()].join(" 1234567890 "), None).unwrap();
    assert!(text.len() >= seq, "{} tokens of text for rows of {seq}", text.len());
    let mut rows: Vec<Vec<i32>> =
        [seq, seq, seq - 1, seq * 3 / 4, seq / 2, 9].iter().map(|&n| text[..n].to_vec()).collect();
    rows.extend((0..2).map(|r| (0..seq).map(|p| (1000 + (r * 131 + p * 17) % (vocab - 1000)) as i32).collect()));
    let t = Tokens::new(&rows, 0);
    let desc = |p| session_desc(rows.len() as u32, seq as u32, p);
    turbo::metal::track_peaks(true);
    let fastest = Session::create(g.m, Some(&desc(TURBO_PRECISION_FASTEST)));
    turbo::metal::track_peaks(false);
    let fastest = fastest.unwrap();
    assert_eq!(fastest.info().compute_dtype, TURBO_DTYPE_F16);
    fastest.write_tokens(&t.batch(), None).unwrap();
    let got = fastest.run().unwrap().rows();
    let peaks = turbo::metal::peaks();
    let cpu = Session::create(c.m, Some(&desc(TURBO_PRECISION_EXACT))).unwrap();
    cpu.write_tokens(&t.batch(), None).unwrap();
    let want = cpu.run().unwrap().rows();
    assert_eq!((got.len(), want.len()), (rows.len(), rows.len()), "a vector for every row");
    let mut lowest = 1f64;
    for (r, (a, b)) in got.iter().zip(&want).enumerate() {
        let cos = cosine(a, b);
        lowest = lowest.min(cos);
        assert!(cos >= FASTEST_FLOOR, "row {r}: cosine {cos} with the cpu");
    }
    println!(
        "{}: {} rows of up to {seq} tokens, 1 - lowest cosine with the cpu {:.3e}",
        dir.display(),
        rows.len(),
        1.0 - lowest
    );
    // Attention keeps Q, K, V and the context in F32 for heads it has no
    // F16 kernel for, and those four slots then stay 0 together.
    let f16_attention = peaks[..4].iter().any(|&p| p > 0.0);
    for (i, (name, peak)) in turbo::metal::PEAK_NAMES.iter().zip(peaks).enumerate() {
        if i < 4 && !f16_attention {
            println!("  {name}: stored in F32");
            continue;
        }
        println!("  {name}: largest magnitude {peak:.1}, {:.1}% of F16's range", 100.0 * peak / F16_MAX);
        assert!(peak > 0.0 && peak < F16_MAX, "{name}: largest magnitude {peak}");
    }
}
