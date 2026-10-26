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
        // FASTEST has five tilings (four tiles and auto) and three
        // attention kernels (staged, registers and auto); the others one path.
        let want = if p == TURBO_PRECISION_FASTEST { "linear=auto;attn=auto;forced=" } else { "" };
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
    for (line, precision, floor, reported) in [
        ("linear=h32", TURBO_PRECISION_FASTEST, FASTEST_FLOOR, "linear=h32;attn=auto;forced=linear"),
        ("linear=h64", TURBO_PRECISION_FASTEST, FASTEST_FLOOR, "linear=h64;attn=auto;forced=linear"),
        ("linear=h128", TURBO_PRECISION_FASTEST, FASTEST_FLOOR, "linear=h128;attn=auto;forced=linear"),
        ("linear=h32n32", TURBO_PRECISION_FASTEST, FASTEST_FLOOR, "linear=h32n32;attn=auto;forced=linear"),
        ("linear=auto", TURBO_PRECISION_FASTEST, FASTEST_FLOOR, "linear=auto;attn=auto;forced=linear"),
        ("attn=staged", TURBO_PRECISION_FASTEST, FASTEST_FLOOR, "linear=auto;attn=staged;forced=attn"),
        ("attn=registers", TURBO_PRECISION_FASTEST, FASTEST_FLOOR, "linear=auto;attn=registers;forced=attn"),
        ("linear=f32", TURBO_PRECISION_EXACT, 0.9999, "linear=f32;forced=linear"),
    ] {
        let s = make(g.m, Some(line), precision, TURBO_AUTOTUNE_OFF).unwrap();
        let i = s.info();
        assert_eq!((field(&i.choices), i.tuned), (reported.to_owned(), TURBO_TUNED_FORCED));
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
    for (line, reported) in [
        ("linear=h32;forced=linear", "linear=h32;attn=auto;forced=linear"),
        ("linear=h32;attn=registers;forced=", "linear=h32;attn=registers;forced=linear,attn"),
    ] {
        let i = make(l.m, Some(line), fastest, TURBO_AUTOTUNE_OFF).unwrap().info();
        assert_eq!(field(&i.choices), reported);
    }
    let exact = TURBO_PRECISION_EXACT;
    for (line, precision) in [("linear=f32", fastest), ("linear=h64", exact), ("attn=registers", exact)] {
        let e = make(l.m, Some(line), precision, TURBO_AUTOTUNE_OFF).err().expect(line);
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{line}: {e:?}");
    }
    for line in ["linear=fast", "attn=fast"] {
        let e = make(l.m, Some(line), fastest, TURBO_AUTOTUNE_OFF).err().expect(line);
        assert_eq!(e.code, INVALID_ARGUMENT, "{line}: {e:?}");
    }
}

/// The session's vectors for t.
fn vectors(s: &Session, t: &Tokens) -> Vec<Vec<f32>> {
    s.write_tokens(&t.batch(), None).unwrap();
    s.run().unwrap().rows()
}

/// A tuned FASTEST session times each choice at each token bin it can
/// hold and runs one of them, or keeps the default when the GPU's times
/// stay apart. Its line, forced back, makes a session of the same kernels:
/// the same vectors, bit for bit, at a few tokens, at 160 and at the
/// largest shape, each within FASTEST's floor of the CPU. EXACT has nothing
/// to time.
#[test]
fn a_tuned_session_runs_one_of_the_choices_and_forces_back() {
    let _t = turn();
    let Some(_) = metal_device("a_tuned_session_runs_one_of_the_choices_and_forces_back") else { return };
    let dir = tiny_bundle();
    let (g, c) = (on_metal(&dir), Loaded::load(&dir).unwrap());
    let cpu = Session::create(c.m, None).unwrap();
    let s = make(g.m, None, TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_RETUNE).unwrap();
    let i = s.info();
    assert!([TURBO_TUNED_MEASURED, TURBO_TUNED_DEFAULT].contains(&i.tuned), "{}", i.tuned);
    let line = field(&i.choices);
    let tiles = ["h32", "h64", "h128", "h32n32", "auto"];
    assert!(tiles.iter().any(|t| line == format!("linear={t};attn=auto;forced=")), "{line}");
    let forced = make(g.m, Some(&line), TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_OFF).unwrap();
    assert_eq!(field(&forced.info().choices), line.replace("forced=", "forced=linear,attn"));
    // The tuning runs wrote the session's memory: its first run is whole.
    for (what, t) in shapes(&dir, i.max_batch as usize, i.max_seq as usize) {
        against_cpu(&s, &cpu, &t, FASTEST_FLOOR, &format!("tuned, {what}"));
        assert_eq!(vectors(&s, &t), vectors(&forced, &t), "{line}, {what}: the forced line runs the same kernels");
    }
    let i = make(g.m, None, TURBO_PRECISION_EXACT, TURBO_AUTOTUNE_RETUNE).unwrap().info();
    assert_eq!((field(&i.choices), i.tuned), (String::new(), TURBO_TUNED_DEFAULT));
}

/// A model with a finite feed-forward output weight past F16's range
/// (65504), in one layer and then in each: FASTEST computes such a layer in F32, as CUDA
/// does, and the others in F16, and stays within its floor of the CPU at
/// every shape; a forced F16 tiling runs in the other layers. With every
/// layer past the range the session computes in F32 and says so, and a
/// forced F16 tiling is refused naming the precision's field.
#[test]
fn a_weight_past_f16_s_range_computes_its_layer_in_f32() {
    let _t = turn();
    let Some(_) = metal_device("a_weight_past_f16_s_range_computes_its_layer_in_f32") else { return };
    for (layers, dtype) in [(&["layer.0."][..], TURBO_DTYPE_F16), (&["layer.0.", "layer.1."][..], TURBO_DTYPE_F32)] {
        let mut weights = tiny_weights(0);
        for at in layers {
            let suffix = format!("{at}output.dense.weight");
            let w = weights.iter_mut().find(|t| t.name.ends_with(&suffix)).unwrap();
            w.data[..4].copy_from_slice(&100000f32.to_le_bytes());
        }
        let mut f = Fixture::model(&format!("metal-f16-past-range-{}", layers.len()));
        f.weights("weights/model.safetensors", &weights);
        let (g, c) = (f.load_on(metal).unwrap(), f.load().unwrap());
        let cpu = Session::create(c.m, None).unwrap();
        for choices in [None, Some("linear=h64")] {
            let s = make(g.m, choices, TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_OFF);
            if dtype == TURBO_DTYPE_F32 && choices.is_some() {
                let e = s.err().expect("a forced F16 tiling with no F16 layer is refused");
                assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{choices:?}: {e:?}");
                continue;
            }
            let s = s.unwrap();
            let i = s.info();
            assert_eq!(i.compute_dtype, dtype, "{} layers past the range, {choices:?}", layers.len());
            for (what, t) in shapes(&f.dir, i.max_batch as usize, i.max_seq as usize) {
                let what = format!("{} layers past the range, {choices:?}, {what}", layers.len());
                let lowest = against_cpu(&s, &cpu, &t, FASTEST_FLOOR, &what);
                println!("{what}: 1 - lowest cosine with the cpu {:.3e}", 1.0 - lowest);
            }
        }
    }
}

/// Heads FASTEST's F16 attention takes past the 64-wide kernel: wider than
/// 64, not a multiple of 8, not a multiple of 4, and 128, the widest; each
/// attention kernel against the CPU within FASTEST's floor at every
/// shape. A head of 136 runs in F32 and reports no attention choice; it is
/// held to the CPU at the two smaller shapes.
#[test]
fn heads_past_64_or_not_a_multiple_of_8_run_f16_attention() {
    let _t = turn();
    let Some(_) = metal_device("heads_past_64_or_not_a_multiple_of_8_run_f16_attention") else { return };
    for (hidden, heads) in [(160u64, 2u64), (144, 4), (104, 4), (256, 2), (272, 2)] {
        let mut m = model_manifest();
        m["architecture"]["layers"] = serde_json::json!(2);
        m["architecture"]["hidden"] = serde_json::json!(hidden);
        m["architecture"]["heads"] = serde_json::json!(heads);
        m["architecture"]["intermediate"] = serde_json::json!(2 * hidden);
        m["embed"]["dim"] = serde_json::json!(hidden);
        let mut f = Fixture::new(&format!("metal-f16-heads-{hidden}-{heads}"), m);
        f.weights("weights/model.safetensors", &bert_weights(2, hidden, 2 * hidden, 1, 0.5));
        f.write();
        let (g, c) = (on_metal(&f.dir), Loaded::load(&f.dir).unwrap());
        let cpu = Session::create(c.m, None).unwrap();
        let hd = hidden / heads;
        for line in ["attn=staged", "attn=registers", "linear=auto"] {
            let s = make(g.m, Some(line), TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_OFF);
            if hd > 128 {
                if line.starts_with("attn") {
                    let e = s.err().expect("attention past 128 runs in F32");
                    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{line}: {e:?}");
                    continue;
                }
                assert_eq!(field(&s.as_ref().unwrap().info().choices), "linear=auto;forced=linear");
            }
            let s = s.unwrap();
            let i = s.info();
            for (what, t) in shapes(&f.dir, i.max_batch as usize, i.max_seq as usize) {
                // The F32 narrow kernel, one query at a time, at the largest
                // shape runs long enough for macOS to stop it on a busy
                // desktop (kIOGPUCommandBufferCallbackErrorImpactingInteractivity);
                // it is main's kernel, not this test's subject.
                if hd > 128 && what == "largest" {
                    continue;
                }
                let what = format!("heads of {hd}, {line}, {what}");
                let lowest = against_cpu(&s, &cpu, &t, FASTEST_FLOOR, &what);
                println!("{what}: 1 - lowest cosine with the cpu {:.3e}", 1.0 - lowest);
            }
        }
    }
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

/// How close FASTEST comes to the CPU on the masked rows below. They agree
/// to 1 - 2.7e-11 on the M2. A kernel that let the masked keys into the
/// scores, while the pooling still left those tokens out, comes to only
/// 1 - 6.0e-7 there: the tiny bundle's attention moves the vectors little,
/// so the bound sits well inside that.
const MASKED_FLOOR: f64 = 1.0 - 1e-8;

/// Keys masked in the middle of a row at FASTEST, in the first chunk of 32
/// keys and in later ones, for each choice: every row comes within
/// MASKED_FLOOR of the CPU, which drops the same keys. The CPU's vectors
/// with those keys masked and with them live are first checked to be at
/// least 100 times MASKED_FLOOR's distance apart, so the gaps change what
/// the rows mean.
#[test]
fn masked_keys_inside_a_row_are_left_out_at_fastest() {
    let _t = turn();
    let Some(_) = metal_device("masked_keys_inside_a_row_are_left_out_at_fastest") else { return };
    let dir = tiny_bundle();
    let (g, c) = (on_metal(&dir), Loaded::load(&dir).unwrap());
    let cpu = Session::create(c.m, None).unwrap();
    let live = rows_of(&dir, &[64, 40, 9, 64]);
    let mut t = rows_of(&dir, &[64, 40, 9, 64]);
    let seq = t.seq as usize;
    let gapped = [(0usize, &[5usize, 6, 40, 41, 42][..]), (1, &[0, 33][..]), (3, &[31, 32, 50][..])];
    for (r, gaps) in gapped {
        for &p in gaps {
            t.mask[r * seq + p] = 0;
        }
    }
    let (masked, unmasked) = (vectors(&cpu, &t), vectors(&cpu, &live));
    for (r, _) in gapped {
        let apart = 1.0 - cosine(&masked[r], &unmasked[r]);
        assert!(apart >= 100.0 * (1.0 - MASKED_FLOOR), "row {r}: masking moves the cpu's vector by only {apart:.3e}");
    }
    let awkward = awkward_rows(&dir);
    let lines =
        ["linear=h32", "linear=h64", "linear=h128", "linear=h32n32", "linear=auto", "attn=staged", "attn=registers"];
    for line in lines {
        let s = make(g.m, Some(line), TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_OFF).unwrap();
        let lowest = against_cpu(&s, &cpu, &t, MASKED_FLOOR, &format!("{line}, masked keys"));
        println!("{line}, masked keys inside rows: 1 - lowest cosine with the cpu {:.3e}", 1.0 - lowest);
        let lowest = against_cpu(&s, &cpu, &awkward, MASKED_FLOOR, &format!("{line}, awkward rows"));
        println!("{line}, awkward rows: 1 - lowest cosine with the cpu {:.3e}", 1.0 - lowest);
    }
}

/// attn=auto runs the staged kernel up to 512 packed tokens and the
/// register kernel past them: its vectors are bit-equal to the forced
/// kernel's on each side of the line, and the two kernels' vectors differ
/// there, so the check can fail.
#[test]
fn attn_auto_switches_past_512_packed_tokens() {
    let _t = turn();
    let Some(_) = metal_device("attn_auto_switches_past_512_packed_tokens") else { return };
    let dir = tiny_bundle();
    let g = on_metal(&dir);
    let fastest = TURBO_PRECISION_FASTEST;
    let auto = make(g.m, None, fastest, TURBO_AUTOTUNE_OFF).unwrap();
    let staged = make(g.m, Some("attn=staged"), fastest, TURBO_AUTOTUNE_OFF).unwrap();
    let registers = make(g.m, Some("attn=registers"), fastest, TURBO_AUTOTUNE_OFF).unwrap();
    // Rows of up to 64 tokens, the session's max_seq, some short and some
    // with keys masked inside them (where the two kernels' sums part), pack
    // to 512; one more token packs to 513, padded to 544.
    let lens = [64, 40, 9, 64, 64, 64, 64, 64, 64, 15];
    let gapped = |lens: &[usize]| {
        let mut t = rows_of(&dir, lens);
        let seq = t.seq as usize;
        for (r, p) in [(0, 5), (0, 40), (1, 0), (1, 33), (3, 31), (3, 50)] {
            t.mask[r * seq + p] = 0;
        }
        t
    };
    let (at, past) = (gapped(&lens), gapped(&[&lens[..], &[1]].concat()));
    assert_ne!(vectors(&staged, &at), vectors(&registers, &at), "the kernels' bits differ at 512 tokens");
    assert_ne!(vectors(&staged, &past), vectors(&registers, &past), "the kernels' bits differ at 513 tokens");
    assert_eq!(vectors(&auto, &at), vectors(&staged, &at), "512 packed tokens run staged");
    assert_eq!(vectors(&auto, &past), vectors(&registers, &past), "513 packed tokens run registers");
}
