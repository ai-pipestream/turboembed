//! Limits, the largest shape, the linear kernels' choices, the reference cases and the profile.

use super::*;

// ---- Limits and the largest shape ----------------------------------------------------------

/// A session longer than attention's local memory holds on this device is
/// refused by field 2. 65536 positions need 256 KiB of scores per
/// work-group, more than any device gives one. 32000 need 125 KiB, which
/// fits a B70's 128 KiB only without the 4 KiB its driver keeps for the
/// kernel's reductions: that is refused too, where a launch would fail. A
/// session just inside the limit runs a row of its full length.
#[test]
fn more_tokens_than_attention_holds_are_refused_by_field() {
    let _t = turn();
    let Some(_) = gpu_device("more_tokens_than_attention_holds_are_refused_by_field") else { return };
    let positions = 65536u64;
    let mut m = model_manifest();
    m["architecture"]["max_positions"] = json!(positions);
    m["embed"]["max_seq"] = json!(positions);
    // A case longer than max_seq, which a manifest needs: 600 paragraphs of
    // about 120 tokens.
    m["reference"]["cases"][8]["text"] = json!(vec![PARAGRAPH; 600].join(" "));
    let mut f = Fixture::new("levelzero-positions", m);
    let weights: Vec<Tensor> = tiny_weights(0)
        .into_iter()
        .map(|t| {
            if t.name != "embeddings.position_embeddings.weight" {
                return t;
            }
            let n = positions as usize * 8;
            let data = (0..n).flat_map(|i| (((i * 37 % 101) as f32 - 50.0) / 500.0).to_le_bytes()).collect();
            Tensor { shape: vec![positions, 8], data, ..t }
        })
        .collect();
    f.weights("weights/model.safetensors", &weights);
    let l = f.load_on(gpu).unwrap();
    let e = Session::create(l.m, Some(&session_desc(1, positions as u32, 0))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 2), "{e:?}");
    assert!(e.message.contains("local memory"), "{e:?}");
    let e = Session::create(l.m, Some(&session_desc(1, 32000, 0))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 2), "{e:?}");
    let s = Session::create(l.m, Some(&session_desc(1, 30000, 0))).unwrap();
    let row: Vec<i32> = (0..30000).map(|i| 1000 + i % 500).collect();
    s.write_tokens(&Tokens::new(&[row], 0).batch(), None).unwrap();
    let v = s.run().unwrap().rows();
    assert!(v[0].iter().all(|x| x.is_finite()), "{:?}", v[0]);
}

/// TURBO_TEST_BUNDLE, a relative path read from the workspace root, as
/// tests/conformance.rs reads it.
fn named_bundle() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE")?);
    Some(if p.is_absolute() { p } else { std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(p) })
}

/// One run at the session's largest shape, max_batch rows of max_seq
/// tokens, on the device and on the CPU: every row's cosine reaches
/// 0.9999. Rows are full or end early, so the padding is exercised too.
fn largest_shape_matches_the_cpu(dir: &std::path::Path) {
    let (g, c) = (on_gpu(dir), Loaded::load(dir).unwrap());
    let cs = Session::create(c.m, None).unwrap();
    // F32 at MODEL, and F16 on the matrix engines at FASTEST, each held
    // to its dtype's floor.
    for (precision, floor) in [(TURBO_PRECISION_MODEL, 0.9999), (TURBO_PRECISION_FASTEST, 0.999)] {
        let gs = Session::create(g.m, Some(&session_desc(0, 0, precision))).unwrap();
        largest_shape_on(dir, &gs, &cs, floor);
    }
}

fn largest_shape_on(dir: &std::path::Path, gs: &Session, cs: &Session, floor: f64) {
    let si = gs.info();
    let (batch, seq) = (si.max_batch as usize, si.max_seq as usize);
    let vocab = Tok::create(dir).unwrap().info().vocab_size as usize;
    let rows: Vec<Vec<i32>> = (0..batch)
        .map(|r| {
            let len = seq - (r * 7) % (seq / 2);
            (0..len).map(|p| (1000 + (r * 131 + p * 17) % (vocab - 1000)) as i32).collect()
        })
        .collect();
    let t = Tokens::new(&rows, 0);
    assert_eq!((t.batch as usize, t.seq as usize), (batch, seq));
    gs.write_tokens(&t.batch(), None).unwrap();
    let got = gs.run().unwrap().rows();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    let mut lowest = 1f64;
    for (r, (a, b)) in got.iter().zip(&want).enumerate() {
        let cos = cosine(a, b);
        lowest = lowest.min(cos);
        assert!(cos >= floor, "row {r}: cosine {cos} with the cpu");
    }
    println!(
        "{}: {batch} rows of {seq} tokens in dtype {}, 1 - lowest cosine with the cpu {:.3e}",
        dir.display(),
        gs.info().compute_dtype,
        1.0 - lowest
    );
}

/// At FASTEST a row's vector does not depend on the rows around it, on
/// either side of the LayerNorm kernels' own switch at 256 tokens: the
/// linear layers never split their sums, each projection back to the
/// hidden width takes its LayerNorm in its epilogue at every batch size,
/// and the one-kernel feed-forward block of large batches computes what
/// its two kernels do. So a row of a few tokens
/// alone, which runs the 8-token tiles, gives the same bits as the same row
/// among others, which run the wider ones.
/// oneDNN's kernel for a batch past the few-token kernels sums in its own
/// order, so this holds for the backend's own kernels, which the test
/// forces.
#[test]
fn a_row_at_fastest_gives_the_same_bits_alone_and_among_others() {
    let _t = turn();
    let Some(_) = gpu_device("a_row_at_fastest_gives_the_same_bits_alone_and_among_others") else { return };
    let g = on_gpu(&tiny_bundle());
    turbo::levelzero::use_choices(Some("linear=own"));
    let s = Session::create(g.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST)));
    turbo::levelzero::use_choices(None);
    let s = s.unwrap();
    assert_eq!(s.info().compute_dtype, TURBO_DTYPE_F16);
    // Rows of 5, 16, 27 and 38 tokens: 86 in all, enough for the fused
    // LayerNorm's groups of several blocks of tokens, and below the
    // LayerNorm kernels' own switch at 256.
    let rows: Vec<Vec<i32>> =
        (0..4i32).map(|r| (0..5 + 11 * r).map(|p| 1000 + (r * 131 + p * 17) % 20000).collect()).collect();
    s.write_tokens(&Tokens::new(&rows, 0).batch(), None).unwrap();
    let together = s.run().unwrap().rows();
    s.write_tokens(&Tokens::new(&rows[..1], 0).batch(), None).unwrap();
    let alone = s.run().unwrap().rows();
    assert_eq!(alone[0], together[0], "a row of 5 tokens alone and among 86");
    // A full batch runs the feed-forward block in one kernel; 8 of its rows
    // run the two kernels. Both are over the LayerNorm kernels' switch.
    let si = s.info();
    let (batch, seq) = (si.max_batch as usize, si.max_seq as usize);
    assert!(batch * seq >= 4096 && (256..4096).contains(&(8 * seq)), "the small model's session spans both");
    let rows: Vec<Vec<i32>> =
        (0..batch).map(|r| (0..seq).map(|p| (1000 + (r * 131 + p * 17) % 20000) as i32).collect()).collect();
    s.write_tokens(&Tokens::new(&rows, 0).batch(), None).unwrap();
    let full = s.run().unwrap().rows();
    s.write_tokens(&Tokens::new(&rows[..8], 0).batch(), None).unwrap();
    let some = s.run().unwrap().rows();
    assert_eq!(some, full[..8], "8 full rows alone and among {batch}");
}

/// Every row full, at FASTEST: the small model's 64 x 64 tokens fill whole
/// groups of the XMX linear kernels and the LayerNorm-fused ones. Then
/// single short rows, for the kernels that take a handful of tokens.
#[test]
fn a_full_batch_at_fastest_matches_the_cpu() {
    let _t = turn();
    let Some(_) = gpu_device("a_full_batch_at_fastest_matches_the_cpu") else { return };
    let dir = tiny_bundle();
    let (g, c) = (on_gpu(&dir), Loaded::load(&dir).unwrap());
    let gs = Session::create(g.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
    let cs = Session::create(c.m, None).unwrap();
    let si = gs.info();
    let (batch, seq) = (si.max_batch as usize, si.max_seq as usize);
    let rows: Vec<Vec<i32>> =
        (0..batch).map(|r| (0..seq).map(|p| (1000 + (r * 131 + p * 17) % 20000) as i32).collect()).collect();
    let t = Tokens::new(&rows, 0);
    gs.write_tokens(&t.batch(), None).unwrap();
    cs.write_tokens(&t.batch(), None).unwrap();
    for (r, (a, b)) in gs.run().unwrap().rows().iter().zip(cs.run().unwrap().rows()).enumerate() {
        let cos = cosine(a, &b);
        assert!(cos >= 0.999, "row {r}: cosine {cos} with the cpu");
    }
    // One short row at a time: 8 tokens at most run the XMX kernels'
    // 8-token tiles.
    for text in TEXTS {
        let (a, b) = (gs.embed(&[text], None).unwrap(), cs.embed(&[text], None).unwrap());
        let cos = cosine(&a[0], &b[0]);
        assert!(cos >= 0.999, "{text:?}: cosine {cos} with the cpu");
    }
}

/// Each precision's sessions with the linear layers' kernels forced each
/// way the build has, held to the dtype's floor against the CPU at the
/// largest shape; then what a session reports of the choice, forced,
/// untuned and tuned.
#[test]
fn each_linear_kernel_matches_the_cpu_and_is_reported() {
    let _t = turn();
    let Some(_) = gpu_device("each_linear_kernel_matches_the_cpu_and_is_reported") else { return };
    let dir = tiny_bundle();
    let (g, c) = (on_gpu(&dir), Loaded::load(&dir).unwrap());
    let cs = Session::create(c.m, None).unwrap();
    let onednn = cfg!(feature = "levelzero-onednn");
    let make = |choices: Option<&str>, precision: u32, tuning: u32| {
        turbo::levelzero::use_choices(choices);
        let mut d = session_desc(0, 0, precision);
        d.tuning = tuning;
        let s = Session::create(g.m, Some(&d));
        turbo::levelzero::use_choices(None);
        s
    };
    for (precision, floor) in
        [(TURBO_PRECISION_MODEL, 0.9999), (TURBO_PRECISION_EXACT, 0.9999), (TURBO_PRECISION_FASTEST, 0.999)]
    {
        for linear in ["own", "onednn"] {
            let line = format!("linear={linear}");
            let s = make(Some(&line), precision, TURBO_AUTOTUNE_OFF);
            if linear == "onednn" && !onednn {
                let e = s.err().expect("linear=onednn without oneDNN");
                assert_eq!(e.code, UNSUPPORTED_OPTION, "{e:?}");
                continue;
            }
            let s = s.unwrap();
            let i = s.info();
            assert_eq!((field(&i.choices), i.tuned), (format!("{line};forced=linear"), TURBO_TUNED_FORCED));
            largest_shape_on(&dir, &s, &cs, floor);
        }
        // Untuned: oneDNN where the build has it, else one path and no
        // choices.
        let i = make(None, precision, TURBO_AUTOTUNE_OFF).unwrap().info();
        let want = if onednn { "linear=onednn;forced=" } else { "" };
        assert_eq!((field(&i.choices), i.tuned), (want.to_owned(), TURBO_TUNED_DEFAULT));
        // Tuned: measured where there are two to time.
        let i = make(None, precision, TURBO_AUTOTUNE_RETUNE).unwrap().info();
        if onednn {
            let line = field(&i.choices);
            assert_eq!(i.tuned, TURBO_TUNED_MEASURED, "{line}");
            assert!(line.starts_with("linear=") && line.ends_with(";forced="), "{line}");
            // Each bin's choice, as reported, forces the same kernels back.
            let back = make(Some(&line), precision, TURBO_AUTOTUNE_OFF).unwrap().info();
            assert_eq!(field(&back.choices), format!("{line}linear"));
        } else {
            assert_eq!((field(&i.choices), i.tuned), (String::new(), TURBO_TUNED_DEFAULT));
        }
    }
    // A reported line forces back; anything else is refused.
    assert!(make(Some("linear=own;forced=linear"), TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_OFF).is_ok());
    if onednn {
        let i = make(Some("le256:linear=own"), TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_OFF).unwrap().info();
        assert_eq!(field(&i.choices), "linear=onednn;le256:linear=own;forced=linear");
    }
    let e = make(Some("linear=fast"), TURBO_PRECISION_FASTEST, TURBO_AUTOTUNE_OFF).err().unwrap();
    assert_eq!(e.code, INVALID_ARGUMENT, "{e:?}");
}

#[test]
fn the_largest_shape_matches_the_cpu() {
    let _t = turn();
    let Some(_) = gpu_device("the_largest_shape_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&tiny_bundle());
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn the_largest_shape_of_a_real_bundle_matches_the_cpu() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = gpu_device("the_largest_shape_of_a_real_bundle_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&dir);
}

/// The bundle's reference cases on the Level Zero device at each
/// precision, against the upstream vectors, within the tolerance of the
/// compute dtype the session reports (tests/common/conformance.rs).
fn reference_cases_match_upstream(dir: &std::path::Path) {
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
        conformance::check(dir, gpu, p);
    }
}

#[test]
fn the_reference_cases_match_upstream() {
    let _t = turn();
    let Some(_) = gpu_device("the_reference_cases_match_upstream") else { return };
    reference_cases_match_upstream(&tiny_bundle());
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn the_reference_cases_of_a_real_bundle_match_upstream() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = gpu_device("the_reference_cases_of_a_real_bundle_match_upstream") else { return };
    reference_cases_match_upstream(&dir);
}

/// Where a real bundle's time goes on the device: TURBO_TEST_BUNDLE at
/// 32 rows of 256 tokens, FASTEST then MODEL, 20 runs each after 5 of
/// warm-up, with the runtime's log printed. With TURBO_LEVELZERO_PROFILE
/// set, each context prints its per-kernel totals when it is released.
#[test]
#[ignore = "measures; needs a real bundle directory in TURBO_TEST_BUNDLE; run with --ignored --nocapture"]
fn the_profile_of_a_real_bundle() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = gpu_device("the_profile_of_a_real_bundle") else { return };
    unsafe extern "C" fn say(_: *mut c_void, level: u32, m: turbo_text) {
        let s = unsafe { std::slice::from_raw_parts(m.ptr as *const u8, m.len as usize) };
        println!("log {level}: {}", String::from_utf8_lossy(s));
    }
    let (batch, seq) = (32usize, 256usize);
    let vocab = Tok::create(&dir).unwrap().info().vocab_size as usize;
    // TURBO_TEST_ROWS=short: rows of 8 to 47 tokens in the same frame.
    let short = std::env::var("TURBO_TEST_ROWS").is_ok_and(|v| v == "short");
    let rows: Vec<Vec<i32>> = (0..batch)
        .map(|r| {
            let len = if short { 8 + (r * 13) % 40 } else { seq };
            (0..len).map(|p| (1000 + (r * 131 + p * 17) % (vocab - 1000)) as i32).collect()
        })
        .collect();
    let t = Tokens::new(&rows, 0);
    // TURBO_TEST_PRECISION=fastest or model runs one of the two.
    let which: Vec<u32> = match std::env::var("TURBO_TEST_PRECISION").as_deref() {
        Ok("fastest") => vec![TURBO_PRECISION_FASTEST],
        Ok("model") => vec![TURBO_PRECISION_MODEL],
        _ => vec![TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL],
    };
    for precision in which {
        let desc = turbo_runtime_desc {
            struct_size: size_of::<turbo_runtime_desc>() as u32,
            reserved: 0,
            log: Some(say),
            log_user_data: ptr::null_mut(),
        };
        let mut err = new_error();
        let mut rt = ptr::null_mut();
        assert_eq!(unsafe { turbo_runtime_create(&desc, &mut rt, &mut err) }, 0);
        let mut ctx = ptr::null_mut();
        let rc = unsafe { turbo_context_create(rt, gpu(rt), &mut ctx, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        let mut m = ptr::null_mut();
        let rc = unsafe { turbo_model_load(ctx, text(dir.to_str().unwrap()), &mut m, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        let l = Loaded { rt, ctx, m };
        let s = Session::create(l.m, Some(&session_desc(batch as u32, seq as u32, precision))).unwrap();
        for _ in 0..5 {
            s.write_tokens(&t.batch(), None).unwrap();
            s.run().unwrap();
        }
        // TURBO_TEST_RUNS: the timed runs, 20 by default.
        let n: usize = std::env::var("TURBO_TEST_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
        let start = std::time::Instant::now();
        for _ in 0..n {
            s.write_tokens(&t.batch(), None).unwrap();
            s.run().unwrap();
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / n as f64;
        println!(
            "{}: precision {precision}: {ms:.3} ms a run of {batch} x {seq} (profile below if enabled)",
            dir.display()
        );
        drop(s);
        drop(l);
    }
}
