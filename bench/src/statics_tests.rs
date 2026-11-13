use super::*;

#[test]
fn a_peers_rates_are_read_from_its_lines() {
    let out = "statembed+rayon: loaded in 3.1 ms\nstatembed+rayon batch 1: 51234 texts/s\n\
               statembed+rayon batch 32: 120000 texts/s\nsomething else\nx batch y: 3 texts/s\n";
    let r = rates("t60", out, &[1, 32]);
    assert_eq!(r.len(), 2);
    assert_eq!(r[0], Rate { set: "t60".into(), batch: 1, texts_per_second: 51234.0 });
    assert_eq!(r[1].batch, 32);
    assert!(rates("golden", "model2vec-zig: not run", &[1]).is_empty());
    // One line a batch, in order; `all` stands for any batch above 1.
    let zig = "model2vec-zig batch 1: 900 texts/s\nmodel2vec-zig batch all (32 threads): 4000 texts/s\n\
               model2vec-zig batch all (32 threads): 4100 texts/s\n";
    let r = rates("golden", zig, &[1, 32, 256]);
    assert_eq!(
        r.iter().map(|r| (r.batch, r.texts_per_second)).collect::<Vec<_>>(),
        [(1, 900.0), (32, 4000.0), (256, 4100.0)]
    );
    assert!(rates("golden", "x batch all (2 threads): 5 texts/s\n", &[1]).is_empty());
}

#[test]
fn vectors_agree_by_bits_cosine_and_refusals() {
    let want = [1.0f32, 0.0, 0.6, 0.8, 0.0, 0.0, 0.5, 0.5];
    let got: Vec<f32> = vec![1.0, 0.0, 0.8, 0.6, 0.0, 0.0, f32::NAN, f32::NAN];
    let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
    let a = agreement("golden", &want, &bytes, 2).unwrap();
    assert_eq!((a.rows, a.bit_equal, a.refused), (4, 2, 1));
    assert!((a.min_cosine - 0.96).abs() < 1e-6, "{}", a.min_cosine);
    assert!(agreement("golden", &want, &bytes[4..], 2).is_err());
}

#[test]
fn one_zero_row_has_zero_agreement_and_two_zero_rows_agree() {
    let want = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
    let got = [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0];
    let bytes: Vec<_> = got.into_iter().flat_map(f32::to_le_bytes).collect();
    let a = agreement("golden", &want, &bytes, 2).unwrap();
    assert_eq!((a.rows, a.bit_equal, a.refused, a.min_cosine), (3, 1, 0, 0.0));
    let zeros = agreement("golden", &[0.0, 0.0], &[0; 8], 2).unwrap();
    assert_eq!((zeros.bit_equal, zeros.refused, zeros.min_cosine), (1, 0, 1.0));
}

#[test]
fn nonfinite_peer_rows_are_refused() {
    let want = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let got = [f32::INFINITY, 0.0, f32::NEG_INFINITY, 0.0, f32::NAN, 0.0, 0.0, 0.0];
    let bytes: Vec<_> = got.into_iter().flat_map(f32::to_le_bytes).collect();
    let a = agreement("golden", &want, &bytes, 2).unwrap();
    assert_eq!((a.rows, a.bit_equal, a.refused, a.min_cosine), (4, 1, 3, 1.0));
}

/// Each committed record parses and passes `check`, and is filed under
/// the name `file_name` gives it.
#[test]
fn committed_records_pass_their_check() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(STATIC_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    for e in entries {
        let path = e.unwrap().path();
        let r: StaticRecord = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        check(&r).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(path.file_name().unwrap().to_str().unwrap(), file_name(&r));
    }
}

#[test]
fn statics_reads_gpu_clock_outside_timed_passes() {
    use turbo::TURBO_PRECISION_MODEL;
    use turbo::record::{ClockReading, GpuClock};

    let before = ClockReading { min_mhz: 2200, max_mhz: 2200, act_mhz: 0 };
    let after = ClockReading { min_mhz: 2200, max_mhz: 2200, act_mhz: 2200 };
    let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let events_r = std::rc::Rc::clone(&events);
    let calls = std::cell::Cell::new(0u32);
    let clock = crate::gpu_clock::with_test_reader(
        move |backend| {
            let n = calls.get();
            calls.set(n + 1);
            events_r.borrow_mut().push(format!("clock:{backend}:{n}"));
            Some(if n == 0 { before } else { after })
        },
        || {
            let mut w = crate::gpu_clock::Window::new("levelzero", 1);
            best_in(Duration::ZERO, &mut w, 2, &mut || {
                events.borrow_mut().push("pass".into());
                Ok(())
            })
            .unwrap();
            w.finish()
        },
    );
    assert_eq!(clock, Some(GpuClock { before, after }));
    assert_eq!(*events.borrow(), ["pass", "clock:levelzero:0", "pass", "pass", "pass", "clock:levelzero:1"]);

    let scratch =
        turbo_scratch::Scratch::at(std::env::temp_dir().join(format!("turbo-bench-static-{}", std::process::id())));
    std::fs::create_dir_all(&scratch).unwrap();
    let set_a = scratch.join("a.json");
    let set_b = scratch.join("b.json");
    std::fs::write(&set_a, r#"["hello world","second row"]"#).unwrap();
    std::fs::write(&set_b, r#"["another row"]"#).unwrap();

    let plan = StaticPlan {
        bundle: Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-static-bundle"),
        precision: TURBO_PRECISION_MODEL,
        sets: vec![("a".into(), set_a), ("b".into(), set_b)],
        batches: vec![1, 2],
        peers: None,
    };
    let p = Provenance { top: String::new(), commit: "d".repeat(40), pushed_to: vec!["origin/main".into()] };

    let cpu_record = measure_for(&plan, &p, "2026-10-22T00:00:00Z".into(), Duration::ZERO).unwrap();
    assert_eq!(cpu_record.device.clock, None);
    check(&cpu_record).unwrap();

    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let seen_r = std::rc::Rc::clone(&seen);
    let clocked = crate::gpu_clock::with_test_reader(
        move |backend| {
            let mut s = seen_r.borrow_mut();
            s.push(backend.to_owned());
            Some(if s.len() == 1 { before } else { after })
        },
        || measure_for(&plan, &p, "2026-10-22T00:00:00Z".into(), Duration::ZERO).unwrap(),
    );
    assert_eq!(*seen.borrow(), ["cpu", "cpu"]);
    assert_eq!(clocked.device.clock, Some(GpuClock { before, after }));
    check(&clocked).unwrap();

    let calls = std::cell::Cell::new(0u32);
    let incomplete = crate::gpu_clock::with_test_reader(
        move |_| {
            let n = calls.get();
            calls.set(n + 1);
            (n == 0).then_some(before)
        },
        || measure_for(&plan, &p, "2026-10-22T00:00:00Z".into(), Duration::ZERO).unwrap(),
    );
    assert_eq!(incomplete.device.clock, None);
    std::fs::remove_dir_all(&scratch).unwrap();
}

#[test]
fn empty_text_count_is_rejected_without_running_a_pass() {
    let mut calls = 0;
    let result = best_in(Duration::ZERO, &mut crate::gpu_clock::Window::none(), 0, &mut || {
        calls += 1;
        if calls > 8 {
            return Err("pass safety limit reached".into());
        }
        Ok(())
    });
    assert_eq!(result.unwrap_err(), "a static text set must contain at least one text");
    assert_eq!(calls, 0);
}

#[test]
fn nonempty_text_count_runs_warmup_and_three_timed_passes() {
    let mut calls = 0;
    let rate = best_in(Duration::ZERO, &mut crate::gpu_clock::Window::none(), 1, &mut || {
        calls += 1;
        if calls > 8 {
            return Err("pass safety limit reached".into());
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(calls, 4);
    assert!(rate.is_finite() && rate > 0.0);
}

#[test]
fn empty_json_measurement_is_rejected() {
    const CHILD: &str = "TURBO_STATIC_EMPTY_TEST_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let scratch =
            turbo_scratch::Scratch::at(std::env::temp_dir().join(format!("turbo-static-empty-{}", std::process::id())));
        std::fs::create_dir_all(&scratch).unwrap();
        let texts = scratch.join("texts.json");
        let plan = StaticPlan {
            bundle: Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-static-bundle"),
            precision: turbo::TURBO_PRECISION_MODEL,
            sets: vec![("test".into(), texts.clone())],
            batches: vec![1],
            peers: None,
        };
        let p = Provenance { top: String::new(), commit: "d".repeat(40), pushed_to: vec!["origin/main".into()] };
        // Prove that this fixture and CPU path work before exercising the empty set.
        std::fs::write(&texts, r#"["hello world"]"#).unwrap();
        let record = measure_for(&plan, &p, "2026-10-22T00:00:00Z".into(), Duration::ZERO).unwrap();
        check(&record).unwrap();
        assert_eq!(record.sets[0].texts, 1);
        assert_eq!(record.device.clock, None);
        std::fs::write(&texts, "[]").unwrap();
        let error = measure_for(&plan, &p, "2026-10-22T00:00:00Z".into(), Duration::ZERO).unwrap_err();
        assert_eq!(error, "a static text set must contain at least one text");
        return;
    }
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "statics::tests::empty_json_measurement_is_rejected", "--nocapture"])
        .env(CHILD, "1")
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "measurement child failed: {status}");
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("empty JSON measurement exceeded 10 seconds; child killed and reaped");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
