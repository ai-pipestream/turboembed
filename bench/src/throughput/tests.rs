use super::*;

fn sample() -> ThroughputRecord {
    ThroughputRecord {
        record_version: THROUGHPUT_RECORD_VERSION,
        recorded_at: "2026-10-22T00:00:00Z".into(),
        machine: Machine { arch: "x86_64-avx512".into(), host_cpu: "A CPU".into(), os: "linux".into() },
        device: Device {
            backend: "cpu".into(),
            kind: "DEVICE_CPU".into(),
            name: "A CPU".into(),
            vendor: "".into(),
            driver_version: "".into(),
            runtime_version: "".into(),
            memory_total: 0,
            clock: None,
        },
        library: Library {
            version: "0.1.0".into(),
            build: "0.1.0 cpu".into(),
            commit: "0123456789abcdef0123456789abcdef01234567".into(),
            pushed_to: vec!["origin/main".into()],
            settings: Vec::new(),
        },
        cargo_build: None,
        procedure: "the tool ran unpinned, the grammar on one thread".into(),
        command: "cargo run --release -p turbo-bench --features grammar -- grammar --rounds 5".into(),
        mode: "grammar".into(),
        subject: Subject { name: "english".into(), sha256: String::new(), load_seconds: None },
        text: RecordText {
            source: "analysis/goldens/data/inputs/{num} in name order, repeated to at least 102400 bytes".into(),
            bytes: 1000,
            chars: 900,
            sha256: "ab".repeat(32),
        },
        rounds: 5,
        runs: ["find", "lex", "parse", "parse-units", "times"]
            .iter()
            .map(|phase| PhaseRun {
                phase: phase.to_string(),
                units: 42,
                seconds: 0.5,
                chars_per_second: 1800.0,
                bytes_per_second: 2000.0,
            })
            .collect(),
    }
}

#[test]
fn a_record_is_named_for_its_machine_mode_subject_and_commit() {
    let r = sample();
    check(&r).unwrap();
    assert_eq!(file_name(&r), "x86-64-avx512.cpu.throughput.grammar.english.0123456789ab.json");
    let back: ThroughputRecord = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    assert_eq!(back, r);
}

#[test]
fn check_refuses_what_a_measurement_cannot_give() {
    let refused = |f: &dyn Fn(&mut ThroughputRecord), why: &str| {
        let mut r = sample();
        f(&mut r);
        assert!(check(&r).is_err(), "{why}");
        let dir = turbo_scratch::Scratch::at(
            std::env::temp_dir().join(format!("turbo-throughput-check-{}", std::process::id())),
        );
        assert!(write(&r, &dir).is_err(), "{why}: written");
    };
    refused(&|r| r.library.commit.truncate(12), "a short commit");
    refused(&|r| r.library.pushed_to.clear(), "a commit on no branch");
    refused(&|r| r.mode = "postag".into(), "another mode");
    refused(&|r| r.mode = "lattice".into(), "a lattice record with no dictionary hash");
    refused(&|r| r.device.backend = "cuda".into(), "a CUDA backend");
    refused(&|r| r.device.kind = "DEVICE_GPU".into(), "a GPU device kind");
    refused(&|r| r.runs[0].phase = "not-a-real-phase".into(), "an unknown phase");
    refused(&|r| r.runs[0].phase = "parse-units".into(), "a grammar record holding only one phase");
    refused(
        &|r| {
            r.runs = ["find", "find", "parse", "parse-units", "times"]
                .iter()
                .map(|p| PhaseRun {
                    phase: p.to_string(),
                    units: 1,
                    seconds: 0.5,
                    chars_per_second: 1800.0,
                    bytes_per_second: 2000.0,
                })
                .collect();
        },
        "a duplicated phase",
    );
    refused(&|r| r.text.chars = r.text.bytes + 1, "more chars than bytes");
    refused(&|r| r.runs[0].units = 0, "a phase that produced nothing");
    refused(&|r| r.runs[0].seconds = f64::NAN, "a NaN time");
    refused(&|r| r.runs[0].chars_per_second = 1.0, "a rate that is not the quotient");
    refused(&|r| r.runs.clear(), "no run");
    refused(&|r| r.command = ["--repo ", "home", "someone", "tree"].join("/"), "a host path");
    refused(&|r| r.record_version = 2, "another version");
}

#[test]
fn historical_records_round_trip_without_invented_build_settings() {
    let r = sample();
    let json = serde_json::to_string(&r).unwrap();
    assert!(!json.contains("cargo_build"));
    let back: ThroughputRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(back.cargo_build, None);
    check(&back).unwrap();
}

#[test]
fn cargo_optimization_is_independent_of_profile_family_and_assertions() {
    for profile in ["debug", "release"] {
        for debug_assertions in [false, true] {
            let mut r = sample();
            r.cargo_build = Some(CargoBuild { profile: profile.into(), opt_level: "0".into(), debug_assertions });
            assert!(check(&r).unwrap_err().contains("opt-level=0"));
            for level in ["1", "2", "3", "s", "z"] {
                r.cargo_build.as_mut().unwrap().opt_level = level.into();
                check(&r).unwrap();
                let json = serde_json::to_vec(&r).unwrap();
                assert_eq!(serde_json::from_slice::<ThroughputRecord>(&json).unwrap(), r);
            }
        }
    }
}

#[cfg(feature = "grammar")]
#[test]
fn grammar_cpu_record_has_no_gpu_clock() {
    let prov = || {
        Ok(crate::git::Provenance { top: String::new(), commit: "1".repeat(40), pushed_to: vec!["origin/main".into()] })
    };
    let run_once = || {
        super::grammar::measure(
            "12 km and $5 on 2020-01-02\n",
            "fixture".into(),
            None,
            1,
            "2026-10-22T00:00:00Z".into(),
            prov,
        )
        .unwrap()
    };

    let mut cpu_record = run_once();
    assert_eq!(cpu_record.device.clock, None);
    cpu_record.cargo_build = None;
    check(&cpu_record).unwrap();
}

#[cfg(feature = "lattice")]
#[test]
fn lattice_cpu_record_has_no_gpu_clock_with_and_without_readings() {
    use turbo_tokenize::lattice::DictionaryFormat;

    let dict = Path::new(env!("CARGO_MANIFEST_DIR")).join("../analysis/goldens/data/fixtures/mecab-fixture-ko");
    for readings in [false, true] {
        let prov = || {
            Ok(crate::git::Provenance {
                top: String::new(),
                commit: "2".repeat(40),
                pushed_to: vec!["origin/main".into()],
            })
        };
        let run_once = || {
            super::lattice::measure_one(
                "ko-dic",
                DictionaryFormat::MecabKoDic,
                &dict,
                "학교\n",
                "fixture".into(),
                "the tool ran unpinned, the tokenizer on one thread".into(),
                None,
                readings,
                1,
                "2026-10-22T00:00:00Z".into(),
                prov,
            )
            .unwrap()
        };

        let mut cpu_record = run_once();
        assert_eq!(cpu_record.device.clock, None);
        cpu_record.cargo_build = None;
        check(&cpu_record).unwrap();
    }
}

#[cfg(all(target_os = "linux", any(feature = "grammar", feature = "lattice")))]
#[test]
fn throughput_clock_reads_enclose_both_timed_phases() {
    let driver = crate::gpu_clock::tests::Driver::new("throughput-timing");
    let mut clock = driver.window(2);
    let mut calls = 0;
    let mut pass = || {
        calls += 1;
        driver.set(calls * 100);
        Ok(5)
    };
    super::measure::median_pass(&mut clock, 2, &mut pass).unwrap();
    super::measure::median_pass(&mut clock, 1, &mut pass).unwrap();
    let clock = clock.finish().unwrap();
    assert_eq!(calls, 5);
    assert_eq!((clock.before.act_mhz, clock.after.act_mhz), (100, 500));
}
