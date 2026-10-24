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
        },
        library: Library {
            version: "0.1.0".into(),
            build: "0.1.0 cpu".into(),
            commit: "0123456789abcdef0123456789abcdef01234567".into(),
            pushed_to: vec!["origin/main".into()],
            settings: Vec::new(),
        },
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
        let dir = std::env::temp_dir().join(format!("turbo-throughput-check-{}", std::process::id()));
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
