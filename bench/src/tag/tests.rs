use super::*;

fn speed(words_per_second: f64) -> Speed {
    Speed { words_per_second, call_p50_ms: 0.5, call_p99_ms: 0.9 }
}

fn sample() -> TagRecord {
    TagRecord {
        record_version: TAG_RECORD_VERSION,
        recorded_at: "2026-10-24T00:00:00Z".into(),
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
        procedure: "the tool ran pinned to CPUs 6, the tagger on one thread".into(),
        command: "cargo run --release -p turbo-bench --features postag -- tag --model <bundle>".into(),
        model: TagModel {
            id: "bilstm".into(),
            revision: "r".into(),
            sha256: "ab".repeat(32),
            source_sha256: "cd".repeat(32),
            bundle_sha256: "12".repeat(32),
            weights: "f32".into(),
            tags: 17,
        },
        tier: Tier {
            device: "cpu".into(),
            precision: "model".into(),
            numerics: "Packed".into(),
            config_hash: "0000000000000001".into(),
        },
        corpus: Treebank { file: "t.conllu".into(), sha256: "ef".repeat(32), sentences: 10, words: 100 },
        sentences_per_call: 32,
        rounds: 3,
        correct: 95,
        cold: speed(1000.0),
        warm: speed(2000.0),
    }
}

#[test]
fn a_record_is_named_for_its_machine_tier_model_and_commit() {
    let r = sample();
    check(&r).unwrap();
    assert_eq!(file_name(&r), "x86-64-avx512.cpu.tag.bilstm-crf.model.cdcdcdcd.0123456789ab.json");
}

#[test]
fn every_cpu_tier_and_its_weights_pass() {
    for (precision, numerics, weights) in CPU_TIERS {
        for w in weights {
            let mut r = sample();
            r.tier.precision = precision.into();
            r.tier.numerics = numerics.into();
            r.model.weights = (*w).into();
            check(&r).unwrap_or_else(|e| panic!("{precision} {w}: {e}"));
        }
    }
}

#[test]
fn check_refuses_what_a_measurement_cannot_give() {
    let bad: [fn(&mut TagRecord); 12] = [
        |r| r.library.pushed_to.clear(),
        |r| r.library.commit = "dirty".into(),
        |r| r.tier.numerics = "JvmParity".into(),
        |r| r.tier.precision = "fastest".into(),
        |r| r.tier.device = "cuda".into(),
        |r| r.model.bundle_sha256 = "zz".into(),
        |r| r.correct = 101,
        |r| r.rounds = 0,
        |r| r.cold.words_per_second = f64::NAN,
        |r| r.warm.call_p50_ms = 0.0,
        |r| r.warm.call_p99_ms = 0.1,
        |r| r.corpus.file = "C:/Users/x/t.conllu".into(),
    ];
    for (i, f) in bad.iter().enumerate() {
        let mut r = sample();
        f(&mut r);
        assert!(check(&r).is_err(), "case {i} passed");
    }
}

#[test]
fn a_record_round_trips_and_is_never_replaced() {
    let dir = std::env::temp_dir().join(format!("turbo-bench-tag-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let r = sample();
    let path = write(&r, &dir).unwrap();
    let back: TagRecord = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(back, r);
    assert!(write(&r, &dir).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
