use super::*;

fn side(words_per_second: f64) -> Side {
    Side { words_per_second, call_p50_ms: 0.5, call_p95_ms: 0.9 }
}

fn sample() -> PosRecord {
    PosRecord {
        record_version: POS_RECORD_VERSION,
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
        procedure: "the tool ran pinned to CPUs 6, the tagger on one thread".into(),
        command: "cargo run --release -p turbo-bench --features postag -- postag --model <model>".into(),
        model: PosModel {
            id: "bilstm".into(),
            revision: "r".into(),
            sha256: "ab".repeat(32),
            file_sha256: "cd".repeat(32),
            file_bytes: 10,
            tags: 17,
            packed_config_hash: "0000000000000001".into(),
            parity_config_hash: "0000000000000002".into(),
        },
        treebank: Treebank { file: "t.conllu".into(), sha256: "ef".repeat(32), sentences: 10, words: 100 },
        rounds: 3,
        accuracy: Accuracy { packed_correct: 90, parity_correct: 90, tags_agree: 100 },
        drift: Drift { max_emission_abs_diff: 1e-6, max_marginal_abs_diff: 1e-7 },
        runs: vec![PosRun {
            sentences_per_call: 32,
            lengths: "all".into(),
            sentences: 10,
            words: 100,
            packed: side(1000.0),
            parity: side(10.0),
            packed_over_parity: 100.0,
        }],
    }
}

#[test]
fn a_record_is_named_for_its_machine_model_and_commit() {
    let r = sample();
    check(&r).unwrap();
    assert_eq!(file_name(&r), "x86-64-avx512.cpu.postag.bilstm-crf.cdcdcdcd.0123456789ab.json");
    assert_eq!(lengths_name(1, Some(9)), "1-9");
    assert_eq!(lengths_name(25, None), "25-");
}

#[test]
fn check_refuses_what_a_measurement_cannot_give() {
    let refused = |f: &dyn Fn(&mut PosRecord), why: &str| {
        let mut r = sample();
        f(&mut r);
        assert!(check(&r).is_err(), "{why}");
    };
    refused(&|r| r.library.commit.truncate(12), "a short commit");
    refused(&|r| r.library.pushed_to.clear(), "a commit on no branch");
    refused(&|r| r.model.parity_config_hash = r.model.packed_config_hash.clone(), "one hash for both");
    refused(&|r| r.accuracy.tags_agree = 101, "more agreeing words than words");
    refused(&|r| r.drift.max_emission_abs_diff = f64::NAN, "a NaN drift");
    refused(&|r| r.runs[0].packed_over_parity = 99.0, "a ratio that is not the quotient");
    refused(&|r| r.runs[0].parity.call_p95_ms = 0.1, "p95 under p50");
    refused(&|r| r.runs.clear(), "no run");
    refused(&|r| r.command = "--model /home/someone/e5.bin".into(), "a host path");
    refused(&|r| r.record_version = 2, "another version");
}
