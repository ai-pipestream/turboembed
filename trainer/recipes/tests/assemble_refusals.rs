//! Refusal regressions for [`assemble_bundle`], [`verify_run`], and
//! `turbo-train-head assemble` (#1726).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo::bundle::sha256_hex;
use turbo::manifest::HeadSpec;
use turbo_tagger::{DepParser, Lemmatizer};
use turbo_trainer_recipes::assemble::{assemble_bundle, verify_run};
use turbo_trainer_recipes::corpus::{Counts, FileRecord};
use turbo_trainer_recipes::receipt::Receipt;
use turbo_trainer_recipes::safetensors;

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-assemble-refuse-1726-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

fn corpus_multiconer(dir: &Path) {
    let names = ["ann", "bob", "eve", "max"];
    let places = ["paris", "rome", "oslo"];
    let mut text = String::new();
    for i in 0..32 {
        let (n, p) = (names[i % names.len()], places[i % places.len()]);
        text += &format!(
            "# id {i}\tdomain=en\nthey _ _ O\nmet _ _ O\n{n} _ _ B-OtherPER\nin _ _ O\n{p} _ _ B-HumanSettlement\n\n"
        );
    }
    fs::create_dir_all(dir.join("EN-English")).unwrap();
    for f in ["en_train", "en_dev", "en_test"] {
        fs::write(dir.join(format!("EN-English/{f}.conll")), &text).unwrap();
    }
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(args).output().unwrap()
}

fn train_ner_run(data: &Path, out: &Path) -> Receipt {
    let b = tiny();
    let args = [
        "train",
        "--corpus",
        "multiconer2-en",
        "--dir",
        data.to_str().unwrap(),
        "--bundle",
        b.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--unrecorded-corpus",
        "--crf",
        "--epochs",
        "4",
        "--batch",
        "8",
        "--lr",
        "0.02",
        "--commit",
        "1726test",
    ];
    let o = run_cli(&args);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap()
}

fn tagger_tensors(prefix: &str, hidden: usize, labels: usize) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let mut w = vec![0.0f32; labels * hidden];
    for i in 0..labels.min(hidden) {
        w[i * hidden + i] = 1.0;
    }
    let mut b = vec![0.0f32; labels];
    if labels > 1 {
        b[1] = 2.0;
    }
    vec![
        (format!("{prefix}.out.weight"), vec![labels as u64, hidden as u64], w),
        (format!("{prefix}.out.bias"), vec![labels as u64], b),
    ]
}

#[test]
fn assemble_refuses_mismatched_encoders_checkpoint_metadata_tampered_files_and_cli_missing_flags() {
    let data = scratch("refuse-data");
    corpus_multiconer(&data);
    let base_run = scratch("refuse-base-run");
    let mut base_receipt = train_ner_run(&data, &base_run);
    assert!(!base_receipt.corpus.recorded);

    let out_dup = scratch("refuse-dup-out");
    assert!(verify_run(&base_run).unwrap_err().0.contains("corpus.recorded is false"));
    assert!(assemble_bundle(&tiny(), &[&base_run], &out_dup).unwrap_err().0.contains("corpus.recorded is false"));
    base_receipt.corpus.recorded = true;
    fs::write(base_run.join("receipt.json"), serde_json::to_vec_pretty(&base_receipt).unwrap()).unwrap();

    assert!(assemble_bundle(&tiny(), &[&base_run], &tiny()).unwrap_err().0.contains("must not equal encoder_bundle"));
    assert!(assemble_bundle(&tiny(), &[&base_run, &base_run], &out_dup).unwrap_err().0.contains("duplicate head name"));

    let collide_run = scratch("refuse-collide-run");
    let mut spec: HeadSpec = serde_json::from_slice(&fs::read(base_run.join("head.json")).unwrap()).unwrap();
    spec.name = "ner_two".into();
    let weights = fs::read(base_run.join("head.safetensors")).unwrap();
    let mut r: Receipt = serde_json::from_slice(&fs::read(base_run.join("receipt.json")).unwrap()).unwrap();
    let spec_bytes = serde_json::to_vec_pretty(&spec).unwrap();
    fs::create_dir_all(&collide_run).unwrap();
    fs::write(collide_run.join("head.json"), &spec_bytes).unwrap();
    fs::write(collide_run.join("head.safetensors"), &weights).unwrap();
    r.head = Some(spec);
    r.outputs[1] =
        FileRecord { path: "head.json".into(), size: spec_bytes.len() as u64, sha256: sha256_hex(&spec_bytes) };
    fs::write(collide_run.join("receipt.json"), serde_json::to_vec_pretty(&r).unwrap()).unwrap();
    assert!(
        assemble_bundle(&tiny(), &[&base_run, &collide_run], &out_dup)
            .unwrap_err()
            .0
            .contains("colliding tensor_prefix")
    );

    let clone_run = |name: &str, files: &[&str]| -> PathBuf {
        let d = scratch(name);
        fs::create_dir_all(&d).unwrap();
        for f in files {
            fs::copy(base_run.join(f), d.join(f)).unwrap();
        }
        d
    };
    let bad_enc = clone_run("refuse-bad-enc", &["head.json", "head.safetensors"]);
    let mut r_enc: Receipt = serde_json::from_slice(&fs::read(base_run.join("receipt.json")).unwrap()).unwrap();
    r_enc.encoder.as_mut().unwrap().manifest_sha256 = "0".repeat(64);
    fs::write(bad_enc.join("receipt.json"), serde_json::to_vec_pretty(&r_enc).unwrap()).unwrap();
    assert!(assemble_bundle(&tiny(), &[&bad_enc], &out_dup).unwrap_err().0.contains("trained on encoder"));

    let bad_epoch = clone_run("refuse-bad-epoch", &["head.json", "head.safetensors"]);
    let mut r_ep: Receipt = serde_json::from_slice(&fs::read(base_run.join("receipt.json")).unwrap()).unwrap();
    let tr = r_ep.training.as_mut().unwrap();
    tr.best_epoch = if tr.best_epoch == 1 { 2 } else { 1 };
    fs::write(bad_epoch.join("receipt.json"), serde_json::to_vec_pretty(&r_ep).unwrap()).unwrap();
    assert!(verify_run(&bad_epoch).unwrap_err().0.contains("__metadata__.best_epoch"));

    let bad_dev = clone_run("refuse-bad-dev", &["head.json", "head.safetensors"]);
    let mut r_dv: Receipt = serde_json::from_slice(&fs::read(base_run.join("receipt.json")).unwrap()).unwrap();
    let tr = r_dv.training.as_mut().unwrap();
    tr.dev[tr.best_epoch - 1] += 0.125;
    fs::write(bad_dev.join("receipt.json"), serde_json::to_vec_pretty(&r_dv).unwrap()).unwrap();
    assert!(verify_run(&bad_dev).unwrap_err().0.contains("__metadata__.dev"));

    let no_meta = clone_run("refuse-no-meta", &["head.json"]);
    let mut r_nm: Receipt = serde_json::from_slice(&fs::read(base_run.join("receipt.json")).unwrap()).unwrap();
    let nm_bytes = safetensors::write(&tagger_tensors("heads.ner", 32, r_nm.head.as_ref().unwrap().labels.len()));
    fs::write(no_meta.join("head.safetensors"), &nm_bytes).unwrap();
    r_nm.outputs[0] =
        FileRecord { path: "head.safetensors".into(), size: nm_bytes.len() as u64, sha256: sha256_hex(&nm_bytes) };
    fs::write(no_meta.join("receipt.json"), serde_json::to_vec_pretty(&r_nm).unwrap()).unwrap();
    assert!(verify_run(&no_meta).unwrap_err().0.contains("missing __metadata__.best_epoch"));

    let tampered = clone_run("refuse-tampered", &["head.json", "receipt.json"]);
    let mut bad_bytes = weights;
    let last = bad_bytes.len() - 1;
    bad_bytes[last] ^= 0xff;
    fs::write(tampered.join("head.safetensors"), &bad_bytes).unwrap();
    assert!(verify_run(&tampered).unwrap_err().0.contains("sha256"));

    let nd_run = clone_run("refuse-nd", &["head.json", "head.safetensors"]);
    let mut r_nd: Receipt = serde_json::from_slice(&fs::read(base_run.join("receipt.json")).unwrap()).unwrap();
    r_nd.corpus.spdx = "CC-BY-ND-4.0".into();
    fs::write(nd_run.join("receipt.json"), serde_json::to_vec_pretty(&r_nd).unwrap()).unwrap();
    assert!(verify_run(&nd_run).unwrap_err().0.contains("refuses"));

    let cli_out = scratch("refuse-cli-out");
    let no_terms = run_cli(&[
        "assemble",
        "--bundle",
        tiny().to_str().unwrap(),
        "--run",
        base_run.to_str().unwrap(),
        "--out",
        cli_out.to_str().unwrap(),
        "--zoo-id",
        "turbo-en-test-1",
        "--url-prefix",
        "https://models.turboembed.org/zoo",
    ]);
    assert!(!no_terms.status.success());
    assert!(String::from_utf8_lossy(&no_terms.stderr).contains("--license-terms is required"));

    let no_prefix = run_cli(&[
        "assemble",
        "--bundle",
        tiny().to_str().unwrap(),
        "--run",
        base_run.to_str().unwrap(),
        "--out",
        cli_out.to_str().unwrap(),
        "--zoo-id",
        "turbo-en-test-1",
        "--license-terms",
        "https://www.apache.org/licenses/LICENSE-2.0",
    ]);
    assert!(!no_prefix.status.success());
    assert!(String::from_utf8_lossy(&no_prefix.stderr).contains("--url-prefix is required"));

    let _ = (Lemmatizer::from_zoo as fn(_, _, _, _, _, _) -> _, DepParser::from_zoo as fn(_, _, _, _, _, _) -> _);
    let _ = Counts::default();
}
