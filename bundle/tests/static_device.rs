//! Device selection for static model distillation: parser rules, refusal
//! without CPU fallback when an accelerator is unavailable, and CUDA vs CPU
//! teacher execution and student vector parity.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::Value;
use turbo::{TURBO_STAGE_DEVICE, TURBO_STAGE_HOST};
use turbo_bundle::api;
use turbo_bundle::distill::from_base::{self, Options};
use turbo_bundle::distill::{self, Device};
use turbo_bundle::recipe::Recipe;
use turbo_scratch::Scratch;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn scratch(name: &str) -> Scratch {
    let d = Scratch::at(std::env::temp_dir().join(format!("turbo-bundle-device-{}-{name}", std::process::id())));
    fs::create_dir_all(&d).unwrap();
    d
}

fn options(id: &str) -> Options {
    Options {
        id: id.into(),
        revision: "1".into(),
        pca_dims: 16,
        sif_coefficient: 1e-4,
        dtype: "F32".into(),
        max_length: 32,
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn embed_static(bundle: &Path, texts: &[&str], dim: usize) -> Vec<Vec<f32>> {
    let rt = api::Runtime::create().unwrap();
    let model = rt.cpu().unwrap().load(bundle).unwrap();
    let info = model.info().unwrap();
    let session = model.session(info.max_batch, info.max_seq).unwrap();
    let opts = api::options(0, 0);
    let mut flat = vec![0.0f32; texts.len() * dim];
    session.texts(texts, &opts, &mut flat).unwrap();
    flat.chunks_exact(dim).map(<[f32]>::to_vec).collect()
}

fn backend_arg(bundle: &Path) -> String {
    let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    m["artifacts"][0]["produced_by"]["args"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|a| a.as_str()?.strip_prefix("backend="))
        .unwrap()
        .to_owned()
}

#[test]
fn device_parser_accepts_each_form_and_refuses_bad_ones() {
    assert_eq!(Device::parse("cpu").unwrap(), Device::Cpu);
    assert_eq!(Device::parse(" select ").unwrap(), Device::Select);
    assert_eq!(Device::parse("0").unwrap(), Device::Index(0));
    assert_eq!(Device::parse("12").unwrap(), Device::Index(12));
    assert_eq!(Device::parse("cuda").unwrap(), Device::Backend { name: "cuda".into(), ordinal: None });
    assert_eq!(Device::parse("cuda:0").unwrap(), Device::Backend { name: "cuda".into(), ordinal: Some(0) });
    assert_eq!(Device::parse("levelzero:2").unwrap(), Device::Backend { name: "levelzero".into(), ordinal: Some(2) });
    assert_eq!(Device::parse("metal").unwrap(), Device::Backend { name: "metal".into(), ordinal: None });
    for bad in ["", "   ", "CUDA", "cuda:", ":0", "cuda:x", "cuda:0:1", "99999999999999999999"] {
        assert!(Device::parse(bad).is_err(), "expected {bad:?} to be refused");
    }
}

#[test]
fn missing_device_or_backend_refuses_without_cpu_fallback() {
    let d = scratch("refuse");
    let base = root().join("testdata/tiny-bert-bundle");
    let recipe = Recipe::load(&root().join("testdata/tiny-static-recipe/recipe.json")).unwrap();

    if std::env::var("TURBO_DISTILL_CHILD_NO_CUDA").as_deref() == Ok("1") {
        for (tag, dev) in [
            ("select", Device::Select),
            ("cuda", Device::Backend { name: "cuda".into(), ordinal: None }),
            ("cuda0", Device::Backend { name: "cuda".into(), ordinal: Some(0) }),
        ] {
            let err = distill::stage_on(&recipe, &base, &d.join(tag), &[], &dev, &mut |_| {}).unwrap_err();
            assert!(err.contains("TURBO_E_DEVICE_NOT_FOUND"), "{dev:?}: {err}");
            assert!(!d.join(tag).join("weights/static.safetensors").exists(), "{dev:?} wrote weights");
        }
        fs::remove_dir_all(d).unwrap();
        return;
    }

    let err = distill::stage_on(&recipe, &base, &d.join("idx"), &[], &Device::Index(9999), &mut |_| {}).unwrap_err();
    assert!(err.contains("TURBO_E_INVALID_ARGUMENT"), "{err}");

    let bogus = Device::Backend { name: "nonexistent".into(), ordinal: None };
    let err = distill::stage_on(&recipe, &base, &d.join("bogus"), &[], &bogus, &mut |_| {}).unwrap_err();
    assert!(err.contains("TURBO_E_DEVICE_NOT_FOUND"), "{err}");

    let bin = env!("CARGO_BIN_EXE_turbo-bundle");
    let out = Command::new(bin)
        .args([
            "distill",
            "--device",
            "nonexistent",
            root().join("testdata/tiny-static-recipe/recipe.json").to_str().unwrap(),
            base.to_str().unwrap(),
            d.join("cli-bogus").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("TURBO_E_DEVICE_NOT_FOUND"), "{stderr}");

    let cuda_linked = turbo::backend::linked().iter().any(|b| b.name() == "cuda");
    if cuda_linked {
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "missing_device_or_backend_refuses_without_cpu_fallback", "--nocapture"])
            .env("CUDA_VISIBLE_DEVICES", "")
            .env("TURBO_DISTILL_CHILD_NO_CUDA", "1")
            .output()
            .unwrap();
        assert!(
            child.status.success(),
            "child failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
    } else {
        for (tag, dev) in [("select", Device::Select), ("cuda", Device::Backend { name: "cuda".into(), ordinal: None })]
        {
            let err = distill::stage_on(&recipe, &base, &d.join(tag), &[], &dev, &mut |_| {}).unwrap_err();
            assert!(err.contains("TURBO_E_DEVICE_NOT_FOUND"), "{dev:?}: {err}");
        }
    }
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn cuda_distillation_executes_on_device_and_matches_cpu_student() {
    if !turbo::backend::linked().iter().any(|b| b.name() == "cuda") {
        return;
    }
    let rt = api::Runtime::create().unwrap();
    let has_cuda = (0..rt.device_count().unwrap()).any(|i| api::field(&rt.device_info(i).unwrap().backend) == "cuda");
    if !has_cuda {
        assert_eq!(
            std::env::var("TURBO_TEST_REQUIRE_CUDA").ok().as_deref(),
            Some("0"),
            "the cuda backend lists no device; TURBO_TEST_REQUIRE_CUDA=0 lets the test skip"
        );
        eprintln!("CUDA unavailable; skipped");
        return;
    }

    let d = scratch("cuda");
    let queries = [
        "neuroimaging scans from turboembed",
        "the quick brown fox jumps over the lazy dog",
        "Café naïve RÉSUMÉ",
        "东京是日本的首都。",
        "reset password in security settings",
    ];

    // 1. BERT encoder (30,522 entries + learned WordPiece terms): Select vs Cpu.
    let bert = root().join("testdata/tiny-bert-bundle");
    let terms = ["neuroimaging".to_owned(), "turboembed".to_owned()];
    let mut r_cpu = from_base::recipe(&bert, &options("test/bert-cpu"), &d.join("w-bert-cpu")).unwrap();
    let mut r_gpu = from_base::recipe(&bert, &options("test/bert-gpu"), &d.join("w-bert-gpu")).unwrap();
    let (b_cpu, b_gpu) = (d.join("bert-cpu"), d.join("bert-gpu"));

    let t0 = Instant::now();
    let s_cpu = distill::stage_on(&r_cpu, &bert, &b_cpu, &terms, &Device::Cpu, &mut |_| {}).unwrap();
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    assert_eq!((s_cpu.backend.as_str(), s_cpu.encode_stage), ("cpu", TURBO_STAGE_HOST));
    distill::seal(&mut r_cpu, &b_cpu).unwrap();
    assert_eq!(backend_arg(&b_cpu), "cpu");

    let t1 = Instant::now();
    let s_gpu = distill::stage_on(&r_gpu, &bert, &b_gpu, &terms, &Device::Select, &mut |_| {}).unwrap();
    let gpu_ms = t1.elapsed().as_secs_f64() * 1e3;
    assert_eq!((s_gpu.backend.as_str(), s_gpu.encode_stage), ("cuda", TURBO_STAGE_DEVICE));
    assert_eq!(s_gpu.added, ["neuroimaging", "turboembed"]);
    distill::seal(&mut r_gpu, &b_gpu).unwrap();
    assert_eq!(backend_arg(&b_gpu), "cuda");

    let (v_cpu, v_gpu) = (embed_static(&b_cpu, &queries, 16), embed_static(&b_gpu, &queries, 16));
    let mut min_bert_cos = 1.0f64;
    for (i, (a, b)) in v_cpu.iter().zip(&v_gpu).enumerate() {
        let cos = cosine(a, b);
        min_bert_cos = min_bert_cos.min(cos);
        assert!(cos > 0.9999, "bert query {i}: cosine {cos}");
    }
    eprintln!("tiny-bert distill: cpu={cpu_ms:.1}ms, cuda={gpu_ms:.1}ms, min_cosine={min_bert_cos:.8}");

    // 2. ModernBERT RoPE + gated-MLP BPE encoder (1,200 entries) on cuda:0 + CLI --device=cuda:0.
    let mbert = root().join("testdata/tiny-modernbert-bundle");
    let mut m_cpu = from_base::recipe(&mbert, &options("test/mbert-cpu"), &d.join("w-mbert-cpu")).unwrap();
    let mut m_gpu = from_base::recipe(&mbert, &options("test/mbert-gpu"), &d.join("w-mbert-gpu")).unwrap();
    let (mb_cpu, mb_gpu) = (d.join("mbert-cpu"), d.join("mbert-gpu"));
    let cuda0 = Device::Backend { name: "cuda".into(), ordinal: Some(0) };
    let sm_cpu = distill::stage_on(&m_cpu, &mbert, &mb_cpu, &[], &Device::Cpu, &mut |_| {}).unwrap();
    let sm_gpu = distill::stage_on(&m_gpu, &mbert, &mb_gpu, &[], &cuda0, &mut |_| {}).unwrap();
    assert_eq!((sm_cpu.backend.as_str(), sm_cpu.encode_stage), ("cpu", TURBO_STAGE_HOST));
    assert_eq!((sm_gpu.backend.as_str(), sm_gpu.encode_stage), ("cuda", TURBO_STAGE_DEVICE));
    distill::seal(&mut m_cpu, &mb_cpu).unwrap();
    distill::seal(&mut m_gpu, &mb_gpu).unwrap();
    let (mv_cpu, mv_gpu) = (embed_static(&mb_cpu, &queries, 16), embed_static(&mb_gpu, &queries, 16));
    for (i, (a, b)) in mv_cpu.iter().zip(&mv_gpu).enumerate() {
        let cos = cosine(a, b);
        assert!(cos > 0.9999, "modernbert query {i}: cosine {cos}");
    }

    let cli_out = d.join("cli-cuda");
    let cli = Command::new(env!("CARGO_BIN_EXE_turbo-bundle"))
        .args([
            "distill",
            "--device=cuda:0",
            root().join("testdata/tiny-static-recipe/recipe.json").to_str().unwrap(),
            bert.to_str().unwrap(),
            cli_out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(cli.status.success(), "stderr: {}", String::from_utf8_lossy(&cli.stderr));
    assert_eq!(backend_arg(&cli_out), "cuda");

    fs::remove_dir_all(d).unwrap();
}
