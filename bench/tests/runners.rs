//! The reference runners without their programs: the commands they build,
//! the output they parse, and what they refuse or record as not run. The
//! programs themselves need docker and, for TensorRT, a GPU.

mod common;

use std::path::Path;

use common::*;
use turbo::manifest::Pooling;
use turbo::{TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32};
use turbo_bench::docker::{self, parse_port};
use turbo_bench::measure::Rows;
use turbo_bench::tei::{self, Tei};
use turbo_bench::tensorrt::{self, TensorRt};

const DIGEST: &str = "4f5c6a1d7f3b2e8a9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b";

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_owned()).collect()
}

// ---- docker ----

#[test]
fn images_are_pinned_by_digest_only() {
    let pinned = format!("ghcr.io/huggingface/text-embeddings-inference@sha256:{DIGEST}");
    assert_eq!(docker::check_pinned("--tei-image", &pinned).unwrap(), pinned);
    for bad in [
        "ghcr.io/huggingface/text-embeddings-inference:1.8".to_owned(),
        format!("ghcr.io/huggingface/text-embeddings-inference@sha256:{}", &DIGEST[1..]),
        format!("ghcr.io/huggingface/text-embeddings-inference@sha256:{}", DIGEST.to_uppercase()),
        format!("text-embeddings-inference:1.8@sha512:{DIGEST}"),
    ] {
        let e = docker::check_pinned("--tei-image", &bad).unwrap_err();
        assert!(e.contains("is not pinned as name@sha256:<64 hex>"), "{e}");
    }
}

#[test]
fn docker_port_output_gives_the_host_port() {
    assert_eq!(parse_port("127.0.0.1:32768\n").unwrap(), 32768);
    assert_eq!(parse_port("127.0.0.1:49153\n[::1]:49153\n").unwrap(), 49153);
    assert!(parse_port("").is_err());
    assert!(parse_port("Error: No public port '80/tcp' published\n").is_err());
}

// ---- text-embeddings-inference ----

#[test]
fn tei_is_started_with_every_setting_on_its_command_line() {
    let image = format!("ghcr.io/huggingface/text-embeddings-inference@sha256:{DIGEST}");
    let a = tei::run_argv(&image, Path::new("/models/minilm"), "turbo-bench-tei-7", Some(1), "float32", "mean", 32, 64);
    let want = strings(&[
        "docker",
        "run",
        "--detach",
        "--rm",
        "--pull",
        "never",
        "--name",
        "turbo-bench-tei-7",
        "--gpus",
        "device=1",
        "--publish",
        "127.0.0.1::80",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--mount",
        "type=bind,src=/models/minilm,dst=/model,readonly",
        &image,
        "--model-id",
        "/model",
        "--port",
        "80",
        "--dtype",
        "float32",
        "--pooling",
        "mean",
        "--max-client-batch-size",
        "32",
        "--max-batch-tokens",
        "16384",
        "--auto-truncate",
        "false",
    ]);
    assert_eq!(a, want);
    // The CPU image: no GPU; a batch past TEI's default token budget raises it.
    let a = tei::run_argv(&image, Path::new("/m"), "c", None, "float32", "cls", 512, 64);
    assert!(!a.iter().any(|s| s == "--gpus"));
    assert_eq!(a[a.len() - 3], "32768");
}

#[test]
fn tei_settings_follow_the_bundle_and_the_compute_dtype() {
    assert_eq!(tei::pooling(Pooling::Mean), "mean");
    assert_eq!(tei::pooling(Pooling::Cls), "cls");
    assert_eq!(tei::pooling(Pooling::Last), "last-token");
    assert_eq!(tei::dtype(TURBO_DTYPE_F32), Ok("float32"));
    assert_eq!(tei::dtype(TURBO_DTYPE_F16), Ok("float16"));
    assert!(tei::dtype(TURBO_DTYPE_BF16).is_err());
}

/// /info as TEI's router serializes its Info struct (router/src/lib.rs),
/// with the example values of its OpenAPI schema.
const TEI_INFO: &str = r#"{"model_id":"thenlper/gte-base","model_sha":"fca14538aa9956a46526bd1d0d11d69e19b5a101",
"model_dtype":"float16","served_model_name":"thenlper/gte-base","model_type":{"embedding":{"pooling":"cls"}},
"max_concurrent_requests":128,"max_input_length":512,"max_batch_tokens":2048,"max_batch_requests":null,
"max_client_batch_size":32,"auto_truncate":false,"tokenization_workers":4,"version":"0.5.0","sha":null,
"docker_label":null}"#;

#[test]
fn tei_info_gives_its_version_and_dtype() {
    let i = tei::parse_info(TEI_INFO).unwrap();
    assert_eq!((i.version.as_str(), i.model_dtype.as_str()), ("0.5.0", "float16"));
    assert!(tei::parse_info(r#"{"model_id":"x"}"#).unwrap_err().contains("no version"));
}

#[test]
fn tei_is_sent_each_rows_live_ids_and_its_answers_are_checked() {
    let m = cpu_measurement();
    let body: serde_json::Value = serde_json::from_str(&tei::embed_body(&m.rows, true)).unwrap();
    assert_eq!(body["normalize"], true);
    assert_eq!(body["truncate"], false);
    let inputs = body["inputs"].as_array().unwrap();
    assert_eq!(inputs.len(), m.rows.batch as usize);
    for (r, row) in inputs.iter().enumerate() {
        let ids: Vec<i32> = row.as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
        assert_eq!(ids, m.rows.live(r), "row {r}: no padding is sent");
        assert_eq!(ids, m.reference.ids[m.rows.cases[r] as usize]);
    }
    // /embed: one vector per row, each as wide as the model's.
    assert_eq!(tei::parse_embed("[[0.5,0.25],[1,0]]", 2, 2).unwrap(), vec![vec![0.5, 0.25], vec![1.0, 0.0]]);
    assert!(tei::parse_embed("[[0.5,0.25]]", 2, 2).is_err());
    assert!(tei::parse_embed("[[0.5]]", 1, 2).is_err());
    assert!(tei::parse_embed(r#"{"error":"Input validation error","error_type":"validation"}"#, 1, 2).is_err());
    // /tokenize, in the form of TEI's TokenizeResponse example.
    let back = tei::parse_tokenize(
        r#"[[{"id":101,"text":"[CLS]","special":true,"start":null,"stop":null},
            {"id":7592,"text":"hello","special":false,"start":0,"stop":5}]]"#,
    )
    .unwrap();
    assert_eq!(back, vec![vec![101, 7592]]);
    assert_eq!(tei::first_changed(&back, &back), None);
    assert_eq!(tei::first_changed(&back, &[vec![101, 7593]]), Some((0, vec![101, 7592], vec![101, 7593])));
}

/// A model directory in the upstream layout, made from the bundle's files.
fn upstream_dir(name: &str) -> Scratch {
    let d = scratch(name);
    std::fs::copy(tiny_bundle().join("tokenizer.json"), d.join("tokenizer.json")).unwrap();
    std::fs::copy(tiny_bundle().join("weights/model.safetensors"), d.join("model.safetensors")).unwrap();
    std::fs::write(d.join("config.json"), "{}").unwrap();
    d
}

#[test]
fn tei_serves_only_the_bundles_own_tokenizer_and_weights() {
    let m = cpu_measurement();
    let d = upstream_dir("tei-model");
    tei::check_model_dir(&d, m).unwrap();
    std::fs::write(d.join("tokenizer.json"), "{}").unwrap();
    assert!(tei::check_model_dir(&d, m).unwrap_err().contains("is not the bundle's tokenizer"));
    let d = upstream_dir("tei-model-weights");
    std::fs::copy(tiny_bundle().join("reference/reference.safetensors"), d.join("model.safetensors")).unwrap();
    assert!(tei::check_model_dir(&d, m).unwrap_err().contains("is not the loaded artifact"));
    let d = upstream_dir("tei-model-config");
    std::fs::remove_file(d.join("config.json")).unwrap();
    assert!(tei::check_model_dir(&d, m).unwrap_err().contains("config.json"));
}

#[test]
fn tei_on_another_models_files_is_recorded_as_not_run_before_docker() {
    let m = cpu_measurement();
    let d = upstream_dir("tei-not-run");
    std::fs::write(d.join("tokenizer.json"), "{}").unwrap();
    let t = Tei {
        image: format!("ghcr.io/huggingface/text-embeddings-inference@sha256:{DIGEST}"),
        model_dir: d.to_owned(),
    };
    let r = tei::run(&t, m, None, 1, 1).unwrap();
    assert!(r.measured.is_none());
    assert!(r.not_run.unwrap().contains("is not the bundle's tokenizer"));
    assert!(r.commands.is_empty(), "nothing ran");
    let t = Tei { image: "text-embeddings-inference:latest".into(), model_dir: ".".into() };
    assert!(tei::run(&t, m, None, 1, 1).unwrap_err().contains("is not pinned"));
}

// ---- TensorRT ----

fn trt(work: &Path) -> TensorRt {
    TensorRt {
        image: format!("nvcr.io/nvidia/tensorrt@sha256:{DIGEST}"),
        trtexec: "trtexec".into(),
        inputs: ["input_ids".into(), "attention_mask".into(), "token_type_ids".into()],
        input_dtype: "int64".into(),
        warmup_ms: 1000,
        work: work.to_owned(),
    }
}

#[test]
fn trtexec_is_run_with_every_setting_on_its_command_line() {
    let rows = Rows { batch: 2, seq: 3, ids: vec![0; 6], mask: vec![0; 6], types: vec![0; 6], cases: vec![0, 1] };
    let t = trt(Path::new("/tmp"));
    let flags = tensorrt::precision_flags(TURBO_DTYPE_F32).unwrap();
    let a = tensorrt::run_argv(&t, Path::new("/b"), Path::new("/w"), "onnx/model.onnx", 0, &rows, 200, &flags);
    let want = strings(&[
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--gpus",
        "device=0",
        "--mount",
        "type=bind,src=/b,dst=/bundle,readonly",
        "--mount",
        "type=bind,src=/w,dst=/work,readonly",
        &t.image,
        "trtexec",
        "--onnx=/bundle/onnx/model.onnx",
        "--shapes=input_ids:2x3,attention_mask:2x3,token_type_ids:2x3",
        "--loadInputs=input_ids:/work/input_ids.bin,attention_mask:/work/attention_mask.bin,token_type_ids:/work/token_type_ids.bin",
        "--warmUp=1000",
        "--iterations=200",
        "--duration=0",
        "--percentile=99",
        "--noTF32",
    ]);
    assert_eq!(a, want);
    assert_eq!(tensorrt::precision_flags(TURBO_DTYPE_F16).unwrap(), strings(&["--fp16"]));
    assert_eq!(tensorrt::precision_flags(TURBO_DTYPE_BF16).unwrap(), strings(&["--bf16"]));
    assert!(tensorrt::precision_flags(8).is_err());
}

#[test]
fn trtexec_inputs_are_the_rows_in_the_exports_element_type() {
    assert_eq!(tensorrt::input_bytes(&[1, -2], "int32").unwrap(), [1, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff]);
    assert_eq!(tensorrt::input_bytes(&[1], "int64").unwrap(), [1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(tensorrt::input_bytes(&[-1], "int64").unwrap(), [0xff; 8]);
    assert!(tensorrt::input_bytes(&[1], "fp32").is_err());
}

/// trtexec's output in the form TensorRT's samples print it: the version
/// line from trtexec.cpp, and the prolog and performance summary from
/// sampleReporting.cpp, with --percentile=99.
const TRTEXEC_OUT: &str = "\
&&&& RUNNING TensorRT.trtexec [TensorRT v100300] [b17] # trtexec --onnx=/bundle/onnx/model.onnx --percentile=99
[09/25/2026-10:00:00] [I] === Model Options ===
[09/25/2026-10:00:00] [I] Format: ONNX
[09/25/2026-10:00:00] [I] TensorRT version: 10.3.0
[09/25/2026-10:00:40] [I] Warmup completed 2710 queries over 1000 ms
[09/25/2026-10:00:40] [I] Timing trace has 200 queries over 0.0741 s
[09/25/2026-10:00:40] [I]
[09/25/2026-10:00:40] [I] === Trace details ===
[09/25/2026-10:00:40] [I] === Performance summary ===
[09/25/2026-10:00:40] [I] Throughput: 2699.06 qps
[09/25/2026-10:00:40] [I] Latency: min = 0.36377 ms, max = 0.52124 ms, mean = 0.374511 ms, median = 0.372559 ms, percentile(99%) = 0.412598 ms
[09/25/2026-10:00:40] [I] Enqueue Time: min = 0.0114746 ms, max = 0.0491943 ms, mean = 0.0136421 ms, median = 0.0130615 ms, percentile(99%) = 0.0249023 ms
[09/25/2026-10:00:40] [I] H2D Latency: min = 0.0107422 ms, max = 0.0244141 ms, mean = 0.0117093 ms, median = 0.0115967 ms, percentile(99%) = 0.0170898 ms
[09/25/2026-10:00:40] [I] GPU Compute Time: min = 0.339966 ms, max = 0.48999 ms, mean = 0.350241 ms, median = 0.348389 ms, percentile(99%) = 0.385986 ms
[09/25/2026-10:00:40] [I] D2H Latency: min = 0.0112305 ms, max = 0.0161133 ms, mean = 0.0125591 ms, median = 0.0124512 ms, percentile(99%) = 0.0146484 ms
[09/25/2026-10:00:40] [I] Total Host Walltime: 0.0741 s
[09/25/2026-10:00:40] [I] Total GPU Compute Time: 0.0700482 s
&&&& PASSED TensorRT.trtexec [TensorRT v100300] [b17] # trtexec --onnx=/bundle/onnx/model.onnx --percentile=99
";

#[test]
fn trtexec_output_gives_its_version_and_summary() {
    let s = tensorrt::parse(TRTEXEC_OUT).unwrap();
    assert_eq!(s.version, "10.3.0");
    assert_eq!(s.queries, 200);
    assert_eq!(s.qps, 2699.06);
    assert_eq!((s.latency_median, s.latency_p99), (0.372559, 0.412598), "Latency, not H2D or D2H Latency");
    assert_eq!((s.compute_median, s.compute_p99), (0.348389, 0.385986));
}

#[test]
fn trtexec_output_without_a_pass_or_a_figure_is_refused() {
    let failed = TRTEXEC_OUT.replace("&&&& PASSED", "&&&& FAILED");
    assert!(tensorrt::parse(&failed).unwrap_err().contains("did not report &&&& PASSED"));
    let no_p99 = TRTEXEC_OUT.replace(", percentile(99%) = 0.412598 ms", "");
    assert!(tensorrt::parse(&no_p99).unwrap_err().contains("has no percentile(99%)"));
    let no_version = TRTEXEC_OUT.replace("TensorRT version: 10.3.0", "Format: ONNX");
    assert!(tensorrt::parse(&no_version).unwrap_err().contains("no TensorRT version line"));
    let no_trace = TRTEXEC_OUT.replace("Timing trace has", "Timing");
    assert!(tensorrt::parse(&no_trace).unwrap_err().contains("no Timing trace line"));
}

#[test]
fn a_bundle_without_onnx_is_recorded_as_trtexec_not_run() {
    let m = cpu_measurement();
    let work = scratch("trt-work");
    let r = tensorrt::run(&trt(&work), m, 0, 10).unwrap();
    assert_eq!(r.name, "tensorrt");
    assert_eq!(r.role, "kernel");
    assert!(r.measured.is_none());
    assert_eq!(
        r.not_run.as_deref(),
        Some("the bundle carries no FORMAT_ONNX artifact for trtexec to build an engine from")
    );
    assert!(r.commands.is_empty(), "nothing ran");
    assert_eq!(std::fs::read_dir(&*work).unwrap().count(), 0, "no input was written");
    let mut unpinned = trt(&work);
    unpinned.image = "nvcr.io/nvidia/tensorrt:24.08-py3".into();
    assert!(tensorrt::run(&unpinned, m, 0, 10).unwrap_err().contains("is not pinned"));
}
