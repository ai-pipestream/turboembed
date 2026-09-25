//! The reference runners without their programs: the commands they build,
//! the output they parse, and what they refuse or record as not run. The
//! programs themselves need docker and, for TensorRT and OpenVINO, a GPU.

mod common;

use std::path::Path;

use common::*;
use turbo::manifest::{Dtype, Pooling};
use turbo::{TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32};
use turbo_bench::cpus::{self, Cpus};
use turbo_bench::docker::{self, parse_port};
use turbo_bench::measure::{RowKind, Rows};
use turbo_bench::openvino::{self, OpenVino};
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
        format!("-v/etc:/x@sha256:{DIGEST}"),
        format!("--privileged@sha256:{DIGEST}"),
        format!("GHCR.io/tei@sha256:{DIGEST}"),
        format!("ghcr.io/tei name@sha256:{DIGEST}"),
        format!("@sha256:{DIGEST}"),
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
    let a = tei::run_argv(
        &image,
        Path::new("/models/minilm"),
        "turbo-bench-tei-7",
        Some(1),
        None,
        "float32",
        "mean",
        32,
        64,
    );
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
    ]);
    assert_eq!(a, want);
    assert!(!a.iter().any(|s| s.contains("truncate")), "the request says truncate: false; the flag is not portable");
    // The CPU image: no GPU; a batch past TEI's default token budget raises it.
    let a = tei::run_argv(&image, Path::new("/m"), "c", None, None, "float32", "cls", 512, 64);
    assert!(!a.iter().any(|s| s == "--gpus"));
    assert!(!a.iter().any(|s| s == "--cpuset-cpus" || s.contains("THREADS")), "unpinned: docker's defaults");
    assert_eq!(a[a.len() - 1], "32768");
}

/// A 9950X3D's topology: 16 cores of two threads, CPU n and n + 16 on
/// one core; the CCD with the stacked cache is cores 0-7.
fn ryzen(name: &str) -> Scratch {
    let d = scratch(name);
    smt_topology(&d, 16);
    d
}

#[test]
fn the_physical_cores_are_the_distinct_cores_of_the_list() {
    let sys = ryzen("topology");
    let cores = |l: &str| Cpus::parse(l, &sys).unwrap().cores;
    assert_eq!(cores("0-7,16-23"), 8, "one CCD: 8 cores, 16 threads");
    assert_eq!(cores("0-15"), 16, "one thread on each of 16 cores");
    assert_eq!(cores("0-31"), 16);
    assert_eq!(cores("3,19"), 1);
    // A second package's core 0 is another core.
    let two = scratch("topology-two");
    smt_topology(&two, 2);
    std::fs::write(two.join("cpu3/topology/physical_package_id"), "1\n").unwrap();
    assert_eq!(Cpus::parse("1,3", &two).unwrap().cores, 2);
    // A processor the tree does not describe is refused, naming it.
    let e = Cpus::parse("0-32", &sys).unwrap_err();
    assert!(e.contains("processor 32") && e.contains("cpu32/topology/physical_package_id"), "{e}");
    assert!(Cpus::parse("3-1", &sys).unwrap_err().contains("runs backwards"));
}

#[test]
fn with_cpus_tei_gets_those_processors_and_every_thread_count_its_image_reads() {
    let image = format!("ghcr.io/huggingface/text-embeddings-inference@sha256:{DIGEST}");
    let sys = ryzen("argv");
    let ccd = Cpus::parse("0-7,16-23", &sys).unwrap();
    let a = tei::run_argv(&image, Path::new("/m"), "c", None, Some(&ccd), "float32", "mean", 32, 256);
    let want = strings(&[
        "docker",
        "run",
        "--detach",
        "--rm",
        "--pull",
        "never",
        "--name",
        "c",
        "--cpuset-cpus",
        "0-7,16-23",
        "--env",
        "OMP_NUM_THREADS=8",
        "--env",
        "MKL_NUM_THREADS=8",
        "--env",
        "RAYON_NUM_THREADS=16",
        "--publish",
        "127.0.0.1::80",
    ]);
    assert_eq!(a[..want.len()], want[..]);
    // Every option before the image: docker's, not the router's.
    let at = a.iter().position(|s| *s == image).unwrap();
    assert!(a[..at].contains(&"--cpuset-cpus".to_owned()));
    let all = Cpus::parse("0-31", &sys).unwrap();
    let a = tei::run_argv(&image, Path::new("/m"), "c", None, Some(&all), "float32", "mean", 32, 256);
    for s in ["0-31", "OMP_NUM_THREADS=16", "MKL_NUM_THREADS=16", "RAYON_NUM_THREADS=32"] {
        assert!(a.contains(&s.to_owned()), "{s}: {a:?}");
    }
}

#[test]
fn the_procedure_says_what_each_side_ran_on() {
    let env = tei::parse_env(IMAGE_ENV).unwrap();
    assert_eq!(cpus::image_value(&env, "RAYON_NUM_THREADS"), Some("8"));
    assert_eq!(cpus::image_value(&env, "OMP_NUM_THREADS"), None);
    assert_eq!(tei::parse_env("null\n").unwrap(), Vec::<String>::new());
    assert!(tei::parse_env("sha256:abc").is_err());

    let sys = ryzen("procedure");
    let c = Cpus::parse("0-7,16-23", &sys).unwrap();
    assert_eq!(
        cpus::procedure(Some(&c), Some(16), &env),
        "the library ran pinned to CPUs 0-7,16-23 with TURBO_CPU_THREADS=16 threads, one per logical CPU; TEI ran \
         with --cpuset-cpus 0-7,16-23 and OMP_NUM_THREADS=8, MKL_NUM_THREADS=8, RAYON_NUM_THREADS=16 (MKL's \
         threads one per physical core of the list, as MKL itself defaults, since two on a core contend for its \
         vector units; candle's rayon threads one per logical CPU), over the image's RAYON_NUM_THREADS=8; its \
         ONNX Runtime and tokenizer threads counted from those CPUs"
    );
    // Without --cpus: today's command, and the counts each side had.
    assert_eq!(
        cpus::procedure(None, Some(32), &env),
        "the library ran unpinned with 32 threads; TEI ran unpinned with the image's thread settings: \
         OMP_NUM_THREADS=unset, MKL_NUM_THREADS=unset, RAYON_NUM_THREADS=8"
    );
    // A GPU measured: no library thread count to state.
    assert_eq!(
        cpus::procedure(None, None, &[]),
        "the tool ran unpinned; TEI ran unpinned with the image's thread settings: OMP_NUM_THREADS=unset, \
         MKL_NUM_THREADS=unset, RAYON_NUM_THREADS=unset"
    );
    let eight = Cpus::parse("0-7", &sys).unwrap();
    assert!(!cpus::procedure(Some(&eight), Some(8), &env).contains("over the image's"), "8 is the image's own");
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
fn teis_timing_headers_are_read_as_its_router_writes_them() {
    assert_eq!(tei::TIMING_HEADERS, ["x-total-time", "x-tokenization-time", "x-queue-time", "x-inference-time"]);
    // As router/src/lib.rs writes them: whole milliseconds, with the
    // compute headers beside them.
    let mut h = ureq::http::HeaderMap::new();
    for (k, v) in [
        ("x-compute-type", "gpu+optimized"),
        ("x-compute-time", "12"),
        ("x-total-time", "12"),
        ("x-tokenization-time", "2"),
        ("x-queue-time", "0"),
        ("x-inference-time", "9"),
    ] {
        h.insert(k, v.parse().unwrap());
    }
    assert_eq!(tei::timing_headers(&h), [Some(12), Some(2), Some(0), Some(9)]);
    h.insert("x-queue-time", "soon".parse().unwrap());
    h.remove("x-inference-time");
    assert_eq!(tei::timing_headers(&h), [Some(12), Some(2), None, None]);

    // Each header's p50 and p99 over the requests, by nearest rank.
    let rt: Vec<f64> = (1..=100).map(f64::from).collect();
    let got: Vec<[Option<u64>; 4]> = (1..=100).map(|i| [Some(i), Some(1), Some(0), Some(i / 2)]).collect();
    assert_eq!(
        tei::timing_text(&rt, &got),
        "round trip (measured) p50 50.000 ms, p99 99.000 ms; TEI's own headers, whole ms, over the same requests: \
         x-total-time p50 50 p99 99, x-tokenization-time p50 1 p99 1, x-queue-time p50 0 p99 0, x-inference-time \
         p50 25 p99 49"
    );
    // One answer without a header: it is not given a figure.
    let mut some = got.clone();
    some[3][2] = None;
    assert!(tei::timing_text(&rt, &some).contains("x-queue-time not sent, x-inference-time p50 25"));
}

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
    // What TEI computes is known only when every row is one length.
    assert_eq!(tei::computed_tokens(&m.rows), None, "cases of different lengths");
    let mut same = rows32();
    for r in 0..32 {
        same.mask[r * 64..r * 64 + 10].fill(1);
    }
    assert_eq!(tei::computed_tokens(&same), Some(32 * 10));
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
    let e = tei::check_model_dir(&d, m).unwrap_err();
    assert!(e.starts_with("<tei-model>/tokenizer.json is not the bundle's tokenizer"), "no host path: {e}");
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
        cpus: None,
    };
    let r = tei::run(&t, m, None, Some(1), 1, 1).unwrap();
    assert!(r.measured.is_none());
    assert!(r.not_run.unwrap().contains("is not the bundle's tokenizer"));
    assert!(r.commands.is_empty(), "nothing ran");
    let t = Tei { image: "text-embeddings-inference:latest".into(), model_dir: ".".into(), cpus: None };
    assert!(tei::run(&t, m, None, Some(1), 1, 1).unwrap_err().contains("is not pinned"));
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
    let rows = Rows {
        kind: RowKind::Mixed,
        batch: 2,
        seq: 3,
        ids: vec![0; 6],
        mask: vec![0; 6],
        types: vec![0; 6],
        cases: vec![0, 1],
    };
    let t = trt(Path::new("/tmp"));
    let (graph, flags) = tensorrt::precision(TURBO_DTYPE_F32).unwrap();
    assert_eq!(graph, None, "F32: the upstream graph");
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
    // TensorRT 11 has no --fp16 or --bf16: F16 and BF16 build strongly
    // typed from the graph converted to that dtype, as TensorRT 10 can.
    assert_eq!(tensorrt::precision(TURBO_DTYPE_F16).unwrap(), (Some(Dtype::F16), strings(&["--stronglyTyped"])));
    assert_eq!(tensorrt::precision(TURBO_DTYPE_BF16).unwrap(), (Some(Dtype::Bf16), strings(&["--stronglyTyped"])));
    assert!(tensorrt::precision(8).is_err());
}

#[test]
fn trtexec_inputs_are_the_rows_in_the_exports_element_type() {
    assert_eq!(tensorrt::input_bytes(&[1, -2], "int32").unwrap(), [1, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff]);
    assert_eq!(tensorrt::input_bytes(&[1], "int64").unwrap(), [1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(tensorrt::input_bytes(&[-1], "int64").unwrap(), [0xff; 8]);
    assert!(tensorrt::input_bytes(&[1], "fp32").is_err());
}

#[test]
fn trtexec_input_names_are_plain_names() {
    let names = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    tensorrt::check_inputs(&names(&["input_ids", "attention_mask", "token_type_ids"])).unwrap();
    tensorrt::check_inputs(&names(&["onnx::Input.0"])).unwrap_err();
    tensorrt::check_inputs(&names(&["input.1", "Mask_2"])).unwrap();
    for bad in [".", "..", "", "../x", "a/b", "a,b", "a:b", "a b", "-x"] {
        let e = tensorrt::check_inputs(&names(&[bad])).unwrap_err();
        assert!(e.contains("is not an input name of [A-Za-z0-9_.]+"), "{bad:?}: {e}");
    }
}

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

// ---- the bundle's ONNX file ----

/// The MiniLM recipe's manifest, sealed over stand-in files: every path it
/// names listed with a size and hash, and the reference's produced_by as a
/// run fills it in.
fn recipe_manifest() -> turbo::manifest::Manifest {
    let r: serde_json::Value =
        serde_json::from_slice(&std::fs::read(workspace().join("bundle/recipes/all-minilm-l6-v2.json")).unwrap())
            .unwrap();
    let mut m = r["manifest"].clone();
    let run =
        serde_json::json!({ "tool": "t", "tool_version": "1", "container": "c", "args": [], "reproducible": false });
    m["reference"]["produced_by"] = run.clone();
    for a in m["artifacts"].as_array_mut().unwrap() {
        if let Some(from) = a.get("produced_by").map(|p| p["from"].clone()) {
            a["produced_by"] = run.clone();
            a["produced_by"]["from"] = from;
        }
    }
    let paths = [
        "tokenizer.json",
        "weights/model.safetensors",
        "onnx/model.onnx",
        "onnx/model-f16.onnx",
        "reference/reference.safetensors",
    ];
    m["files"] = paths.iter().map(|p| serde_json::json!({ "path": p, "size": 1, "sha256": "0".repeat(64) })).collect();
    turbo::manifest::Manifest::parse(&serde_json::to_vec(&m).unwrap()).unwrap_or_else(|e| panic!("{}", e.message))
}

#[test]
fn the_runners_find_the_recipes_onnx_file_by_its_format() {
    let m = recipe_manifest();
    assert_eq!(turbo_bench::onnx::file(&m, None, "for a program").unwrap(), "onnx/model.onnx");
    assert_eq!(turbo_bench::onnx::file(&m, Some(Dtype::F16), "for a program").unwrap(), "onnx/model-f16.onnx");
    assert_eq!(
        turbo_bench::onnx::file(&m, Some(Dtype::Bf16), "for a program").unwrap_err(),
        "the bundle carries no FORMAT_ONNX artifact in DTYPE_BF16 for a program"
    );
    let bundle = turbo::bundle::Bundle::open(&tiny_bundle()).unwrap();
    assert_eq!(
        turbo_bench::onnx::file(&bundle.manifest, None, "for a program").unwrap_err(),
        "the bundle carries no FORMAT_ONNX artifact for a program"
    );
    // The inputs both runners default to are the export's.
    assert_eq!(trt(Path::new("/w")).inputs, ["input_ids", "attention_mask", "token_type_ids"]);
    assert_eq!(ov(Path::new("/w")).inputs, trt(Path::new("/w")).inputs);
    assert_eq!(
        (trt(Path::new("/w")).input_dtype.as_str(), ov(Path::new("/w")).input_dtype.as_str()),
        ("int64", "int64")
    );
}

// ---- OpenVINO ----

fn ov(work: &Path) -> OpenVino {
    OpenVino {
        image: format!("openvino/ubuntu24_dev@sha256:{DIGEST}"),
        benchmark_app: "benchmark_app".into(),
        inputs: ["input_ids".into(), "attention_mask".into(), "token_type_ids".into()],
        input_dtype: "int64".into(),
        dri: "/dev/dri".into(),
        work: work.to_owned(),
    }
}

#[test]
fn benchmark_app_is_run_with_every_setting_on_its_command_line() {
    let rows = Rows {
        kind: RowKind::Mixed,
        batch: 2,
        seq: 3,
        ids: vec![0; 6],
        mask: vec![0; 6],
        types: vec![0; 6],
        cases: vec![0, 1],
    };
    let o = ov(Path::new("/tmp"));
    let precision = openvino::infer_precision(TURBO_DTYPE_F32).unwrap();
    let a = openvino::run_argv(&o, Path::new("/b"), Path::new("/w"), "onnx/model.onnx", 993, &rows, 200, precision, 99);
    let want = strings(&[
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--device",
        "/dev/dri",
        "--group-add",
        "993",
        "--mount",
        "type=bind,src=/b,dst=/bundle,readonly",
        "--mount",
        "type=bind,src=/w,dst=/work,readonly",
        &o.image,
        "benchmark_app",
        "-m",
        "/bundle/onnx/model.onnx",
        "-d",
        "GPU",
        "-hint",
        "latency",
        "-api",
        "sync",
        "-nireq",
        "1",
        "-niter",
        "200",
        "-shape",
        "input_ids[2,3],attention_mask[2,3],token_type_ids[2,3]",
        "-i",
        "input_ids:/work/input_ids.bin,attention_mask:/work/attention_mask.bin,token_type_ids:/work/token_type_ids.bin",
        "-infer_precision",
        "f32",
        "-latency_percentile",
        "99",
    ]);
    assert_eq!(a, want);
    // The two runs differ in their percentile and nothing else.
    let b = openvino::run_argv(&o, Path::new("/b"), Path::new("/w"), "onnx/model.onnx", 993, &rows, 200, precision, 50);
    let differ: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
    assert_eq!((a.len(), differ), (b.len(), vec![a.len() - 1]));
    assert_eq!(openvino::PERCENTILES, [50, 99]);
    assert_eq!(openvino::infer_precision(TURBO_DTYPE_F16).unwrap(), "f16");
    let e = openvino::infer_precision(TURBO_DTYPE_BF16).unwrap_err();
    assert!(e.contains("no inference precision"), "{e}");
}

#[test]
fn the_container_gets_the_render_nodes_group() {
    let d = scratch("dri");
    let e = openvino::render_group(&d).unwrap_err();
    assert!(e.contains("no render node"), "{e}");
    std::fs::write(d.join("card0"), "").unwrap();
    assert!(openvino::render_group(&d).is_err(), "a card node is not a render node");
    std::fs::write(d.join("renderD128"), "").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let gid = std::fs::metadata(d.join("renderD128")).unwrap().gid();
        assert_eq!(openvino::render_group(&d).unwrap(), gid);
    }
    assert!(openvino::render_group(&d.join("absent")).is_err());
}

#[test]
fn benchmark_app_output_gives_its_version_and_report() {
    let r = openvino::parse(BENCHMARK_APP_OUT, 50).unwrap();
    assert_eq!(r.version, "2025.3.0-19807-44526285f24-releases/2025/3");
    assert_eq!(r.count, 200);
    assert_eq!(r.duration_ms, 412.64);
    assert_eq!((r.latency_ms, r.average_ms), (2.03, 2.05));
    assert_eq!(r.throughput_fps, 484.68);
    let r = openvino::parse(&p99_out(), 99).unwrap();
    assert!((r.latency_ms - 0.9612).abs() < 1e-12 && (r.average_ms - 0.90433).abs() < 1e-12, "{r:?}");

    // benchmark_app's C++ form: the version as ov::Version prints it, the
    // latency block as LatencyMetrics::write_to_slog does.
    let cpp = BENCHMARK_APP_OUT
        .replace(
            "[ INFO ] Build ................................. 2025.3.0-19807-44526285f24-releases/2025/3\n[ INFO ] \n[ INFO ] Device info:",
            "[ INFO ] OpenVINO Runtime\n    Version : 2025.3.0\n    Build   : 2025.3.0-19807-44526285f24-releases/2025/3\n[ INFO ] Device info:",
        )
        .replace("Count:            200", "Count:               200")
        .replace("   Median:        2.03 ms", "   Median:           2.03 ms");
    let r = openvino::parse(&cpp, 50).unwrap();
    assert_eq!((r.version.as_str(), r.latency_ms, r.count), ("2025.3.0-19807-44526285f24-releases/2025/3", 2.03, 200));
}

#[test]
fn benchmark_app_output_without_a_figure_is_refused() {
    let e = openvino::parse(BENCHMARK_APP_OUT, 99).unwrap_err();
    assert!(e.contains("no 99 percentile: line"), "the percentile asked for, not another: {e}");
    let e = openvino::parse(&p99_out(), 50).unwrap_err();
    assert!(e.contains("no Median: line"), "{e}");
    let no_version = BENCHMARK_APP_OUT.replace("[ INFO ] OpenVINO:\n", "");
    assert!(openvino::parse(&no_version, 50).unwrap_err().contains("no Build line under OpenVINO:"));
    let no_count = BENCHMARK_APP_OUT.replace("Count:", "Counted:");
    assert!(openvino::parse(&no_count, 50).unwrap_err().contains("no Count line"));
    let no_fps = BENCHMARK_APP_OUT.replace("484.68 FPS", "484.68");
    assert!(openvino::parse(&no_fps, 50).unwrap_err().contains("no Throughput line"));
    let seconds = BENCHMARK_APP_OUT.replace("Median:        2.03 ms", "Median:        2.03 s");
    assert!(openvino::parse(&seconds, 50).unwrap_err().contains("no Median: line"));
    let failed = "[ ERROR ] Exception from src/inference/src/cpp/core.cpp:112\n";
    assert!(openvino::parse(failed, 50).is_err());
}

/// 32 padded rows of 64 tokens, as benchmark_app's static shape takes them.
fn rows32() -> Rows {
    let n = 32 * 64;
    Rows {
        kind: RowKind::Mixed,
        batch: 32,
        seq: 64,
        ids: vec![0; n],
        mask: vec![0; n],
        types: vec![0; n],
        cases: vec![0; 32],
    }
}

#[test]
fn the_two_runs_make_one_measured_reference() {
    let p50 = openvino::parse(BENCHMARK_APP_OUT, 50).unwrap();
    let mut p99 = p50.clone();
    p99.latency_ms = 2.6;
    let image = ov(Path::new("/w")).image;
    let log = docker::Log { commands: vec![strings(&["docker", "run"]), strings(&["docker", "run"])] };
    let r = openvino::measured(&image, log.clone(), "two runs", p50.clone(), p99.clone(), &rows32()).unwrap();
    assert_eq!((r.name.as_str(), r.role.as_str()), ("openvino", "kernel"));
    assert!(turbo::record::REFERENCES.contains(&(r.name.as_str(), r.role.as_str())));
    assert_eq!(r.version, "2025.3.0-19807-44526285f24-releases/2025/3");
    assert_eq!(r.commands.len(), 2, "both runs are recorded");
    let m = r.measured.unwrap();
    assert_eq!((m.iterations, m.p50_ms, m.p99_ms, m.min_cosine), (200, 2.03, 2.6, None));
    assert_eq!(m.computed_tokens, Some(32 * 64), "every position of the static shape");
    assert!((m.rows_per_second - 200.0 * 32.0 / 0.41264).abs() < 1e-6, "{}", m.rows_per_second);
    assert!(r.procedure.contains("484.68 FPS"));

    let mut under = p99.clone();
    under.latency_ms = 2.0;
    let e = openvino::measured(&image, log.clone(), "", p50.clone(), under, &rows32()).unwrap_err();
    assert!(e.contains("under its median run's"), "{e}");
    let mut other = p99;
    other.count = 199;
    assert!(openvino::measured(&image, log, "", p50, other, &rows32()).unwrap_err().contains("two runs differ"));
}

#[test]
fn a_bundle_without_onnx_is_recorded_as_benchmark_app_not_run() {
    let m = cpu_measurement();
    let work = scratch("ov-work");
    let r = openvino::run(&ov(&work), m, 10).unwrap();
    assert_eq!((r.name.as_str(), r.role.as_str()), ("openvino", "kernel"));
    assert!(r.measured.is_none());
    assert_eq!(r.not_run.as_deref(), Some("the bundle carries no FORMAT_ONNX artifact for benchmark_app to compile"));
    assert!(r.commands.is_empty(), "nothing ran");
    assert_eq!(std::fs::read_dir(&*work).unwrap().count(), 0, "no input was written");
    let mut unpinned = ov(&work);
    unpinned.image = "openvino/ubuntu24_dev:2025.3.0".into();
    assert!(openvino::run(&unpinned, m, 10).unwrap_err().contains("--openvino-image"));
}

// ---- which programs a backend gets ----

#[test]
fn each_backend_gets_its_reference_programs() {
    use turbo_bench::{applies, wanted};
    assert_eq!(applies("cuda"), [TEI, TRT]);
    assert_eq!(applies("cpu"), [TEI]);
    assert_eq!(applies("levelzero"), [TEI, openvino::NAME], "OpenVINO, and TEI's CPU image as the end-to-end baseline");
    assert!(applies("metal").is_empty(), "none yet");
    for backend in ["cuda", "cpu", "levelzero", "metal"] {
        for name in applies(backend) {
            assert!(turbo::record::REFERENCES.iter().any(|r| r.0 == *name), "{name}");
        }
    }
    wanted("levelzero", |n| n == TEI || n == openvino::NAME).unwrap();
    let e = wanted("levelzero", |n| n == TEI).unwrap_err();
    assert_eq!(e, "levelzero: OpenVINO is a reference here: give --openvino-image, or --no-openvino");
    let e = wanted("levelzero", |_| true).unwrap_err();
    assert_eq!(e, "levelzero: TensorRT is not a reference for this backend");
    let e = wanted("cuda", |n| n != openvino::NAME).map(|_| ());
    assert!(e.is_ok());
    let e = wanted("cuda", |_| true).unwrap_err();
    assert_eq!(e, "cuda: OpenVINO is not a reference for this backend");
    wanted("metal", |_| false).unwrap();
    assert_eq!(wanted("metal", |n| n == TEI).unwrap_err(), "metal: TEI is not a reference for this backend");
}
