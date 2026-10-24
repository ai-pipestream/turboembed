//! The reference runners without their programs: the commands they build,
//! the output they parse, and what they refuse or record as not run. The
//! programs themselves need docker and, for TensorRT and OpenVINO, a GPU.

mod common;

use std::path::Path;

use common::*;
use openvino_tests::{ov, rows32};
use turbo::manifest::{Dtype, Pooling};
use turbo::{TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32};
use turbo_bench::cpus::{self, Cpus};
use turbo_bench::docker::{self, parse_port};
use turbo_bench::measure::{Reference, RowKind, Rows};
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
    assert_eq!(tei::dtype(TURBO_DTYPE_F32, true), Ok("float32"));
    assert_eq!(tei::dtype(TURBO_DTYPE_F16, true), Ok("float16"));
    assert!(tei::dtype(TURBO_DTYPE_BF16, true).is_err());
    // On the CPU, F32 runs and F16 is not run, saying why.
    assert_eq!(tei::dtype(TURBO_DTYPE_F32, false), Ok("float32"));
    let why = tei::dtype(TURBO_DTYPE_F16, false).unwrap_err();
    assert!(why.contains("emulates float16"), "{why}");
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
    let text = tei::timing_text(&rt, &got);
    assert_eq!(
        text,
        "round trip (measured) p50 50.000 ms, p99 99.000 ms; TEI's own headers, whole ms, over the same requests: \
         x-total-time p50 50 p99 99, x-tokenization-time p50 1 p99 1, x-queue-time p50 0 p99 0, x-inference-time \
         p50 25 p99 49"
    );
    assert_eq!(tei::total_time_p50(&format!("TEI's procedure; {text}; 4 threads")), Some(50.0));
    // One answer without a header: it is not given a figure.
    let mut some = got.clone();
    some[3][2] = None;
    assert!(tei::timing_text(&rt, &some).contains("x-queue-time not sent, x-inference-time p50 25"));
    some[3][0] = None;
    let text = tei::timing_text(&rt, &some);
    assert!(text.contains("x-total-time not sent"), "{text}");
    assert_eq!(tei::total_time_p50(&text), None);
}

/// A histogram as metrics-exporter-prometheus writes it into TEI's
/// /metrics: its TYPE line, each bucket's cumulative count, +Inf, the sum
/// and the count, for the values recorded in it.
fn prom_histogram(name: &str, les: &[u64], values: &[u64]) -> String {
    let mut t = format!("# TYPE {name} histogram\n");
    for le in les {
        t += &format!("{name}_bucket{{le=\"{le}\"}} {}\n", values.iter().filter(|&&v| v <= *le).count());
    }
    let sum: u64 = values.iter().sum();
    t + &format!("{name}_bucket{{le=\"+Inf\"}} {}\n{name}_sum {sum}\n{name}_count {}\n", values.len(), values.len())
}

/// TEI's /metrics after its batcher recorded these batches, each (inputs,
/// tokens), an empty poll being (0, 0) (core/src/queue.rs), with the
/// buckets router/src/prometheus.rs sets and some of TEI's other metrics
/// around them.
fn tei_metrics(batches: &[(u64, u64)]) -> String {
    let sizes: Vec<u64> = batches.iter().map(|b| b.0).collect();
    let tokens: Vec<u64> = batches.iter().map(|b| b.1).collect();
    let les = |n: u32| (0..n).map(|i| 1u64 << i).collect::<Vec<u64>>();
    format!(
        "# TYPE te_queue_size gauge\nte_queue_size 0\n\n# TYPE te_request_count counter\n\
         te_request_count{{method=\"batch\"}} 22\n\n# TYPE te_embed_duration histogram\n\
         te_embed_duration_bucket{{le=\"+Inf\"}} 22\nte_embed_duration_sum 0.21\nte_embed_duration_count 22\n\n{}\n{}",
        prom_histogram("te_batch_next_size", &les(13), &sizes),
        prom_histogram("te_batch_next_tokens", &les(21), &tokens)
    )
}

#[test]
fn teis_batch_size_histogram_is_read_from_its_metrics() {
    assert_eq!((tei::BATCH_SIZE_METRIC, tei::BATCH_TOKENS_METRIC), ("te_batch_next_size", "te_batch_next_tokens"));
    // Two warmup requests of 32 inputs of 20 tokens, each run whole and
    // followed by an empty poll.
    let mut polls = vec![(32, 640), (0, 0), (32, 640), (0, 0)];
    let before = tei::parse_batch_metrics(&tei_metrics(&polls)).unwrap();
    assert_eq!((before.size.count, before.size.sum), (4.0, 64.0));
    assert_eq!(before.size.buckets.len(), 14);
    assert_eq!(before.size.buckets[0], (1.0, 2.0));
    assert_eq!(before.size.buckets[13], (f64::INFINITY, 4.0));
    assert_eq!(before.tokens.as_ref().unwrap().buckets.len(), 22);

    // Ten timed requests, each run as a batch of 1 then one of 31, then
    // an empty poll.
    let mut split = polls.clone();
    for _ in 0..10 {
        split.extend([(1, 20), (31, 620), (0, 0)]);
    }
    let after = tei::parse_batch_metrics(&tei_metrics(&split)).unwrap();
    assert_eq!(after.size.since(Some(&before.size)).count, 30.0);
    assert_eq!(
        tei::batches_text(Some(&before), Some(&after), 10, 2, 20),
        "TEI's /metrics over the timed requests: 320 inputs in 10 requests, 20 backend batches (te_batch_next_size's \
         30 samples less the 10 empty polls te_batch_next_tokens shows), 2.00 batches per request, mean batch size \
         16.00 inputs (batches by size, 1: 10, 17 to 32: 10)"
    );

    // Two samples, 32 inputs, one in le="1" and one in 17 to 32: one
    // batch of 32 and an empty poll, not batches of 1 and 31.
    polls.extend([(32, 640), (0, 0)]);
    let one = tei::parse_batch_metrics(&tei_metrics(&polls)).unwrap();
    assert_eq!(
        tei::batches_text(Some(&before), Some(&one), 1, 2, 20),
        "TEI's /metrics over the timed requests: 32 inputs in 1 requests, 1 backend batches (te_batch_next_size's 2 \
         samples less the 1 empty polls te_batch_next_tokens shows), 1.00 batches per request, mean batch size 32.00 \
         inputs (batches by size, 17 to 32: 1)"
    );

    // Without the tokens histogram, or with a row of one token, the
    // empty polls cannot be told apart: the inputs only.
    let no_tokens = tei::BatchMetrics { tokens: None, ..one.clone() };
    let text = tei::batches_text(Some(&before), Some(&no_tokens), 1, 2, 20);
    assert!(text.starts_with("TEI's /metrics over the timed requests: 32 inputs in 1 requests; how many"), "{text}");
    assert!(text.ends_with("te_batch_next_tokens, which tells them apart, was not given"), "{text}");
    let text = tei::batches_text(Some(&before), Some(&one), 1, 2, 1);
    assert!(text.contains("is not known") && text.contains("with a row of 1 token"), "{text}");

    // Missing, or not read before the timed requests after a warmup:
    // said, not failed. With no warmup, counting from the server's start
    // is counting the timed requests.
    let none = "# TYPE te_queue_size gauge\nte_queue_size 0\n";
    assert_eq!(tei::parse_batch_metrics(none), None);
    assert!(tei::batches_text(Some(&before), None, 10, 2, 20).contains("gave no te_batch_next_size after"));
    let text = tei::batches_text(None, Some(&after), 10, 2, 20);
    assert!(text.contains("gave no te_batch_next_size before the timed requests, after 2 warmup"), "{text}");
    let cold = tei::parse_batch_metrics(&tei_metrics(&split[4..])).unwrap();
    assert!(tei::batches_text(None, Some(&cold), 10, 0, 20).contains(", 20 backend batches ("));
    assert!(tei::batches_text(Some(&before), Some(&before), 10, 2, 20).ends_with("recorded no backend batch"));
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
        binary: None,
    };
    let r = tei::run(&t, m, None, Some(1), 1, 1).unwrap();
    assert!(r.measured.is_none());
    assert!(r.not_run.unwrap().contains("is not the bundle's tokenizer"));
    assert!(r.commands.is_empty(), "nothing ran");
    let t = Tei { image: "text-embeddings-inference:latest".into(), model_dir: ".".into(), cpus: None, binary: None };
    assert!(tei::run(&t, m, None, Some(1), 1, 1).unwrap_err().contains("is not pinned"));
}

#[test]
fn the_native_router_is_pinned_by_its_bytes_and_started_on_the_loopback() {
    let d = upstream_dir("tei-native-pin");
    let bin = d.join("text-embeddings-router");
    std::fs::write(&bin, b"a router").unwrap();
    let pin = tei::native_pin(&bin).unwrap();
    assert_eq!(pin, format!("text-embeddings-router@sha256:{}", turbo::bundle::sha256_hex(b"a router")));
    assert!(turbo::record::pinned(&pin).is_some(), "{pin} is pinned as a record reads it");
    let a = tei::native_argv(tei::TEI_BIN, turbo_bench::docker::TEI_MODEL, 4242, "float32", "mean", 32, 256);
    assert_eq!(a[0], "<tei-bin>");
    let at = |flag: &str| a[a.iter().position(|x| x == flag).unwrap() + 1].clone();
    assert_eq!(at("--hostname"), "127.0.0.1", "the loopback only");
    assert_eq!(at("--port"), "4242");
    assert_eq!(
        (at("--model-id"), at("--dtype"), at("--pooling")),
        ("<tei-model>".into(), "float32".into(), "mean".into())
    );
    assert_eq!((at("--max-client-batch-size"), at("--max-batch-tokens")), ("32".into(), "16384".into()));
    assert!(tei::native_pin(&d.join("absent")).unwrap_err().contains("--tei-bin"));
}

#[cfg(unix)]
#[test]
fn a_native_router_that_exits_is_an_error_with_its_output_and_leaves_nothing() {
    use std::os::unix::fs::PermissionsExt;
    let m = cpu_measurement();
    let d = upstream_dir("tei-native-exits");
    let bin = d.join("router");
    std::fs::write(&bin, "#!/bin/sh\necho no model here\nexit 3\n").unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    let t = Tei { image: String::new(), model_dir: d.to_owned(), cpus: None, binary: Some(bin) };
    let e = tei::run(&t, m, None, Some(1), 1, 1).unwrap_err();
    assert!(e.contains("TEI did not become healthy") && e.contains("no model here"), "{e}");
    let log = std::env::temp_dir().join(format!("turbo-bench-tei-{}.log", std::process::id()));
    assert!(!log.exists(), "the router's output file is removed");
}

#[path = "runners/openvino.rs"]
mod openvino_tests;
#[path = "runners/routing.rs"]
mod routing;
#[path = "runners/tensorrt.rs"]
mod tensorrt_tests;
