//! No record holds a path of the machine that made it. Every reference
//! runner is run through its own code on a bundle, a model directory and
//! a work directory under a path with a sentinel user name in it, with
//! `docker` a script that prints each program's report as the program
//! prints it and TEI's HTTP API answered from a local socket. The
//! commands run name the real paths; the record names none.

#![cfg(unix)]

mod common;

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

use common::*;
use serde_json::{Value, json};
use turbo::TURBO_PRECISION_MODEL;
use turbo::bundle::sha256_hex;
use turbo_bench::cpus::Cpus;
use turbo_bench::measure::{Measurement, Plan, RowKind, measure};
use turbo_bench::openvino::{self, OpenVino};
use turbo_bench::tei::{self, Tei};
use turbo_bench::tensorrt::{self, TensorRt};

const SENTINEL: &str = "/home/sentinel-user/x";
const DIGEST: &str = "4f5c6a1d7f3b2e8a9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b";

/// A copy of the small bundle at `dir` with an ONNX artifact added, the
/// manifest's files list and hashes kept true.
fn bundle_with_onnx(dir: &Path) -> PathBuf {
    let b = bundle_copy(dir);
    let onnx = b"an ONNX graph, as far as the runners are concerned";
    std::fs::create_dir_all(b.join("onnx")).unwrap();
    std::fs::write(b.join("onnx/model.onnx"), onnx).unwrap();
    let mut m: Value = serde_json::from_slice(&std::fs::read(b.join("manifest.json")).unwrap()).unwrap();
    m["artifacts"].as_array_mut().unwrap().push(json!({
        "name": "onnx-f32", "format": "FORMAT_ONNX", "files": ["onnx/model.onnx"], "backends": [],
        "graph_input": "INPUT_TOKEN_IDS", "graph_output": "OUTPUT_HIDDEN_STATES"
    }));
    let files = m["files"].as_array_mut().unwrap();
    files.push(json!({ "path": "onnx/model.onnx", "size": onnx.len(), "sha256": sha256_hex(onnx) }));
    files.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    std::fs::write(b.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    b
}

/// TEI's HTTP API on a local port, answering as its router does: /health,
/// /info, /decode and /tokenize as a round trip that gives the ids back,
/// and /embed with each row's expected vector and TEI's timing headers.
fn tei_server(ms: &[&Measurement]) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let vectors: HashMap<Vec<i32>, Vec<f32>> = ms
        .iter()
        .flat_map(|m| (0..m.rows.batch as usize).map(|r| (m.rows.live(r).to_vec(), m.expected[r].clone())))
        .collect();
    std::thread::spawn(move || {
        let mut decoded: HashMap<String, Vec<i32>> = HashMap::new();
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            let mut length = 0;
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if h.trim().is_empty() {
                    break;
                }
                if let Some((k, v)) = h.split_once(':')
                    && k.eq_ignore_ascii_case("content-length")
                {
                    length = v.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let path = request.split_whitespace().nth(1).unwrap_or("");
            let answer = match path {
                "/health" => json!(null),
                "/info" => json!({ "version": "1.8.0", "model_dtype": "float32" }),
                "/decode" => {
                    let rows: Vec<Vec<i32>> = serde_json::from_value(body["ids"].clone()).unwrap();
                    let texts: Vec<String> = (0..rows.len()).map(|i| format!("text {i}")).collect();
                    decoded = texts.iter().cloned().zip(rows).collect();
                    json!(texts)
                }
                "/tokenize" => {
                    let texts: Vec<String> = serde_json::from_value(body["inputs"].clone()).unwrap();
                    json!(
                        texts
                            .iter()
                            .map(|t| decoded[t].iter().map(|id| json!({ "id": id })).collect::<Vec<_>>())
                            .collect::<Vec<_>>()
                    )
                }
                "/embed" => {
                    let rows: Vec<Vec<i32>> = serde_json::from_value(body["inputs"].clone()).unwrap();
                    json!(rows.iter().map(|r| vectors[r].clone()).collect::<Vec<_>>())
                }
                _ => json!({ "error": "not found" }),
            };
            let text = if path == "/health" { String::new() } else { answer.to_string() };
            let timing = if path == "/embed" {
                "x-compute-type: cpu\r\nx-total-time: 7\r\nx-tokenization-time: 1\r\nx-queue-time: 0\r\n\
                 x-inference-time: 5\r\n"
            } else {
                ""
            };
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n{timing}Content-Length: {}\r\nConnection: \
                 close\r\n\r\n{text}",
                text.len()
            );
        }
    });
    port
}

/// A `docker` that logs its arguments to `calls` and answers each command
/// the runners give: the image is present, the TEI container starts and
/// publishes `port`, and trtexec and benchmark_app print their reports.
fn fake_docker(bin: &Path, out: &Path, port: u16) {
    std::fs::create_dir_all(bin).unwrap();
    std::fs::write(out.join("trtexec.out"), TRTEXEC_OUT).unwrap();
    std::fs::write(out.join("p50.out"), BENCHMARK_APP_OUT).unwrap();
    let p99 = BENCHMARK_APP_OUT.replace("   Median:        2.03 ms", "   99 percentile:     2.61 ms");
    std::fs::write(out.join("p99.out"), p99).unwrap();
    let o = out.display();
    let script = format!(
        "#!/bin/sh\n\
         printf '%s\\n' \"$*\" >> {o}/calls\n\
         case \"$*\" in\n\
         'image inspect --format {{{{json .Config.Env}}}}'*) echo '{IMAGE_ENV}' ;;\n\
         'image inspect'*) echo sha256:{DIGEST} ;;\n\
         'port '*) echo 127.0.0.1:{port} ;;\n\
         'rm '*) ;;\n\
         *' --detach '*) echo 0123456789ab ;;\n\
         *' trtexec '*) cat {o}/trtexec.out ;;\n\
         *'-latency_percentile 99'*) cat {o}/p99.out ;;\n\
         *' benchmark_app '*) cat {o}/p50.out ;;\n\
         *) echo \"unexpected: $*\" >&2; exit 1 ;;\n\
         esac\n"
    );
    let path = bin.join("docker");
    std::fs::write(&path, script).unwrap();
    let mut p = std::fs::metadata(&path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut p, 0o755);
    std::fs::set_permissions(&path, p).unwrap();
}

#[test]
fn a_record_names_no_host_path_and_the_commands_run_do() {
    let root = scratch("redaction");
    let home = root.join(SENTINEL.trim_start_matches('/'));
    assert!(home.display().to_string().contains(SENTINEL));
    let bundle = bundle_with_onnx(&home.join("bundle"));
    let m = measure(&Plan {
        bundle: bundle.clone(),
        device: "cpu".into(),
        precision: TURBO_PRECISION_MODEL,
        batch: Some(4),
        seq: None,
        rows: RowKind::Mixed,
        warmup: 1,
        iterations: 5,
    })
    .unwrap_or_else(|e| panic!("{e}"));
    // Dense rows, the long cases cut to 40 tokens.
    let dense = measure(&Plan {
        bundle: bundle.clone(),
        device: "cpu".into(),
        precision: TURBO_PRECISION_MODEL,
        batch: Some(4),
        seq: Some(40),
        rows: RowKind::Dense,
        warmup: 1,
        iterations: 5,
    })
    .unwrap_or_else(|e| panic!("{e}"));

    let model = home.join("upstream");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::copy(bundle.join("tokenizer.json"), model.join("tokenizer.json")).unwrap();
    std::fs::copy(bundle.join("weights/model.safetensors"), model.join("model.safetensors")).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    let work = home.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let dri = root.join("dri");
    std::fs::create_dir_all(&dri).unwrap();
    std::fs::write(dri.join("renderD128"), "").unwrap();

    let port = tei_server(&[&m, &dense]);
    fake_docker(&root.join("bin"), &root, port);
    let path = format!("{}:{}", root.join("bin").display(), std::env::var("PATH").unwrap_or_default());
    // The only test in this binary, so no other thread reads the
    // environment while it changes.
    unsafe { std::env::set_var("PATH", path) };

    let inputs: [String; 3] = ["input_ids".into(), "attention_mask".into(), "token_type_ids".into()];
    let tei_run = tei::run(
        &Tei {
            image: format!("ghcr.io/huggingface/text-embeddings-inference@sha256:{DIGEST}"),
            model_dir: model.clone(),
            cpus: Some(Cpus::parse("0-1", &smt_topology(&root.join("sys"), 1)).unwrap()),
        },
        &m,
        None,
        Some(2),
        1,
        3,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    // Dense rows, before the model directory is spoiled below.
    let dense_run = tei::run(
        &Tei {
            image: format!("ghcr.io/huggingface/text-embeddings-inference@sha256:{DIGEST}"),
            model_dir: model.clone(),
            cpus: None,
        },
        &dense,
        None,
        Some(2),
        1,
        3,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let trt_run = tensorrt::run(
        &TensorRt {
            image: format!("nvcr.io/nvidia/tensorrt@sha256:{DIGEST}"),
            trtexec: "trtexec".into(),
            inputs: inputs.clone(),
            input_dtype: "int64".into(),
            warmup_ms: 100,
            work: work.clone(),
        },
        &m,
        0,
        5,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let ov_run = openvino::run(
        &OpenVino {
            image: format!("openvino/ubuntu24_dev@sha256:{DIGEST}"),
            benchmark_app: "benchmark_app".into(),
            inputs,
            input_dtype: "int64".into(),
            dri,
            work: work.clone(),
        },
        &m,
        5,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    // TEI on another model's files: the reason names no path either.
    std::fs::write(model.join("tokenizer.json"), "{}").unwrap();
    let tei_not_run = tei::run(
        &Tei {
            image: format!("ghcr.io/huggingface/text-embeddings-inference@sha256:{DIGEST}"),
            model_dir: model,
            cpus: None,
        },
        &m,
        None,
        Some(2),
        1,
        1,
    )
    .unwrap();
    assert!(tei_not_run.not_run.as_deref().unwrap().starts_with("<tei-model>/tokenizer.json"));

    for r in [&tei_run, &trt_run, &ov_run] {
        assert!(r.measured.is_some(), "{r:?}");
    }
    let joined =
        |r: &turbo::record::ReferenceRun| r.commands.iter().map(|c| c.join(" ")).collect::<Vec<_>>().join("\n");
    assert!(joined(&tei_run).contains("type=bind,src=<tei-model>,dst=/model,readonly"), "{}", joined(&tei_run));
    // --cpus: the container's processors and threads are in the command
    // run, and both sides' are in the procedure, over the image's own.
    assert!(
        joined(&tei_run)
            .contains("--cpuset-cpus 0-1 --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env RAYON_NUM_THREADS=2"),
        "{}",
        joined(&tei_run)
    );
    assert!(joined(&tei_run).contains("docker image inspect --format {{json .Config.Env}}"));
    assert!(
        tei_run.procedure.ends_with(
            "; the library ran pinned to CPUs 0-1 with TURBO_CPU_THREADS=2 threads, one per logical CPU; TEI ran \
             with --cpuset-cpus 0-1 and OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, RAYON_NUM_THREADS=2 (MKL's threads \
             one per physical core of the list, as MKL itself defaults, since two on a core contend for its vector \
             units; candle's rayon threads one per logical CPU), over the image's RAYON_NUM_THREADS=8; its ONNX \
             Runtime and tokenizer threads counted from those CPUs"
        ),
        "{}",
        tei_run.procedure
    );
    // TEI's own timing, beside the round trip that is the measurement.
    let measured = tei_run.measured.as_ref().unwrap();
    let round_trip = format!("round trip (measured) p50 {:.3} ms, p99 {:.3} ms;", measured.p50_ms, measured.p99_ms);
    assert!(tei_run.procedure.contains(&round_trip), "{}", tei_run.procedure);
    assert!(
        tei_run.procedure.contains(
            "TEI's own headers, whole ms, over the same requests: x-total-time p50 7 p99 7, x-tokenization-time \
             p50 1 p99 1, x-queue-time p50 0 p99 0, x-inference-time p50 5 p99 5;"
        ),
        "{}",
        tei_run.procedure
    );
    assert!(
        tei_run.procedure.contains("min_cosine is against the bundle's reference vectors;"),
        "{}",
        tei_run.procedure
    );
    assert!(measured.min_cosine.unwrap() > 0.999_999);
    // Rows of different lengths: what TEI pads them to is its own choice.
    assert_eq!(measured.computed_tokens, None);
    assert_eq!(trt_run.measured.as_ref().unwrap().computed_tokens, Some(4 * m.rows.seq as u64));
    assert_eq!(ov_run.measured.as_ref().unwrap().computed_tokens, Some(4 * m.rows.seq as u64));

    // Dense rows go to TEI as they are, cut, through the same check, and
    // its vectors are compared with the library's of each cut row alone.
    assert!(dense_run.measured.as_ref().unwrap().min_cosine.unwrap() > 0.999_999, "{dense_run:?}");
    assert_eq!(dense_run.measured.as_ref().unwrap().computed_tokens, Some(4 * 40), "every row 40 tokens");
    assert!(dense_run.procedure.contains("the library's vector of the row alone for a row cut to seq"));
    assert!(dense_run.procedure.contains("POST /embed with the batch's 4 rows as token ids"));
    // trtexec is given the dense rows' shape, every position live.
    let dense_trt = tensorrt::run(
        &TensorRt {
            image: format!("nvcr.io/nvidia/tensorrt@sha256:{DIGEST}"),
            trtexec: "trtexec".into(),
            inputs: ["input_ids".into(), "attention_mask".into(), "token_type_ids".into()],
            input_dtype: "int64".into(),
            warmup_ms: 100,
            work: work.clone(),
        },
        &dense,
        0,
        5,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        joined(&dense_trt).contains("--shapes=input_ids:4x40,attention_mask:4x40,token_type_ids:4x40"),
        "{}",
        joined(&dense_trt)
    );
    assert!(dense_trt.procedure.contains("the engine is built from onnx/model.onnx with --noTF32"));
    assert_eq!(dense_trt.measured.as_ref().unwrap().computed_tokens, Some(4 * 40));
    let r = turbo_bench::record(
        &dense,
        &provenance("redaction-dense"),
        vec![dense_run, dense_trt],
        "2026-01-02T03:04:05Z".into(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((r.rows.kind.as_str(), r.rows.live_tokens), ("ROWS_DENSE", 4 * 40));
    for r in [&trt_run, &ov_run] {
        let j = joined(r);
        assert!(j.contains("type=bind,src=<bundle>,dst=/bundle,readonly"), "{j}");
        assert!(j.contains("type=bind,src=<work>,dst=/work,readonly"), "{j}");
    }
    assert_eq!(ov_run.commands.iter().filter(|c| c.contains(&"benchmark_app".to_owned())).count(), 2);

    // Each runner's own record, and one of all three.
    let p = provenance("redaction-git");
    for refs in [vec![tei_run.clone()], vec![trt_run.clone()], vec![ov_run.clone()], vec![tei_not_run]] {
        let r = turbo_bench::record(&m, &p, refs, "2026-01-02T03:04:05Z".into()).unwrap_or_else(|e| panic!("{e}"));
        let text = serde_json::to_string_pretty(&r).unwrap();
        assert!(!text.contains("sentinel-user"), "{text}");
        assert!(!text.contains(&home.display().to_string()), "{text}");
        assert!(!text.contains(&work.display().to_string()), "{text}");
    }

    // What ran named the real directories.
    let calls = std::fs::read_to_string(root.join("calls")).unwrap();
    let real = |p: &Path| std::fs::canonicalize(p).unwrap().display().to_string();
    assert!(calls.contains(&format!("src={},dst=/bundle,readonly", m.bundle_dir.display())), "{calls}");
    assert!(calls.contains(&format!("src={},dst=/model,readonly", real(&home.join("upstream")))), "{calls}");
    assert!(calls.contains(SENTINEL), "{calls}");
    assert_eq!(std::fs::read_dir(&work).unwrap().count(), 0, "the input files are removed");
}
