//! docs/bundle.md loader rules 1 to 5, each broken on a real bundle
//! directory and opened through turbo_tokenizer_create.

mod common;

use std::fs;

use common::*;
use serde_json::{Value, json};
use turbo::status::*;

fn refused(f: &Fixture) -> Failure {
    match f.open() {
        Ok(_) => panic!("the bundle loaded"),
        Err(e) => e,
    }
}

fn with(name: &str, edit: impl FnOnce(&mut Value)) -> Failure {
    let mut f = Fixture::standard(name);
    edit(&mut f.manifest);
    refused(&f)
}

#[test]
fn the_standard_bundle_loads() {
    Fixture::standard("loads").open().expect("loads");
}

// Rule 1

#[test]
fn no_directory_or_no_manifest_is_not_found() {
    let e = Tok::create(std::path::Path::new("/nonexistent/turbo-bundle")).unwrap_err();
    assert_eq!(e.code, BUNDLE_NOT_FOUND, "{e:?}");
    let f = Fixture::standard("no-manifest");
    let e = Tok::create(&f.dir).unwrap_err();
    assert!(e.is(BUNDLE_NOT_FOUND, "no manifest.json"), "{e:?}");
}

// Rule 2

#[test]
fn unknown_fields_are_invalid_and_named() {
    let e = with("unknown-top", |m| m["extra"] = json!(1));
    assert!(e.is(BUNDLE_INVALID, "extra"), "{e:?}");
    let e = with("unknown-nested", |m| m["tokenizer"]["normalizer"]["nfkc"] = json!(true));
    assert!(e.is(BUNDLE_INVALID, "tokenizer.normalizer"), "{e:?}");
    assert!(e.message.contains("nfkc"), "{e:?}");
    let e = with("unknown-bpe", |m| m["tokenizer"]["bpe"] = json!({}));
    assert!(e.is(BUNDLE_INVALID, "bpe"), "{e:?}");
}

#[test]
fn unknown_enum_values_are_invalid_and_named() {
    let e = with("enum-pooling", |m| m["embed"]["pooling"] = json!("POOLING_MAX"));
    assert!(e.is(BUNDLE_INVALID, "embed.pooling"), "{e:?}");
    let e = with("enum-lowercase", |m| m["embed"]["normalize"] = json!("l2"));
    assert!(e.is(BUNDLE_INVALID, "embed.normalize"), "{e:?}");
    let e = with("enum-role", |m| m["tokenizer"]["special_tokens"][0]["role"] = json!("SPECIAL_SEP"));
    assert!(e.is(BUNDLE_INVALID, "tokenizer.special_tokens[0].role"), "{e:?}");
    let e = with("enum-unicode", |m| m["tokenizer"]["normalizer"]["unicode_form"] = json!("UNICODE_NFC"));
    assert!(e.is(BUNDLE_INVALID, "unicode_form"), "{e:?}");
    let e = with("enum-number", |m| m["embed"]["pooling"] = json!(1));
    assert!(e.is(BUNDLE_INVALID, "embed.pooling"), "{e:?}");
}

#[test]
fn missing_required_fields_are_invalid_and_named() {
    let e = with("missing-dim", |m| {
        m["embed"].as_object_mut().unwrap().remove("dim");
    });
    assert!(e.is(BUNDLE_INVALID, "embed"), "{e:?}");
    assert!(e.message.contains("dim"), "{e:?}");
    let e = with("missing-embed", |m| {
        m.as_object_mut().unwrap().remove("embed");
    });
    assert!(e.is(BUNDLE_INVALID, "embed"), "{e:?}");
    let e = with("empty-id", |m| m["model"]["id"] = json!(""));
    assert!(e.is(BUNDLE_INVALID, "model.id: required"), "{e:?}");
    let e = with("missing-architecture", |m| {
        m.as_object_mut().unwrap().remove("architecture");
    });
    assert!(e.is(BUNDLE_INVALID, "architecture: required by raw weights"), "{e:?}");
    let e = with("missing-reproducible", |m| {
        m["reference"]["produced_by"].as_object_mut().unwrap().remove("reproducible");
    });
    assert!(e.is(BUNDLE_INVALID, "reference.produced_by"), "{e:?}");
}

#[test]
fn another_bundle_version_is_invalid() {
    let e = with("version", |m| m["bundle_version"] = json!(2));
    assert!(e.is(BUNDLE_INVALID, "bundle_version"), "{e:?}");
}

#[test]
fn a_task_this_build_does_not_have_is_unsupported() {
    let e = with("task-rerank", |m| m["task"] = json!("TASK_RERANK"));
    assert_eq!(e.code, UNSUPPORTED_TASK, "{e:?}");
    // Its own task block is unknown to this build; the task still decides.
    let e = with("task-rerank-block", |m| {
        m["task"] = json!("TASK_RERANK");
        m["rerank"] = json!({ "max_seq": 512 });
    });
    assert_eq!(e.code, UNSUPPORTED_TASK, "{e:?}");
    let e = with("task-garbage", |m| m["task"] = json!("embed"));
    assert_eq!(e.code, BUNDLE_INVALID, "{e:?}");
}

#[test]
fn strings_longer_than_their_header_buffer_are_invalid() {
    let e = with("long-id", |m| m["model"]["id"] = json!("m".repeat(128)));
    assert!(e.is(BUNDLE_INVALID, "model.id"), "{e:?}");
    let e = with("long-prefix", |m| m["embed"]["prefix_query"] = json!("q".repeat(128)));
    assert!(e.is(BUNDLE_INVALID, "embed.prefix_query"), "{e:?}");
    let e = with("long-target", |m| m["artifacts"][0]["target"] = json!("t".repeat(32)));
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].target"), "{e:?}");
    // One byte under the buffer fits with its NUL.
    let mut f = Fixture::standard("id-fits");
    f.manifest["model"]["id"] = json!("m".repeat(127));
    f.open().expect("127 bytes fits a 128-byte buffer");
}

#[test]
fn bad_paths_are_invalid() {
    let e = with("dotdot", |m| m["tokenizer"]["file"] = json!("../tokenizer.json"));
    assert!(e.is(BUNDLE_INVALID, "tokenizer.file"), "{e:?}");
    let e = with("absolute", |m| m["reference"]["file"] = json!("/etc/passwd"));
    assert!(e.is(BUNDLE_INVALID, "reference.file"), "{e:?}");
    let e = with("dot", |m| m["tokenizer"]["file"] = json!("./tokenizer.json"));
    assert!(e.is(BUNDLE_INVALID, "tokenizer.file"), "{e:?}");
    let e = with("backslash", |m| m["artifacts"][0]["files"][0] = json!("weights\\model.safetensors"));
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].files[0]"), "{e:?}");
    let e = with("files-dotdot", |m| m["files"][0]["path"] = json!("a/../b"));
    assert!(e.is(BUNDLE_INVALID, "files[0].path"), "{e:?}");
}

#[test]
fn a_path_not_in_files_is_invalid() {
    let e = with("unlisted", |m| m["model"]["license_file"] = json!("LICENSE"));
    assert!(e.is(BUNDLE_INVALID, "model.license_file"), "{e:?}");
    assert!(e.message.contains("not in files"), "{e:?}");
}

#[test]
fn cross_references_are_checked() {
    let e = with("host-weights", |m| {
        m["artifacts"][0]["graph_input"] = json!("INPUT_EMBEDDINGS");
    });
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].host_weights"), "{e:?}");
    let e = with("from", |m| {
        m["artifacts"][0]["produced_by"] = json!({
            "tool": "t", "tool_version": "1", "container": "c", "from": "onnx-f32", "reproducible": true
        });
    });
    assert!(e.is(BUNDLE_INVALID, "no artifact named \"onnx-f32\""), "{e:?}");
    let e = with("template", |m| m["tokenizer"]["template"] = json!(["[CLS]", "$TEXT", "</s>"]));
    assert!(e.is(BUNDLE_INVALID, "tokenizer.template[2]"), "{e:?}");
    let e = with("layer", |m| m["artifacts"][0]["tensor_names"]["q_weight"] = json!("encoder.q.weight"));
    assert!(e.is(BUNDLE_INVALID, "tensor_names.q_weight"), "{e:?}");
    let e = with("role", |m| m["artifacts"][0]["tensor_names"]["lm_head"] = json!("lm_head.weight"));
    assert!(e.is(BUNDLE_INVALID, "tensor_names"), "{e:?}");
    let e = with("max-seq", |m| m["embed"]["max_seq"] = json!(1024));
    assert!(e.is(BUNDLE_INVALID, "architecture.max_positions"), "{e:?}");
}

#[test]
fn the_example_artifacts_parse() {
    // docs/bundle.md's OpenVINO and HEF entries, including DTYPE_I8, which
    // the header does not have yet. Their files are listed and absent.
    let mut f = Fixture::standard("example-artifacts");
    let zero = "0".repeat(64);
    let files = f.manifest["files"].as_array_mut().unwrap();
    for p in [
        "openvino/model.xml",
        "openvino/model.bin",
        "hailo/model-hailo10h-s128.hef",
        "onnx/model.onnx",
        "calibration/texts.txt",
    ] {
        files.push(json!({ "path": p, "size": 1, "sha256": zero }));
    }
    let arts = f.manifest["artifacts"].as_array_mut().unwrap();
    arts.push(json!({
      "name": "openvino-f16", "format": "FORMAT_OPENVINO_IR",
      "files": ["openvino/model.xml", "openvino/model.bin"], "backends": ["openvino"],
      "compute_dtype": "DTYPE_F16", "graph_input": "INPUT_TOKEN_IDS", "graph_output": "OUTPUT_HIDDEN_STATES",
      "produced_by": { "tool": "ovc", "tool_version": "v", "container": "c", "from": "onnx-f32",
        "args": ["onnx/model.onnx", "--compress_to_fp16=True"], "reproducible": true }
    }));
    arts.push(json!({
      "name": "hef-hailo10h-s128", "format": "FORMAT_HEF", "files": ["hailo/model-hailo10h-s128.hef"],
      "backends": ["hailo"], "target": "hailo10h", "fixed_seq": 128, "compute_dtype": "DTYPE_I8",
      "graph_input": "INPUT_EMBEDDINGS", "host_weights": "weights-f32", "graph_output": "OUTPUT_HIDDEN_STATES",
      "produced_by": { "tool": "hailo-dataflow-compiler", "tool_version": "v", "container": "c", "from": "onnx-f32",
        "inputs": ["calibration/texts.txt"], "args": ["--hw-arch", "hailo10h"], "reproducible": false }
    }));
    arts.push(json!({
      "name": "onnx-f32", "format": "FORMAT_ONNX", "files": ["onnx/model.onnx"], "backends": [],
      "graph_input": "INPUT_TOKEN_IDS", "graph_output": "OUTPUT_HIDDEN_STATES"
    }));
    f.open().expect("the example's artifacts are well formed");
}

// Rule 3

#[test]
fn a_link_out_of_the_bundle_is_invalid() {
    let outside = std::env::temp_dir().join(format!("turbo-test-{}-outside.json", std::process::id()));
    fs::copy(upstream_tokenizer_json(), &outside).unwrap();
    let f = Fixture::standard("symlink-out");
    fs::remove_file(f.dir.join("tokenizer.json")).unwrap();
    std::os::unix::fs::symlink(&outside, f.dir.join("tokenizer.json")).unwrap();
    // Same bytes, same hash: the hash never vouches for a file elsewhere.
    let e = refused(&f);
    fs::remove_file(&outside).unwrap();
    assert!(e.is(BUNDLE_INVALID, "outside the bundle"), "{e:?}");
}

#[test]
fn a_link_inside_the_bundle_is_followed() {
    let f = Fixture::standard("symlink-in");
    fs::create_dir(f.dir.join("real")).unwrap();
    fs::rename(f.dir.join("tokenizer.json"), f.dir.join("real/tokenizer.json")).unwrap();
    std::os::unix::fs::symlink("real/tokenizer.json", f.dir.join("tokenizer.json")).unwrap();
    f.open().expect("a link that stays inside is fine");
}

// Rule 5, and the integrity errors

#[test]
fn a_changed_tokenizer_is_integrity() {
    let f = Fixture::standard("tokenizer-bytes");
    let path = f.dir.join("tokenizer.json");
    let mut bytes = fs::read(&path).unwrap();
    let i = bytes.len() / 2;
    bytes[i] ^= 1;
    fs::write(&path, &bytes).unwrap();
    let e = refused(&f);
    assert!(e.is(BUNDLE_INTEGRITY, "tokenizer.json: SHA-256 is"), "{e:?}");
    assert!(e.message.contains(f.manifest["files"][1]["sha256"].as_str().unwrap()), "names both: {e:?}");
}

#[test]
fn a_resized_reference_is_integrity() {
    let f = Fixture::standard("reference-size");
    let path = f.dir.join("reference/reference.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    bytes.push(0);
    let len = bytes.len();
    fs::write(&path, &bytes).unwrap();
    let e = refused(&f);
    assert!(e.is(BUNDLE_INTEGRITY, &format!("size is {len}")), "{e:?}");
}

#[test]
fn an_absent_listed_file_is_not_found() {
    let f = Fixture::standard("absent");
    fs::remove_file(f.dir.join("tokenizer.json")).unwrap();
    let e = refused(&f);
    assert!(e.is(BUNDLE_NOT_FOUND, "tokenizer.json is absent"), "{e:?}");
}

#[test]
fn a_reference_id_that_differs_names_the_case_and_position() {
    let mut f = Fixture::standard("reference-ids");
    let mut ids = reference_ids(&f.manifest);
    ids[1][3] += 1;
    let width = ids.iter().map(Vec::len).max().unwrap();
    fs::write(f.dir.join("reference/reference.safetensors"), reference_file(&ids, width, 0, 384)).unwrap();
    f.list("reference/reference.safetensors");
    let e = refused(&f);
    assert!(e.is(BUNDLE_INVALID, "reference case 1"), "{e:?}");
    assert!(e.message.contains("position 3"), "{e:?}");
}

#[test]
fn a_reference_row_that_is_short_names_the_position() {
    let mut f = Fixture::standard("reference-short");
    let mut ids = reference_ids(&f.manifest);
    ids[2].pop();
    let width = ids.iter().map(Vec::len).max().unwrap();
    fs::write(f.dir.join("reference/reference.safetensors"), reference_file(&ids, width, 0, 384)).unwrap();
    f.list("reference/reference.safetensors");
    let e = refused(&f);
    assert!(e.is(BUNDLE_INVALID, "reference case 2"), "{e:?}");
}

#[test]
fn a_reference_with_the_wrong_shape_is_invalid() {
    let mut f = Fixture::standard("reference-shape");
    let ids = reference_ids(&f.manifest);
    let width = ids.iter().map(Vec::len).max().unwrap();
    fs::write(f.dir.join("reference/reference.safetensors"), reference_file(&ids, width, 0, 128)).unwrap();
    f.list("reference/reference.safetensors");
    let e = refused(&f);
    assert!(e.is(BUNDLE_INVALID, "embeddings"), "{e:?}");
}

#[test]
fn a_reference_without_a_long_case_is_invalid() {
    let mut m = manifest();
    m["reference"]["cases"].as_array_mut().unwrap().pop();
    let e = refused(&Fixture::new("no-long-case", m));
    assert!(e.is(BUNDLE_INVALID, "truncation is not checked"), "{e:?}");
}

#[test]
fn a_bundle_that_does_not_cut_is_invalid() {
    // The reference must carry a case longer than max_seq, so a bundle
    // that never cuts could not load; it is refused as a bundle, not as
    // the long case's CAPACITY.
    let e = with("truncate-none", |m| m["tokenizer"]["truncation"] = json!("TRUNCATE_NONE"));
    assert!(e.is(BUNDLE_INVALID, "tokenizer.truncation"), "{e:?}");
}

#[test]
fn a_bundle_that_cuts_on_the_left_loads() {
    let mut m = manifest();
    m["tokenizer"]["truncation"] = json!("TRUNCATE_LEFT");
    let f = Fixture::new("truncate-left", m);
    f.open().expect("the reference ids were cut on the left too");
}

#[test]
fn a_manifest_that_disagrees_with_tokenizer_json_is_invalid() {
    let e = with("no-lowercase", |m| m["tokenizer"]["normalizer"]["lowercase"] = json!(false));
    assert!(e.is(BUNDLE_INVALID, "tokenizer.json: normalizer"), "{e:?}");
    let e = with("prefix", |m| m["tokenizer"]["wordpiece"]["continuing_prefix"] = json!("@@"));
    assert!(e.is(BUNDLE_INVALID, "continuing_subword_prefix"), "{e:?}");
    let e = with("special-id", |m| m["tokenizer"]["special_tokens"][4]["id"] = json!(104));
    assert!(e.is(BUNDLE_INVALID, "[MASK]"), "{e:?}");
}

#[test]
fn files_not_opened_are_not_checked() {
    // The weights are listed with a size and hash they do not have, and
    // are absent; the tokenizer never opens them.
    let f = Fixture::standard("unopened");
    assert!(!f.dir.join("weights/model.safetensors").exists());
    f.open().expect("loads");
}
