//! turbo_model_load, turbo_model_get_info and turbo_model_release on the
//! CPU backend this build links, over a small BERT bundle each test writes:
//! docs/bundle.md loader rules 6 to 8 on top of 1 to 5, and where the
//! weights live once loaded.

mod common;

use std::ffi::c_void;
use std::fs;
use std::ptr;

use common::*;
use serde_json::{Value, json};
use turbo::status::*;
use turbo::*;

fn refused(f: &Fixture) -> Failure {
    match f.load() {
        Ok(_) => panic!("the model loaded"),
        Err(e) => e,
    }
}

fn with(name: &str, edit: impl FnOnce(&mut Value)) -> Failure {
    let mut f = Fixture::model(name);
    edit(&mut f.manifest);
    refused(&f)
}

/// The small BERT's tensors with `edit` applied to the one named `name`.
fn edited(name: &str, edit: impl FnOnce(&mut Tensor)) -> Vec<Tensor> {
    let mut t = tiny_weights(0);
    edit(t.iter_mut().find(|t| t.name == name).expect("the small BERT has it"));
    t
}

fn with_weights(name: &str, tensors: &[Tensor]) -> Failure {
    let mut f = Fixture::model(name);
    f.weights("weights/model.safetensors", tensors);
    refused(&f)
}

#[test]
fn a_model_reports_what_its_manifest_and_files_say() {
    let mut f = Fixture::new("info", {
        let mut m = model_manifest();
        m["embed"]["prefix_query"] = json!("query: ");
        m["embed"]["prefix_document"] = json!("passage: ");
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let l = f.load().unwrap();
    let info = l.info();
    let read = |p: &str| sha256_hex(&fs::read(f.dir.join(p)).unwrap());
    assert_eq!(info.struct_size as usize, size_of::<turbo_model_info>());
    assert_eq!(info.task, TURBO_TASK_EMBED);
    assert_eq!(info.dim, 8);
    assert_eq!(info.pooling, TURBO_POOLING_MEAN);
    assert_eq!(info.normalize, TURBO_NORMALIZE_L2);
    assert_eq!(info.max_seq, MAX_SEQ as u32);
    assert_eq!(info.max_batch, 64);
    assert_eq!(info.dtype, TURBO_DTYPE_F32, "no compute_dtype: the weights' storage dtype");
    assert_eq!(field(&info.model_id), "sentence-transformers/all-MiniLM-L6-v2");
    assert_eq!(field(&info.revision), "3");
    assert_eq!(field(&info.manifest_sha256), read("manifest.json"));
    assert_eq!(field(&info.artifact_sha256), read("weights/model.safetensors"));
    assert_eq!(field(&info.tokenizer_sha256), read("tokenizer.json"));
    assert_eq!(field(&info.prefix_query), "query: ");
    assert_eq!(field(&info.prefix_document), "passage: ");
}

#[test]
fn a_fixed_shape_is_what_is_reported() {
    let mut f = Fixture::model("fixed");
    let a = &mut f.manifest["artifacts"][0];
    a["fixed_seq"] = json!(128);
    a["fixed_batch"] = json!(8);
    f.manifest["embed"]["pooling"] = json!("POOLING_CLS");
    f.manifest["embed"]["normalize"] = json!("NORMALIZE_NONE");
    let info = f.load().unwrap().info();
    assert_eq!(info.dtype, TURBO_DTYPE_F32);
    assert_eq!((info.max_seq, info.max_batch), (128, 8));
    assert_eq!((info.pooling, info.normalize), (TURBO_POOLING_CLS, TURBO_NORMALIZE_NONE));
}

#[test]
fn a_fixed_shape_larger_than_the_models_changes_nothing() {
    let mut f = Fixture::model("fixed-large");
    f.manifest["artifacts"][0]["fixed_seq"] = json!(512);
    f.manifest["artifacts"][0]["fixed_batch"] = json!(1000);
    let info = f.load().unwrap().info();
    assert_eq!((info.max_seq, info.max_batch), (MAX_SEQ as u32, 64));
}

#[test]
fn release_in_any_order_is_safe() {
    let f = Fixture::model("release-order");
    let l = f.load().unwrap();
    let want = field(&l.info().artifact_sha256);
    let (rt, ctx, m) = l.into_raw();
    // The runtime, then the context, then the model: it keeps both.
    unsafe {
        turbo_runtime_release(rt);
        turbo_context_release(ctx);
    }
    let still = Loaded::model(m);
    assert_eq!(field(&still.info().artifact_sha256), want);
    let ModelWeights { files, held } = unsafe { model_weights(m) }.unwrap();
    assert!(!files.is_empty() && held.is_some());
    drop(still);

    // The model before its context and runtime.
    let (rt, ctx, m) = f.load().unwrap().into_raw();
    unsafe {
        turbo_model_release(m);
        turbo_context_release(ctx);
        turbo_runtime_release(rt);
    }
}

#[test]
fn release_of_null_is_a_no_op() {
    unsafe { turbo_model_release(ptr::null_mut()) };
}

#[test]
fn the_same_bundle_loaded_twice_is_two_models() {
    let f = Fixture::model("twice");
    f.write();
    let a = Loaded::load(&f.dir).unwrap();
    // The second on the first's context.
    let mut b = ptr::null_mut();
    let mut err = new_error();
    let rc = unsafe { turbo_model_load(a.ctx, text(f.dir.to_str().unwrap()), &mut b, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    assert_ne!(a.m, b);
    let ModelWeights { files: fa, held: ha } = unsafe { model_weights(a.m) }.unwrap();
    let ModelWeights { files: fb, held: hb } = unsafe { model_weights(b) }.unwrap();
    assert_ne!(fa[0].as_ptr(), fb[0].as_ptr(), "each model has its own weights");
    assert_eq!(fa[0], fb[0]);
    assert_ne!(ha, hb);
    let ia = a.info();
    drop(a);
    let second = Loaded::model(b);
    let ib = second.info();
    assert_eq!(field(&ia.manifest_sha256), field(&ib.manifest_sha256));
    assert_eq!(field(&ia.artifact_sha256), field(&ib.artifact_sha256));
    assert_eq!(
        unsafe { model_weights(b) }.unwrap().files[0],
        fs::read(f.dir.join("weights/model.safetensors")).unwrap()
    );
}

/// Where each tensor's bytes start in a safetensors file.
fn offsets(file: &[u8]) -> Vec<(String, usize)> {
    let n = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&file[8..8 + n]).unwrap();
    header
        .into_iter()
        .filter(|(k, _)| k != "__metadata__")
        .map(|(k, v)| (k, 8 + n + v["data_offsets"][0].as_u64().unwrap() as usize))
        .collect()
}

/// The names of the tensors the backend is handed, in turbo_backend.h's
/// order: the small BERT's own order, without the pooler.
fn bert_order() -> Vec<String> {
    tiny_weights(0).into_iter().map(|t| t.name).filter(|n| !n.starts_with("pooler")).collect()
}

#[test]
fn the_cpu_reads_the_weights_where_the_core_verified_them() {
    let f = Fixture::model("in-place");
    let l = f.load().unwrap();
    let on_disk = fs::read(f.dir.join("weights/model.safetensors")).unwrap();
    let ModelWeights { files, held } = unsafe { model_weights(l.m) }.unwrap();
    let held = held.expect("a model on the cpu backend");
    // One host copy: the bytes that were hashed.
    assert_eq!(files.len(), 1);
    assert_eq!(files[0], on_disk.as_slice());
    // Every tensor the backend keeps is an address inside that copy, at
    // the tensor's own offset: the backend copied nothing.
    let at = offsets(files[0]);
    let order = bert_order();
    assert_eq!(held.len(), order.len());
    assert_eq!(held.len(), 5 + 2 * 16);
    let base = files[0].as_ptr() as usize;
    for (name, p) in order.iter().zip(&held) {
        let off = at.iter().find(|(n, _)| n == name).unwrap().1;
        assert_eq!(*p as usize, base + off, "{name}");
    }
    assert_eq!(base % 64, 0, "the weights start on a 64-byte boundary");
    assert!(held.iter().all(|&p| (p as usize).is_multiple_of(4)), "every F32 tensor is aligned to 4 bytes");
}

#[test]
fn weights_in_two_files_are_one_artifact() {
    let mut f = Fixture::model("two-files");
    let mut all = tiny_weights(0);
    let second = all.split_off(5);
    f.weights("weights/model-1.safetensors", &all);
    f.weights("weights/model-2.safetensors", &second);
    f.manifest["artifacts"][0]["files"] = json!(["weights/model-1.safetensors", "weights/model-2.safetensors"]);
    let l = f.load().unwrap();
    let (h1, h2) = (f.sha256("weights/model-1.safetensors"), f.sha256("weights/model-2.safetensors"));
    assert_eq!(field(&l.info().artifact_sha256), sha256_hex(format!("{h1}{h2}").as_bytes()));
    let ModelWeights { files, held } = unsafe { model_weights(l.m) }.unwrap();
    let held = held.unwrap();
    assert_eq!(files.len(), 2);
    let inside = |p: *const c_void, f: &[u8]| f.as_ptr_range().contains(&(p as *const u8));
    assert!(held[..5].iter().all(|&p| inside(p, files[0])));
    assert!(held[5..].iter().all(|&p| inside(p, files[1])));
}

#[test]
fn a_tensor_in_two_files_is_invalid() {
    let mut f = Fixture::model("two-files-twice");
    f.weights("weights/model-2.safetensors", &tiny_weights(0)[..1]);
    f.manifest["artifacts"][0]["files"] = json!(["weights/model.safetensors", "weights/model-2.safetensors"]);
    let e = refused(&f);
    assert!(e.is(BUNDLE_INVALID, "embeddings.word_embeddings.weight\" (word_embeddings) is in both"), "{e:?}");
}

#[test]
fn models_load_from_many_threads() {
    let f = Fixture::model("threads");
    f.write();
    let l = Loaded::load(&f.dir).unwrap();
    let (ctx, path) = (l.ctx as usize, f.dir.to_str().unwrap().to_owned());
    std::thread::scope(|s| {
        for _ in 0..4 {
            let path = path.clone();
            s.spawn(move || {
                let mut m = ptr::null_mut();
                let rc = unsafe { turbo_model_load(ctx as *mut _, text(&path), &mut m, ptr::null_mut()) };
                assert_eq!(rc, 0);
                unsafe { turbo_model_release(m) };
            });
        }
    });
}

// Rules 1 to 5, reached through turbo_model_load.

#[test]
fn a_missing_bundle_or_manifest_is_not_found() {
    let e = Loaded::load(std::path::Path::new("/nonexistent/turbo-bundle")).err().unwrap();
    assert_eq!(e.code, BUNDLE_NOT_FOUND, "{e:?}");
    let f = Fixture::model("no-manifest");
    let e = Loaded::load(&f.dir).err().unwrap();
    assert!(e.is(BUNDLE_NOT_FOUND, "no manifest.json"), "{e:?}");
}

#[test]
fn a_bad_manifest_is_invalid() {
    let e = with("bad-field", |m| m["embed"]["extra"] = json!(1));
    assert!(e.is(BUNDLE_INVALID, "embed.extra"), "{e:?}");
    let f = Fixture::model("not-json");
    fs::write(f.dir.join("manifest.json"), b"{ not json").unwrap();
    let e = Loaded::load(&f.dir).err().unwrap();
    assert_eq!(e.code, BUNDLE_INVALID, "{e:?}");
}

#[test]
fn a_string_longer_than_its_info_field_is_invalid() {
    let e = with("long-id", |m| m["model"]["id"] = json!("m".repeat(128)));
    assert!(e.is(BUNDLE_INVALID, "model.id: 128 bytes is over the header's 127"), "{e:?}");
    let e = with("long-revision", |m| m["model"]["revision"] = json!("r".repeat(64)));
    assert!(e.is(BUNDLE_INVALID, "model.revision"), "{e:?}");
    let e = with("long-prefix", |m| m["embed"]["prefix_document"] = json!("p".repeat(128)));
    assert!(e.is(BUNDLE_INVALID, "embed.prefix_document"), "{e:?}");
    // One byte less fits, and is reported whole.
    let mut f = Fixture::model("longest-id");
    f.manifest["model"]["id"] = json!("m".repeat(127));
    assert_eq!(field(&f.load().unwrap().info().model_id), "m".repeat(127));
}

#[test]
fn another_task_is_unsupported() {
    let e = with("task", |m| m["task"] = json!("TASK_RERANK"));
    assert_eq!(e.code, UNSUPPORTED_TASK, "{e:?}");
}

#[test]
fn the_reference_ids_are_checked_on_load() {
    // The reference was made before the prefix was set.
    let e = with("reference", |m| m["embed"]["prefix_query"] = json!("query: "));
    assert!(e.is(BUNDLE_INVALID, "reference case 5"), "{e:?}");
}

// Rule 6

#[test]
fn no_artifact_for_the_cpu_is_no_artifact_and_says_why() {
    let e = with("no-cpu", |m| m["artifacts"][0]["backends"] = json!(["cuda", "metal"]));
    assert!(e.is(BUNDLE_NO_ARTIFACT, "weights-f32: backends [\"cuda\", \"metal\"] has no cpu"), "{e:?}");
    let e = with("target", |m| m["artifacts"][0]["target"] = json!("rtx4080"));
    assert!(e.is(BUNDLE_NO_ARTIFACT, "target rtx4080 is not this device's"), "{e:?}");
}

/// An artifact listing cpu that this build cannot hand the cpu backend,
/// ahead of the weights.
fn openvino(f: &mut Fixture) {
    fs::create_dir_all(f.dir.join("openvino")).unwrap();
    fs::write(f.dir.join("openvino/model.xml"), b"<net/>").unwrap();
    f.list("openvino/model.xml");
    let ir = json!({
        "name": "openvino-f16",
        "format": "FORMAT_OPENVINO_IR",
        "files": ["openvino/model.xml"],
        "backends": ["openvino", "cpu"],
        "compute_dtype": "DTYPE_F16",
        "graph_input": "INPUT_TOKEN_IDS",
        "graph_output": "OUTPUT_HIDDEN_STATES"
    });
    f.manifest["artifacts"].as_array_mut().unwrap().insert(0, ir);
}

#[test]
fn the_first_artifact_the_device_can_load_is_chosen() {
    let mut f = Fixture::model("order");
    openvino(&mut f);
    // A second weights artifact for the cpu, after the first.
    f.weights("weights/other.safetensors", &tiny_weights(1));
    let mut other = f.manifest["artifacts"][1].clone();
    other["name"] = json!("weights-other");
    other["files"] = json!(["weights/other.safetensors"]);
    f.manifest["artifacts"].as_array_mut().unwrap().push(other);
    let info = f.load().unwrap().info();
    assert_eq!(field(&info.artifact_sha256), f.sha256("weights/model.safetensors"));
    assert_eq!(info.dtype, TURBO_DTYPE_F32, "the chosen artifact's dtype, not the one skipped");

    // Only what cannot be loaded: each is named with why.
    let mut f = Fixture::model("order-none");
    openvino(&mut f);
    f.manifest["artifacts"][1]["backends"] = json!(["cuda"]);
    let e = refused(&f);
    assert!(
        e.is(BUNDLE_NO_ARTIFACT, "openvino-f16: FORMAT_OPENVINO_IR is not a format the cpu backend loads"),
        "{e:?}"
    );
    assert!(e.message.contains("weights-f32: backends"), "{e:?}");
}

#[test]
fn raw_weights_that_start_at_embeddings_or_fix_a_dtype_are_invalid() {
    let e = with("embeddings", |m| {
        m["artifacts"][0]["graph_input"] = json!("INPUT_EMBEDDINGS");
        m["artifacts"][0]["host_weights"] = json!("weights-f32");
    });
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].graph_input: raw weights start at INPUT_TOKEN_IDS"), "{e:?}");
    // What raw weights compute in is the session's precision to say.
    let e = with("compute-dtype", |m| m["artifacts"][0]["compute_dtype"] = json!("DTYPE_F16"));
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].compute_dtype: fixed by a compilation"), "{e:?}");
}

// Rule 7

#[test]
fn absent_weights_are_not_found() {
    let f = Fixture::model("absent");
    fs::remove_file(f.dir.join("weights/model.safetensors")).unwrap();
    let e = refused(&f);
    assert!(e.is(BUNDLE_NOT_FOUND, "weights/model.safetensors is absent"), "{e:?}");
}

#[test]
fn weights_that_disagree_with_their_hash_are_refused() {
    let f = Fixture::model("hash");
    let p = f.dir.join("weights/model.safetensors");
    let mut bytes = fs::read(&p).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&p, &bytes).unwrap();
    let e = refused(&f);
    assert!(e.is(BUNDLE_INTEGRITY, "weights/model.safetensors: SHA-256 is"), "{e:?}");
    bytes.push(0);
    fs::write(&p, &bytes).unwrap();
    let e = refused(&f);
    assert!(e.is(BUNDLE_INTEGRITY, "weights/model.safetensors: size is"), "{e:?}");
}

#[test]
fn a_host_weights_artifact_is_verified_too() {
    let mut f = Fixture::model("host-weights");
    f.weights("weights/other.safetensors", &tiny_weights(1));
    let mut other = f.manifest["artifacts"][0].clone();
    other["name"] = json!("weights-other");
    other["files"] = json!(["weights/other.safetensors"]);
    other["backends"] = json!([]);
    f.manifest["artifacts"].as_array_mut().unwrap().push(other);
    // host_weights is for an artifact whose graph starts at embeddings; the
    // loader verifies the files it names whichever artifact names it.
    f.manifest["artifacts"][0]["host_weights"] = json!("weights-other");
    f.load().expect("both verify");
    fs::write(f.dir.join("weights/other.safetensors"), b"changed").unwrap();
    let e = refused(&f);
    assert!(e.is(BUNDLE_INTEGRITY, "weights/other.safetensors"), "{e:?}");
}

// Rule 8

#[test]
fn a_missing_tensor_is_invalid_and_named() {
    let mut t = tiny_weights(0);
    t.retain(|t| t.name != "encoder.layer.1.attention.self.key.bias");
    let e = with_weights("missing", &t);
    assert!(e.is(BUNDLE_INVALID, "no tensor \"encoder.layer.1.attention.self.key.bias\" (k_bias of layer 1)"), "{e:?}");
}

#[test]
fn a_tensor_of_the_wrong_shape_is_invalid_and_named() {
    // The same bytes, transposed.
    let t = edited("encoder.layer.0.intermediate.dense.weight", |t| t.shape = vec![8, 16]);
    let e = with_weights("shape", &t);
    assert!(
        e.is(
            BUNDLE_INVALID,
            "encoder.layer.0.intermediate.dense.weight (ffn_in_weight of layer 0) has shape [8, 16]; the architecture implies [16, 8]"
        ),
        "{e:?}"
    );
    let t = edited("embeddings.position_embeddings.weight", |t| {
        t.shape = vec![256, 8];
        t.data.truncate(256 * 8 * 4);
    });
    let e = with_weights("positions", &t);
    assert!(e.is(BUNDLE_INVALID, "implies [512, 8]"), "{e:?}");
}

#[test]
fn a_tensor_of_the_wrong_dtype_is_invalid_and_named() {
    let t = edited("embeddings.LayerNorm.bias", |t| t.dtype = "I32");
    let e = with_weights("dtype-i32", &t);
    assert!(e.is(BUNDLE_INVALID, "embeddings.LayerNorm.bias (embeddings_ln_bias) is I32"), "{e:?}");
    let t = edited("encoder.layer.1.output.LayerNorm.weight", |t| {
        t.dtype = "F16";
        t.data.truncate(8 * 2);
    });
    let e = with_weights("dtype-mixed", &t);
    assert!(e.is(BUNDLE_INVALID, "is F16; the other weights are F32"), "{e:?}");
    let t = edited("encoder.layer.1.output.LayerNorm.weight", |t| t.dtype = "I64");
    let e = with_weights("dtype-other", &t);
    assert!(e.is(BUNDLE_INVALID, "(ffn_ln_weight of layer 1) is I64; weights are F32, F16 or BF16"), "{e:?}");
}

#[test]
fn a_tensor_not_aligned_to_its_elements_is_invalid() {
    // A tensor the model does not use, three bytes long, first in the data.
    let mut t = tiny_weights(0);
    t.insert(0, Tensor { name: "extra".into(), dtype: "U8", shape: vec![3], data: vec![1, 2, 3] });
    let e = with_weights("misaligned", &t);
    assert!(e.message.ends_with(", not a multiple of its 4-byte elements"), "{e:?}");
    assert!(e.is(BUNDLE_INVALID, "embeddings.word_embeddings.weight (word_embeddings) starts at byte "), "{e:?}");
    // Four bytes, and every tensor after it is aligned again.
    t[0] = Tensor { name: "extra".into(), dtype: "U8", shape: vec![4], data: vec![1, 2, 3, 4] };
    let mut f = Fixture::model("aligned-again");
    f.weights("weights/model.safetensors", &t);
    f.load().expect("aligned");
}

#[test]
fn a_header_length_not_a_multiple_of_8_is_invalid() {
    let mut bytes = safetensors_file(&tiny_weights(0));
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    // One more space in the header, and the data one byte later.
    bytes.insert(8 + n, b' ');
    bytes[..8].copy_from_slice(&(n as u64 + 1).to_le_bytes());
    let mut f = Fixture::model("header-length");
    fs::write(f.dir.join("weights/model.safetensors"), &bytes).unwrap();
    f.list("weights/model.safetensors");
    let e = refused(&f);
    assert!(
        e.is(BUNDLE_INVALID, &format!("weights/model.safetensors: header length {} is not a multiple of 8", n + 1)),
        "{e:?}"
    );
}

#[test]
fn half_precision_weights_load_as_their_dtype() {
    let mut t = tiny_weights(0);
    for t in &mut t {
        t.dtype = "BF16";
        // The upper half of each F32 is its BF16.
        t.data = t.data.chunks_exact(4).flat_map(|b| [b[2], b[3]]).collect();
    }
    let mut f = Fixture::model("bf16");
    f.weights("weights/model.safetensors", &t);
    let l = f.load().unwrap();
    assert_eq!(l.info().dtype, TURBO_DTYPE_BF16);
    let ModelWeights { files, held } = unsafe { model_weights(l.m) }.unwrap();
    assert_eq!(files[0].as_ptr() as usize % 64, 0);
    assert!(held.unwrap().iter().all(|&p| (p as usize).is_multiple_of(2)), "every BF16 tensor is aligned to 2 bytes");
}

#[test]
fn every_role_a_bert_encoder_needs_is_named() {
    let e = with("role", |m| {
        m["artifacts"][0]["tensor_names"].as_object_mut().unwrap().remove("q_bias");
    });
    assert!(e.is(BUNDLE_INVALID, "tensor_names: no q_bias, which a BERT encoder needs"), "{e:?}");
}

#[test]
fn a_dim_other_than_the_hidden_width_is_invalid() {
    let mut f = Fixture::new("dim", {
        let mut m = model_manifest();
        m["embed"]["dim"] = json!(16);
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let e = refused(&f);
    assert!(e.is(BUNDLE_INVALID, "embed.dim: 16 is not architecture.hidden 8"), "{e:?}");
}

#[test]
fn a_weights_file_that_is_not_safetensors_is_invalid() {
    let mut f = Fixture::model("garbage");
    fs::write(f.dir.join("weights/model.safetensors"), b"not a safetensors file").unwrap();
    f.list("weights/model.safetensors");
    let e = refused(&f);
    assert!(e.is(BUNDLE_INVALID, "weights/model.safetensors"), "{e:?}");
}

// The calling convention

#[test]
fn null_and_wrong_handles_are_refused() {
    let f = Fixture::model("handles");
    let l = f.load().unwrap();
    let path = f.dir.to_str().unwrap();
    let mut m = ptr::null_mut();
    let mut err = new_error();
    let rc = unsafe { turbo_model_load(ptr::null_mut(), text(path), &mut m, &mut err) };
    assert!(failure(rc, &err).is(INVALID_HANDLE, "not a turbo_context"));
    let rc = unsafe { turbo_model_load(l.m as *mut turbo_context, text(path), &mut m, &mut err) };
    assert!(failure(rc, &err).is(INVALID_HANDLE, "not a turbo_context"));
    assert!(m.is_null());
    let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_model_info>() as u32;
    let rc = unsafe { turbo_model_get_info(ptr::null_mut(), &mut info, &mut err) };
    assert!(failure(rc, &err).is(INVALID_HANDLE, "not a turbo_model"));
    let rc = unsafe { turbo_model_get_info(l.ctx as *mut turbo_model, &mut info, &mut err) };
    assert!(failure(rc, &err).is(INVALID_HANDLE, "not a turbo_model"));
    // Not a model: nothing is released.
    unsafe { turbo_model_release(l.ctx as *mut turbo_model) };
    assert_eq!(l.info().dim, 8);
}

#[test]
fn null_outputs_and_bad_paths_are_refused() {
    let f = Fixture::model("outs");
    let l = f.load().unwrap();
    let mut err = new_error();
    let rc = unsafe { turbo_model_load(l.ctx, text(f.dir.to_str().unwrap()), ptr::null_mut(), &mut err) };
    assert!(failure(rc, &err).is(INVALID_ARGUMENT, "out is NULL"));
    let rc = unsafe { turbo_model_get_info(l.m, ptr::null_mut(), &mut err) };
    assert!(failure(rc, &err).is(INVALID_ARGUMENT, "out is NULL"));
    let mut m = ptr::null_mut();
    let null_path = turbo_text { ptr: ptr::null(), len: 4 };
    let rc = unsafe { turbo_model_load(l.ctx, null_path, &mut m, &mut err) };
    assert!(failure(rc, &err).is(INVALID_ARGUMENT, "bundle_path"));
    let bad = [0x2fu8, 0xff];
    let rc = unsafe { turbo_model_load(l.ctx, turbo_text { ptr: bad.as_ptr() as *const _, len: 2 }, &mut m, &mut err) };
    assert!(failure(rc, &err).is(INVALID_UTF8, "bundle_path"));
    let rc = unsafe { turbo_model_load(l.ctx, text(""), &mut m, &mut err) };
    assert_eq!(failure(rc, &err).code, BUNDLE_NOT_FOUND);
    assert!(m.is_null());
    // err may be NULL.
    let rc = unsafe { turbo_model_get_info(l.m, ptr::null_mut(), ptr::null_mut()) };
    assert_eq!(rc, INVALID_ARGUMENT);
}

#[test]
fn a_wrong_struct_size_is_refused_and_the_output_left_alone() {
    let f = Fixture::model("struct-size");
    let l = f.load().unwrap();
    for size in [0, 8, size_of::<turbo_model_info>() as u32 - 4, size_of::<turbo_model_info>() as u32 + 4] {
        let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
        info.struct_size = size;
        info.dim = 77;
        let mut err = new_error();
        let rc = unsafe { turbo_model_get_info(l.m, &mut info, &mut err) };
        assert!(failure(rc, &err).is(INVALID_STRUCT_SIZE, "turbo_model_info.struct_size"));
        assert_eq!((info.struct_size, info.dim), (size, 77));
    }
    let mut err = new_error();
    err.struct_size = 4;
    let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_model_info>() as u32;
    assert_eq!(unsafe { turbo_model_get_info(l.m, &mut info, &mut err) }, INVALID_STRUCT_SIZE);
}
