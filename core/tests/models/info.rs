use super::*;

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
    assert_eq!((info.output_dims_count, info.output_dims), (0, [0; TURBO_OUTPUT_DIMS_MAX]));
}

#[test]
fn the_output_dims_are_reported_ascending() {
    let mut f = Fixture::model("output-dims");
    f.manifest["embed"]["output_dims"] = json!([6, 2, 4]);
    let info = f.load().unwrap().info();
    let mut want = [0; TURBO_OUTPUT_DIMS_MAX];
    want[..3].copy_from_slice(&[2, 4, 6]);
    assert_eq!((info.output_dims_count, info.output_dims), (3, want));
}

#[test]
fn the_struct_before_output_dims_is_accepted_and_its_end_left_alone() {
    let mut f = Fixture::model("old-struct-size");
    f.manifest["embed"]["output_dims"] = json!([4]);
    let l = f.load().unwrap();
    assert_eq!(TURBO_MODEL_INFO_SIZE_V1, 696);
    let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
    info.struct_size = TURBO_MODEL_INFO_SIZE_V1 as u32;
    info.output_dims_count = 99;
    info.output_dims = [7; TURBO_OUTPUT_DIMS_MAX];
    let mut err = new_error();
    assert_eq!(unsafe { turbo_model_get_info(l.m, &mut info, &mut err) }, OK);
    assert_eq!((info.struct_size as usize, info.dim), (TURBO_MODEL_INFO_SIZE_V1, 8));
    assert_eq!(field(&info.model_id), "sentence-transformers/all-MiniLM-L6-v2");
    assert_eq!((info.output_dims_count, info.output_dims), (99, [7; TURBO_OUTPUT_DIMS_MAX]));
}

#[test]
fn more_output_dims_than_the_header_holds_are_invalid() {
    let e = with("output-dims-17", |m| m["embed"]["output_dims"] = json!((1..=17).collect::<Vec<u32>>()));
    assert!(e.is(BUNDLE_INVALID, "embed.output_dims: 17 widths, and turbo_model_info holds 16"), "{e:?}");
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
    // fixed_batch is the frame the backend runs, not a limit.
    assert_eq!((info.max_seq, info.max_batch), (128, 64));
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
    let e = with("task", |m| m["task"] = json!("TASK_FUTURE"));
    assert_eq!(e.code, UNSUPPORTED_TASK, "{e:?}");
}

#[test]
fn the_reference_ids_are_checked_on_load() {
    // The reference was made before the prefix was set.
    let e = with("reference", |m| m["embed"]["prefix_query"] = json!("query: "));
    assert!(e.is(BUNDLE_INVALID, "reference case 5"), "{e:?}");
}

#[test]
fn the_pre_rerank_model_info_keeps_its_tail_untouched() {
    let f = Fixture::model("pre-rerank-info");
    let loaded = f.load().unwrap();
    let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
    info.struct_size = TURBO_MODEL_INFO_SIZE_V2 as u32;
    info.rerank_query_tokens = 123;
    info.rerank_pair_template = 456;
    let mut err = new_error();
    assert_eq!(unsafe { turbo_model_get_info(loaded.m, &mut info, &mut err) }, OK);
    assert_eq!(info.dim, 8);
    assert_eq!((info.rerank_query_tokens, info.rerank_pair_template), (123, 456));
}
