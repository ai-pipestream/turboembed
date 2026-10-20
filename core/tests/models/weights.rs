use super::*;

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
    let t = edited("encoder.layer.1.output.LayerNorm.weight", |t| t.dtype = "U32");
    let e = with_weights("dtype-other", &t);
    assert!(e.is(BUNDLE_INVALID, "(ffn_ln_weight of layer 1) is U32; weights are F32, F16 or BF16"), "{e:?}");
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
        t.data = t.data.as_chunks::<4>().0.iter().flat_map(|b| [b[2], b[3]]).collect();
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
