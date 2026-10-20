use super::*;

const HEF: &[u8] = b"not a real HEF: the core hashes these bytes and hands them on unread";

/// The CPU's architecture label, which a HEF's target must equal for the
/// cpu to get as far as its format.
fn cpu_arch() -> String {
    let mut rt = ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, ptr::null_mut()) }, OK);
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    assert_eq!(unsafe { turbo_runtime_device_info(rt, cpu(rt), &mut info, ptr::null_mut()) }, OK);
    unsafe { turbo_runtime_release(rt) };
    field(&info.arch)
}

/// The small BERT with a HEF for `target` ahead of its weights: it starts
/// at the word-embedding rows and looks them up in the weights artifact.
fn hef(name: &str, target: &str) -> Fixture {
    let mut f = Fixture::model(name);
    fs::create_dir_all(f.dir.join("hailo")).unwrap();
    fs::write(f.dir.join("hailo/model.hef"), HEF).unwrap();
    f.list("hailo/model.hef");
    let art = json!({
        "name": "hef-s128",
        "format": "FORMAT_HEF",
        "files": ["hailo/model.hef"],
        "backends": ["hailo", "cpu"],
        "target": target,
        "fixed_seq": 128,
        "fixed_batch": 1,
        "compute_dtype": "DTYPE_I8",
        "graph_input": "INPUT_EMBEDDINGS",
        "host_weights": "weights-f32",
        "graph_output": "OUTPUT_HIDDEN_STATES"
    });
    f.manifest["artifacts"].as_array_mut().unwrap().insert(0, art);
    f.write();
    f
}

/// The CPU backend's table, saying it loads `formats`: the table a
/// backend that loads HEFs gives, around a backend that is linked in.
fn cpu_table(formats: u32) -> backend::turbo_backend {
    let mut t = unsafe { ptr::read(&cpu::BACKEND) };
    t.formats = formats;
    t
}

fn open(f: &Fixture) -> bundle::Bundle {
    f.write();
    bundle::Bundle::open(&f.dir).unwrap()
}

#[test]
fn a_hef_is_never_chosen_for_a_backend_that_does_not_load_it() {
    let arch = cpu_arch();
    let f = hef("hef-cpu", &arch);
    // The cpu backend lists the HEF and the device matches its target: its
    // format alone rules it out, and the weights after it load.
    let l = f.load().unwrap();
    let info = l.info();
    assert_eq!(field(&info.artifact_sha256), f.sha256("weights/model.safetensors"));
    assert_eq!(info.dtype, TURBO_DTYPE_F32);
    assert_eq!((info.max_seq, info.max_batch), (MAX_SEQ as u32, 64), "not the HEF's fixed shape");

    let b = open(&f);
    let cpu_formats = cpu::BACKEND.formats();
    assert_eq!(cpu_formats, backend::format_bit(backend::TURBO_FORMAT_SAFETENSORS));
    assert_eq!(model::choose(&b.manifest, "cpu", cpu_formats, &arch).unwrap(), 1);
    // A table from before formats, or one that leaves it 0, loads raw
    // weights alone.
    let mut old = cpu_table(backend::format_bit(backend::TURBO_FORMAT_HEF));
    old.struct_size = std::mem::offset_of!(backend::turbo_backend, formats) as u32;
    assert_eq!(model::choose(&b.manifest, "cpu", old.formats(), &arch).unwrap(), 1);
    assert_eq!(model::choose(&b.manifest, "cpu", cpu_table(0).formats(), &arch).unwrap(), 1);

    // With nothing after it, the HEF is skipped and says why.
    let mut f = hef("hef-cpu-only", &arch);
    f.manifest["artifacts"][1]["backends"] = json!(["cuda"]);
    let e = refused(&f);
    assert!(e.is(BUNDLE_NO_ARTIFACT, "hef-s128: FORMAT_HEF is not a format the cpu backend loads"), "{e:?}");
}

#[test]
fn a_backend_that_loads_hefs_gets_the_hashed_bytes_and_the_embedding_tensors() {
    let f = hef("hef-loaded", "hailo10h");
    let b = open(&f);
    let t = cpu_table(
        backend::format_bit(backend::TURBO_FORMAT_SAFETENSORS) | backend::format_bit(backend::TURBO_FORMAT_HEF),
    );
    // On another device the target rules it out.
    assert_eq!(model::choose(&b.manifest, "cpu", t.formats(), "rtx4080").unwrap(), 1);
    let i = model::choose(&b.manifest, "cpu", t.formats(), "hailo10h").unwrap();
    assert_eq!(b.manifest.artifacts[i].name, "hef-s128");

    let w = model::Weights::load(&b, i).unwrap();
    let tensors = w.tensors();
    let d = w.desc(&tensors);
    assert_eq!(d.struct_size as usize, size_of::<backend::turbo_backend_model>());
    assert_eq!(d.format, backend::TURBO_FORMAT_HEF);
    assert_eq!(d.graph_input, backend::TURBO_INPUT_EMBEDDINGS);
    assert_eq!(d.graph_output, backend::TURBO_OUTPUT_HIDDEN_STATES);
    assert_eq!(d.compute_dtype, TURBO_DTYPE_I8);
    assert_eq!((d.fixed_seq, d.fixed_batch), (128, 1));
    assert_eq!(w.info_dtype(), TURBO_DTYPE_I8, "turbo_model_info reports the compiled dtype");
    // The architecture, as for raw weights.
    assert_eq!((d.family, d.layers, d.hidden, d.heads, d.intermediate), (backend::TURBO_FAMILY_BERT, 2, 8, 2, 16));
    assert_eq!((d.vocab_size, d.max_positions, d.token_types), (30522, 512, 2));

    // The HEF's bytes as hashed.
    let hef = unsafe { std::slice::from_raw_parts(d.artifact as *const u8, d.artifact_bytes as usize) };
    assert_eq!(hef, HEF);
    assert_eq!(sha256_hex(hef), f.sha256("hailo/model.hef"));

    // The host_weights artifact's embedding tensors, in TURBO_BERT_* order,
    // where they lie in its verified file.
    assert_eq!(d.tensor_count, backend::TURBO_BERT_EMBEDDING_TENSORS);
    assert_eq!(d.dtype, TURBO_DTYPE_F32);
    let handed = unsafe { std::slice::from_raw_parts(d.tensors, d.tensor_count as usize) };
    let files = w.files();
    assert_eq!(files, [fs::read(f.dir.join("weights/model.safetensors")).unwrap().as_slice()]);
    let file = files[0].as_ptr_range();
    let weights = tiny_weights(0);
    for (k, t) in handed.iter().enumerate() {
        let want = &weights[k];
        let name = unsafe { std::ffi::CStr::from_ptr(t.name) }.to_str().unwrap();
        assert_eq!(name, want.name, "tensor {k}");
        assert_eq!(&t.shape[..want.shape.len()], want.shape.as_slice(), "{name}");
        assert_eq!((t.ndim as usize, t.dtype, t.bytes), (want.shape.len(), TURBO_DTYPE_F32, want.data.len() as u64));
        let data = unsafe { std::slice::from_raw_parts(t.data as *const u8, t.bytes as usize) };
        assert_eq!(data, want.data.as_slice(), "{name}");
        assert!(file.contains(&(t.data as *const u8)), "{name} is read where the core verified it");
    }
    assert_eq!(handed[0].shape, [30522, 8]);
}

#[test]
fn raw_weights_are_described_as_before_with_no_artifact_bytes() {
    let mut f = Fixture::model("raw-desc");
    f.manifest["artifacts"][0]["fixed_seq"] = json!(64);
    let b = open(&f);
    let w = model::Weights::load(&b, 0).unwrap();
    let tensors = w.tensors();
    let d = w.desc(&tensors);
    assert_eq!(d.format, backend::TURBO_FORMAT_SAFETENSORS);
    assert_eq!(d.graph_input, backend::TURBO_INPUT_TOKEN_IDS);
    assert_eq!(d.graph_output, backend::TURBO_OUTPUT_HIDDEN_STATES);
    assert_eq!((d.compute_dtype, d.fixed_seq, d.fixed_batch), (0, 64, 0));
    assert!(d.artifact.is_null());
    assert_eq!(d.artifact_bytes, 0);
    assert_eq!(d.tensor_count, backend::TURBO_BERT_EMBEDDING_TENSORS + 2 * backend::TURBO_BERT_LAYER_TENSORS);
    assert_eq!(w.info_dtype(), TURBO_DTYPE_F32);
    assert!(w.artifact().is_none());
}

#[test]
fn a_changed_hef_or_host_weights_file_is_refused() {
    let hef_formats =
        backend::format_bit(backend::TURBO_FORMAT_SAFETENSORS) | backend::format_bit(backend::TURBO_FORMAT_HEF);
    let load = |f: &Fixture| {
        let b = open(f);
        let i = model::choose(&b.manifest, "hailo", hef_formats, "hailo10h").unwrap();
        model::Weights::load(&b, i).err().expect("refused")
    };
    let f = hef("hef-changed", "hailo10h");
    let mut bytes = HEF.to_vec();
    bytes[0] ^= 1;
    fs::write(f.dir.join("hailo/model.hef"), &bytes).unwrap();
    let e = load(&f);
    assert_eq!(e.code, BUNDLE_INTEGRITY, "{e:?}");
    assert!(e.message.contains("hailo/model.hef: SHA-256 is"), "{e:?}");
    bytes.push(0);
    fs::write(f.dir.join("hailo/model.hef"), &bytes).unwrap();
    let e = load(&f);
    assert!(e.code == BUNDLE_INTEGRITY && e.message.contains("hailo/model.hef: size is"), "{e:?}");
    fs::remove_file(f.dir.join("hailo/model.hef")).unwrap();
    let e = load(&f);
    assert!(e.code == BUNDLE_NOT_FOUND && e.message.contains("hailo/model.hef is absent"), "{e:?}");

    // The host_weights file is verified with the HEF.
    let f = hef("hef-host-changed", "hailo10h");
    let p = f.dir.join("weights/model.safetensors");
    let mut bytes = fs::read(&p).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&p, &bytes).unwrap();
    let e = load(&f);
    assert!(e.code == BUNDLE_INTEGRITY && e.message.contains("weights/model.safetensors: SHA-256 is"), "{e:?}");
}

#[test]
fn a_host_lookup_needs_only_the_embedding_tensors() {
    // host_weights whose layer tensors are absent, and not named: the
    // lookup reads the five embedding tensors and nothing else.
    let mut f = hef("hef-embeddings-only", "hailo10h");
    f.weights("weights/model.safetensors", &tiny_weights(0)[..5]);
    let names = f.manifest["artifacts"][1]["tensor_names"].as_object_mut().unwrap();
    let embeddings = model::BERT_EMBEDDING_ROLES.map(manifest::role_name);
    names.retain(|k, _| embeddings.contains(k));
    assert_eq!(names.len(), 5);
    f.manifest["artifacts"][1]["backends"] = json!([]);
    let b = open(&f);
    let formats = backend::format_bit(backend::TURBO_FORMAT_HEF);
    let i = model::choose(&b.manifest, "hailo", formats, "hailo10h").unwrap();
    let w = model::Weights::load(&b, i).unwrap();
    assert_eq!(w.tensors().len(), 5);

    // A word embedding table of the wrong shape is refused as raw weights' is.
    let mut t = tiny_weights(0);
    t[0].shape = vec![30521, 8];
    t[0].data.truncate(30521 * 8 * 4);
    f.weights("weights/model.safetensors", &t[..5]);
    let b = open(&f);
    let e = model::Weights::load(&b, i).err().unwrap();
    assert!(e.code == BUNDLE_INVALID && e.message.contains("word_embeddings"), "{e:?}");
}

#[test]
fn a_hef_is_one_file_with_a_compute_dtype_over_raw_weights() {
    let e = {
        let mut f = hef("hef-two-files", "hailo10h");
        f.manifest["artifacts"][0]["files"] = json!(["hailo/model.hef", "weights/model.safetensors"]);
        refused(&f)
    };
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].files: a HEF is one file"), "{e:?}");
    let e = {
        let mut f = hef("hef-no-dtype", "hailo10h");
        f.manifest["artifacts"][0].as_object_mut().unwrap().remove("compute_dtype");
        refused(&f)
    };
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].compute_dtype: required for a compiled HEF"), "{e:?}");
    let e = {
        let mut f = hef("hef-host-hef", "hailo10h");
        f.manifest["artifacts"][0]["host_weights"] = json!("hef-s128");
        refused(&f)
    };
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].host_weights: \"hef-s128\" is not FORMAT_SAFETENSORS"), "{e:?}");
    for field in ["fixed_seq", "fixed_batch"] {
        let e = {
            let mut f = hef(&format!("hef-no-{field}"), "hailo10h");
            f.manifest["artifacts"][0][field] = json!(0);
            refused(&f)
        };
        assert!(e.is(BUNDLE_INVALID, &format!("artifacts[0].{field}: required for a compiled HEF")), "{e:?}");
    }
}

// Rule 7
