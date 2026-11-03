pub fn options(omit_special_tokens: u32, truncate: u32, max_tokens: u32, prompt_role: u32) -> turbo_encode_options {
    turbo_encode_options {
        struct_size: size_of::<turbo_encode_options>() as u32,
        omit_special_tokens,
        truncate,
        max_tokens,
        prompt_role,
    }
}

/// The texts in testdata/tokenizer-texts.jsonl.
pub fn parity_texts() -> Vec<String> {
    fs::read_to_string(testdata().join("tokenizer-texts.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap()["text"].as_str().unwrap().to_owned())
        .collect()
}

// ---- A small BERT --------------------------------------------------------

/// Two layers, hidden 8, two heads, intermediate 16, over the upstream
/// vocabulary: the shapes of a BERT encoder at a size a test writes in
/// a moment.
pub fn tiny_architecture() -> Value {
    json!({
        "family": "FAMILY_BERT",
        "layers": 2,
        "hidden": 8,
        "heads": 2,
        "intermediate": 16,
        "activation": "ACTIVATION_GELU_ERF",
        "layer_norm_eps": 1e-12,
        "position_embedding": "POSITION_ABSOLUTE",
        "max_positions": 512,
        "token_types": 2,
        "vocab_size": 30522
    })
}

/// The standard manifest for the small BERT, loadable on the CPU: its
/// architecture, a dim of its hidden width, and cpu among the weights
/// artifact's backends, as the MiniLM recipe has it.
pub fn model_manifest() -> Value {
    let mut m = manifest();
    m["architecture"] = tiny_architecture();
    m["embed"]["dim"] = json!(8);
    m["artifacts"][0]["backends"] = json!(["cuda", "metal", "levelzero", "cpu"]);
    m["files"] = json!([]);
    m
}

pub struct Tensor {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

/// Every tensor of the small BERT under the upstream names, F32, with
/// values that depend on `seed` and the element's place and nothing else.
pub fn tiny_weights(seed: u32) -> Vec<Tensor> {
    bert_weights(2, 8, 16, seed, 1.0)
}

/// Every tensor of a BERT of `layers` layers, hidden width `h` and
/// intermediate width `i` under the upstream names, F32, as tiny_weights
/// makes them, with each weight matrix scaled by `scale`.
pub fn bert_weights(layers: usize, h: u64, i: u64, seed: u32, scale: f32) -> Vec<Tensor> {
    let (v, p, t) = (30522u64, 512u64, 2u64);
    let mut shapes: Vec<(String, Vec<u64>)> = vec![
        ("embeddings.word_embeddings.weight".into(), vec![v, h]),
        ("embeddings.position_embeddings.weight".into(), vec![p, h]),
        ("embeddings.token_type_embeddings.weight".into(), vec![t, h]),
        ("embeddings.LayerNorm.weight".into(), vec![h]),
        ("embeddings.LayerNorm.bias".into(), vec![h]),
    ];
    for l in 0..layers {
        let at = |s: &str| format!("encoder.layer.{l}.{s}");
        for (s, shape) in [
            ("attention.self.query.weight", vec![h, h]),
            ("attention.self.query.bias", vec![h]),
            ("attention.self.key.weight", vec![h, h]),
            ("attention.self.key.bias", vec![h]),
            ("attention.self.value.weight", vec![h, h]),
            ("attention.self.value.bias", vec![h]),
            ("attention.output.dense.weight", vec![h, h]),
            ("attention.output.dense.bias", vec![h]),
            ("attention.output.LayerNorm.weight", vec![h]),
            ("attention.output.LayerNorm.bias", vec![h]),
            ("intermediate.dense.weight", vec![i, h]),
            ("intermediate.dense.bias", vec![i]),
            ("output.dense.weight", vec![h, i]),
            ("output.dense.bias", vec![h]),
            ("output.LayerNorm.weight", vec![h]),
            ("output.LayerNorm.bias", vec![h]),
        ] {
            shapes.push((at(s), shape));
        }
    }
    // A tensor upstream carries that a BERT encoder does not use.
    shapes.push(("pooler.dense.bias".into(), vec![h]));
    let mut k = 0u64;
    shapes
        .into_iter()
        .map(|(name, shape)| {
            let n: u64 = shape.iter().product();
            let s = if name.starts_with("encoder.") && shape.len() == 2 { scale } else { 1.0 };
            let data = (0..n)
                .flat_map(|_| {
                    k += 1;
                    let x = ((k * 7919 + seed as u64 * 104729) % 2001) as f32 / 1000.0 - 1.0;
                    (x * s).to_le_bytes()
                })
                .collect();
            Tensor { name, dtype: "F32", shape, data }
        })
        .collect()
}

/// A safetensors file of `tensors`, in the order given.
pub fn safetensors_file(tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!({ "format": "pt" }));
    let mut at = 0usize;
    for t in tensors {
        let end = at + t.data.len();
        header.insert(t.name.clone(), json!({ "dtype": t.dtype, "shape": t.shape, "data_offsets": [at, end] }));
        at = end;
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for t in tensors {
        out.extend(&t.data);
    }
    out
}

/// A model loaded through the C interface on the CPU, with the runtime and
/// context it was loaded with.
pub struct Loaded {
    pub rt: *mut turbo_runtime,
    pub ctx: *mut turbo_context,
    pub m: *mut turbo_model,
}

impl Loaded {
    pub fn load(dir: &Path) -> Result<Loaded, Failure> {
        Loaded::load_on(dir, cpu)
    }

    /// The bundle loaded on the device `device` picks from the runtime.
    pub fn load_on(dir: &Path, device: impl Fn(*mut turbo_runtime) -> u32) -> Result<Loaded, Failure> {
        let mut err = new_error();
        let mut rt = ptr::null_mut();
        assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, &mut err) }, 0);
        let mut ctx = ptr::null_mut();
        let rc = unsafe { turbo_context_create(rt, device(rt), &mut ctx, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        let mut m = ptr::null_mut();
        let path = dir.to_str().unwrap();
        let rc = unsafe { turbo_model_load(ctx, text(path), &mut m, &mut err) };
        let loaded = Loaded { rt, ctx, m };
        if rc != 0 {
            assert!(m.is_null(), "a failed load wrote out");
            return Err(failure(rc, &err));
        }
        Ok(loaded)
    }

    /// The handles, which the caller now releases.
    pub fn into_raw(self) -> (*mut turbo_runtime, *mut turbo_context, *mut turbo_model) {
        let l = std::mem::ManuallyDrop::new(self);
        (l.rt, l.ctx, l.m)
    }

    /// A model handle alone, released on drop.
    pub fn model(m: *mut turbo_model) -> Loaded {
        Loaded { rt: ptr::null_mut(), ctx: ptr::null_mut(), m }
    }

    pub fn info(&self) -> turbo_model_info {
        let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_model_info>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_model_get_info(self.m, &mut info, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        info
    }
}

impl Drop for Loaded {
    fn drop(&mut self) {
        unsafe {
            turbo_model_release(self.m);
            turbo_context_release(self.ctx);
            turbo_runtime_release(self.rt);
        }
    }
}

/// The index of the device the cpu backend lists.
pub fn cpu(rt: *mut turbo_runtime) -> u32 {
    let mut n = 0;
    assert_eq!(unsafe { turbo_runtime_device_count(rt, &mut n, ptr::null_mut()) }, 0);
    (0..n)
        .find(|&i| {
            let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
            info.struct_size = size_of::<turbo_device_info>() as u32;
            assert_eq!(unsafe { turbo_runtime_device_info(rt, i, &mut info, ptr::null_mut()) }, 0);
            info.kind == TURBO_DEVICE_CPU
        })
        .expect("the cpu backend lists the host")
}

/// The index of the first device the backend named lists, if any.
pub fn first_of(rt: *mut turbo_runtime, backend: &str) -> Option<u32> {
    let mut n = 0;
    assert_eq!(unsafe { turbo_runtime_device_count(rt, &mut n, ptr::null_mut()) }, 0);
    (0..n).find(|&i| {
        let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_device_info>() as u32;
        assert_eq!(unsafe { turbo_runtime_device_info(rt, i, &mut info, ptr::null_mut()) }, 0);
        field(&info.backend) == backend
    })
}

/// A fixed-size C string field as a String.
pub fn field(b: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = b.iter().take_while(|&&c| c != 0).map(|&c| u8::from_ne_bytes(c.to_ne_bytes())).collect();
    String::from_utf8(bytes).unwrap()
}

// ---- Sessions ------------------------------------------------------------

/// The sealed bundle in testdata/tiny-bert-bundle: a small BERT whose
/// reference vectors are upstream sentence-transformers outputs.
pub fn tiny_bundle() -> PathBuf {
    testdata().join("tiny-bert-bundle")
}
