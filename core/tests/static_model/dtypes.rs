//! The table in each dtype: what a restored bundle gives against the
//! arithmetic on its values.

use super::*;

/// The bundle copied with `tensors` as its table file, and its manifest's
/// hashes, sizes and tensor names made to match.
fn restored(name: &str, tensors: &[Tensor]) -> Scratch {
    let src = bundle();
    let dir = Scratch::at(std::env::temp_dir().join(format!("turbo-static-{name}-{}", std::process::id())));
    for sub in ["", "weights", "reference", "quality"] {
        fs::create_dir_all(dir.join(sub)).unwrap();
    }
    for f in ["tokenizer.json", "reference/reference.safetensors", "quality/texts.jsonl"] {
        fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    let file = safetensors_file(tensors);
    fs::write(dir.join("weights/static.safetensors"), &file).unwrap();
    let mut m: Value = serde_json::from_slice(&fs::read(src.join("manifest.json")).unwrap()).unwrap();
    for e in m["files"].as_array_mut().unwrap() {
        if e["path"] == "weights/static.safetensors" {
            e["size"] = file.len().into();
            e["sha256"] = sha256_hex(&file).into();
        }
    }
    let names = &mut m["artifacts"][0]["tensor_names"];
    for t in tensors {
        let role = match t.name.as_str() {
            "embeddings" => "static_embeddings",
            "weights" => "static_weights",
            _ => "static_mapping",
        };
        names[role] = t.name.clone().into();
    }
    if let Some(map) = tensors.iter().find(|t| t.name == "mapping") {
        let rows = tensors.iter().find(|t| t.name == "embeddings").unwrap().shape[0];
        m["static_embedding"]["rows"] = rows.into();
        assert_eq!(map.shape[0], m["static_embedding"]["vocab_size"].as_u64().unwrap());
    }
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

/// The bundle's F16 table, widened, and its shape.
fn table_values() -> (Vec<f32>, Vec<u64>) {
    let bytes = fs::read(bundle().join("weights/static.safetensors")).unwrap();
    let f = File::parse("static.safetensors", &bytes).unwrap();
    let t = f.tensor("embeddings").unwrap();
    (t.data.chunks(2).map(|c| from_f16(u16::from_le_bytes([c[0], c[1]]))).collect(), t.shape.clone())
}

fn tensor(name: &str, dtype: &'static str, shape: Vec<u64>, data: Vec<u8>) -> Tensor {
    Tensor { name: name.into(), dtype, shape, data }
}

/// The table stored in F32, BF16, F64 and I8, and in F32 behind a token
/// mapping with F64 weights, as Model2Vec's vocabulary quantization
/// stores it: each against the arithmetic written plainly.
#[test]
fn the_table_in_each_dtype_gives_its_values_arithmetic() {
    let f16 = session(&load(&bundle())).embed(&TEXTS, None).unwrap();
    let (wide, shape) = table_values();
    let (vocab, dim) = (shape[0] as usize, shape[1] as usize);
    let max = wide.iter().fold(0f32, |m, v| m.max(v.abs()));
    // Rows in reverse order, each token's weight a value of its own.
    let reversed: Vec<f32> = (0..vocab).rev().flat_map(|r| wide[r * dim..(r + 1) * dim].to_vec()).collect();
    let mapping: Vec<u8> = (0..vocab as i64).flat_map(|id| (vocab as i64 - 1 - id).to_le_bytes()).collect();
    let weights: Vec<u8> = (0..vocab).flat_map(|id| (0.5 + (id % 7) as f64 / 8.0).to_le_bytes()).collect();
    let cases: Vec<(&str, u32, Vec<Tensor>)> = vec![
        (
            "f32",
            TURBO_DTYPE_F32,
            vec![tensor("embeddings", "F32", shape.clone(), wide.iter().flat_map(|v| v.to_le_bytes()).collect())],
        ),
        (
            "bf16",
            TURBO_DTYPE_BF16,
            vec![tensor(
                "embeddings",
                "BF16",
                shape.clone(),
                // Rounded to nearest even.
                wide.iter()
                    .flat_map(|v| {
                        let b = v.to_bits();
                        (((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16).to_le_bytes()
                    })
                    .collect(),
            )],
        ),
        (
            "f64",
            TURBO_DTYPE_F64,
            vec![tensor(
                "embeddings",
                "F64",
                shape.clone(),
                wide.iter().flat_map(|&v| (v as f64).to_le_bytes()).collect(),
            )],
        ),
        (
            "i8",
            TURBO_DTYPE_I8,
            vec![tensor(
                "embeddings",
                "I8",
                shape.clone(),
                wide.iter().map(|v| (v / max * 127.0).round() as i8 as u8).collect(),
            )],
        ),
        (
            "mapped",
            TURBO_DTYPE_F32,
            vec![
                tensor("mapping", "I64", vec![vocab as u64], mapping),
                tensor("weights", "F64", vec![vocab as u64], weights),
                tensor("embeddings", "F32", shape.clone(), reversed.iter().flat_map(|v| v.to_le_bytes()).collect()),
            ],
        ),
    ];
    for (name, dtype, tensors) in cases {
        let dir = restored(name, &tensors);
        let plain = Plain::new(&dir);
        let l = load(&dir);
        assert_eq!(l.info().dtype, dtype, "{name}");
        let s = session(&l);
        let tok = Tok::create(&dir).unwrap();
        let got = s.embed(&TEXTS, None).unwrap();
        for (r, t) in TEXTS.iter().enumerate() {
            let row = tok.row(t, None).unwrap();
            plain.close(
                &got[r],
                &plain.embed(&row, &vec![1; row.len()], TURBO_POOLING_MEAN, true),
                &format!("{name} row {r}"),
            );
            if name == "f32" {
                // The F16 table's values: its vectors, but for the F16
                // table's rounding of the mean and of the result.
                let f16_plain = Plain { tolerance: 1e-3, ..Plain::new(&dir) };
                let want: Vec<f64> = got[r].iter().map(|&v| v as f64).collect();
                f16_plain.close(&f16[r], &want, &format!("{name} row {r} against F16"));
            }
        }
        drop(s);
        drop(l);
        fs::remove_dir_all(&dir).unwrap();
    }
}
