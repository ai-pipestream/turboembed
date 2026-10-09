//! A static model through the C interface on the CPU backend this build
//! links, over the sealed bundle in testdata/tiny-static-bundle: what it
//! reports, every option against the same arithmetic written plainly in
//! f64, rows with no tokens, every precision, and the table stored in each
//! dtype. tests/conformance.rs holds its vectors to the Model2Vec
//! reference with TURBO_TEST_BUNDLE.

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::*;
use serde_json::Value;
use turbo::safetensors::{Dtype, File};
use turbo::*;

fn bundle() -> PathBuf {
    testdata().join("tiny-static-bundle")
}

fn load(dir: &Path) -> Loaded {
    Loaded::load(dir).unwrap_or_else(|e| panic!("{e:?}"))
}

fn session(l: &Loaded) -> Session {
    Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_MODEL))).unwrap_or_else(|e| panic!("{e:?}"))
}

fn opts(pooling: u32, normalize: u32) -> turbo_embed_options {
    let mut o = embed_options();
    o.pooling = pooling;
    o.normalize = normalize;
    o
}

fn from_f16(h: u16) -> f32 {
    let (sign, exp, man) = ((h as u32 & 0x8000) << 16, (h >> 10) & 0x1f, h as u32 & 0x3ff);
    match (exp, man) {
        (0, 0) => f32::from_bits(sign),
        // Subnormal: man x 2^-24.
        (0, _) => f32::from_bits(sign) + if sign != 0 { -1.0 } else { 1.0 } * man as f32 * 2f32.powi(-24),
        _ => f32::from_bits(sign | ((exp as u32 + 112) << 23) | (man << 13)),
    }
}

/// The bundle's table, weights and mapping widened to f64, and its
/// width: StaticModel's arithmetic written plainly, in f64.
struct Plain {
    table: Vec<f64>,
    weights: Option<Vec<f64>>,
    mapping: Option<Vec<usize>>,
    dim: usize,
    /// The library rounds an F16 table's vectors to F16, as StaticModel
    /// does; the plain arithmetic does not.
    tolerance: f64,
}

impl Plain {
    fn new(dir: &Path) -> Plain {
        let bytes = fs::read(dir.join("weights/static.safetensors")).unwrap();
        let f = File::parse("static.safetensors", &bytes).unwrap();
        let wide = |name: &str| -> Option<(Vec<f64>, Vec<u64>, Dtype)> {
            let t = f.tensor(name)?;
            let v = match t.dtype {
                Dtype::F16 => t.data.chunks(2).map(|c| from_f16(u16::from_le_bytes([c[0], c[1]])) as f64).collect(),
                Dtype::Bf16 => t
                    .data
                    .chunks(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
                    .collect(),
                Dtype::F32 => t.f32s().into_iter().map(f64::from).collect(),
                Dtype::F64 => t.data.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
                Dtype::I8 => t.data.iter().map(|&b| b as i8 as f64).collect(),
                Dtype::I64 => t.data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
                other => panic!("{other:?}"),
            };
            Some((v, t.shape.clone(), t.dtype))
        };
        let (table, shape, dtype) = wide("embeddings").unwrap();
        Plain {
            table,
            weights: wide("weights").map(|w| w.0),
            mapping: wide("mapping").map(|m| m.0.iter().map(|&v| v as usize).collect()),
            dim: shape[1] as usize,
            tolerance: if dtype == Dtype::F16 { 1e-3 } else { 1e-5 },
        }
    }

    fn scaled(&self, id: i32) -> Vec<f64> {
        let (id, d) = (id as usize, self.dim);
        let row = self.mapping.as_ref().map_or(id, |m| m[id]);
        let w = self.weights.as_ref().map_or(1.0, |w| w[id]);
        self.table[row * d..(row + 1) * d].iter().map(|v| v * w).collect()
    }

    /// One row of `ids` under `mask`: the static model's vector in f64.
    fn embed(&self, ids: &[i32], mask: &[i32], pooling: u32, normalize: bool) -> Vec<f64> {
        let live: Vec<i32> = ids.iter().zip(mask).filter(|&(_, &m)| m == 1).map(|(&id, _)| id).collect();
        let mut v = if live.is_empty() {
            vec![0.0; self.dim]
        } else {
            match pooling {
                TURBO_POOLING_CLS => self.scaled(ids[0]),
                TURBO_POOLING_LAST => self.scaled(*live.last().unwrap()),
                _ => {
                    let mut acc = vec![0.0; self.dim];
                    for &id in &live {
                        for (a, x) in acc.iter_mut().zip(self.scaled(id)) {
                            *a += x;
                        }
                    }
                    acc.iter().map(|a| a / live.len() as f64).collect()
                }
            }
        };
        if normalize {
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-32;
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }

    fn close(&self, got: &[f32], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (g, w) in got.iter().zip(want) {
            assert!((*g as f64 - w).abs() < self.tolerance * (1.0 + w.abs()), "{what}: {g} vs {w}");
        }
    }
}

const TEXTS: [&str; 4] =
    ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a", "Café naïve RÉSUMÉ, 东京"];

#[test]
fn a_static_model_reports_what_its_manifest_says() {
    let l = load(&bundle());
    let info = l.info();
    let m: Value = serde_json::from_slice(&fs::read(bundle().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(info.task, TURBO_TASK_EMBED);
    assert_eq!(info.dim, 16);
    assert_eq!(info.max_seq, m["embed"]["max_seq"].as_u64().unwrap() as u32);
    assert_eq!(info.dtype, TURBO_DTYPE_F16);
    assert_eq!((info.pooling, info.normalize), (TURBO_POOLING_MEAN, TURBO_NORMALIZE_L2));
    let s = session(&l);
    let si = s.info();
    assert_eq!(si.compute_dtype, TURBO_DTYPE_F32);
}

/// Every pooling, with and without normalization, with a masked token in
/// the middle of a row and padding on either side, against Plain in f64.
#[test]
fn every_option_matches_the_arithmetic_written_plainly() {
    let dir = bundle();
    let plain = Plain::new(&dir);
    let l = load(&dir);
    let s = session(&l);
    let tok = Tok::create(&dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let seq = rows.iter().map(Vec::len).max().unwrap() + 2;
    let mut t = Tokens::new(&vec![vec![0; seq]; rows.len()], 0);
    t.mask.fill(0);
    for (r, row) in rows.iter().enumerate() {
        let start = if r == 1 { seq - row.len() } else { 0 };
        for (p, &id) in row.iter().enumerate() {
            t.ids[r * seq + start + p] = id;
            t.mask[r * seq + start + p] = 1;
        }
    }
    t.mask[1] = 0;

    for pooling in [TURBO_POOLING_MEAN, TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
            let o = opts(pooling, normalize);
            s.write_tokens(&t.batch(), Some(&o)).unwrap();
            let got = s.run().unwrap().rows();
            for (r, row) in got.iter().enumerate() {
                let at = r * seq..(r + 1) * seq;
                let want = plain.embed(&t.ids[at.clone()], &t.mask[at], pooling, normalize == TURBO_NORMALIZE_L2);
                plain.close(row, &want, &format!("pooling {pooling} normalize {normalize} row {r}"));
            }
        }
    }

    // Text is tokenized with no special tokens: the same as its ids.
    let from_text = s.embed(&TEXTS, None).unwrap();
    for (r, row) in rows.iter().enumerate() {
        plain.close(
            &from_text[r],
            &plain.embed(row, &vec![1; row.len()], TURBO_POOLING_MEAN, true),
            &format!("text {r}"),
        );
    }
}

#[test]
fn a_text_of_no_tokens_or_only_unknown_ones_is_the_zero_vector() {
    let dir = bundle();
    let l = load(&dir);
    let s = session(&l);
    let tok = Tok::create(&dir).unwrap();
    // A character the vocabulary does not hold is [UNK], which is dropped
    // as StaticModel drops it.
    let unk = "\u{2603}\u{2603}";
    assert_eq!(tok.row(unk, None).unwrap(), Vec::<i32>::new());
    for pooling in [TURBO_POOLING_MEAN, TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        let got = s.embed(&["", "   ", "a", unk], Some(&opts(pooling, TURBO_NORMALIZE_L2))).unwrap();
        assert!(got[0].iter().all(|&x| x == 0.0), "pooling {pooling}: {:?}", got[0]);
        assert!(got[1].iter().all(|&x| x == 0.0), "pooling {pooling}: {:?}", got[1]);
        assert!(got[2].iter().any(|&x| x != 0.0), "pooling {pooling}");
        assert!(got[3].iter().all(|&x| x == 0.0), "pooling {pooling}: {:?}", got[3]);
    }
    // Only empty texts: one column of padding, no live token.
    let got = s.embed(&["", ""], None).unwrap();
    assert!(got.iter().flatten().all(|&x| x == 0.0));
}

#[test]
fn every_precision_computes_the_same_f32() {
    let l = load(&bundle());
    let want = session(&l).embed(&TEXTS, None).unwrap();
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let s = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap();
        assert_eq!(s.info().compute_dtype, TURBO_DTYPE_F32, "precision {p}");
        assert_eq!(s.embed(&TEXTS, None).unwrap(), want, "precision {p}");
    }
}

#[test]
fn a_run_reports_its_stages() {
    let l = load(&bundle());
    let s = session(&l);
    for (normalize, stage) in [(TURBO_NORMALIZE_L2, TURBO_STAGE_HOST), (TURBO_NORMALIZE_NONE, TURBO_STAGE_UNUSED)] {
        s.write_text(&TEXTS, Some(&opts(0, normalize))).unwrap();
        let info = s.run().unwrap().info();
        assert_eq!(info.stage_count as usize, TURBO_EMBED_STAGE_COUNT);
        let st = &info.stage;
        assert_eq!(st[TURBO_EMBED_STAGE_TOKENIZE], TURBO_STAGE_HOST);
        assert_eq!(st[TURBO_EMBED_STAGE_LOOKUP], TURBO_STAGE_HOST);
        assert_eq!(st[TURBO_EMBED_STAGE_ENCODE], TURBO_STAGE_UNUSED);
        assert_eq!(st[TURBO_EMBED_STAGE_POOL], TURBO_STAGE_FUSED);
        assert_eq!(st[TURBO_EMBED_STAGE_NORMALIZE], stage);
    }
}

/// A batch large enough that write_text tokenizes it on several threads
/// gives the rows one text at a time gives.
#[test]
fn a_large_batch_gives_what_one_text_at_a_time_gives() {
    let l = load(&bundle());
    let s = Session::create(l.m, Some(&session_desc(256, 0, TURBO_PRECISION_MODEL))).unwrap();
    let texts: Vec<String> = (0..256).map(|i| format!("{} number {i}", TEXTS[i % TEXTS.len()])).collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let all = s.embed(&refs, None).unwrap();
    for (i, t) in refs.iter().enumerate() {
        assert_eq!(s.embed(&[t], None).unwrap()[0], all[i], "text {i}");
    }
    // The first text that fails is the one named, as on one thread.
    let mut bad = refs.clone();
    let long = "word ".repeat(400);
    bad[200] = &long;
    bad[77] = &long;
    let mut o = embed_options();
    o.truncate = TURBO_TRUNCATE_NONE;
    let e = s.write_text(&bad, Some(&o)).unwrap_err();
    assert!(e.message.starts_with("texts[77]:"), "{e:?}");
}

/// The bundle copied with `tensors` as its table file, and its manifest's
/// hashes, sizes and tensor names made to match.
fn restored(name: &str, tensors: &[Tensor]) -> PathBuf {
    let src = bundle();
    let dir = std::env::temp_dir().join(format!("turbo-static-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
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

// ---- The manifest ------------------------------------------------------------------

/// The tiny static bundle's manifest with `edit` applied, parsed.
fn parsed(edit: impl FnOnce(&mut Value)) -> Result<turbo::manifest::Manifest, turbo::status::Error> {
    let mut m: Value = serde_json::from_slice(&fs::read(bundle().join("manifest.json")).unwrap()).unwrap();
    edit(&mut m);
    turbo::manifest::Manifest::parse(&serde_json::to_vec(&m).unwrap())
}

fn refused(edit: impl FnOnce(&mut Value), containing: &str) {
    match parsed(edit) {
        Ok(_) => panic!("the manifest parsed; wanted {containing:?}"),
        Err(e) => {
            assert_eq!(e.code, turbo::status::BUNDLE_INVALID, "{}", e.message);
            assert!(e.message.contains(containing), "{} does not say {containing:?}", e.message);
        }
    }
}

#[test]
fn a_static_manifest_is_checked_field_by_field() {
    parsed(|_| {}).expect("the bundle's own manifest");
    let mut arch = tiny_architecture();
    arch["hidden"] = 16.into();
    refused(|m| m["architecture"] = arch, "static_embedding: a static model has no architecture");
    refused(|m| m["static_embedding"]["vocab_size"] = 0.into(), "static_embedding.vocab_size");
    refused(|m| m["static_embedding"]["distilled_from"]["model_id"] = "".into(), "distilled_from.model_id");
    refused(|m| m["static_embedding"]["distilled_from"]["manifest_sha256"] = "ABC".into(), "manifest_sha256");
    refused(|m| m["static_embedding"]["quality"]["texts"] = "quality/other.jsonl".into(), "quality.texts");
    refused(|m| m["static_embedding"]["quality"]["static_top1"] = 1.5.into(), "quality.static_top1");
    refused(|m| m["static_embedding"]["quality"]["similarity_spearman"] = (-2.0).into(), "similarity_spearman");
    refused(|m| m["static_embedding"]["vocab_size"] = 101.into(), "is not under static_embedding.vocab_size 101");
    refused(|m| m["static_embedding"]["quality"]["extra"] = 1.into(), "extra");
    refused(|m| m["static_embedding"]["max_length"] = 0.into(), "static_embedding.max_length");
    refused(|m| m["static_embedding"]["max_length"] = 4096.into(), "is over embed.max_seq");
    refused(|m| m["tokenizer"]["template"] = serde_json::json!(["[CLS]", "$TEXT"]), "a static model's rows");
    refused(|m| m["static_embedding"]["rows"] = 30.into(), "static_mapping and static_embedding.rows come together");
    refused(
        |m| m["artifacts"][0]["tensor_names"]["static_mapping"] = "mapping".into(),
        "static_mapping and static_embedding.rows come together",
    );
    refused(
        |m| m["tokenizer"]["special_tokens"][0]["single_word"] = true.into(),
        "single_word and rstrip are read with normalized",
    );
    refused(|m| m["artifacts"][0]["format"] = "FORMAT_ONNX".into(), "a static model's artifacts are raw weights");
    refused(
        |m| m["artifacts"][0]["tensor_names"]["word_embeddings"] = "embeddings".into(),
        "an encoder's role, and the bundle is not one",
    );
    refused(
        |m| {
            m.as_object_mut().unwrap().remove("static_embedding");
        },
        "architecture: required by raw weights",
    );
}

#[test]
fn every_reference_case_matches_model2vec() {
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        assert_eq!(common::conformance::check(&bundle(), cpu, p), 0, "precision {p}: every case fits the session");
    }
}
