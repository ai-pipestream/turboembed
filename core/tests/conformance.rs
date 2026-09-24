//! The conformance check: a bundle's reference cases run through the C
//! interface on one device, and compared with the vectors the upstream
//! pipeline produced. The same test serves every backend; docs/conformance.md
//! says how to run it and what it checks.
//!
//! TURBO_TEST_BUNDLE names the bundle directory; without it, the small
//! sealed bundle in testdata/tiny-bert-bundle. TURBO_TEST_DEVICE names the
//! device, as a runtime device index or a backend name (the first device
//! that backend lists); without it, the CPU.

mod common;

use std::path::{Path, PathBuf};
use std::ptr;

use common::*;
use serde_json::Value;
use turbo::*;

fn bundle() -> PathBuf {
    std::env::var_os("TURBO_TEST_BUNDLE").map_or_else(tiny_bundle, PathBuf::from)
}

/// The device TURBO_TEST_DEVICE names, or the CPU.
fn device(rt: *mut turbo_runtime) -> u32 {
    let Ok(want) = std::env::var("TURBO_TEST_DEVICE") else {
        return cpu(rt);
    };
    if let Ok(i) = want.parse() {
        return i;
    }
    let mut n = 0;
    assert_eq!(unsafe { turbo_runtime_device_count(rt, &mut n, ptr::null_mut()) }, 0);
    (0..n)
        .find(|&i| {
            let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
            info.struct_size = size_of::<turbo_device_info>() as u32;
            assert_eq!(unsafe { turbo_runtime_device_info(rt, i, &mut info, ptr::null_mut()) }, 0);
            field(&info.backend) == want
        })
        .unwrap_or_else(|| panic!("TURBO_TEST_DEVICE={want}: no device of that backend is listed"))
}

/// The lowest cosine a session's compute dtype must reach against the
/// fp32 reference.
fn tolerance(compute_dtype: u32) -> f64 {
    match compute_dtype {
        TURBO_DTYPE_F32 => 0.9999,
        TURBO_DTYPE_F16 | TURBO_DTYPE_BF16 => 0.999,
        d => panic!("compute dtype {d} has no tolerance"),
    }
}

/// The worst of a set of comparisons.
#[derive(Default)]
struct Worst {
    cosine: f64,
    abs: f64,
    rows: usize,
}

impl Worst {
    fn new() -> Worst {
        Worst { cosine: 1.0, abs: 0.0, rows: 0 }
    }

    fn add(&mut self, what: &str, got: &[f32], want: &[f32], floor: f64) {
        assert_eq!(got.len(), want.len(), "{what}: width");
        let c = cosine(got, want);
        assert!(c >= floor, "{what}: cosine {c} is under {floor}");
        self.cosine = self.cosine.min(c);
        self.abs = self.abs.max(max_abs_diff(got, want));
        self.rows += 1;
    }
}

struct Case {
    text: String,
    role: u32,
    ids: Vec<i32>,
    vector: Vec<f32>,
}

fn cases(dir: &Path, m: &Value) -> Vec<Case> {
    let r = Reference::read(dir, m);
    m["reference"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .zip(r.ids)
        .zip(r.embeddings)
        .map(|((c, ids), vector)| Case {
            text: c["text"].as_str().unwrap().to_owned(),
            role: match c["prompt_role"].as_str().unwrap() {
                "PROMPT_QUERY" => TURBO_PROMPT_QUERY,
                "PROMPT_DOCUMENT" => TURBO_PROMPT_DOCUMENT,
                _ => TURBO_PROMPT_NONE,
            },
            ids,
            vector,
        })
        .collect()
}

/// Returns how many cases were longer than the session takes.
fn check(dir: &Path) -> usize {
    let m: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let cases = cases(dir, &m);

    let mut err = new_error();
    let mut rt = ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, &mut err) }, 0);
    let dev = device(rt);
    let mut ctx = ptr::null_mut();
    let rc = unsafe { turbo_context_create(rt, dev, &mut ctx, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    let mut model = ptr::null_mut();
    let rc = unsafe { turbo_model_load(ctx, text(dir.to_str().unwrap()), &mut model, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    let loaded = Loaded { rt, ctx, m: model };
    let mi = loaded.info();
    let s = Session::create(model, None).unwrap_or_else(|e| panic!("{e:?}"));
    let si = s.info();
    let floor = tolerance(si.compute_dtype);
    let prefix = |role: u32| match role {
        TURBO_PROMPT_QUERY => field(&mi.prefix_query),
        TURBO_PROMPT_DOCUMENT => field(&mi.prefix_document),
        _ => String::new(),
    };

    // The ids, through the bundle's tokenizer, with each case's role.
    let tok = Tok::create(dir).unwrap_or_else(|e| panic!("{e:?}"));
    for (i, c) in cases.iter().enumerate() {
        let o = options(0, 0, 0, c.role);
        assert_eq!(tok.row(&c.text, Some(&o)).unwrap(), c.ids, "case {i}: ids");
    }

    // A case longer than the session takes is refused whole, never cut
    // differently on this device (docs/bundle.md).
    let fits = |c: &Case| c.ids.len() <= si.max_seq as usize;
    for (i, c) in cases.iter().enumerate().filter(|(_, c)| !fits(c)) {
        let o = turbo_embed_options { prompt_role: c.role, ..embed_options() };
        let e = s.write_text(&[&c.text], Some(&o)).unwrap_err();
        assert_eq!(e.code, status::CAPACITY, "case {i}: {e:?}");
        let e = s.write_tokens(&Tokens::new(std::slice::from_ref(&c.ids), 0).batch(), None).unwrap_err();
        assert_eq!(e.code, status::CAPACITY, "case {i}: {e:?}");
    }
    let run: Vec<(usize, &Case)> = cases.iter().enumerate().filter(|(_, c)| fits(c)).collect();

    // write_text, one case at a time, with the case's prompt role.
    let mut text1 = Worst::new();
    for &(i, c) in &run {
        let o = turbo_embed_options { prompt_role: c.role, ..embed_options() };
        let got = s.embed(&[&c.text], Some(&o)).unwrap_or_else(|e| panic!("case {i}: {e:?}"));
        text1.add(&format!("write_text case {i}"), &got[0], &c.vector, floor);
    }

    // write_text, every case in batches as large as the session takes: one
    // write has one prompt role, so each text carries its prefix itself.
    let mut text_full = Worst::new();
    for group in run.chunks(si.max_batch as usize) {
        let texts: Vec<String> = group.iter().map(|(_, c)| format!("{}{}", prefix(c.role), c.text)).collect();
        let views: Vec<&str> = texts.iter().map(String::as_str).collect();
        let got = s.embed(&views, None).unwrap_or_else(|e| panic!("{e:?}"));
        for ((i, c), v) in group.iter().zip(&got) {
            text_full.add(&format!("write_text batch, case {i}"), v, &c.vector, floor);
        }
    }

    // write_tokens with the reference's ids, one at a time and batched.
    let pad = mi_pad(&tok);
    let mut tokens1 = Worst::new();
    for &(i, c) in &run {
        s.write_tokens(&Tokens::new(std::slice::from_ref(&c.ids), pad).batch(), None).unwrap();
        tokens1.add(&format!("write_tokens case {i}"), &s.run().unwrap().rows()[0], &c.vector, floor);
    }
    let mut tokens_full = Worst::new();
    for group in run.chunks(si.max_batch as usize) {
        let rows: Vec<Vec<i32>> = group.iter().map(|(_, c)| c.ids.clone()).collect();
        s.write_tokens(&Tokens::new(&rows, pad).batch(), None).unwrap();
        for ((i, c), v) in group.iter().zip(s.run().unwrap().rows()) {
            tokens_full.add(&format!("write_tokens batch, case {i}"), &v, &c.vector, floor);
        }
    }

    println!(
        "{}: {} cases on device {dev}, compute dtype {}, cosine floor {floor}",
        dir.display(),
        cases.len(),
        si.compute_dtype
    );
    for (what, w) in [
        ("write_text, batch 1", &text1),
        ("write_text, full batch", &text_full),
        ("write_tokens, batch 1", &tokens1),
        ("write_tokens, full batch", &tokens_full),
    ] {
        println!("  {what}: {} rows, 1 - min cosine {:.3e}, max abs diff {:.3e}", w.rows, 1.0 - w.cosine, w.abs);
        assert_eq!(w.rows, run.len(), "{what}");
    }
    assert!(!run.is_empty(), "no case fits the session");
    println!("  {} cases longer than max_seq {} refused with TURBO_E_CAPACITY", cases.len() - run.len(), si.max_seq);
    cases.len() - run.len()
}

fn mi_pad(tok: &Tok) -> i32 {
    tok.info().pad_id.max(0)
}

#[test]
fn every_reference_case_matches_upstream() {
    check(&bundle());
}

/// The small bundle as a fixed-shape artifact of 32 tokens would have it:
/// the cases longer than that are refused with TURBO_E_CAPACITY, and the
/// rest still match.
#[test]
fn a_fixed_shape_refuses_the_cases_it_cannot_hold() {
    let dir = std::env::temp_dir().join(format!("turbo-conformance-{}-fixed", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors", "weights/model.safetensors"] {
        std::fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        std::fs::copy(tiny_bundle().join(f), dir.join(f)).unwrap();
    }
    let mut m: Value = serde_json::from_slice(&std::fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    m["artifacts"][0]["fixed_seq"] = 32.into();
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    let long = check(&dir);
    assert!(long > 0, "some case is longer than 32 tokens");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn a_real_bundle_matches_its_reference() {
    let dir = std::env::var_os("TURBO_TEST_BUNDLE").expect("TURBO_TEST_BUNDLE is not set");
    check(Path::new(&dir));
}
