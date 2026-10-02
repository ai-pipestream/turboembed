//! The conformance check: a bundle's reference cases run through the C
//! interface on one device, and compared with the vectors the upstream
//! pipeline produced. tests/conformance.rs runs it on the device
//! TURBO_TEST_DEVICE names; each device's own tests run it on theirs.

use std::path::Path;
use std::ptr;

use serde_json::Value;
use turbo::*;

use super::*;

/// The lowest cosine an int8 session must reach against the fp32
/// reference. Benchmark records have no int8 tolerance; this test does.
pub const I8_MIN_COSINE: f64 = 0.93;

/// What a session's compute dtype must reach against the fp32 reference:
/// the rule benchmark records are judged by too (turbo::record), and
/// I8_MIN_COSINE for int8.
pub fn tolerance(compute_dtype: u32) -> record::Tolerance {
    if compute_dtype == TURBO_DTYPE_I8 {
        return record::Tolerance { min_cosine: I8_MIN_COSINE, max_abs_diff: None };
    }
    record::tolerance(compute_dtype).unwrap_or_else(|| panic!("compute dtype {compute_dtype} has no tolerance"))
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

    fn add(&mut self, what: &str, got: &[f32], want: &[f32], floor: record::Tolerance) {
        assert_eq!(got.len(), want.len(), "{what}: width");
        let c = cosine(got, want);
        assert!(c >= floor.min_cosine, "{what}: cosine {c} is under {}", floor.min_cosine);
        let d = max_abs_diff(got, want);
        if let Some(most) = floor.max_abs_diff {
            assert!(d <= most, "{what}: max abs diff {d:e} is over {most:e}");
        }
        self.cosine = self.cosine.min(c);
        self.abs = self.abs.max(d);
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

/// Every reference case of the bundle at `dir`, through the C interface on
/// the device `device` picks, in a session at `precision`, against the
/// vectors the upstream pipeline produced. Returns how many cases were
/// longer than the session takes.
pub fn check(dir: &Path, device: impl Fn(*mut turbo_runtime) -> u32, precision: u32) -> usize {
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
    let s = Session::create(model, Some(&session_desc(0, 0, precision))).unwrap_or_else(|e| panic!("{e:?}"));
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
        "{}: {} cases on device {dev}, precision {}, compute dtype {}, cosine floor {}, max abs diff {:?}",
        dir.display(),
        cases.len(),
        si.precision,
        si.compute_dtype,
        floor.min_cosine,
        floor.max_abs_diff
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
