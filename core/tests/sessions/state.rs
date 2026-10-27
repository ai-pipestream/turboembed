//! The session's state: one write per run, what a failed call leaves, and
//! what a held result holds.

use std::ptr;

use crate::common::*;
use crate::support::*;
use turbo::status::*;
use turbo::*;

#[test]
fn a_run_takes_one_write() {
    let l = tiny();
    let s = session(&l);
    let e = s.run().err().unwrap();
    assert!(e.is(INVALID_STATE, "nothing is written"), "{e:?}");
    s.write_text(&TEXTS, None).unwrap();
    drop(s.run().unwrap());
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE, "the write was taken by the run");
}

#[test]
fn a_write_replaces_the_one_before() {
    let l = tiny();
    let s = session(&l);
    let alone = s.embed(&TEXTS[1..2], None).unwrap();
    s.write_text(&TEXTS, None).unwrap();
    s.write_text(&TEXTS[1..2], None).unwrap();
    let r = s.run().unwrap();
    assert_eq!(r.info().batch, 1);
    assert_eq!(r.rows(), alone);
}

#[test]
fn a_failed_write_leaves_nothing_written() {
    let l = tiny();
    let s = session(&l);
    s.write_text(&TEXTS, None).unwrap();
    let bad = opts(|o| o.pooling = 9);
    assert_eq!(s.write_text(&TEXTS, Some(&bad)).err().unwrap().code, INVALID_ENUM);
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE);
    s.write_text(&TEXTS, None).unwrap();
    let b = Tokens::new(&[vec![101, 99999, 102]], 0);
    assert_eq!(s.write_tokens(&b.batch(), None).err().unwrap().code, INVALID_ARGUMENT);
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE);
}

#[test]
fn a_write_refused_before_its_rows_are_read_leaves_nothing_written() {
    let l = tiny();
    let s = session(&l);
    s.write_text(&TEXTS, None).unwrap();
    let mut err = new_error();
    let rc = unsafe { turbo_embed_write_tokens(s.0, ptr::null(), ptr::null(), &mut err) };
    assert_eq!(rc, INVALID_ARGUMENT, "{:?}", failure(rc, &err));
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE, "a NULL batch drops the write before it");
    s.write_text(&TEXTS, None).unwrap();
    let b = Tokens::new(&[vec![101, 102]], 0);
    let mut tb = b.batch();
    tb.struct_size = 8;
    assert_eq!(s.write_tokens(&tb, None).err().unwrap().code, INVALID_STRUCT_SIZE);
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE, "a short struct_size drops the write before it");
}

#[test]
fn a_failed_run_takes_the_write() {
    let l = tiny();
    let s = session(&l);
    s.write_text(&TEXTS, None).unwrap();
    let mut err = new_error();
    let rc = unsafe { turbo_session_run(s.0, ptr::null_mut(), &mut err) };
    assert_eq!(rc, INVALID_ARGUMENT, "{:?}", failure(rc, &err));
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE, "the failed run took the write");
}

#[test]
fn a_bundle_without_a_pad_token_pads_text_rows_with_ids_in_the_vocabulary() {
    let mut tj: serde_json::Value = serde_json::from_slice(&std::fs::read(upstream_tokenizer_json()).unwrap()).unwrap();
    tj["added_tokens"].as_array_mut().unwrap().retain(|t| t["content"] != "[PAD]");
    let tj_path = std::env::temp_dir().join(format!("turbo-test-{}-no-pad-tokenizer.json", std::process::id()));
    std::fs::write(&tj_path, serde_json::to_vec(&tj).unwrap()).unwrap();
    let mut m = model_manifest();
    m["tokenizer"]["special_tokens"].as_array_mut().unwrap().retain(|t| t["role"] != "SPECIAL_PAD");
    let mut f = Fixture::with_tokenizer("no-pad", m, &tj_path);
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let l = f.load().unwrap_or_else(|e| panic!("{e:?}"));
    let s = session(&l);
    let tok = Tok::create(&f.dir).unwrap();
    assert_eq!(tok.info().pad_id, -1);
    // Rows of different lengths, so the shorter ones are padded.
    let together = s.embed(&TEXTS, None).unwrap();
    for (i, t) in TEXTS.iter().enumerate() {
        let alone = s.embed(&[t], None).unwrap();
        assert_eq!(together[i], alone[0], "row {i}: padding changed the vector");
    }
    // The tokenizer pads with pad_id, -1; the write names what to do instead.
    let e = tok.encode(&TEXTS, None, 64).unwrap();
    let rows: Vec<Vec<i32>> = (0..TEXTS.len()).map(|i| e.row(i)).collect();
    let padded = Tokens::new(&rows, -1);
    let err = s.write_tokens(&padded.batch(), None).err().unwrap();
    assert!(err.is(INVALID_ARGUMENT, "the bundle has no pad token"), "{err:?}");
    let padded = Tokens::new(&rows, tok.info().unk_id);
    s.write_tokens(&padded.batch(), None).unwrap();
    assert_eq!(s.run().unwrap().rows(), together);
}

#[test]
fn a_held_result_or_its_buffer_makes_the_session_busy() {
    let l = tiny();
    let s = session(&l);
    s.write_text(&TEXTS, None).unwrap();
    let r = s.run().unwrap();
    let e = s.write_text(&TEXTS, None).err().unwrap();
    assert!(e.is(BUSY, "result is held"), "{e:?}");
    assert_eq!(s.run().err().unwrap().code, BUSY);
    let b = Tokens::new(&[vec![101, 102]], 0);
    assert_eq!(s.write_tokens(&b.batch(), None).err().unwrap().code, BUSY);
    assert_eq!(s.info().max_batch, 64, "reading the session's info is not a write");

    let mut buf = ptr::null_mut();
    assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buf, null_err()) }, 0);
    let want = r.rows();
    drop(r);
    assert_eq!(s.write_text(&TEXTS, None).err().unwrap().code, BUSY, "the buffer holds the result");
    let mut host = ptr::null_mut();
    assert_eq!(unsafe { turbo_buffer_host_ptr(buf, &mut host, null_err()) }, 0);
    let held = unsafe { std::slice::from_raw_parts(host as *const f32, 3 * want[0].len()) };
    assert_eq!(held, want.concat().as_slice(), "the vectors are still there");
    unsafe { turbo_buffer_release(buf) };
    s.write_text(&TEXTS, None).unwrap();
    drop(s.run().unwrap());
}
