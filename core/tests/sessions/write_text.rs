//! write_text: the refusals of a text write, every option checked by
//! name, and the prompt roles a bundle declares.

use std::ptr;

use crate::common::*;
use crate::support::*;
use turbo::status::*;
use turbo::*;

#[test]
fn write_text_refuses_what_it_cannot_take() {
    let l = tiny();
    let s = Session::create(l.m, Some(&session_desc(2, 16, 0))).unwrap();
    let e = s.write_text(&[], None).err().unwrap();
    assert!(e.is(INVALID_ARGUMENT, "count is 0"), "{e:?}");
    assert_eq!(unsafe { turbo_embed_write_text(s.0, ptr::null(), 1, ptr::null(), null_err()) }, INVALID_ARGUMENT);
    let e = s.write_text(&TEXTS, None).err().unwrap();
    assert!(e.is(CAPACITY, "3 texts is over the session's max_batch 2"), "{e:?}");
    let bad = [0xffu8, 0xfe];
    let t = turbo_text { ptr: bad.as_ptr() as *const _, len: 2 };
    let mut err = new_error();
    assert_eq!(unsafe { turbo_embed_write_text(s.0, &t, 1, ptr::null(), &mut err) }, INVALID_UTF8);
    assert!(failure(INVALID_UTF8, &err).message.contains("texts[0]"));
    // Cut at the bundle's max_seq, 64, as on every device; then longer
    // than this session's 16.
    let long = PARAGRAPH;
    let e = s.write_text(&["short", long], None).err().unwrap();
    assert!(e.is(CAPACITY, "texts[1]: 64 tokens is over the session's max_seq 16"), "{e:?}");
    // Asked to fit, it fits.
    s.write_text(&[long], Some(&opts(|o| o.max_tokens = 16))).unwrap();
    let e = s.write_text(&[long], Some(&opts(|o| o.max_tokens = 17))).err().unwrap();
    assert!(e.is(CAPACITY, "max_tokens 17 is over the session's max_seq 16"), "{e:?}");
    let e = s.write_text(&[long], Some(&opts(|o| o.truncate = TURBO_TRUNCATE_NONE))).err().unwrap();
    assert!(e.is(CAPACITY, "TRUNCATE_NONE"), "{e:?}");
    let e = s.write_text(&["a"], Some(&opts(|o| o.max_tokens = 1))).err().unwrap();
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 2), "{e:?}");
}

/// What a write made of each text, read back (issue #2511).
fn row_tokens(s: &Session) -> std::result::Result<Vec<turbo_row_tokens>, Failure> {
    let mut rows = [turbo_row_tokens::default(); 8];
    let (mut n, mut err) = (0u32, new_error());
    let rc = unsafe { turbo_session_row_tokens(s.0, rows.as_mut_ptr(), rows.len() as u32, &mut n, &mut err) };
    if rc != 0 {
        return Err(failure(rc, &err));
    }
    Ok(rows[..n as usize].to_vec())
}

/// A text cut at the limit says so beside its row: the tokens kept, the
/// tokens the whole text gives, and that it was cut. Before, the write
/// succeeded and nothing told a cut row from a whole one.
#[test]
fn a_cut_text_is_told_row_by_row() {
    let l = tiny();
    let s = Session::create(l.m, Some(&session_desc(2, 64, 0))).unwrap();
    // Nothing was written yet.
    assert_eq!(row_tokens(&s).err().unwrap().code, INVALID_STATE);
    s.write_text(&["short", PARAGRAPH], None).unwrap();
    let rows = row_tokens(&s).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!((rows[0].truncated, rows[0].seen_known), (0, 1), "{rows:?}");
    assert_eq!(rows[0].kept, rows[0].seen, "{rows:?}");
    // The bundle's max_seq is 64: TURBO_TRUNCATE_MODEL keeps that many.
    assert_eq!((rows[1].kept, rows[1].truncated, rows[1].seen_known), (64, 1, 1), "{rows:?}");
    assert!(rows[1].seen > 64, "{rows:?}");
    // The counts stay readable until the next write: a result holds the
    // session (busy, like every session call), and once it is released
    // the same counts are read again.
    let out = s.run().unwrap();
    assert_eq!(row_tokens(&s).err().unwrap().code, BUSY);
    drop(out);
    assert_eq!(row_tokens(&s).unwrap(), rows, "the counts outlive the run");
    // Left truncation keeps as many and reports the same whole count.
    s.write_text(&[PARAGRAPH], Some(&opts(|o| o.truncate = TURBO_TRUNCATE_LEFT))).unwrap();
    let left = row_tokens(&s).unwrap();
    assert_eq!(left, [rows[1]]);
    // A capacity under the rows written is refused, never a short answer.
    let (mut n, mut one) = (0u32, turbo_row_tokens::default());
    s.write_text(&["a", "b"], None).unwrap();
    assert_eq!(unsafe { turbo_session_row_tokens(s.0, &mut one, 1, &mut n, null_err()) }, CAPACITY);
    // A write of the caller's own tokens has no text to have cut.
    let (ids, mask) = ([101i32, 102], [1i32, 1]);
    let batch = turbo_token_batch {
        struct_size: size_of::<turbo_token_batch>() as u32,
        batch: 1,
        seq: 2,
        row_stride: 2,
        ids: ids.as_ptr(),
        mask: mask.as_ptr(),
        types: ptr::null(),
    };
    assert_eq!(unsafe { turbo_embed_write_tokens(s.0, &batch, ptr::null(), null_err()) }, 0);
    assert_eq!(row_tokens(&s).err().unwrap().code, INVALID_STATE);
}

/// A text of exactly the limit is whole, one token more is cut, and in a
/// batch beside a short and a long text each row keeps its own counts, in
/// order.
#[test]
fn a_text_at_the_limit_is_whole_and_one_over_is_cut() {
    let l = tiny();
    let s = Session::create(l.m, Some(&session_desc(4, 64, 0))).unwrap();
    let words = |n: usize| vec!["a"; n].join(" ");
    let whole = |n: usize| {
        s.write_text(&[words(n).as_str()], None).unwrap();
        row_tokens(&s).unwrap()[0]
    };
    // The bundle's max_seq is 64, special tokens included.
    let at = (1..64).find(|&n| whole(n).seen == 64).expect("a text of exactly 64 tokens");
    let r = whole(at);
    assert_eq!((r.kept, r.seen, r.seen_known, r.truncated), (64, 64, 1, 0), "{r:?}");
    let r = whole(at + 1);
    assert_eq!((r.kept, r.seen, r.seen_known, r.truncated), (64, 65, 1, 1), "{r:?}");
    let (short, exact, over) = (words(1), words(at), words(at + 1));
    s.write_text(&[short.as_str(), exact.as_str(), PARAGRAPH, over.as_str()], None).unwrap();
    let rows = row_tokens(&s).unwrap();
    let counts: Vec<_> = rows.iter().map(|r| (r.kept, r.truncated)).collect();
    assert_eq!(counts, [(rows[0].seen, 0), (64, 0), (64, 1), (64, 1)], "{rows:?}");
    assert_eq!((rows[1].seen, rows[3].seen), (64, 65), "{rows:?}");
}

#[test]
fn every_option_is_checked_by_name() {
    let l = tiny();
    let s = session(&l);
    let refused = |o: turbo_embed_options| s.write_text(&["a"], Some(&o)).err().unwrap();
    for (o, name) in [
        (opts(|o| o.truncate = 4), "truncate: 4"),
        (opts(|o| o.prompt_role = 3), "prompt_role: 3"),
        (opts(|o| o.normalize = 3), "normalize: 3"),
        (opts(|o| o.pooling = 4), "pooling: 4"),
    ] {
        let e = refused(o);
        assert!(e.is(INVALID_ENUM, name), "{e:?}");
    }
    let e = refused(opts(|o| o.output_dim = 33));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 6), "{e:?}");
    assert!(e.message.contains("output_dim 33 is over the model's dim 32"), "{e:?}");
    let e = refused(opts(|o| o.output_dim = 16));
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6), "{e:?}");
    let mut o = embed_options();
    o.struct_size += 4;
    assert_eq!(refused(o).code, INVALID_STRUCT_SIZE);
    // The model's own width is always there.
    assert_eq!(s.embed(&["a"], Some(&opts(|o| o.output_dim = 32))).unwrap()[0].len(), 32);
}

#[test]
fn prompt_roles_prefix_the_text_as_the_bundle_says() {
    let l = tiny();
    assert_eq!(field(&l.info().prefix_query), "query: ");
    let s = session(&l);
    let q = s.embed(&["reset"], Some(&opts(|o| o.prompt_role = TURBO_PROMPT_QUERY))).unwrap();
    assert_eq!(q, s.embed(&["query: reset"], None).unwrap());
    assert_ne!(q, s.embed(&["reset"], None).unwrap());
    // The bundle has no document prefix.
    let d = s.embed(&["reset"], Some(&opts(|o| o.prompt_role = TURBO_PROMPT_DOCUMENT))).unwrap();
    assert_eq!(d, s.embed(&["reset"], None).unwrap());
}
