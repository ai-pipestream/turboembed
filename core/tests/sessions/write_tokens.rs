//! write_tokens: the refusals of a token-batch write, and tokens against
//! text.

use std::ptr;

use crate::common::*;
use crate::support::*;
use turbo::status::*;
use turbo::*;

#[test]
fn write_tokens_refuses_what_it_cannot_take() {
    let l = tiny();
    let s = Session::create(l.m, Some(&session_desc(2, 8, 0))).unwrap();
    let ok = Tokens::new(&[vec![101, 7592, 102], vec![101, 102]], 0);
    s.write_tokens(&ok.batch(), None).unwrap();
    let fails = |b: &turbo_token_batch, o: Option<&turbo_embed_options>| s.write_tokens(b, o).err().unwrap();

    assert_eq!(unsafe { turbo_embed_write_tokens(s.0, ptr::null(), ptr::null(), null_err()) }, INVALID_ARGUMENT);
    let mut b = ok.batch();
    b.struct_size = 8;
    assert_eq!(fails(&b, None).code, INVALID_STRUCT_SIZE);
    for (edit, code, what) in [
        ((|b: &mut turbo_token_batch| b.batch = 0) as fn(&mut turbo_token_batch), INVALID_SHAPE, "neither may be 0"),
        (|b| b.seq = 0, INVALID_SHAPE, "neither may be 0"),
        (|b| b.batch = 3, CAPACITY, "batch 3 is over the session's max_batch 2"),
        (|b| b.seq = 9, CAPACITY, "seq 9 is over the session's max_seq 8"),
        (|b| b.row_stride = 2, INVALID_SHAPE, "row_stride 2 is under seq 3"),
        (|b| b.ids = ptr::null(), INVALID_ARGUMENT, "may not be NULL"),
        (|b| b.mask = ptr::null(), INVALID_ARGUMENT, "may not be NULL"),
    ] {
        let mut b = ok.batch();
        edit(&mut b);
        let e = fails(&b, None);
        assert!(e.is(code, what), "{what}: {e:?}");
    }

    let with = |edit: fn(&mut Tokens)| {
        let mut t = Tokens::new(&[vec![101, 7592, 102], vec![101, 102]], 0);
        edit(&mut t);
        t
    };
    for (t, what) in [
        (with(|t| t.ids[1] = -1), "ids[0][1] is -1"),
        (with(|t| t.ids[4] = 30522), "ids[1][1] is 30522, outside the model's vocabulary of 30522"),
        (with(|t| t.mask[2] = 2), "mask[0][2] is 2"),
        (with(|t| t.mask[3..].fill(0)), "mask row 1 has no 1"),
        (with(|t| t.types = Some(vec![0, 2, 0, 0, 0, 0])), "types[0][1] is 2, and the model has 2 token types"),
    ] {
        let e = fails(&t.batch(), None);
        assert!(e.is(INVALID_ARGUMENT, what), "{what}: {e:?}");
    }
    let e = fails(&ok.batch(), Some(&opts(|o| o.truncate = TURBO_TRUNCATE_RIGHT)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 1), "{e:?}");
    let e = fails(&ok.batch(), Some(&opts(|o| o.prompt_role = TURBO_PROMPT_QUERY)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 3), "{e:?}");
    let e = fails(&ok.batch(), Some(&opts(|o| o.max_tokens = 2)));
    assert!(e.is(CAPACITY, "row 0: 3 tokens through its last live one is over max_tokens 2"), "{e:?}");
    let e = fails(&ok.batch(), Some(&opts(|o| o.max_tokens = 9)));
    assert!(e.is(CAPACITY, "max_tokens 9 is over the session's max_seq 8"), "{e:?}");
    s.write_tokens(&ok.batch(), Some(&opts(|o| o.max_tokens = 3))).unwrap();
}

#[test]
fn tokens_and_text_give_the_same_vectors() {
    let l = tiny();
    let s = session(&l);
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let from_text = s.embed(&TEXTS, None).unwrap();
    let t = Tokens::new(&rows, 0);
    s.write_tokens(&t.batch(), None).unwrap();
    assert_eq!(s.run().unwrap().rows(), from_text);

    // The same rows at a wider stride, and with explicit zero types.
    let seq = t.seq as usize;
    let mut wide = Tokens::new(&rows, 0);
    wide.stride = seq as u32 + 5;
    wide.ids = vec![-9; rows.len() * wide.stride as usize];
    wide.mask = vec![-9; rows.len() * wide.stride as usize];
    for r in 0..rows.len() {
        let at = r * wide.stride as usize;
        wide.ids[at..at + seq].copy_from_slice(&t.ids[r * seq..(r + 1) * seq]);
        wide.mask[at..at + seq].copy_from_slice(&t.mask[r * seq..(r + 1) * seq]);
    }
    wide.types = Some(vec![0; wide.ids.len()]);
    s.write_tokens(&wide.batch(), None).unwrap();
    assert_eq!(s.run().unwrap().rows(), from_text, "the stride's gap is never read");
}
