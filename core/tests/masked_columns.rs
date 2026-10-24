//! Columns whose mask is 0 are checked as live ones are. turbo_backend.h
//! promises a backend that every id is below vocab_size and every type
//! below token_types, and the Metal, Level Zero, NPU and Hailo backends
//! look every packed column up without reading its mask, so a padding
//! column with an id or type the model lacks is refused before a backend
//! sees it.

mod common;

use common::*;
use turbo::status::INVALID_ARGUMENT;

#[test]
fn padding_with_an_id_or_type_the_model_lacks_is_refused() {
    let l = Loaded::load(&tiny_bundle()).unwrap_or_else(|e| panic!("{e:?}"));
    let s = Session::create(l.m, Some(&session_desc(2, 8, 0))).unwrap();
    // Row 1 is [101, 102] and one column of padding, at index 5.
    let with = |edit: fn(&mut Tokens)| {
        let mut t = Tokens::new(&[vec![101, 7592, 102], vec![101, 102]], 0);
        edit(&mut t);
        t
    };
    s.write_tokens(&with(|_| {}).batch(), None).unwrap();
    for (t, what) in [
        (with(|t| t.ids[5] = -1), "ids[1][2] is -1, outside the model's vocabulary of 30522"),
        (with(|t| t.ids[5] = 30522), "ids[1][2] is 30522, outside the model's vocabulary of 30522"),
        (with(|t| t.types = Some(vec![0, 0, 0, 0, 0, 99])), "types[1][2] is 99, and the model has 2 token types"),
    ] {
        assert_eq!(t.mask[5], 0, "the edited column is padding");
        let e = s.write_tokens(&t.batch(), None).err().unwrap();
        assert!(e.is(INVALID_ARGUMENT, what), "{what}: {e:?}");
    }
}
