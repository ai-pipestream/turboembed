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
