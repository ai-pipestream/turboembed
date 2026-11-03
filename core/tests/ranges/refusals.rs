//! The refusals of a range write.

use std::ptr;

use crate::common::*;
use crate::support::*;
use turbo::status::*;
use turbo::*;

#[test]
fn a_range_write_refuses_what_it_cannot_take() {
    let l = tiny();
    let s = session_of(&l, 4);
    let (t, lens) = tokens_of(&tiny_bundle(), &RANGES_TEXTS);
    let (seq, l1) = (t.seq, lens[1] as u32);
    assert!(l1 < seq);
    let good = [range(0, 0, 2), range(1, 0, 3)];
    let with_third = |third: turbo_token_range| [good[0], good[1], third, range(2, 0, 2)];

    // After a write the run takes, a refused write leaves nothing to run.
    let refused = |ranges: &[turbo_token_range], o: Option<&turbo_embed_options>| -> Failure {
        s.write_ranges(&t.batch(), &good, None).unwrap();
        let e = s.write_ranges(&t.batch(), ranges, o).expect_err("the write is refused");
        assert_eq!(s.run().err().unwrap().code, INVALID_STATE, "{e:?}");
        e
    };

    let e = refused(&[], None);
    assert_eq!(e.code, INVALID_ARGUMENT, "count 0: {e:?}");
    let five = [good[0], good[1], good[0], good[1], good[0]];
    let e = refused(&five, None);
    assert_eq!(e.code, CAPACITY, "count over max_ranges: {e:?}");

    for pooling in [TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        let e = refused(&good, Some(&opts(|o| o.pooling = pooling)));
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 5), "pooling {pooling}: {e:?}");
    }
    let e = refused(&good, Some(&opts(|o| o.pooling = 99)));
    assert_eq!(e.code, INVALID_ENUM, "pooling 99 is no TURBO_POOLING_* value: {e:?}");
    let e = refused(&good, Some(&opts(|o| o.truncate = TURBO_TRUNCATE_RIGHT)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 1), "{e:?}");
    let e = refused(&good, Some(&opts(|o| o.prompt_role = TURBO_PROMPT_QUERY)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 3), "{e:?}");

    for (what, bad) in [
        ("row 3 of 3", range(3, 0, 1)),
        ("start 2 is not under end 2", range(0, 2, 2)),
        ("start 3 is not under end 2", range(0, 3, 2)),
        ("end past seq", range(0, 0, seq + 1)),
    ] {
        let e = refused(&with_third(bad), None);
        assert_eq!(e.code, INVALID_ARGUMENT, "{what}: {e:?}");
        assert!(e.message.contains("ranges[2]"), "{what}: {e:?}");
    }
    // No live token is not a refusal on cpu BERT: the mask-0 columns
    // pool their own states (turbo.h).
    s.write_ranges(&t.batch(), &with_third(range(1, l1, seq)), None).unwrap();
    assert_eq!(s.run().unwrap().info().batch, 4);

    let mut err = new_error();
    s.write_ranges(&t.batch(), &good, None).unwrap();
    let rc = unsafe { turbo_embed_write_ranges(s.0, &t.batch(), ptr::null(), 1, ptr::null(), &mut err) };
    assert_eq!(rc, INVALID_ARGUMENT, "{:?}", failure(rc, &err));
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE);

    // A good write after the refusals runs.
    s.write_ranges(&t.batch(), &good, None).unwrap();
    assert_eq!(s.run().unwrap().info().batch, 2);
}
