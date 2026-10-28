//! Ranges: the refusals of a range write, the mask read through the row
//! stride, and a range against the row's own vector.

use std::ptr;

use crate::common::*;
use turbo::status::*;
use turbo::*;

/// A session of 2 rows by 8 tokens that takes 4 ranges, and two rows of 3
/// columns: row 0 of three tokens, row 1 of two and a padding column.
fn ranges_session(l: &Loaded) -> (Session, Tokens) {
    let mut d = session_desc(2, 8, 0);
    d.max_ranges = 4;
    let s = Session::create(l.m, Some(&d)).unwrap_or_else(|e| panic!("{e:?}"));
    (s, Tokens::new(&[vec![101, 7592, 102], vec![101, 102]], 0))
}

#[test]
fn write_ranges_refuses_what_it_cannot_take() {
    let l = tiny();
    let (s, t) = ranges_session(&l);
    let ok = [range(0, 0, 3), range(1, 0, 2)];
    s.write_ranges(&t.batch(), &ok, None).unwrap();
    let fails = |ranges: &[turbo_token_range], o: Option<&turbo_embed_options>| {
        s.write_ranges(&t.batch(), ranges, o).err().unwrap()
    };

    let mut err = new_error();
    let rc = unsafe { turbo_embed_write_ranges(s.0, &t.batch(), ptr::null(), 2, ptr::null(), &mut err) };
    assert!(failure(rc, &err).is(INVALID_ARGUMENT, "ranges is NULL and count is 2"), "{:?}", failure(rc, &err));
    let rc = unsafe { turbo_embed_write_ranges(s.0, &t.batch(), ptr::null(), 0, ptr::null(), &mut err) };
    assert!(failure(rc, &err).is(INVALID_ARGUMENT, "count is 0"), "{:?}", failure(rc, &err));
    let e = fails(&[], None);
    assert!(e.is(INVALID_ARGUMENT, "count is 0: a run needs at least one range"), "{e:?}");
    let e = fails(&[range(0, 0, 3); 5], None);
    assert!(e.is(CAPACITY, "5 ranges is over the session's max_ranges 4"), "{e:?}");
    s.write_ranges(&t.batch(), &[range(0, 0, 3); 4], None).unwrap();

    for pooling in [TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        let e = fails(&ok, Some(&opts(|o| o.pooling = pooling)));
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 5), "{e:?}");
        assert!(e.message.contains("a range is pooled by the mean alone"), "{e:?}");
    }
    s.write_ranges(&t.batch(), &ok, Some(&opts(|o| o.pooling = TURBO_POOLING_MEAN))).unwrap();
    let e = fails(&ok, Some(&opts(|o| o.truncate = TURBO_TRUNCATE_RIGHT)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 1), "{e:?}");
    let e = fails(&ok, Some(&opts(|o| o.prompt_role = TURBO_PROMPT_QUERY)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 3), "{e:?}");
    // The rows are checked as turbo_embed_write_tokens checks them.
    let mut bad = Tokens::new(&[vec![101, 7592, 102], vec![101, 102]], 0);
    bad.ids[1] = 30522;
    let e = s.write_ranges(&bad.batch(), &ok, None).err().unwrap();
    assert!(e.is(INVALID_ARGUMENT, "ids[0][1] is 30522"), "{e:?}");

    // Each range is named by its index, after a good one.
    for (r, what) in [
        (range(2, 0, 1), "ranges[1]: row 2 is not under batch 2"),
        (range(0, 1, 1), "ranges[1]: start 1 is not under end 1"),
        (range(0, 2, 1), "ranges[1]: start 2 is not under end 1"),
        (range(0, 2, 4), "ranges[1]: end 4 is over seq 3"),
    ] {
        let e = fails(&[range(0, 0, 3), r], None);
        assert!(e.is(INVALID_ARGUMENT, what), "{what}: {e:?}");
    }
    // A range holding no live token pools its mask-0 columns' own
    // states on this cpu BERT session (turbo.h): the write is taken.
    s.write_ranges(&t.batch(), &[range(1, 2, 3)], None).unwrap();
}

/// A range's mask is read through the row stride: row 1's column 0 is
/// mask 0 through the stride (the gap columns of the row before, with
/// mask 1, are not its), so its range pools that column's own state
/// and differs from the range over the live columns beside it.
#[test]
fn a_ranges_mask_is_read_through_the_row_stride() {
    let l = tiny();
    let (s, _) = ranges_session(&l);
    let mut t = Tokens::new(&[vec![101, 7592, 102], vec![101, 102, 0]], 0);
    t.stride = 5;
    t.ids = vec![101, 7592, 102, 0, 0, 0, 101, 102, 0, 0];
    t.mask = vec![1, 1, 1, 1, 1, 0, 1, 1, 0, 0];
    s.write_ranges(&t.batch(), &[range(1, 1, 3)], None).unwrap();
    let live = s.run().unwrap().rows();
    s.write_ranges(&t.batch(), &[range(1, 0, 1)], None).unwrap();
    let masked = s.run().unwrap().rows();
    assert!(masked[0].iter().all(|v| v.is_finite()), "{masked:?}");
    assert_ne!(masked, live, "the mask-0 column pools its own state, not a live neighbour's");
}

#[test]
fn a_failed_range_write_leaves_nothing_written() {
    let l = tiny();
    let (s, t) = ranges_session(&l);
    s.write_tokens(&t.batch(), None).unwrap();
    let e = s.write_ranges(&t.batch(), &[range(0, 0, 4)], None).err().unwrap();
    assert_eq!(e.code, INVALID_ARGUMENT, "{e:?}");
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE);
    s.write_ranges(&t.batch(), &[range(0, 0, 3)], None).unwrap();
    let e = s.write_ranges(&t.batch(), &[range(0, 0, 3)], Some(&opts(|o| o.pooling = TURBO_POOLING_CLS)));
    assert_eq!(e.err().unwrap().code, UNSUPPORTED_OPTION);
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE);
}

/// A range over a whole row is that row's vector under mean pooling, to
/// the bit, since it is the same sum over the same tokens in the same
/// order. The ranges come back in the order given, repeats and all, and
/// the result counts ranges, not rows. A range write replaces a token
/// write before it, and a token write replaces a range write.
#[test]
fn a_range_over_a_whole_row_is_the_rows_mean_vector() {
    let l = tiny();
    let (s, t) = ranges_session(&l);
    let mean = opts(|o| o.pooling = TURBO_POOLING_MEAN);
    s.write_tokens(&t.batch(), Some(&mean)).unwrap();
    let rows = s.run().unwrap().rows();
    let dim = rows[0].len();

    s.write_tokens(&t.batch(), Some(&mean)).unwrap();
    // Row 1's padding column is inside the range and pooled as nothing.
    s.write_ranges(&t.batch(), &[range(1, 0, 3), range(0, 0, 3), range(1, 0, 2)], None).unwrap();
    let r = s.run().unwrap();
    let info = r.info();
    assert_eq!((info.batch, info.dim), (3, dim as u32));
    assert_eq!(info.bytes, 3 * dim as u64 * 4);
    let got = r.rows();
    drop(r);
    assert_eq!(got, [rows[1].clone(), rows[0].clone(), rows[1].clone()]);

    s.write_ranges(&t.batch(), &[range(0, 0, 3)], None).unwrap();
    s.write_tokens(&t.batch(), Some(&mean)).unwrap();
    let r = s.run().unwrap();
    assert_eq!(r.info().batch, 2, "the token write replaced the range write");
    assert_eq!(r.rows(), rows);
}
