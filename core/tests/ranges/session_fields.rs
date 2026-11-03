//! The session's max_ranges and cpu_threads fields.

use crate::common::*;
use crate::support::*;
use turbo::status::*;
use turbo::*;

#[test]
fn max_ranges_defaults_to_max_batch_and_is_bounded() {
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let l = tiny();
    let (t, lens) = tokens_of(&dir, &RANGES_TEXTS[1..]);
    let (l0, l1) = (lens[0] as u32, lens[1] as u32);
    let o = normalized(TURBO_NORMALIZE_NONE);

    // 0 is max_batch: four ranges in a session of four rows, not five.
    let s = Session::create(l.m, Some(&session_desc(4, 16, TURBO_PRECISION_MODEL))).unwrap();
    let four = [range(0, 0, l0), range(1, 0, l1), range(0, 1, 3), range(1, 2, 4)];
    assert_eq!(run_ranges(&s, &t, &four, &o).len(), 4);
    let five = [four[0], four[1], four[2], four[3], four[0]];
    let e = s.write_ranges(&t.batch(), &five, Some(&o)).err().unwrap();
    assert_eq!(e.code, CAPACITY, "{e:?}");

    // The most is max_batch * max_seq.
    let mut d = session_desc(4, 16, TURBO_PRECISION_MODEL);
    d.max_ranges = 65;
    let e = Session::create(l.m, Some(&d)).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 7), "{e:?}");
    d.max_ranges = 64;
    Session::create(l.m, Some(&d)).unwrap();

    // More ranges than rows: every token of one row alone.
    let mut d = session_desc(2, 32, TURBO_PRECISION_MODEL);
    d.max_ranges = 10;
    let s = Session::create(l.m, Some(&d)).unwrap();
    let (one, _) = tokens_of(&dir, &RANGES_TEXTS[..1]);
    let ranges: Vec<_> = (0..10).map(|p| range(0, p, p + 1)).collect();
    let got = run_ranges(&s, &one, &ranges, &o);
    assert_eq!(got.len(), 10);
    for (p, g) in got.iter().enumerate() {
        let (ids, mask, ty) = row_of(&one, 0);
        let want = plain.embed_range(ids, mask, &ty, p, p + 1, 32, false);
        for (a, b) in g.iter().zip(&want) {
            assert!((*a as f64 - b).abs() < 1e-5 * (1.0 + b.abs()), "token {p}: {a} vs {b}");
        }
    }
}

#[test]
fn cpu_threads_sets_the_sessions_thread_count() {
    let l = tiny();
    let (t, lens) = tokens_of(&tiny_bundle(), &RANGES_TEXTS);
    let ranges = [range(0, 0, 6), range(0, 6, lens[0] as u32), range(1, 0, lens[1] as u32), range(2, 2, 5)];
    let o = normalized(TURBO_NORMALIZE_L2);
    let with_threads = |threads: u32| {
        let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
        d.cpu_threads = threads;
        Session::create(l.m, Some(&d)).unwrap_or_else(|e| panic!("{threads}: {e:?}"))
    };
    let base = with_threads(0);
    base.write_tokens(&t.batch(), Some(&o)).unwrap();
    let want_rows = base.run().unwrap().rows();
    let want_ranges = run_ranges(&base, &t, &ranges, &o);
    for threads in [1, 2, 3] {
        let s = with_threads(threads);
        s.write_tokens(&t.batch(), Some(&o)).unwrap();
        let rows = s.run().unwrap().rows();
        for (g, w) in rows.iter().zip(&want_rows) {
            assert_eq!(bits(g), bits(w), "{threads} threads, rows");
        }
        let got = run_ranges(&s, &t, &ranges, &o);
        for (g, w) in got.iter().zip(&want_ranges) {
            assert_eq!(bits(g), bits(w), "{threads} threads, ranges");
        }
    }

    let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
    d.cpu_threads = 1025;
    let e = Session::create(l.m, Some(&d)).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6), "{e:?}");
    d.cpu_threads = 1024;
    Session::create(l.m, Some(&d)).unwrap();

    // A desc of 24 bytes ends before the new fields, which then read as 0.
    let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
    d.struct_size = 24;
    d.cpu_threads = 1025;
    d.max_ranges = 1000;
    let s = Session::create(l.m, Some(&d)).unwrap();
    assert_eq!(run_ranges(&s, &t, &ranges, &o).len(), 4);
}
