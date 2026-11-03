//! A range over a whole row.

use crate::common::*;
use crate::support::*;
use turbo::*;

#[test]
fn a_range_over_a_whole_row_is_the_rows_mean_vector() {
    let l = tiny();
    let s = session_of(&l, 0);
    let (t, lens) = tokens_of(&tiny_bundle(), &RANGES_TEXTS);
    assert!(lens[0] > lens[1] && lens[1] > lens[2], "rows of different lengths: {lens:?}");
    let whole: Vec<_> = (0..3).map(|r| range(r, 0, t.seq)).collect();
    for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
        let o = opts(|o| {
            o.pooling = TURBO_POOLING_MEAN;
            o.normalize = normalize;
        });
        s.write_tokens(&t.batch(), Some(&o)).unwrap();
        let rows = s.run().unwrap().rows();

        s.write_ranges(&t.batch(), &whole, Some(&o)).unwrap();
        let r = s.run().unwrap();
        let i = r.info();
        assert_eq!((i.task, i.batch, i.dim), (TURBO_TASK_EMBED, 3, 32), "normalize {normalize}");
        let got = r.rows();
        for (g, w) in got.iter().zip(&rows) {
            assert_eq!(bits(g), bits(w), "normalize {normalize}");
        }
    }
}
