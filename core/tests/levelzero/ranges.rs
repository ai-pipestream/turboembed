//! Range pooling, against the row path and the CPU.

use super::*;

// ---- Range pooling -------------------------------------------------------------------------

/// Ranges of awkward_rows, each holding a token whose mask is 1: whole
/// rows; one live token; a range that starts in row 1's leading padding;
/// row 2's two live tokens apart, around its masked column, one of them
/// in a range that runs past the row's last live token; and two ranges of
/// row 0 side by side.
fn awkward_ranges(t: &Tokens) -> Vec<turbo_token_range> {
    let seq = t.seq;
    let first_live = |r: u32| (0..seq).find(|&p| t.mask[(r * seq + p) as usize] == 1).unwrap();
    let lead = first_live(1);
    let v = vec![
        range(0, 0, seq),
        range(0, 2, 3),
        range(1, 0, lead + 1),
        range(1, lead, seq),
        range(2, 0, 2),
        range(2, 1, seq),
        range(0, 0, 3),
        range(0, 3, seq),
    ];
    for g in &v {
        let at = (g.row * seq) as usize;
        assert!(t.mask[at + g.start as usize..at + g.end as usize].contains(&1), "{g:?} holds a live token");
    }
    v
}

/// A range over a whole row is that row's MEAN vector, bit for bit, at
/// MODEL and at FASTEST: the kernel sums a range as pool sums a row. Every
/// other range is held to the CPU backend's range write on the same rows,
/// with L2 off and on, for more ranges than rows and for fewer. A plain
/// write after a range write replaces it.
#[test]
fn ranges_pool_as_the_row_path_and_the_cpu_do() {
    let _t = turn();
    let Some(_) = gpu_device("ranges_pool_as_the_row_path_and_the_cpu_do") else { return };
    let dir = tiny_bundle();
    let (g, c) = (on_gpu(&dir), Loaded::load(&dir).unwrap());
    let cs = Session::create(c.m, None).unwrap();
    let t = awkward_rows(&dir);
    let all = awkward_ranges(&t);
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let gs = Session::create(g.m, Some(&session_desc(0, 0, precision))).unwrap();
        let floor = if precision == TURBO_PRECISION_FASTEST { 0.999 } else { 0.9999 };
        for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
            let o = opts(|o| {
                o.pooling = TURBO_POOLING_MEAN;
                o.normalize = normalize;
            });
            gs.write_tokens(&t.batch(), Some(&o)).unwrap();
            let rows = gs.run().unwrap().rows();
            let whole: Vec<turbo_token_range> = (0..t.batch).map(|r| range(r, 0, t.seq)).collect();
            gs.write_ranges(&t.batch(), &whole, Some(&o)).unwrap();
            assert_eq!(gs.run().unwrap().rows(), rows, "precision {precision} normalize {normalize}: whole rows");
            // More ranges than rows, then fewer.
            for ranges in [&all[..], &all[1..3]] {
                gs.write_ranges(&t.batch(), ranges, Some(&o)).unwrap();
                let run = gs.run().unwrap();
                assert_eq!(run.info().batch as usize, ranges.len());
                let got = run.rows();
                drop(run);
                cs.write_ranges(&t.batch(), ranges, Some(&o)).unwrap();
                let want = cs.run().unwrap().rows();
                assert_eq!(got.len(), ranges.len());
                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(a.len(), b.len());
                    let cos = cosine(a, b);
                    assert!(
                        cos >= floor,
                        "precision {precision} normalize {normalize} {:?}: cosine {cos} with the cpu",
                        ranges[i]
                    );
                    if normalize == TURBO_NORMALIZE_L2 {
                        let norm: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                        assert!((norm - 1.0).abs() < 1e-5, "{:?}: norm {norm}", ranges[i]);
                    }
                }
            }
            // A plain write after a range write is the one the run pools.
            gs.write_ranges(&t.batch(), &all, Some(&o)).unwrap();
            gs.write_tokens(&t.batch(), Some(&o)).unwrap();
            assert_eq!(gs.run().unwrap().rows(), rows, "a row write replaces a range write");
        }
    }
}

/// max_ranges past max_batch sizes the output for that many vectors: a
/// session of 2 rows writes 8 ranges, each a vector the CPU backend's
/// session of the same shape gives. One more is refused by the core.
#[test]
fn max_ranges_past_max_batch_gives_that_many_vectors() {
    let _t = turn();
    let Some(_) = gpu_device("max_ranges_past_max_batch_gives_that_many_vectors") else { return };
    let dir = tiny_bundle();
    let (g, c) = (on_gpu(&dir), Loaded::load(&dir).unwrap());
    let desc = turbo_session_desc { max_ranges: 8, ..session_desc(2, 0, TURBO_PRECISION_MODEL) };
    let (gs, cs) = (Session::create(g.m, Some(&desc)).unwrap(), Session::create(c.m, Some(&desc)).unwrap());
    let tok = Tok::create(&dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS[..2].iter().map(|t| tok.row(t, None).unwrap()).collect();
    let t = Tokens::new(&rows, 0);
    let ranges: Vec<turbo_token_range> =
        (0..8).map(|i| range(i % 2, i / 2, (i / 2 + 1 + i % 3).min(rows[(i % 2) as usize].len() as u32))).collect();
    let o = opts(|o| o.normalize = TURBO_NORMALIZE_L2);
    gs.write_ranges(&t.batch(), &ranges, Some(&o)).unwrap();
    let got = gs.run().unwrap().rows();
    cs.write_ranges(&t.batch(), &ranges, Some(&o)).unwrap();
    let want = cs.run().unwrap().rows();
    assert_eq!(got.len(), 8);
    for (g, (a, b)) in ranges.iter().zip(got.iter().zip(&want)) {
        let cos = cosine(a, b);
        assert!(cos >= 0.9999, "{g:?}: cosine {cos} with the cpu");
    }
    let nine: Vec<turbo_token_range> = (0..9).map(|_| range(0, 0, 1)).collect();
    let e = gs.write_ranges(&t.batch(), &nine, Some(&o)).err().unwrap();
    assert_eq!(e.code, CAPACITY, "9 ranges past max_ranges 8: {e:?}");
}

/// The device is not a CPU: a session that names a thread count is
/// refused by field 6.
#[test]
fn a_thread_count_is_refused_by_field_6() {
    let _t = turn();
    let Some(_) = gpu_device("a_thread_count_is_refused_by_field_6") else { return };
    let g = on_gpu(&tiny_bundle());
    let desc = turbo_session_desc { cpu_threads: 1, ..session_desc(0, 0, 0) };
    let e = Session::create(g.m, Some(&desc)).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6), "{e:?}");
    assert!(e.message.contains("not a CPU"), "{e:?}");
}
