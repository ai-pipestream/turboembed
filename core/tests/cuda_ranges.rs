//! Late pooling's ranges on CUDA against the row path's vectors, the f64
//! arithmetic and the CPU backend's ranges, and packed rows against the
//! CPU at every precision and pooling.
//!
//! Built with the `cuda` feature only. A test that needs a device says it
//! was skipped, and passes, when the backend lists none; nothing is run
//! on anything else in its place. With TURBO_TEST_REQUIRE_CUDA=1 it fails
//! instead, so a run on a GPU machine cannot pass by finding no GPU.
//! docs/cuda.md says how to run them; tests/cuda_support/ holds what the
//! files share.

#![cfg(feature = "cuda")]

mod common;
#[path = "cuda_support/mod.rs"]
mod cuda_support;

use common::*;
use cuda_support::*;
use serde_json::json;
use turbo::status::UNSUPPORTED_OPTION;
use turbo::*;

// ---- Range pooling ---------------------------------------------------------------------

/// A session on `m` of the model's size at `precision` that takes
/// `max_ranges` ranges.
fn ranged_session(m: *mut turbo_model, precision: u32, max_ranges: u32) -> Session {
    let mut d = session_desc(0, 0, precision);
    d.max_ranges = max_ranges;
    Session::create(m, Some(&d)).unwrap_or_else(|e| panic!("{e:?}"))
}

/// A range over a whole row, padding and a masked token inside it
/// included, is that row's MEAN vector from embed_write, bit for bit, at
/// EXACT and FASTEST and with either pooling kernel (the default's groups
/// of tokens and TURBO_CUDA_POOL=columns): a range is pooled by the row's
/// own kernel over the same tokens in the same order.
#[test]
fn a_range_over_a_whole_row_is_the_rows_mean_vector_bit_for_bit() {
    let _t = turn();
    let Some(_) = cuda_device("a_range_over_a_whole_row_is_the_rows_mean_vector_bit_for_bit") else { return };
    let l = on_cuda(&tiny_bundle());
    let t = awkward_rows(&tiny_bundle());
    let whole: Vec<_> = (0..t.batch).map(|r| range(r, 0, t.seq)).collect();
    for precision in [TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
        for columns in [false, true] {
            let mut d = session_desc(0, 0, precision);
            d.max_ranges = 8;
            turbo::cuda::use_column_pool(Some(columns));
            let s = Session::create(l.m, Some(&d));
            turbo::cuda::use_column_pool(None);
            let s = s.unwrap();
            for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
                let what = format!("precision {precision} columns {columns} normalize {normalize}");
                let o = opts(|o| {
                    o.pooling = TURBO_POOLING_MEAN;
                    o.normalize = normalize;
                });
                s.write_tokens(&t.batch(), Some(&o)).unwrap();
                let rows = s.run().unwrap().rows();
                s.write_ranges(&t.batch(), &whole, Some(&o)).unwrap();
                let got = s.run().unwrap().rows();
                assert_eq!(got.len(), rows.len());
                for (r, (g, w)) in got.iter().zip(&rows).enumerate() {
                    assert_eq!(bits(g), bits(w), "{what} row {r}");
                }
                // In another order, and a row twice: each vector is its row's.
                let shuffled = [whole[2], whole[0], whole[2], whole[1]];
                s.write_ranges(&t.batch(), &shuffled, Some(&o)).unwrap();
                let got = s.run().unwrap().rows();
                assert_eq!(got.len(), shuffled.len());
                for (g, r) in got.iter().zip([2, 0, 2, 1]) {
                    assert_eq!(bits(g), bits(&rows[r]), "{what} row {r} out of order");
                }
            }
        }
    }
}

/// Ranges of every kind (one live token, a span over a masked token and
/// padding, two on one row, a row left out, the same range twice), more
/// ranges than rows and fewer, with and without L2: the result's shape,
/// each vector against the f64 encoder of tests/common and against the
/// CPU backend's range write on the same rows. A plain write after a
/// range write pools rows again.
#[test]
fn ranges_match_the_arithmetic_and_the_cpu() {
    let _t = turn();
    let Some(_) = cuda_device("ranges_match_the_arithmetic_and_the_cpu") else { return };
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let (g, c) = (on_cuda(&dir), Loaded::load(&dir).unwrap());
    // EXACT: F32 FMAs throughout, which the f64 encoder's bound needs.
    let (gs, cs) = (ranged_session(g.m, TURBO_PRECISION_EXACT, 16), ranged_session(c.m, TURBO_PRECISION_MODEL, 16));
    let t = awkward_rows(&dir);
    let seq = t.seq;
    let types = t.types.clone().unwrap();
    let live = |r: u32| -> Vec<u32> { (0..seq).filter(|&p| t.mask[(r * seq + p) as usize] == 1).collect() };
    let (l0, l1, l2) = (live(0), live(1), live(2));
    // Row 1 is padded on the left, row 2 has its second column masked.
    assert!(l1[0] > 0 && l2[1] > 1, "awkward rows: {l1:?} {l2:?}");
    let many = vec![
        range(0, l0[1], l0[1] + 1),
        range(2, 0, l2[1] + 1),
        range(2, 1, seq),
        range(1, 0, seq),
        range(1, seq - 2, seq),
        range(0, 0, 3),
        range(0, 2, *l0.last().unwrap() + 1),
        range(0, l0[1], l0[1] + 1),
        range(1, l1[0], l1[0] + 1),
    ];
    let few = vec![range(1, seq - 1, seq)];
    let (mut worst, mut lowest) = (0f64, 1f64);
    for ranges in [&many, &few] {
        for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
            let o = opts(|o| o.normalize = normalize);
            gs.write_ranges(&t.batch(), ranges, Some(&o)).unwrap();
            let run = gs.run().unwrap();
            let info = run.info();
            assert_eq!(info.batch as usize, ranges.len(), "one vector per range");
            assert_eq!(info.bytes, ranges.len() as u64 * info.dim as u64 * 4);
            assert_eq!(info.h2d_bytes, 3 * t.ids.len() as u64 * 4 + ranges.len() as u64 * 12, "rows and ranges");
            let got = run.rows();
            drop(run);
            cs.write_ranges(&t.batch(), ranges, Some(&o)).unwrap();
            let cpu = cs.run().unwrap().rows();
            assert_eq!(got.len(), cpu.len());
            for (i, (v, g)) in got.iter().zip(ranges.iter()).enumerate() {
                let at = (g.row * seq) as usize..((g.row + 1) * seq) as usize;
                let want = plain.embed_range(
                    &t.ids[at.clone()],
                    &t.mask[at.clone()],
                    &types[at],
                    g.start as usize,
                    g.end as usize,
                    v.len(),
                    normalize == TURBO_NORMALIZE_L2,
                );
                for (a, w) in v.iter().zip(&want) {
                    let d = (*a as f64 - w).abs();
                    worst = worst.max(d);
                    assert!(d < 1e-5 * (1.0 + w.abs()), "normalize {normalize} ranges[{i}] {g:?}: {a} vs {w}");
                }
                let c = cosine(v, &cpu[i]);
                lowest = lowest.min(c);
                assert!(c >= 0.9999, "normalize {normalize} ranges[{i}] {g:?}: cosine {c} with the cpu");
            }
        }
    }
    println!(
        "ranges: largest difference from the f64 encoder {worst:.3e}; 1 - lowest cosine with the cpu {:.3e}",
        1.0 - lowest
    );

    // A plain write replaces the range write: the run pools rows.
    gs.write_tokens(&t.batch(), None).unwrap();
    let rows = gs.run().unwrap();
    assert_eq!(rows.info().batch, t.batch, "a vector per row again");
}

/// Rows in page-locked memory go to the device from where they are, and
/// the ranges, ordinary memory, from the session's staging after them:
/// the same vectors as rows the staging carried.
#[test]
fn ranges_beside_rows_in_pinned_memory_give_the_same_vectors() {
    let _t = turn();
    let Some(_) = cuda_device("ranges_beside_rows_in_pinned_memory_give_the_same_vectors") else { return };
    let l = on_cuda(&tiny_bundle());
    let s = ranged_session(l.m, TURBO_PRECISION_EXACT, 8);
    let t = awkward_rows(&tiny_bundle());
    let ranges = [range(2, 0, t.seq), range(0, 1, 3), range(1, t.seq - 2, t.seq), range(0, 0, 1), range(2, 2, 4)];
    s.write_ranges(&t.batch(), &ranges, None).unwrap();
    let want = s.run().unwrap().rows();

    let ctx = Ctx(l.ctx, false);
    let n = t.ids.len() as u64;
    let bufs: Vec<Buf> = (0..3).map(|_| ctx.alloc(TURBO_PLACE_PINNED, n).unwrap()).collect();
    let at: Vec<*mut i32> = bufs.iter().map(|b| b.host().unwrap() as *mut i32).collect();
    let types = t.types.as_ref().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(t.ids.as_ptr(), at[0], t.ids.len());
        std::ptr::copy_nonoverlapping(t.mask.as_ptr(), at[1], t.mask.len());
        std::ptr::copy_nonoverlapping(types.as_ptr(), at[2], types.len());
    }
    let mut b = t.batch();
    b.ids = at[0];
    b.mask = at[1];
    b.types = at[2];
    s.write_ranges(&b, &ranges, None).unwrap();
    let r = s.run().unwrap();
    assert_eq!(bits(&r.rows().concat()), bits(&want.concat()));
    assert_eq!(r.info().h2d_bytes, 3 * n * 4 + ranges.len() as u64 * 12);
}

/// The GPU takes no thread count: cpu_threads other than 0 is refused by
/// field 6, and 0 makes the session.
#[test]
fn cpu_threads_is_refused_by_field_6() {
    let _t = turn();
    let Some(_) = cuda_device("cpu_threads_is_refused_by_field_6") else { return };
    let l = on_cuda(&tiny_bundle());
    for threads in [1, 4] {
        let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
        d.cpu_threads = threads;
        let e = Session::create(l.m, Some(&d)).err().unwrap();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6), "{e:?}");
        assert!(e.message.contains("not a CPU"), "{e:?}");
    }
    Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_MODEL))).unwrap();
}

// ---- Packed rows --------------------------------------------------------------------

/// A full batch of rows of very different lengths, a row of CLS alone
/// among them, at every precision and with every pooling, with and
/// without normalization: the vectors are the CPU's within the bound of
/// the session's compute dtype (F32: cosine 0.9999 and 1e-4; F16: cosine
/// 0.999), each row alone gives its vector in the batch, and the same
/// rows run again give the same bits.
#[test]
fn packed_rows_match_the_cpu_at_every_precision_and_pooling() {
    let _t = turn();
    let Some(_) = cuda_device("packed_rows_match_the_cpu_at_every_precision_and_pooling") else { return };
    let dir = tiny_bundle();
    let (g, c) = (on_cuda(&dir), Loaded::load(&dir).unwrap());
    let cs = Session::create(c.m, None).unwrap();
    let mi = g.info();
    let t = ragged_rows(&dir, mi.max_batch as usize, mi.max_seq as usize);
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
        let gs = strict(|| Session::create(g.m, Some(&session_desc(0, 0, precision)))).unwrap();
        let dtype = gs.info().compute_dtype;
        assert_eq!(dtype, if precision == TURBO_PRECISION_FASTEST { TURBO_DTYPE_F16 } else { TURBO_DTYPE_F32 });
        let tol = record::tolerance(dtype).unwrap();
        for pooling in [TURBO_POOLING_MEAN, TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
            for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
                let o = opts(|o| {
                    o.pooling = pooling;
                    o.normalize = normalize;
                });
                let what = format!("precision {precision} pooling {pooling} normalize {normalize}");
                gs.write_tokens(&t.batch(), Some(&o)).unwrap();
                let got = gs.run().unwrap().rows();
                cs.write_tokens(&t.batch(), Some(&o)).unwrap();
                let want = cs.run().unwrap().rows();
                let (cos, abs) = within(&what, &got, &want, tol);
                println!("{what}: 1 - lowest cosine with the cpu {cos:.3e}, max abs diff {abs:.3e}");

                gs.write_tokens(&t.batch(), Some(&o)).unwrap();
                assert_eq!(gs.run().unwrap().rows(), got, "{what}: the same rows give the same bits");

                // The shortest rows, the longest and one in between, alone.
                for r in [0, 1, 2, 3, 4] {
                    gs.write_tokens(&one_row(&t, r).batch(), Some(&o)).unwrap();
                    let alone = gs.run().unwrap().rows();
                    within(&format!("{what}: row {r} alone"), &alone, &got[r..r + 1], tol);
                }
            }
        }
    }
}

/// More rows than the packing's scan gives one thread each (600, where
/// its block has 256 threads), of lengths from 1 to the session's longest,
/// at MODEL and FASTEST: the CPU's vectors within the bound.
#[test]
fn many_rows_pack_as_the_cpu_packs_them() {
    let _t = turn();
    let Some(_) = cuda_device("many_rows_pack_as_the_cpu_packs_them") else { return };
    let mut m = model_manifest();
    m["embed"]["max_batch"] = json!(600);
    let mut f = Fixture::new("cuda-many-rows", m);
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let (g, c) = (f.load_on(cuda).unwrap(), f.load().unwrap());
    let t = ragged_rows(&f.dir, 600, 40);
    let cs = Session::create(c.m, Some(&session_desc(600, 40, 0))).unwrap();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let gs = strict(|| Session::create(g.m, Some(&session_desc(600, 40, precision)))).unwrap();
        let tol = record::tolerance(gs.info().compute_dtype).unwrap();
        gs.write_tokens(&t.batch(), None).unwrap();
        let got = gs.run().unwrap().rows();
        let (cos, abs) = within(&format!("precision {precision}"), &got, &want, tol);
        println!("600 rows at precision {precision}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
    }
}

/// The encoder at FASTEST: F16 in the cell and the session, and its
/// vectors within the F16 bound of the CPU's
/// (the_reference_cases_match_upstream holds them to the upstream
/// vectors).
#[test]
fn fastest_computes_in_f16_within_its_bound() {
    let _t = turn();
    let Some(_) = cuda_device("fastest_computes_in_f16_within_its_bound") else { return };
    let dir = tiny_bundle();
    let (g, c) = (on_cuda(&dir), Loaded::load(&dir).unwrap());
    let fast = Session::create(g.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
    assert_eq!(fast.info().compute_dtype, TURBO_DTYPE_F16);
    assert!(unsafe { model_converted_weights(g.m) }.is_some(), "the F16 copy of an F32 model's GEMM weights");
    let exact = Session::create(g.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
    assert_eq!(exact.info().compute_dtype, TURBO_DTYPE_F32);
    let cpu = Session::create(c.m, None).unwrap().embed(&TEXTS, None).unwrap();
    let f16 = record::tolerance(TURBO_DTYPE_F16).unwrap();
    let f32 = record::tolerance(TURBO_DTYPE_F32).unwrap();
    let (cos, _) = within("FASTEST", &fast.embed(&TEXTS, None).unwrap(), &cpu, f16);
    within("EXACT", &exact.embed(&TEXTS, None).unwrap(), &cpu, f32);
    println!("FASTEST: 1 - lowest cosine with the cpu {cos:.3e}");
    let r = {
        fast.write_text(&TEXTS, None).unwrap();
        fast.run().unwrap()
    };
    assert_eq!((r.info().compute_dtype, r.info().dtype), (TURBO_DTYPE_F16, TURBO_DTYPE_F32), "F32 vectors out");
}

/// MODEL on an F32 model takes EXACT's FMA kernels by default, so it
/// gives EXACT's bits; TURBO_CUDA_TF32=1 computes its GEMMs as TF32 on
/// the tensor cores (sm_80 and newer), which still reports F32 and
/// reaches F32's cosine (not its absolute bound) against the CPU, on the tiny bundle and on rows
/// of very different lengths with masked tokens inside, every tile.
#[test]
fn model_is_exact_unless_tf32_is_asked_for() {
    let _t = turn();
    let Some(_) = cuda_device("model_is_exact_unless_tf32_is_asked_for") else { return };
    use turbo::cuda::Tile;
    let dir = tiny_bundle();
    let (g, c) = (on_cuda(&dir), Loaded::load(&dir).unwrap());
    let cpu = Session::create(c.m, None).unwrap().embed(&TEXTS, None).unwrap();
    let exact = Session::create(g.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
    let model = strict(|| Session::create(g.m, Some(&session_desc(0, 0, TURBO_PRECISION_MODEL)))).unwrap();
    assert_eq!(model.embed(&TEXTS, None).unwrap(), exact.embed(&TEXTS, None).unwrap(), "MODEL is EXACT");
    // TF32 rounds each operand to 10 bits of mantissa: F32's cosine, with
    // no bound on the largest absolute difference.
    let tol = record::Tolerance { max_abs_diff: None, ..record::tolerance(TURBO_DTYPE_F32).unwrap() };
    let t = awkward_rows(&dir);
    let want = {
        let cs = Session::create(c.m, None).unwrap();
        cs.write_tokens(&t.batch(), None).unwrap();
        cs.run().unwrap().rows()
    };
    for tile in [Tile::Default, Tile::T64x64, Tile::T128x64, Tile::T128x128] {
        turbo::cuda::use_tf32(Some(true));
        turbo::cuda::use_tile(Some(tile));
        let tf32 = Session::create(g.m, Some(&session_desc(0, 0, TURBO_PRECISION_MODEL)));
        turbo::cuda::use_tile(None);
        turbo::cuda::use_tf32(None);
        let tf32 = tf32.unwrap();
        assert_eq!(tf32.info().compute_dtype, TURBO_DTYPE_F32);
        let (cos, abs) = within(&format!("TF32, tile {tile:?}"), &tf32.embed(&TEXTS, None).unwrap(), &cpu, tol);
        println!("TF32, tile {tile:?}: 1 - lowest cosine with the cpu {cos:.3e}, max abs diff {abs:.3e}");
        tf32.write_tokens(&t.batch(), None).unwrap();
        let got = tf32.run().unwrap().rows();
        within(&format!("TF32, tile {tile:?}, awkward rows"), &got, &want, tol);
        tf32.write_tokens(&t.batch(), None).unwrap();
        assert_eq!(tf32.run().unwrap().rows(), got, "TF32, tile {tile:?}: the same bits again");
    }
}
