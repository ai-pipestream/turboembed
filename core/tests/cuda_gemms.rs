//! The CUDA backend's GEMMs: against cuBLAS, F16 sums over the whole of
//! k, and every tile and stream-K mode giving the same vectors.
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
use cuda_support::rows::bert_weights;
use cuda_support::*;
use serde_json::json;
use turbo::*;

/// The error F16 sums add to a GEMM, against cuBLAS's F32 sums of the same
/// F16 operands (uniform in [-1, 1]), at MiniLM's shapes for the
/// benchmark's 1353 tokens and at 8193: F32 accumulators, F16 ones over
/// each 64 terms of k added in F32, and F16 ones over the whole of a
/// block's k (`f16k`, `f16k3`, `f16k256`), with the work shared among as many blocks
/// as the device holds, and among 7 and 1 (so a block's segment runs to
/// the whole of k). Each line prints the largest difference relative to
/// the largest value; the whole-k sums stay within 1e-2 of it, the
/// others within their bounds in the_gemms_match_cublas.
#[test]
fn f16_sums_over_the_whole_of_k_stay_within_their_bound() {
    let _t = turn();
    let Some(dev) = cuda_device("f16_sums_over_the_whole_of_k_stay_within_their_bound") else { return };
    if !cublas_or_skip("f16_sums_over_the_whole_of_k_stay_within_their_bound", "the test") {
        return;
    }
    let ordinal = Rt::new().info(dev).ordinal;
    use turbo::cuda::Epilogue::*;
    use turbo::cuda::Tile;
    for (m, n, k, epilogue, heads) in [
        (1353, 1152, 384, Qkv, 12),
        (1353, 384, 384, Plain, 1),
        (1353, 1536, 384, Gelu, 1),
        (1353, 384, 1536, Plain, 1),
        (8193, 384, 1536, Plain, 1),
    ] {
        let counts: &[i32] = if m > 4096 { &[0, 7] } else { &[0, 7, 1] };
        for &blocks in counts {
            let mut worst = [0.0f64; 3];
            for (sums, tiles) in [
                (0, &[Tile::SwizzledEightWarps][..]),
                (1, &[Tile::SwizzledEightWarpsF16Accumulate][..]),
                (2, &[Tile::F16WholeK, Tile::F16WholeK3, Tile::F16WholeK256, Tile::CtK][..]),
            ] {
                for &tile in tiles {
                    let (diff, reference) =
                        turbo::cuda::gemm_check(ordinal, m, n, k, epilogue, true, true, tile, blocks, heads).unwrap();
                    let relative = diff / reference.max(1.0);
                    worst[sums] = worst[sums].max(relative);
                    let what = format!("{epilogue:?} [{m}, {k}] x [{n}, {k}], tile {tile:?}, {blocks} blocks");
                    println!("{what}: largest difference {diff:.3e} of values up to {reference:.3}, {relative:.3e}");
                    let bound = match sums {
                        0 if matches!(epilogue, Plain) => 1e-5,
                        0 => 2e-3,
                        _ => 1e-2,
                    };
                    assert!(relative <= bound, "{what}: {relative:e} over {bound:e}");
                }
            }
            println!(
                "{epilogue:?} [{m}, {k}] x [{n}, {k}], {blocks} blocks: relative error with F32 sums {:.3e}, F16 over \
                 64 {:.3e}, F16 over the whole of k {:.3e}",
                worst[0], worst[1], worst[2]
            );
        }
    }
}

/// TURBO_CUDA_TILE picks the GEMMs' tile for measuring: every tile gives
/// the vectors of the default within the bound of the precision, at MODEL
/// and FASTEST, on MiniLM-like heads of 32 and ragged rows. (Not the same
/// bits: a tile shares the k steps among the blocks at other points, so
/// the partial sums differ.) Each tile repeats its own bits.
#[test]
fn every_gemm_tile_gives_the_same_vectors() {
    let _t = turn();
    let Some(_) = cuda_device("every_gemm_tile_gives_the_same_vectors") else { return };
    use turbo::cuda::Tile;
    let mut m = model_manifest();
    m["architecture"]["hidden"] = json!(64);
    m["architecture"]["heads"] = json!(2);
    m["architecture"]["intermediate"] = json!(256);
    m["embed"]["dim"] = json!(64);
    m["embed"]["max_seq"] = json!(160);
    m["embed"]["max_batch"] = json!(40);
    let mut f = Fixture::new("cuda-tiles", m);
    f.weights("weights/model.safetensors", &bert_weights(64, 256));
    let g = f.load_on(cuda).unwrap();
    let t = ragged_rows(&f.dir, 40, 160);
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let own = strict(|| Session::create(g.m, Some(&session_desc(40, 160, precision)))).unwrap();
        let tol = record::tolerance(own.info().compute_dtype).unwrap();
        own.write_tokens(&t.batch(), None).unwrap();
        let want = own.run().unwrap().rows();
        for tile in [
            Tile::T64x64,
            Tile::T128x64,
            Tile::T128x128,
            Tile::T128x128Thread16x8,
            Tile::T128x128Warps4,
            Tile::T256x128,
            Tile::EightWarps,
            Tile::Swizzled,
            Tile::SwizzledEightWarps,
            Tile::Swizzled256x128,
            Tile::F16WholeK,
            Tile::F16WholeK3,
            Tile::F16WholeKRows,
            Tile::F16WholeK256,
            Tile::Ct,
            Tile::CtK,
            Tile::Cs,
            Tile::Kt,
        ] {
            // The whole-k tiles sum in F16, the F16 accumulators' experiment.
            let whole_k = matches!(
                tile,
                Tile::F16WholeK | Tile::F16WholeK3 | Tile::F16WholeKRows | Tile::F16WholeK256 | Tile::CtK
            );
            turbo::cuda::use_f16_accumulate(whole_k.then_some(true));
            turbo::cuda::use_tile(Some(tile));
            let s = strict(|| Session::create(g.m, Some(&session_desc(40, 160, precision))));
            turbo::cuda::use_tile(None);
            turbo::cuda::use_f16_accumulate(None);
            let s = s.unwrap();
            s.write_tokens(&t.batch(), None).unwrap();
            let got = s.run().unwrap().rows();
            s.write_tokens(&t.batch(), None).unwrap();
            assert_eq!(s.run().unwrap().rows(), got, "precision {precision}, tile {tile:?}: the same bits again");
            let (cos, abs) = within(&format!("precision {precision}, tile {tile:?}"), &got, &want, tol);
            println!("precision {precision}, tile {tile:?}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
        }
    }
}

/// TURBO_CUDA_SK_STEPS shares the GEMMs' k steps out among the blocks at
/// other points, or gives each block whole tiles (`tiles`): every choice
/// gives the vectors of the default within the bound of the precision,
/// at every precision, on ragged rows, and repeats its own bits. Forcing
/// the kernels' own fewest steps (`Default`, 0) forces `Steps(4)`,
/// SK_MIN_STEPS, bit for bit; it is not the default's choices, which hold
/// whole tiles for some GEMMs at FASTEST from the le4k bin, and on a
/// device with few SMs the two differ (on one with many, stream-K at
/// these shapes never splits a tile, and they happen to agree).
#[test]
fn every_stream_k_mode_gives_the_same_vectors() {
    let _t = turn();
    let Some(_) = cuda_device("every_stream_k_mode_gives_the_same_vectors") else { return };
    use turbo::cuda::StreamK;
    let mut m = model_manifest();
    m["architecture"]["hidden"] = json!(64);
    m["architecture"]["heads"] = json!(2);
    m["architecture"]["intermediate"] = json!(256);
    m["embed"]["dim"] = json!(64);
    m["embed"]["max_seq"] = json!(160);
    m["embed"]["max_batch"] = json!(40);
    let mut f = Fixture::new("cuda-stream-k", m);
    f.weights("weights/model.safetensors", &bert_weights(64, 256));
    let g = f.load_on(cuda).unwrap();
    let t = ragged_rows(&f.dir, 40, 160);
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let own = strict(|| Session::create(g.m, Some(&session_desc(40, 160, precision)))).unwrap();
        let tol = record::tolerance(own.info().compute_dtype).unwrap();
        own.write_tokens(&t.batch(), None).unwrap();
        let want = own.run().unwrap().rows();
        let mut own = None;
        for sk in [
            StreamK::Default,
            StreamK::Steps(4),
            StreamK::Steps(1),
            StreamK::Steps(2),
            StreamK::Steps(8),
            StreamK::Steps(16),
            StreamK::Tiles,
        ] {
            turbo::cuda::use_stream_k(Some(sk));
            let s = strict(|| Session::create(g.m, Some(&session_desc(40, 160, precision))));
            turbo::cuda::use_stream_k(None);
            let s = s.unwrap();
            s.write_tokens(&t.batch(), None).unwrap();
            let got = s.run().unwrap().rows();
            s.write_tokens(&t.batch(), None).unwrap();
            assert_eq!(s.run().unwrap().rows(), got, "precision {precision}, {sk:?}: the same bits again");
            match sk {
                StreamK::Default => own = Some(got.clone()),
                StreamK::Steps(4) => assert_eq!(
                    own.as_ref(),
                    Some(&got),
                    "precision {precision}: the kernels' own steps, forced, are Steps(4), SK_MIN_STEPS"
                ),
                _ => {}
            }
            let (cos, abs) = within(&format!("precision {precision}, {sk:?}"), &got, &want, tol);
            println!("precision {precision}, {sk:?}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
        }
    }
}
