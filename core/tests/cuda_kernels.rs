//! The CUDA backend's limits and its own kernels: launch limits, chunked
//! keys, the packing, layer norm, and the FASTEST switches.
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
use turbo::status::UNSUPPORTED_OPTION;
use turbo::*;

// ---- Limits and the largest shape ----------------------------------------------------

/// A session over 65535 rows is refused by field 1: a grid dimension of
/// the kernels that run one block per row.
#[test]
fn more_rows_than_a_launch_takes_are_refused_by_field() {
    let _t = turn();
    let Some(_) = cuda_device("more_rows_than_a_launch_takes_are_refused_by_field") else { return };
    let mut m = model_manifest();
    m["embed"]["max_batch"] = json!(70000);
    let mut f = Fixture::new("cuda-rows", m);
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let l = f.load_on(cuda).unwrap();
    let e = Session::create(l.m, Some(&session_desc(65536, 8, 0))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 1), "{e:?}");
    assert!(e.message.contains("at most 65535 rows"), "{e:?}");
    Session::create(l.m, Some(&session_desc(65535, 8, 0))).unwrap();
}

/// Attention streams a row's keys through shared memory a chunk at a
/// time, so no length is refused for it: a session of 65536 positions is
/// made, and a row of 3000 tokens, 24 chunks of keys, gives the CPU's
/// vector at MODEL and FASTEST.
#[test]
fn rows_longer_than_a_chunk_of_keys_run_in_chunks() {
    let _t = turn();
    let Some(_) = cuda_device("rows_longer_than_a_chunk_of_keys_run_in_chunks") else { return };
    let positions = 65536u64;
    let mut m = model_manifest();
    m["architecture"]["max_positions"] = json!(positions);
    m["embed"]["max_seq"] = json!(positions);
    // A case longer than max_seq, which a manifest needs: 600 paragraphs of
    // about 120 tokens.
    m["reference"]["cases"][8]["text"] = json!(vec![PARAGRAPH; 600].join(" "));
    let mut f = Fixture::new("cuda-positions", m);
    let weights: Vec<Tensor> = tiny_weights(0)
        .into_iter()
        .map(|t| {
            if t.name != "embeddings.position_embeddings.weight" {
                return t;
            }
            let n = positions as usize * 8;
            let data = (0..n).flat_map(|i| (((i * 37 % 101) as f32 - 50.0) / 500.0).to_le_bytes()).collect();
            Tensor { shape: vec![positions, 8], data, ..t }
        })
        .collect();
    f.weights("weights/model.safetensors", &weights);
    let (g, c) = (f.load_on(cuda).unwrap(), f.load().unwrap());
    Session::create(g.m, Some(&session_desc(1, positions as u32, 0))).unwrap();
    let t = ragged_rows(&f.dir, 1, 3000);
    let cs = Session::create(c.m, Some(&session_desc(1, 3000, 0))).unwrap();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let gs = strict(|| Session::create(g.m, Some(&session_desc(1, 3000, precision)))).unwrap();
        let tol = record::tolerance(gs.info().compute_dtype).unwrap();
        gs.write_tokens(&t.batch(), None).unwrap();
        let (cos, abs) = within(&format!("precision {precision}"), &gs.run().unwrap().rows(), &want, tol);
        println!("a row of 3000 at precision {precision}: 1 - cosine {cos:.3e}, max abs diff {abs:.3e}");
    }
}

// ---- The backend's own kernels -------------------------------------------------------

/// The packing's edges on one session of 32 rows of 512: one row of 512
/// live tokens, 32 rows of 3, and the two together, at every precision.
/// Each is the CPU's within the bound; rows of 3 give the same bits
/// whether the rows are 3 wide or 512 (the run's width is not part of
/// the graph); and each shape run again after the others gives the same
/// bits, the one graph serving them all.
#[test]
fn the_packing_takes_one_long_row_and_many_short_ones() {
    let _t = turn();
    let Some(_) = cuda_device("the_packing_takes_one_long_row_and_many_short_ones") else { return };
    let mut m = model_manifest();
    m["embed"]["max_seq"] = json!(512);
    m["embed"]["max_batch"] = json!(32);
    m["reference"]["cases"][8]["text"] = json!([PARAGRAPH; 8].join(" "));
    let mut f = Fixture::new("cuda-packing-edges", m);
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let (g, c) = (f.load_on(cuda).unwrap(), f.load().unwrap());
    let cs = Session::create(c.m, Some(&session_desc(32, 512, 0))).unwrap();
    let long = rows_of(&f.dir, &[512], 512, true);
    let short = rows_of(&f.dir, &[3; 32], 3, false);
    let short_wide = rows_of(&f.dir, &[3; 32], 512, false);
    let mut lens = vec![3; 32];
    lens[7] = 512;
    let mixed = rows_of(&f.dir, &lens, 512, true);
    let cpu = |t: &Tokens| {
        cs.write_tokens(&t.batch(), None).unwrap();
        cs.run().unwrap().rows()
    };
    let want = [cpu(&long), cpu(&short), cpu(&mixed)];
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
        let gs = strict(|| Session::create(g.m, Some(&session_desc(32, 512, precision)))).unwrap();
        let tol = record::tolerance(gs.info().compute_dtype).unwrap();
        let run = |t: &Tokens| {
            gs.write_tokens(&t.batch(), None).unwrap();
            gs.run().unwrap().rows()
        };
        let got = [run(&long), run(&short), run(&mixed)];
        for (what, (a, b)) in ["one row of 512", "32 rows of 3", "both"].iter().zip(got.iter().zip(&want)) {
            let (cos, abs) = within(&format!("precision {precision}: {what}"), a, b, tol);
            println!("precision {precision}, {what}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
        }
        assert_eq!(run(&short_wide), got[1], "precision {precision}: rows of 3, 3 or 512 wide");
        assert_eq!(run(&long), got[0], "precision {precision}: the long row again");
        assert_eq!(run(&mixed), got[2], "precision {precision}: both again");
        assert_eq!(run(&short), got[1], "precision {precision}: the short rows again");
    }
}

/// The whole-row tiles, TURBO_CUDA_TILE=swrow and f16krow: the attention
/// output and second feed-forward GEMMs add the residual and run the
/// LayerNorm in their epilogue, summing each row in another order than
/// the separate kernel, so FASTEST's vectors are the default's within
/// FASTEST's bound (cosine 0.999), on hidden widths of 64 and 384 and
/// rows of very different lengths; a row alone gives its vector in the
/// batch within the bound, and the same rows the same bits again. A
/// model wider than the tile runs the eight-warp shapes instead (f16k3's
/// for f16krow).
#[test]
fn whole_row_layer_norm_matches_the_separate_kernel() {
    let _t = turn();
    let Some(_) = cuda_device("whole_row_layer_norm_matches_the_separate_kernel") else { return };
    use turbo::cuda::Tile;
    for hidden in [64, 384, 512] {
        let mut m = model_manifest();
        m["architecture"]["hidden"] = json!(hidden);
        m["architecture"]["heads"] = json!(hidden / 32);
        m["architecture"]["intermediate"] = json!(4 * hidden);
        m["embed"]["dim"] = json!(hidden);
        m["embed"]["max_seq"] = json!(300);
        m["embed"]["max_batch"] = json!(6);
        m["reference"]["cases"][8]["text"] = json!([PARAGRAPH; 8].join(" "));
        let mut f = Fixture::new(&format!("cuda-row-ln-{hidden}"), m);
        f.weights("weights/model.safetensors", &bert_weights(hidden, 4 * hidden));
        let g = f.load_on(cuda).unwrap();
        let t = rows_of(&f.dir, &[300, 129, 64, 63, 17, 1], 300, true);
        let base = Session::create(g.m, Some(&session_desc(6, 300, TURBO_PRECISION_FASTEST))).unwrap();
        base.write_tokens(&t.batch(), None).unwrap();
        let want = base.run().unwrap().rows();
        let tol = record::tolerance(base.info().compute_dtype).unwrap();
        for tile in [Tile::SwizzledRows, Tile::F16WholeKRows] {
            turbo::cuda::use_tile(Some(tile));
            let s = Session::create(g.m, Some(&session_desc(6, 300, TURBO_PRECISION_FASTEST)));
            turbo::cuda::use_tile(None);
            let s = s.unwrap();
            s.write_tokens(&t.batch(), None).unwrap();
            let got = s.run().unwrap().rows();
            let what = format!("hidden {hidden}, tile {tile:?}");
            let (cos, abs) = within(&what, &got, &want, tol);
            println!("{what}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
            s.write_tokens(&t.batch(), None).unwrap();
            assert_eq!(s.run().unwrap().rows(), got, "{what}: the same bits again");
            for r in [0, 3, 5] {
                s.write_tokens(&one_row(&t, r).batch(), None).unwrap();
                let alone = s.run().unwrap().rows();
                within(&format!("{what}: row {r} alone"), &alone, &got[r..r + 1], tol);
            }
        }
    }
}

/// FASTEST with F16 accumulators over each 64 terms of k
/// (TURBO_CUDA_F16_ACCUMULATE=1), and over the whole of k (the tiles
/// `f16k`, `f16k3`, `f16krow` and `f16k256`), on a model of MiniLM's widths, the LayerNorm
/// separate and in the GEMMs: the CPU's vectors within FASTEST's bound
/// (cosine 0.999), and the same bits when run again.
#[test]
fn f16_accumulators_hold_fastest_s_bound() {
    let _t = turn();
    let Some(_) = cuda_device("f16_accumulators_hold_fastest_s_bound") else { return };
    let mut m = model_manifest();
    m["architecture"]["hidden"] = json!(384);
    m["architecture"]["heads"] = json!(12);
    m["architecture"]["intermediate"] = json!(1536);
    m["embed"]["dim"] = json!(384);
    m["embed"]["max_seq"] = json!(300);
    m["embed"]["max_batch"] = json!(6);
    m["reference"]["cases"][8]["text"] = json!([PARAGRAPH; 8].join(" "));
    let mut f = Fixture::new("cuda-f16-accumulate", m);
    f.weights("weights/model.safetensors", &bert_weights(384, 1536));
    let (g, c) = (f.load_on(cuda).unwrap(), f.load().unwrap());
    let t = rows_of(&f.dir, &[300, 129, 64, 63, 17, 1], 300, true);
    let cs = Session::create(c.m, Some(&session_desc(6, 300, 0))).unwrap();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    use turbo::cuda::Tile;
    for (tile, separate) in [
        None,
        Some(Tile::F16WholeK),
        Some(Tile::F16WholeK3),
        Some(Tile::F16WholeKRows),
        Some(Tile::F16WholeK256),
        Some(Tile::CtK),
    ]
    .into_iter()
    .flat_map(|tile| [(tile, true), (tile, false)])
    {
        // The whole-k tiles are the F16 accumulators' experiment too.
        turbo::cuda::use_f16_accumulate(Some(true));
        turbo::cuda::use_tile(tile);
        turbo::cuda::use_separate_layer_norm(Some(separate));
        let gs = Session::create(g.m, Some(&session_desc(6, 300, TURBO_PRECISION_FASTEST)));
        turbo::cuda::use_separate_layer_norm(None);
        turbo::cuda::use_tile(None);
        turbo::cuda::use_f16_accumulate(None);
        let gs = gs.unwrap();
        let tol = record::tolerance(gs.info().compute_dtype).unwrap();
        gs.write_tokens(&t.batch(), None).unwrap();
        let got = gs.run().unwrap().rows();
        let sums = match tile {
            None => "F16 accumulators".to_string(),
            Some(tile) => format!("tile {tile:?}"),
        };
        let what = format!("{sums}, separate LayerNorm {separate}");
        let (cos, abs) = within(&what, &got, &want, tol);
        println!("{what}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
        gs.write_tokens(&t.batch(), None).unwrap();
        assert_eq!(gs.run().unwrap().rows(), got, "{what}: the same bits again");
    }
}

/// FASTEST's residual stream, F16 alone by default and F32 as well as
/// TURBO_CUDA_RESIDUAL=f32 keeps it, gives the CPU's vectors within
/// FASTEST's bound either way and the same bits when run again; MODEL and
/// EXACT, whose stream is F32 whatever the switch says, give the same
/// bits with it set as without.
#[test]
fn the_residual_switch_moves_only_fastest() {
    let _t = turn();
    let Some(_) = cuda_device("the_residual_switch_moves_only_fastest") else { return };
    let mut m = model_manifest();
    m["architecture"]["hidden"] = json!(64);
    m["architecture"]["heads"] = json!(2);
    m["architecture"]["intermediate"] = json!(256);
    m["embed"]["dim"] = json!(64);
    m["embed"]["max_seq"] = json!(300);
    m["embed"]["max_batch"] = json!(6);
    m["reference"]["cases"][8]["text"] = json!([PARAGRAPH; 8].join(" "));
    let mut f = Fixture::new("cuda-residual", m);
    f.weights("weights/model.safetensors", &bert_weights(64, 256));
    let (g, c) = (f.load_on(cuda).unwrap(), f.load().unwrap());
    let t = rows_of(&f.dir, &[300, 129, 64, 63, 17, 1], 300, true);
    let cs = Session::create(c.m, Some(&session_desc(6, 300, 0))).unwrap();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    let mut got = Vec::new();
    for f16 in [true, false] {
        turbo::cuda::use_residual16(Some(f16));
        let gs = Session::create(g.m, Some(&session_desc(6, 300, TURBO_PRECISION_FASTEST)));
        turbo::cuda::use_residual16(None);
        let gs = gs.unwrap();
        let tol = record::tolerance(gs.info().compute_dtype).unwrap();
        gs.write_tokens(&t.batch(), None).unwrap();
        let rows = gs.run().unwrap().rows();
        let what = format!("F16 residual stream {f16}");
        let (cos, abs) = within(&what, &rows, &want, tol);
        println!("{what}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
        gs.write_tokens(&t.batch(), None).unwrap();
        assert_eq!(gs.run().unwrap().rows(), rows, "{what}: the same bits again");
        got.push(rows);
    }
    let differ = got[0].iter().zip(&got[1]).filter(|(a, b)| a != b).count();
    println!("the F16 stream's vectors against the F32 stream's: {differ} rows differ");
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT] {
        let mut bits = Vec::new();
        for f16 in [true, false] {
            turbo::cuda::use_residual16(Some(f16));
            let s = strict(|| Session::create(g.m, Some(&session_desc(6, 300, precision))));
            turbo::cuda::use_residual16(None);
            let s = s.unwrap();
            s.write_tokens(&t.batch(), None).unwrap();
            bits.push(s.run().unwrap().rows());
        }
        assert_eq!(bits[0], bits[1], "precision {precision}: TURBO_CUDA_RESIDUAL moved the bits");
    }
}

/// FASTEST's attention output and second feed-forward GEMMs write their
/// product with its bias in F16 by default, which their LayerNorm kernel
/// reads; TURBO_CUDA_PRODUCT=f32 keeps the F32 product and the kernel's
/// bias. Either way the vectors are the CPU's within FASTEST's bound and
/// the same bits when run again, with the product from the GEMM's
/// epilogue and from cuBLAS (TURBO_CUDA_CUBLAS=out,ffn2, F32 either way);
/// MODEL and EXACT, whose products are F32, give the same bits with the
/// switch set as without.
#[test]
fn the_product_switch_moves_only_fastest() {
    let _t = turn();
    let Some(_) = cuda_device("the_product_switch_moves_only_fastest") else { return };
    let mut m = model_manifest();
    m["architecture"]["hidden"] = json!(64);
    m["architecture"]["heads"] = json!(2);
    m["architecture"]["intermediate"] = json!(256);
    m["embed"]["dim"] = json!(64);
    m["embed"]["max_seq"] = json!(300);
    m["embed"]["max_batch"] = json!(6);
    m["reference"]["cases"][8]["text"] = json!([PARAGRAPH; 8].join(" "));
    let mut f = Fixture::new("cuda-product", m);
    f.weights("weights/model.safetensors", &bert_weights(64, 256));
    let (g, c) = (f.load_on(cuda).unwrap(), f.load().unwrap());
    let t = rows_of(&f.dir, &[300, 129, 64, 63, 17, 1], 300, true);
    let cs = Session::create(c.m, Some(&session_desc(6, 300, 0))).unwrap();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    for cublas in [None, Some(2 | 8)] {
        if cublas.is_some() && !cublas_or_skip("the_product_switch_moves_only_fastest", "its cuBLAS half") {
            continue;
        }
        let mut got = Vec::new();
        for f16 in [true, false] {
            turbo::cuda::use_cublas(cublas);
            turbo::cuda::use_product16(Some(f16));
            let gs = Session::create(g.m, Some(&session_desc(6, 300, TURBO_PRECISION_FASTEST)));
            turbo::cuda::use_product16(None);
            turbo::cuda::use_cublas(None);
            let gs = gs.unwrap();
            let tol = record::tolerance(gs.info().compute_dtype).unwrap();
            gs.write_tokens(&t.batch(), None).unwrap();
            let rows = gs.run().unwrap().rows();
            let what = format!("F16 product {f16}, cuBLAS {cublas:?}");
            let (cos, abs) = within(&what, &rows, &want, tol);
            println!("{what}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
            gs.write_tokens(&t.batch(), None).unwrap();
            assert_eq!(gs.run().unwrap().rows(), rows, "{what}: the same bits again");
            got.push(rows);
        }
        let differ = got[0].iter().zip(&got[1]).filter(|(a, b)| a != b).count();
        println!("cuBLAS {cublas:?}: the F16 product's vectors against the F32 product's: {differ} rows differ");
        // cuBLAS's product is F32 whatever the switch says.
        if cublas.is_some() {
            assert_eq!(got[0], got[1], "TURBO_CUDA_PRODUCT moved cuBLAS's bits");
        }
    }
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT] {
        let mut bits = Vec::new();
        for f16 in [true, false] {
            turbo::cuda::use_product16(Some(f16));
            let s = strict(|| Session::create(g.m, Some(&session_desc(6, 300, precision))));
            turbo::cuda::use_product16(None);
            let s = s.unwrap();
            s.write_tokens(&t.batch(), None).unwrap();
            bits.push(s.run().unwrap().rows());
        }
        assert_eq!(bits[0], bits[1], "precision {precision}: TURBO_CUDA_PRODUCT moved the bits");
    }
}

/// FASTEST's GELU, the tanh form on the hardware tanh by default, erff or
/// erf from a fit as TURBO_CUDA_GELU=erf or poly picks it, gives the CPU's
/// vectors within FASTEST's bound each way, in the GEMM's epilogue and
/// after cuBLAS's product (TURBO_CUDA_CUBLAS=ffn1), and the same bits when
/// run again; the default is the tanh form's bits; MODEL and EXACT, whose
/// GELU writes F32 and takes erff whatever the switch says, give the same
/// bits with it set as without.
#[test]
fn the_gelu_switch_moves_only_fastest() {
    let _t = turn();
    let Some(_) = cuda_device("the_gelu_switch_moves_only_fastest") else { return };
    let mut m = model_manifest();
    m["architecture"]["hidden"] = json!(64);
    m["architecture"]["heads"] = json!(2);
    m["architecture"]["intermediate"] = json!(256);
    m["embed"]["dim"] = json!(64);
    m["embed"]["max_seq"] = json!(300);
    m["embed"]["max_batch"] = json!(6);
    m["reference"]["cases"][8]["text"] = json!([PARAGRAPH; 8].join(" "));
    let mut f = Fixture::new("cuda-gelu", m);
    f.weights("weights/model.safetensors", &bert_weights(64, 256));
    let (g, c) = (f.load_on(cuda).unwrap(), f.load().unwrap());
    let t = rows_of(&f.dir, &[300, 129, 64, 63, 17, 1], 300, true);
    let cs = Session::create(c.m, Some(&session_desc(6, 300, 0))).unwrap();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    use turbo::cuda::Gelu;
    const KINDS: [Gelu; 3] = [Gelu::Erf, Gelu::Poly, Gelu::Tanh];
    for cublas in [None, Some(4)] {
        if cublas.is_some() && !cublas_or_skip("the_gelu_switch_moves_only_fastest", "its cuBLAS half") {
            continue;
        }
        let mut got = Vec::new();
        for kind in KINDS {
            turbo::cuda::use_cublas(cublas);
            turbo::cuda::use_gelu(Some(kind));
            let gs = Session::create(g.m, Some(&session_desc(6, 300, TURBO_PRECISION_FASTEST)));
            turbo::cuda::use_gelu(None);
            turbo::cuda::use_cublas(None);
            let gs = gs.unwrap();
            let tol = record::tolerance(gs.info().compute_dtype).unwrap();
            gs.write_tokens(&t.batch(), None).unwrap();
            let rows = gs.run().unwrap().rows();
            let what = format!("cuBLAS {cublas:?}, GELU {kind:?}");
            let (cos, abs) = within(&what, &rows, &want, tol);
            println!("{what}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
            gs.write_tokens(&t.batch(), None).unwrap();
            assert_eq!(gs.run().unwrap().rows(), rows, "{what}: the same bits again");
            got.push(rows);
        }
        for k in 1..KINDS.len() {
            let differ = got[k].iter().zip(&got[0]).filter(|(a, b)| a != b).count();
            println!("cuBLAS {cublas:?}: {:?}'s vectors against erff's: {differ} rows differ", KINDS[k]);
        }
        // The default is the tanh form.
        turbo::cuda::use_cublas(cublas);
        let ds = Session::create(g.m, Some(&session_desc(6, 300, TURBO_PRECISION_FASTEST)));
        turbo::cuda::use_cublas(None);
        let ds = ds.unwrap();
        ds.write_tokens(&t.batch(), None).unwrap();
        assert_eq!(ds.run().unwrap().rows(), got[2], "cuBLAS {cublas:?}: the default is not the tanh form");
    }
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT] {
        let mut bits = Vec::new();
        for kind in KINDS {
            turbo::cuda::use_gelu(Some(kind));
            let s = strict(|| Session::create(g.m, Some(&session_desc(6, 300, precision))));
            turbo::cuda::use_gelu(None);
            let s = s.unwrap();
            s.write_tokens(&t.batch(), None).unwrap();
            bits.push(s.run().unwrap().rows());
        }
        assert_eq!(bits[0], bits[1], "precision {precision}: TURBO_CUDA_GELU moved the bits");
        assert_eq!(bits[0], bits[2], "precision {precision}: TURBO_CUDA_GELU=tanh moved the bits");
    }
}

/// A model of heads 32 wide, MiniLM's, so FASTEST's attention runs on the
/// tensor cores and EXACT's streams a 512-token row's keys in chunks:
/// rows of 512, 300, 129, 64, 63, 17 and 1 tokens, masked tokens inside
/// the longer ones, give the CPU's vectors within the bound of each
/// precision, and the same bits when run again.
#[test]
fn heads_of_32_match_the_cpu() {
    let _t = turn();
    let Some(_) = cuda_device("heads_of_32_match_the_cpu") else { return };
    let mut m = model_manifest();
    m["architecture"]["hidden"] = json!(64);
    m["architecture"]["heads"] = json!(2);
    m["architecture"]["intermediate"] = json!(128);
    m["embed"]["dim"] = json!(64);
    m["embed"]["max_seq"] = json!(512);
    m["embed"]["max_batch"] = json!(8);
    m["reference"]["cases"][8]["text"] = json!([PARAGRAPH; 8].join(" "));
    let mut f = Fixture::new("cuda-heads-32", m);
    f.weights("weights/model.safetensors", &bert_weights(64, 128));
    let (g, c) = (f.load_on(cuda).unwrap(), f.load().unwrap());
    let t = rows_of(&f.dir, &[512, 300, 129, 64, 63, 17, 1], 512, true);
    let cs = Session::create(c.m, Some(&session_desc(8, 512, 0))).unwrap();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let gs = strict(|| Session::create(g.m, Some(&session_desc(8, 512, precision)))).unwrap();
        let tol = record::tolerance(gs.info().compute_dtype).unwrap();
        gs.write_tokens(&t.batch(), None).unwrap();
        let got = gs.run().unwrap().rows();
        let (cos, abs) = within(&format!("precision {precision}"), &got, &want, tol);
        println!("heads of 32 at precision {precision}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
        gs.write_tokens(&t.batch(), None).unwrap();
        assert_eq!(gs.run().unwrap().rows(), got, "precision {precision}: the same bits again");
    }
    attention_matches(&g, &t, 8, 512, &want, "heads of 32");
}
