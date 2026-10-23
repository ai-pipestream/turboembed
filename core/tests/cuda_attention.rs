//! The CUDA backend's attention kernels against the CPU: wide heads,
//! 128 queries on whole chunks and holes, layer norm in the GEMMs, and
//! both FMA attentions.
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

/// Heads 64 wide: the attention of 128 queries to a block, and of 64,
/// match the CPU within FASTEST's bound on rows of 300, 129, 64, 63, 17 and 1
/// tokens with masked tokens inside.
#[test]
fn wide_attention_matches_the_cpu_at_heads_of_64() {
    let _t = turn();
    let Some(_) = cuda_device("wide_attention_matches_the_cpu_at_heads_of_64") else { return };
    let mut m = model_manifest();
    m["architecture"]["hidden"] = json!(128);
    m["architecture"]["heads"] = json!(2);
    m["architecture"]["intermediate"] = json!(256);
    m["embed"]["dim"] = json!(128);
    m["embed"]["max_seq"] = json!(300);
    m["embed"]["max_batch"] = json!(6);
    m["reference"]["cases"][8]["text"] = json!([PARAGRAPH; 8].join(" "));
    let mut f = Fixture::new("cuda-wide-attention-64", m);
    f.weights("weights/model.safetensors", &bert_weights(128, 256));
    let (g, c) = (f.load_on(cuda).unwrap(), f.load().unwrap());
    let t = rows_of(&f.dir, &[300, 129, 64, 63, 17, 1], 300, true);
    let cs = Session::create(c.m, Some(&session_desc(6, 300, 0))).unwrap();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    attention_matches(&g, &t, 6, 300, &want, "heads of 64");
}

/// The attention of 128 queries, heads 32 and 64 wide, on rows that fill
/// its chunks of keys (64 at heads of 32, 32 at heads of 64) with no
/// masked token (512, 256, 192, 128 and 64 tokens, every chunk taking
/// the path that tests and masks no score) and on rows with a masked
/// token and a last chunk part full (300, 129, 65, 63, 17 and 1): the
/// CPU's vectors within FASTEST's bound, with its default of F16 sums,
/// with TURBO_CUDA_ATTENTION=acc32's F32 sums and with
/// TURBO_CUDA_ATTENTION=exact's, which is the earlier arithmetic; the
/// same bits again; and each of the two against the default within the
/// bound. At heads of 32, TURBO_CUDA_ATTENTION=fa32's kernel of 32
/// queries to a warp gives the default's bits, so its bound too. Each
/// session reports the kernel it runs.
#[test]
fn attention_of_128_queries_matches_on_whole_chunks_and_holes() {
    let _t = turn();
    let Some(_) = cuda_device("attention_of_128_queries_matches_on_whole_chunks_and_holes") else { return };
    for hidden in [64usize, 128] {
        let mut m = model_manifest();
        m["architecture"]["hidden"] = json!(hidden);
        m["architecture"]["heads"] = json!(2);
        m["architecture"]["intermediate"] = json!(2 * hidden);
        m["embed"]["dim"] = json!(hidden);
        m["embed"]["max_seq"] = json!(512);
        m["embed"]["max_batch"] = json!(6);
        m["reference"]["cases"][8]["text"] = json!([PARAGRAPH; 8].join(" "));
        let mut f = Fixture::new(&format!("cuda-attention-128-{hidden}"), m);
        f.weights("weights/model.safetensors", &bert_weights(hidden as u64, 2 * hidden as u64));
        let (g, c) = (f.load_on(cuda).unwrap(), f.load().unwrap());
        let cs = Session::create(c.m, Some(&session_desc(6, 512, 0))).unwrap();
        for (lens, hole) in [(&[512, 256, 192, 128, 64][..], false), (&[300, 129, 65, 63, 17, 1][..], true)] {
            let t = rows_of(&f.dir, lens, 512, hole);
            cs.write_tokens(&t.batch(), None).unwrap();
            let want = cs.run().unwrap().rows();
            let mut runs = Vec::new();
            for variant in ["default", "acc32", "exact"] {
                turbo::cuda::use_acc32_attention(Some(variant == "acc32"));
                turbo::cuda::use_exact_attention(Some(variant == "exact"));
                let gs = Session::create(g.m, Some(&session_desc(6, 512, TURBO_PRECISION_FASTEST)));
                turbo::cuda::use_acc32_attention(None);
                turbo::cuda::use_exact_attention(None);
                let gs = gs.unwrap();
                let info = gs.info();
                let choices = field(&info.choices);
                let attn = match variant {
                    "acc32" => "attn=mma128-acc32,",
                    "exact" => "attn=mma128-exact,",
                    _ => "attn=mma128,",
                };
                assert!(choices.contains(attn), "{choices}");
                let tol = record::tolerance(info.compute_dtype).unwrap();
                gs.write_tokens(&t.batch(), None).unwrap();
                let got = gs.run().unwrap().rows();
                let what = format!("heads of {}, rows {lens:?}, {variant}", hidden / 2);
                let (cos, abs) = within(&what, &got, &want, tol);
                println!("{what}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
                gs.write_tokens(&t.batch(), None).unwrap();
                assert_eq!(gs.run().unwrap().rows(), got, "{what}: the same bits again");
                runs.push((got, tol));
            }
            for (i, variant) in ["acc32", "exact"].into_iter().enumerate() {
                let what = format!("heads of {}, rows {lens:?}, {variant} against the default", hidden / 2);
                let (cos, abs) = within(&what, &runs[i + 1].0, &runs[0].0, runs[0].1);
                println!("{what}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
            }
            if hidden == 64 {
                turbo::cuda::use_fa32_attention(Some(true));
                let gs = Session::create(g.m, Some(&session_desc(6, 512, TURBO_PRECISION_FASTEST)));
                turbo::cuda::use_fa32_attention(None);
                let gs = gs.unwrap();
                let choices = field(&gs.info().choices);
                assert!(choices.contains("attn=mma128-fa32,"), "{choices}");
                gs.write_tokens(&t.batch(), None).unwrap();
                let got = gs.run().unwrap().rows();
                let what = format!("heads of 32, rows {lens:?}, fa32");
                within(&what, &got, &want, runs[0].1);
                assert_eq!(got, runs[0].0, "{what}: the default's bits");
            }
        }
    }
}

/// The LayerNorm inside the attention output and second feed-forward
/// GEMMs, which TURBO_CUDA_LAYER_NORM=fused picks, gives the bits of the
/// separate kernel, the default, at every precision, with every tile
/// (rows split among blocks of 64, 128 and 256), on hidden widths of 64 and 384 and rows of
/// very different lengths with masked tokens inside, with whole tiles to
/// a block. The fused kernel holds more than the plain one, so fewer of
/// its blocks fit a device, and where that launch is what bounds the
/// blocks (the 8 SMs of a Jetson Orin Nano at these shapes; never the 66
/// of an RTX 4080 SUPER) stream-K splits a tile's k steps at other points
/// than the plain kernel's, and the F32 sums round apart (on the Orin at
/// FASTEST, tile T64x64: 2.7e-7 at hidden 64, 1.4e-5 at 384, the same
/// bits on every run). So under the default stream-K the fused vectors
/// are held to the separate kernel's within the bound of the precision
/// instead, and the distance is printed. And the pooling of
/// a thread per column, TURBO_CUDA_POOL=columns, gives the default's
/// vectors within the bound (it adds the tokens in another order), at
/// every pooling.
#[test]
fn layer_norm_in_the_gemms_gives_the_separate_bits() {
    let _t = turn();
    let Some(_) = cuda_device("layer_norm_in_the_gemms_gives_the_separate_bits") else { return };
    use turbo::cuda::{StreamK, Tile};
    for hidden in [64, 384] {
        let mut m = model_manifest();
        m["architecture"]["hidden"] = json!(hidden);
        m["architecture"]["heads"] = json!(hidden / 32);
        m["architecture"]["intermediate"] = json!(256);
        m["embed"]["dim"] = json!(hidden);
        m["embed"]["max_seq"] = json!(300);
        m["embed"]["max_batch"] = json!(6);
        m["reference"]["cases"][8]["text"] = json!([PARAGRAPH; 8].join(" "));
        let mut f = Fixture::new(&format!("cuda-fused-ln-{hidden}"), m);
        f.weights("weights/model.safetensors", &bert_weights(hidden, 256));
        let g = f.load_on(cuda).unwrap();
        let t = rows_of(&f.dir, &[300, 129, 64, 63, 17, 1], 300, true);
        for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
            let gs = Session::create(g.m, Some(&session_desc(6, 300, precision))).unwrap();
            let tol = record::tolerance(gs.info().compute_dtype).unwrap();
            for tile in [
                Tile::Default,
                Tile::T64x64,
                Tile::T128x64,
                Tile::T128x128,
                Tile::T256x128,
                Tile::EightWarps,
                Tile::EightWarpsF16Accumulate,
                Tile::Swizzled,
                Tile::SwizzledEightWarps,
                Tile::Swizzled256x128,
                Tile::SwizzledEightWarpsF16Accumulate,
            ] {
                // A run with the LayerNorm separate or fused, under a
                // stream-K forced, or the default's when none is.
                let run = |separate: bool, sk: Option<StreamK>| {
                    // The epilogues read the F32 residual stream and the
                    // F32 product: the separate kernel is held to them here.
                    turbo::cuda::use_residual16(Some(false));
                    turbo::cuda::use_product16(Some(false));
                    turbo::cuda::use_tile(Some(tile));
                    turbo::cuda::use_stream_k(sk);
                    turbo::cuda::use_separate_layer_norm(Some(separate));
                    let s = Session::create(g.m, Some(&session_desc(6, 300, precision)));
                    turbo::cuda::use_separate_layer_norm(None);
                    turbo::cuda::use_stream_k(None);
                    turbo::cuda::use_tile(None);
                    turbo::cuda::use_product16(None);
                    turbo::cuda::use_residual16(None);
                    let s = s.unwrap();
                    s.write_tokens(&t.batch(), None).unwrap();
                    s.run().unwrap().rows()
                };
                let what = format!("hidden {hidden}, precision {precision}, tile {tile:?}");
                // Whole tiles to a block: no split, so the two kernels sum
                // the same FMA chains, and the epilogues alone are compared.
                assert_eq!(run(true, Some(StreamK::Tiles)), run(false, Some(StreamK::Tiles)), "{what}: other bits");
                // The default stream-K: the fused kernel's launch may
                // split k at other points, so within the bound.
                let (cos, abs) = within(&format!("{what}, stream-K"), &run(false, None), &run(true, None), tol);
                println!("{what}: fused under stream-K, 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
            }
            turbo::cuda::use_column_pool(Some(true));
            let cols = Session::create(g.m, Some(&session_desc(6, 300, precision)));
            turbo::cuda::use_column_pool(None);
            let cols = cols.unwrap();
            for pooling in [TURBO_POOLING_MEAN, TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
                for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
                    let o = opts(|o| {
                        o.pooling = pooling;
                        o.normalize = normalize;
                    });
                    gs.write_tokens(&t.batch(), Some(&o)).unwrap();
                    let got = gs.run().unwrap().rows();
                    cols.write_tokens(&t.batch(), Some(&o)).unwrap();
                    let want = cols.run().unwrap().rows();
                    let what = format!("hidden {hidden}, precision {precision}, pooling {pooling} {normalize}");
                    within(&what, &got, &want, tol);
                }
            }
        }
    }
}

/// Both FMA attentions, the default register-tiled one and the key-split
/// one TURBO_CUDA_ATTENTION=split picks, on heads 16, 24, 32 and 64
/// wide: rows of 300 tokens (five chunks of keys for the tiled kernel,
/// three for the split one), 129, 64, 63, 17 and 1, masked tokens inside
/// the longer ones, give the CPU's vectors within the F32 bound, and the
/// same bits when run again.
#[test]
fn both_fma_attentions_match_the_cpu() {
    let _t = turn();
    let Some(_) = cuda_device("both_fma_attentions_match_the_cpu") else { return };
    for (hidden, heads) in [(64, 4), (72, 3), (64, 2), (128, 2)] {
        let mut m = model_manifest();
        m["architecture"]["hidden"] = json!(hidden);
        m["architecture"]["heads"] = json!(heads);
        m["architecture"]["intermediate"] = json!(128);
        m["embed"]["dim"] = json!(hidden);
        m["embed"]["max_seq"] = json!(300);
        m["embed"]["max_batch"] = json!(6);
        m["reference"]["cases"][8]["text"] = json!([PARAGRAPH; 8].join(" "));
        let mut f = Fixture::new(&format!("cuda-attention-{hidden}-{heads}"), m);
        f.weights("weights/model.safetensors", &bert_weights(hidden, 128));
        let (g, c) = (f.load_on(cuda).unwrap(), f.load().unwrap());
        let t = rows_of(&f.dir, &[300, 129, 64, 63, 17, 1], 300, true);
        let cs = Session::create(c.m, Some(&session_desc(6, 300, 0))).unwrap();
        cs.write_tokens(&t.batch(), None).unwrap();
        let want = cs.run().unwrap().rows();
        for split in [false, true] {
            turbo::cuda::use_split_attention(Some(split));
            let gs = Session::create(g.m, Some(&session_desc(6, 300, TURBO_PRECISION_EXACT)));
            turbo::cuda::use_split_attention(None);
            let gs = gs.unwrap();
            let tol = record::tolerance(gs.info().compute_dtype).unwrap();
            gs.write_tokens(&t.batch(), None).unwrap();
            let got = gs.run().unwrap().rows();
            let what = format!("heads of {}, split {split}", hidden / heads);
            let (cos, abs) = within(&what, &got, &want, tol);
            println!("{what}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
            gs.write_tokens(&t.batch(), None).unwrap();
            assert_eq!(gs.run().unwrap().rows(), got, "{what}: the same bits again");
        }
    }
}

/// The backend's GEMMs against cuBLAS on random operands, every epilogue,
/// F32 with FMAs, F32 as TF32 on the tensor cores (MODEL), F16 on the
/// tensor cores and F16 with FMAs (the path of devices before sm_80),
/// every tile, at MiniLM's shapes for the benchmark's 1353
/// tokens, at shapes no tile divides, and at a token count past 8192
/// that no tile divides either, with the work shared among as
/// many blocks as the device holds and among 1, 7 and 33 (so tiles split
/// between blocks at other points): F32 within 1e-5 of the largest value,
/// F16 outputs and TF32 products within 2e-3. Each GEMM runs twice and
/// repeats its bits.
#[test]
fn the_gemms_match_cublas() {
    let _t = turn();
    let Some(dev) = cuda_device("the_gemms_match_cublas") else { return };
    if !cublas_or_skip("the_gemms_match_cublas", "the test") {
        return;
    }
    let ordinal = Rt::new().info(dev).ordinal;
    use turbo::cuda::Epilogue::*;
    use turbo::cuda::Tile;
    for (m, n, k, epilogue, heads) in [
        (1353, 1152, 384, Qkv, 12),
        (1353, 384, 384, Plain, 1),
        (1353, 384, 384, Bias, 1),
        (1353, 1536, 384, Gelu, 1),
        (1353, 1536, 384, GeluErf, 1),
        (1353, 1536, 384, GeluTanh, 1),
        (1353, 384, 1536, Plain, 1),
        (1353, 384, 1536, Bias, 1),
        (1, 1152, 384, Qkv, 12),
        (37, 96, 32, Qkv, 4),
        (65, 24, 8, Qkv, 2),
        (100, 16, 8, Gelu, 1),
        (129, 64, 128, Gelu, 1),
        (200, 136, 72, Gelu, 1),
        (200, 136, 72, GeluErf, 1),
        (200, 136, 72, GeluTanh, 1),
        (33, 8, 16, Plain, 1),
        (200, 72, 96, Plain, 1),
        (200, 72, 96, Bias, 1),
        (300, 384, 1536, Plain, 1),
        (8193, 384, 1536, Plain, 1),
    ] {
        for (half, tensor_cores) in [(false, false), (false, true), (true, true), (true, false)] {
            // The F16 product with its bias is the tensor cores' at F16.
            if matches!(epilogue, Bias) && !(half && tensor_cores) {
                continue;
            }
            let tiles: &[Tile] = if tensor_cores {
                &[
                    Tile::Default,
                    Tile::T64x64,
                    Tile::T128x64,
                    Tile::T128x128,
                    Tile::T128x128Warps4,
                    Tile::T256x128,
                    Tile::EightWarps,
                    Tile::EightWarpsF16Accumulate,
                    Tile::Swizzled,
                    Tile::SwizzledEightWarps,
                    Tile::Swizzled256x128,
                    Tile::SwizzledEightWarpsF16Accumulate,
                    Tile::F16WholeK,
                    Tile::F16WholeK3,
                    Tile::F16WholeK256,
                    Tile::Ct,
                    Tile::CtK,
                ]
            } else {
                &[
                    Tile::Default,
                    Tile::T64x64,
                    Tile::T128x64,
                    Tile::T128x128,
                    Tile::T128x128Thread16x8,
                    Tile::Cs,
                    Tile::Kt,
                ]
            };
            for &tile in tiles {
                // A token count past a few thousand only as many blocks
                // as fit and 7: one block would take seconds.
                let counts: &[i32] = if m > 4096 { &[0, 7] } else { &[0, 1, 7, 33] };
                for &blocks in counts {
                    let (diff, reference) =
                        turbo::cuda::gemm_check(ordinal, m, n, k, epilogue, half, tensor_cores, tile, blocks, heads)
                            .unwrap();
                    // TF32 rounds each operand to 10 bits of mantissa, which
                    // cuBLAS's F32 product does not.
                    let f16_out = half && !matches!(epilogue, Plain);
                    let tf32 = !half && tensor_cores;
                    // F16 accumulators round each 64 terms' sum to 11 bits,
                    // or with the whole-k tiles every sum of a segment's k.
                    let f16_sums = half
                        && tensor_cores
                        && matches!(
                            tile,
                            Tile::EightWarpsF16Accumulate
                                | Tile::SwizzledEightWarpsF16Accumulate
                                | Tile::F16WholeK
                                | Tile::F16WholeK3
                                | Tile::F16WholeK256
                                | Tile::CtK
                        );
                    let bound = if f16_sums {
                        1e-2
                    } else if f16_out || tf32 {
                        2e-3
                    } else {
                        1e-5
                    } * reference.max(1.0);
                    let what = format!(
                        "{epilogue:?} [{m}, {k}] x [{n}, {k}], tile {tile:?}, {blocks} blocks, F16 {half}, mma \
                         {tensor_cores}"
                    );
                    println!("{what}: largest difference {diff:.3e} of values up to {reference:.3}");
                    assert!(diff <= bound, "{what}: {diff:e} over {bound:e}");
                }
            }
        }
    }
}
