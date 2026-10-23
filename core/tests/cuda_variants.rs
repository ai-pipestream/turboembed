//! Every CUDA kernel variant and cuBLAS giving the same vectors, the
//! largest shape against the CPU, and the bundle's reference cases
//! against upstream (tests/common/conformance.rs).
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
use turbo::status::{INVALID_ARGUMENT, UNSUPPORTED_OPTION};
use turbo::*;

/// A session reports the kernels it chose, per token bin, as one line;
/// forcing that line back with TURBO_CUDA_CHOICES makes a session of the
/// same kernels, every knob forced, which gives the same bits. So do the
/// choices the switches force, and a line read back reports itself.
#[test]
fn a_sessions_choices_forced_back_give_its_bits() {
    let _t = turn();
    let Some(_) = cuda_device("a_sessions_choices_forced_back_give_its_bits") else { return };
    use turbo::cuda::{StreamK, Tile};
    let (f, g) = small_model("cuda-choices");
    let t = ragged_rows(&f.dir, 40, 160);
    let run = |s: &Session| {
        s.write_tokens(&t.batch(), None).unwrap();
        s.run().unwrap().rows()
    };
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        for switched in [false, true] {
            if switched {
                turbo::cuda::use_tile(Some(if precision == TURBO_PRECISION_FASTEST {
                    Tile::SwizzledEightWarps
                } else {
                    Tile::T64x64
                }));
                turbo::cuda::use_stream_k(Some(StreamK::Tiles));
            }
            let s = strict(|| Session::create(g.m, Some(&session_desc(40, 160, precision))));
            turbo::cuda::use_tile(None);
            turbo::cuda::use_stream_k(None);
            let s = s.unwrap();
            let info = s.info();
            let choices = field(&info.choices);
            println!("precision {precision}: {choices}");
            assert!(choices.starts_with("le256:") && choices.contains(";le16k:"), "{choices}");
            assert!(!choices.contains("gt16k"), "40 x 160 tokens have no bin past 16384: {choices}");
            if switched {
                assert!(choices.ends_with(";forced=tile,sk"), "{choices}");
                assert!(choices.contains("/tiles,"), "{choices}");
            } else {
                assert!(choices.ends_with(";forced="), "{choices}");
                assert_eq!(info.tuned, TURBO_TUNED_DEFAULT);
            }
            assert_eq!(info.tune_ms, 0);
            let want = run(&s);
            let back = forcing(&choices, || strict(|| Session::create(g.m, Some(&session_desc(40, 160, precision)))));
            let back = back.unwrap();
            let bi = back.info();
            assert_eq!(bi.tuned, TURBO_TUNED_FORCED, "precision {precision}: every knob named");
            assert_eq!(kernels_of(&field(&bi.choices)), kernels_of(&choices));
            assert_eq!(run(&back), want, "precision {precision}, {choices}: the same bits");
            // The line a forced session reports forces the same again.
            let again = forcing(&field(&bi.choices), || {
                strict(|| Session::create(g.m, Some(&session_desc(40, 160, precision))))
            });
            assert_eq!(field(&again.unwrap().info().choices), field(&bi.choices));
        }
    }
}

/// F32 operands take the transposed FMA tile (`kt`) for the four GEMMs
/// at every bin, stream-K, but the second feed-forward GEMM with whole
/// tiles from the le16k bin where its k is 3072 or more; F16 operands
/// keep their own. The default gives the vectors of the FMA kernel's
/// 128 x 64 forced everywhere, within EXACT's bound, and repeats its bits.
#[test]
fn f32_defaults_take_the_transposed_tile() {
    let _t = turn();
    let Some(_) = cuda_device("f32_defaults_take_the_transposed_tile") else { return };
    for (hidden, heads, inter) in [(1024u64, 16u64, 4096u64), (64, 2, 256)] {
        let mut m = model_manifest();
        m["architecture"]["hidden"] = json!(hidden);
        m["architecture"]["heads"] = json!(heads);
        m["architecture"]["intermediate"] = json!(inter);
        m["embed"]["dim"] = json!(hidden);
        m["embed"]["max_seq"] = json!(160);
        m["embed"]["max_batch"] = json!(40);
        let mut f = Fixture::new(&format!("cuda-f32-defaults-{hidden}"), m);
        f.weights("weights/model.safetensors", &bert_weights(hidden, inter));
        let g = f.load_on(cuda).unwrap();
        let s = strict(|| Session::create(g.m, Some(&session_desc(40, 160, TURBO_PRECISION_EXACT)))).unwrap();
        let choices = kernels_of(&field(&s.info().choices));
        let kt = "qkv=kt/sk4,out=kt/sk4,ffn1=kt/sk4,ffn2=kt/sk4";
        for bin in ["le256", "le1k", "le4k"] {
            assert!(choices.contains(&format!("{bin}:{kt},")), "hidden {hidden}: {choices}");
        }
        let last = if inter >= 3072 { "qkv=kt/sk4,out=kt/sk4,ffn1=kt/sk4,ffn2=kt/tiles" } else { kt };
        assert!(choices.contains(&format!("le16k:{last},")), "hidden {hidden}: {choices}");
        let t = ragged_rows(&f.dir, 40, 160);
        s.write_tokens(&t.batch(), None).unwrap();
        let got = s.run().unwrap().rows();
        s.write_tokens(&t.batch(), None).unwrap();
        assert_eq!(s.run().unwrap().rows(), got, "the same bits again");
        let all = "all:qkv=128x64,out=128x64,ffn1=128x64,ffn2=128x64";
        let o = forcing(all, || strict(|| Session::create(g.m, Some(&session_desc(40, 160, TURBO_PRECISION_EXACT)))));
        let o = o.unwrap();
        o.write_tokens(&t.batch(), None).unwrap();
        let want = o.run().unwrap().rows();
        let tol = record::tolerance(o.info().compute_dtype).unwrap();
        let (cos, abs) = within("the default against the FMA kernel's 128 x 64", &got, &want, tol);
        println!("hidden {hidden}: default against 128x64: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
    }
}

/// Bins forced apart run apart: a batch in the bin up to 256 tokens runs
/// that bin's kernels, one in another bin the others', each within the
/// precision's bound of the default and repeating its bits. A bin the
/// session does not have is left out of what it forces.
#[test]
fn bins_forced_apart_run_their_own_kernels() {
    let _t = turn();
    let Some(_) = cuda_device("bins_forced_apart_run_their_own_kernels") else { return };
    let (f, g) = small_model("cuda-bins");
    let small = ragged_rows(&f.dir, 3, 60);
    let large = ragged_rows(&f.dir, 40, 160);
    for precision in [TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let own = strict(|| Session::create(g.m, Some(&session_desc(40, 160, precision)))).unwrap();
        let tol = record::tolerance(own.info().compute_dtype).unwrap();
        let tile = if precision == TURBO_PRECISION_FASTEST { "sw8w" } else { "64x64" };
        // The bin past 16384 tokens, which the session does not have, is
        // left out: one line forces sessions of any size.
        let apart = format!(
            "le256:qkv={tile}/sk2,out={tile}/tiles,ffn1={tile}/sk8,ffn2={tile}/sk1,ln=fused;gt16k:qkv=64x64/tiles"
        );
        let s = forcing(&apart, || strict(|| Session::create(g.m, Some(&session_desc(40, 160, precision)))));
        let s = s.unwrap();
        let choices = field(&s.info().choices);
        assert!(choices.starts_with(&format!("le256:qkv={tile}/sk2,out={tile}/tiles,")), "{choices}");
        assert!(choices.ends_with(";forced=tile,sk,tf32,ln"), "{choices}");
        assert!(!choices.contains("gt16k"), "{choices}");
        for t in [&small, &large] {
            own.write_tokens(&t.batch(), None).unwrap();
            let want = own.run().unwrap().rows();
            s.write_tokens(&t.batch(), None).unwrap();
            let got = s.run().unwrap().rows();
            s.write_tokens(&t.batch(), None).unwrap();
            assert_eq!(s.run().unwrap().rows(), got, "precision {precision}: the same bits again");
            within(&format!("precision {precision}, {} rows", t.batch().batch), &got, &want, tol);
        }
    }
}

/// A variant is eligible for a precision by the numeric class it
/// computes in, which the core's table allows: every variant has one of
/// the four classes; one outside the precision's is refused when forced,
/// naming it, unless the experiment's switch widens the session's set;
/// and TURBO_CUDA_TF32 at EXACT changes nothing. An unknown item is
/// refused.
#[test]
fn a_variant_outside_the_precision_is_refused() {
    let _t = turn();
    let Some(dev) = cuda_device("a_variant_outside_the_precision_is_refused") else { return };
    let ordinal = Rt::new().info(dev).ordinal;
    use turbo::backend::*;
    let (_f, g) = small_model("cuda-numerics");
    let classes = [TURBO_NUMERIC_F32_FMA, TURBO_NUMERIC_TF32, TURBO_NUMERIC_F16_F32ACC, TURBO_NUMERIC_F16_CHUNKACC];
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let vs = turbo::cuda::variants(ordinal, precision).unwrap();
        assert!(!vs.is_empty());
        for v in &vs {
            assert!(classes.contains(&v.numeric), "{v:?}");
            let make = || {
                let all = format!("all:qkv={0},out={0},ffn1={0},ffn2={0}", v.name);
                forcing(&all, || Session::create(g.m, Some(&session_desc(40, 160, precision))))
            };
            if v.numeric & turbo::tuning::numerics_allowed(precision) != 0 {
                make().unwrap();
                continue;
            }
            let e = make().err().unwrap();
            assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{v:?}: {}", e.message);
            // The refusal names the kernel a GEMM runs: the variant, or
            // on a GEMM it does not serve, the tile it gives way to there.
            let named = e.message.contains(v.name.split('/').next().unwrap());
            // The experiment's switch widens the session's classes.
            let mut widened = None;
            if v.numeric == TURBO_NUMERIC_TF32 && precision == TURBO_PRECISION_MODEL {
                turbo::cuda::use_tf32(Some(true));
                let s = make();
                turbo::cuda::use_tf32(None);
                let choices = field(&s.unwrap().info().choices);
                assert!(choices.contains("/tf32"), "{choices}");
                widened = Some(choices);
            }
            if v.numeric == TURBO_NUMERIC_F16_CHUNKACC {
                turbo::cuda::use_f16_accumulate(Some(true));
                let s = make();
                turbo::cuda::use_f16_accumulate(None);
                let choices = field(&s.unwrap().info().choices);
                assert!(choices.contains(&format!("={}/", v.name)), "{choices}");
                widened = Some(choices);
            }
            if !named {
                // "<bin>:<gemm>=<tile>/..." is what that GEMM of that bin
                // runs once the switch allows it.
                let item = e.message.split(' ').find(|w| w.contains(":") && w.contains("=")).unwrap();
                let (bin, run) = item.split_once(':').unwrap();
                let choices = widened.unwrap_or_else(|| panic!("{v:?}: {}", e.message));
                let in_bin = choices.split(';').find(|b| b.starts_with(&format!("{bin}:"))).unwrap();
                assert!(in_bin.contains(run), "{v:?}: {} runs as {in_bin}", e.message);
            }
        }
    }
    // Neither accuracy-changing class is a default candidate's anywhere.
    turbo::cuda::use_tf32(Some(true));
    let s = Session::create(g.m, Some(&session_desc(40, 160, TURBO_PRECISION_EXACT)));
    turbo::cuda::use_tf32(None);
    let choices = field(&s.unwrap().info().choices);
    assert!(!choices.contains("/tf32") && choices.ends_with("forced="), "{choices}");
    for bad in ["le9k:qkv=8w", "le256:qkv=9w", "le256:qkv=8w/sk99", "le256:gelu=8w", "pool=rows", "tiles"] {
        let e = forcing(bad, || Session::create(g.m, Some(&session_desc(40, 160, TURBO_PRECISION_FASTEST))));
        let e = e.err().unwrap();
        assert_eq!(e.code, INVALID_ARGUMENT, "{bad}: {}", e.message);
        assert!(e.message.contains("TURBO_CUDA_CHOICES"), "{}", e.message);
    }
}

/// The whole-k tiles sum in F16, which FASTEST allows: one forced through
/// TURBO_CUDA_CHOICES is taken with no switch, and the session's choices
/// name it; the default is `ctk`, F16 sums over the whole of a block's k
/// on CUTLASS's mainloop, in a bin of any size.
#[test]
fn a_whole_k_tile_is_taken_at_fastest() {
    let _t = turn();
    let Some(_) = cuda_device("a_whole_k_tile_is_taken_at_fastest") else { return };
    let (_f, g) = small_model("cuda-whole-k");
    let make =
        |line: &str| forcing(line, || Session::create(g.m, Some(&session_desc(40, 160, TURBO_PRECISION_FASTEST))));
    // (f16k256 is 256 x 128 only for QKV and GELU.)
    for (gemm, tile) in [("ffn2", "f16k"), ("ffn2", "f16k3"), ("ffn2", "f16krow"), ("ffn1", "f16k256"), ("out", "ctk")]
    {
        let line = format!("all:{gemm}={tile}");
        let choices = field(&make(&line).unwrap().info().choices);
        assert!(choices.contains(&format!("{gemm}={tile}/")), "{choices}");
    }
    let s = Session::create(g.m, Some(&session_desc(40, 160, TURBO_PRECISION_FASTEST))).unwrap();
    let choices = field(&s.info().choices);
    for gemm in ["qkv", "out", "ffn1", "ffn2"] {
        assert!(choices.contains(&format!("{gemm}=ctk/")), "{choices}");
    }
}

/// Every GEMM variant a precision allows, forced for every GEMM in every
/// bin, gives the default's vectors within the precision's bound, at
/// every precision, and repeats its own bits.
#[test]
fn every_variant_gives_the_same_vectors() {
    let _t = turn();
    let Some(dev) = cuda_device("every_variant_gives_the_same_vectors") else { return };
    let ordinal = Rt::new().info(dev).ordinal;
    let (f, g) = small_model("cuda-variants");
    let t = ragged_rows(&f.dir, 40, 160);
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let own = strict(|| Session::create(g.m, Some(&session_desc(40, 160, precision)))).unwrap();
        let tol = record::tolerance(own.info().compute_dtype).unwrap();
        own.write_tokens(&t.batch(), None).unwrap();
        let want = own.run().unwrap().rows();
        let allowed = turbo::tuning::numerics_allowed(precision);
        for v in turbo::cuda::variants(ordinal, precision).unwrap().iter().filter(|v| v.numeric & allowed != 0) {
            let all = format!("all:qkv={0},out={0},ffn1={0},ffn2={0}", v.name);
            let s = forcing(&all, || Session::create(g.m, Some(&session_desc(40, 160, precision)))).unwrap();
            s.write_tokens(&t.batch(), None).unwrap();
            let got = s.run().unwrap().rows();
            s.write_tokens(&t.batch(), None).unwrap();
            assert_eq!(s.run().unwrap().rows(), got, "precision {precision}, {}: the same bits again", v.name);
            // What the session reports it ran, forced back, runs the same
            // kernels: the reported names are the kernels', not the asked.
            let line = field(&s.info().choices);
            let back = forcing(&line, || Session::create(g.m, Some(&session_desc(40, 160, precision)))).unwrap();
            assert_eq!(
                kernels_of(&field(&back.info().choices)),
                kernels_of(&line),
                "precision {precision}, {}",
                v.name
            );
            back.write_tokens(&t.batch(), None).unwrap();
            assert_eq!(back.run().unwrap().rows(), got, "precision {precision}, {} forced back as {line}", v.name);
            let (cos, abs) = within(&format!("precision {precision}, {}", v.name), &got, &want, tol);
            println!("precision {precision}, {}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}", v.name);
        }
    }
}

/// TURBO_CUDA_CUBLAS hands GEMMs to cuBLAS for measuring: every choice of
/// them gives the vectors of the backend's own GEMMs within the bound of
/// the precision, at MODEL and FASTEST.
#[test]
fn gemms_handed_to_cublas_give_the_same_vectors() {
    let _t = turn();
    let Some(_) = cuda_device("gemms_handed_to_cublas_give_the_same_vectors") else { return };
    if !cublas_or_skip("gemms_handed_to_cublas_give_the_same_vectors", "the test") {
        return;
    }
    let dir = tiny_bundle();
    let g = on_cuda(&dir);
    let mi = g.info();
    let t = ragged_rows(&dir, mi.max_batch as usize, mi.max_seq as usize);
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let own = strict(|| Session::create(g.m, Some(&session_desc(0, 0, precision)))).unwrap();
        let tol = record::tolerance(own.info().compute_dtype).unwrap();
        own.write_tokens(&t.batch(), None).unwrap();
        let want = own.run().unwrap().rows();
        for gemms in [1, 2, 4, 8, 15] {
            turbo::cuda::use_cublas(Some(gemms));
            let s = strict(|| Session::create(g.m, Some(&session_desc(0, 0, precision))));
            turbo::cuda::use_cublas(None);
            let s = s.unwrap();
            s.write_tokens(&t.batch(), None).unwrap();
            let (cos, abs) =
                within(&format!("precision {precision}, cuBLAS {gemms}"), &s.run().unwrap().rows(), &want, tol);
            println!("precision {precision}, cuBLAS for {gemms}: 1 - lowest cosine {cos:.3e}, max abs diff {abs:.3e}");
        }
    }
}

/// One run at the session's largest shape, max_batch rows of max_seq
/// tokens, on the device and on the CPU, at each precision: every row is
/// within the bound of the session's compute dtype. Rows are full or end
/// early, so the padding is exercised too.
fn largest_shape_matches_the_cpu(dir: &std::path::Path) {
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
        largest_shape_matches_the_cpu_at(dir, p);
    }
}

fn largest_shape_matches_the_cpu_at(dir: &std::path::Path, precision: u32) {
    let (g, c) = (on_cuda(dir), Loaded::load(dir).unwrap());
    let (gs, cs) = (
        strict(|| Session::create(g.m, Some(&session_desc(0, 0, precision)))).unwrap(),
        Session::create(c.m, None).unwrap(),
    );
    let si = gs.info();
    let tol = record::tolerance(si.compute_dtype).unwrap();
    let (batch, seq) = (si.max_batch as usize, si.max_seq as usize);
    let vocab = Tok::create(dir).unwrap().info().vocab_size as usize;
    let rows: Vec<Vec<i32>> = (0..batch)
        .map(|r| {
            let len = seq - (r * 7) % (seq / 2);
            (0..len).map(|p| (1000 + (r * 131 + p * 17) % (vocab - 1000)) as i32).collect()
        })
        .collect();
    let t = Tokens::new(&rows, 0);
    assert_eq!((t.batch as usize, t.seq as usize), (batch, seq));
    gs.write_tokens(&t.batch(), None).unwrap();
    let got = gs.run().unwrap().rows();
    cs.write_tokens(&t.batch(), None).unwrap();
    let want = cs.run().unwrap().rows();
    let (mut lowest, mut worst) = (1f64, 0f64);
    for (r, (a, b)) in got.iter().zip(&want).enumerate() {
        let cos = cosine(a, b);
        lowest = lowest.min(cos);
        assert!(cos >= tol.min_cosine, "precision {precision} row {r}: cosine {cos} with the cpu");
        let d = max_abs_diff(a, b);
        worst = worst.max(d);
        if let Some(most) = tol.max_abs_diff {
            assert!(d <= most, "precision {precision} row {r}: max abs diff {d:e} with the cpu");
        }
    }
    println!(
        "{}: precision {precision}, compute dtype {}, {batch} rows of {seq} tokens, 1 - lowest cosine with the cpu \
         {:.3e}, max abs diff {worst:.3e}",
        dir.display(),
        si.compute_dtype,
        1.0 - lowest
    );
}

#[test]
fn the_largest_shape_matches_the_cpu() {
    let _t = turn();
    let Some(_) = cuda_device("the_largest_shape_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&tiny_bundle());
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn the_largest_shape_of_a_real_bundle_matches_the_cpu() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = cuda_device("the_largest_shape_of_a_real_bundle_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&dir);
}

/// The bundle's reference cases on the CUDA device at each precision,
/// against the upstream vectors, within the tolerance of the compute
/// dtype the session reports (tests/common/conformance.rs).
fn reference_cases_match_upstream(dir: &std::path::Path) {
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
        strict(|| conformance::check(dir, cuda, p));
    }
}

#[test]
fn the_reference_cases_match_upstream() {
    let _t = turn();
    let Some(_) = cuda_device("the_reference_cases_match_upstream") else { return };
    reference_cases_match_upstream(&tiny_bundle());
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn the_reference_cases_of_a_real_bundle_match_upstream() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = cuda_device("the_reference_cases_of_a_real_bundle_match_upstream") else { return };
    reference_cases_match_upstream(&dir);
}
