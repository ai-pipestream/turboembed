//! The Metal backend's late pooling, its session limits and its largest
//! shape: ranges held to the row path's vectors, the f64 arithmetic and the
//! CPU backend's ranges; the largest shape and the bundle's reference
//! vectors (tests/common/conformance.rs) on the device.
//!
//! Built with the `metal` feature only, on macOS; tests/metal.rs says how a
//! test without a device is skipped.

#![cfg(feature = "metal")]

mod common;
mod metal_support;

use common::*;
use metal_support::*;
use serde_json::json;
use turbo::status::UNSUPPORTED_OPTION;
use turbo::*;

// ---- Range pooling ---------------------------------------------------------------------

/// A session on `m` of the model's size that takes `max_ranges` ranges.
fn ranged_session(m: *mut turbo_model, max_ranges: u32) -> Session {
    let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
    d.max_ranges = max_ranges;
    Session::create(m, Some(&d)).unwrap_or_else(|e| panic!("{e:?}"))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A range over a whole row, padding and a masked token inside it
/// included, is that row's MEAN vector from embed_write, bit for bit:
/// the range kernel sums the same tokens in the same order.
#[test]
fn a_range_over_a_whole_row_is_the_rows_mean_vector_bit_for_bit() {
    let _t = turn();
    let Some(_) = metal_device("a_range_over_a_whole_row_is_the_rows_mean_vector_bit_for_bit") else { return };
    let l = on_metal(&tiny_bundle());
    let s = ranged_session(l.m, 8);
    let t = awkward_rows(&tiny_bundle());
    let whole: Vec<_> = (0..t.batch).map(|r| range(r, 0, t.seq)).collect();
    for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
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
            assert_eq!(bits(g), bits(w), "normalize {normalize} row {r}");
        }
        // In another order, and a row twice: each vector is its row's.
        let shuffled = [whole[2], whole[0], whole[2], whole[1]];
        s.write_ranges(&t.batch(), &shuffled, Some(&o)).unwrap();
        let got = s.run().unwrap().rows();
        for (g, r) in got.iter().zip([2, 0, 2, 1]) {
            assert_eq!(bits(g), bits(&rows[r]), "normalize {normalize} row {r} out of order");
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
    let Some(_) = metal_device("ranges_match_the_arithmetic_and_the_cpu") else { return };
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let (g, c) = (on_metal(&dir), Loaded::load(&dir).unwrap());
    let (gs, cs) = (ranged_session(g.m, 16), ranged_session(c.m, 16));
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

/// The GPU takes no thread count: cpu_threads other than 0 is refused by
/// field 6, and 0 makes the session.
#[test]
fn cpu_threads_is_refused_by_field_6() {
    let _t = turn();
    let Some(_) = metal_device("cpu_threads_is_refused_by_field_6") else { return };
    let l = on_metal(&tiny_bundle());
    for threads in [1, 4] {
        let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
        d.cpu_threads = threads;
        let e = Session::create(l.m, Some(&d)).err().unwrap();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 6), "{e:?}");
        assert!(e.message.contains("not a CPU"), "{e:?}");
    }
    Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_MODEL))).unwrap();
}

// ---- Limits and the largest shape ----------------------------------------------------

/// A session longer than attention's threadgroup memory holds on this
/// device is refused by field 2. 65536 positions need 256 KiB of scores
/// per threadgroup, more than any Apple GPU gives one.
#[test]
fn more_tokens_than_attention_holds_are_refused_by_field() {
    let _t = turn();
    let Some(_) = metal_device("more_tokens_than_attention_holds_are_refused_by_field") else { return };
    let positions = 65536u64;
    let mut m = model_manifest();
    m["architecture"]["max_positions"] = json!(positions);
    m["embed"]["max_seq"] = json!(positions);
    // A case longer than max_seq, which a manifest needs: 600 paragraphs of
    // about 120 tokens.
    m["reference"]["cases"][8]["text"] = json!(vec![PARAGRAPH; 600].join(" "));
    let mut f = Fixture::new("metal-positions", m);
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
    let l = f.load_on(metal).unwrap();
    let e = Session::create(l.m, Some(&session_desc(1, positions as u32, 0))).err().unwrap();
    assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 2), "{e:?}");
    assert!(e.message.contains("threadgroup memory"), "{e:?}");
    Session::create(l.m, Some(&session_desc(1, 512, 0))).unwrap();
}

/// One run at the session's largest shape, max_batch rows of max_seq
/// tokens, on the device and on the CPU: every row's cosine reaches
/// 0.9999. Rows are full or end early, so the padding is exercised too.
fn largest_shape_matches_the_cpu(dir: &std::path::Path) {
    let (g, c) = (on_metal(dir), Loaded::load(dir).unwrap());
    let (gs, cs) = (Session::create(g.m, None).unwrap(), Session::create(c.m, None).unwrap());
    let si = gs.info();
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
    let mut lowest = 1f64;
    for (r, (a, b)) in got.iter().zip(&want).enumerate() {
        let cos = cosine(a, b);
        lowest = lowest.min(cos);
        assert!(cos >= 0.9999, "row {r}: cosine {cos} with the cpu");
    }
    println!("{}: {batch} rows of {seq} tokens, 1 - lowest cosine with the cpu {:.3e}", dir.display(), 1.0 - lowest);
}

#[test]
fn the_largest_shape_matches_the_cpu() {
    let _t = turn();
    let Some(_) = metal_device("the_largest_shape_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&tiny_bundle());
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn the_largest_shape_of_a_real_bundle_matches_the_cpu() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = metal_device("the_largest_shape_of_a_real_bundle_matches_the_cpu") else { return };
    largest_shape_matches_the_cpu(&dir);
}

/// LayerNorm keeps a row of up to 1024 values in registers and reads a
/// wider row from device memory: one-layer models at widths on either side
/// of that edge, a lane's last value unused (1000) and every value used
/// (1024), match the CPU at the model's precision and at FASTEST.
#[test]
fn layer_norm_widths_on_either_side_of_the_register_edge_match_the_cpu() {
    let _t = turn();
    let Some(_) = metal_device("layer_norm_widths_on_either_side_of_the_register_edge_match_the_cpu") else { return };
    for (hidden, heads) in [(1000u64, 8), (1024, 16), (1056, 16)] {
        let mut m = model_manifest();
        m["architecture"]["layers"] = json!(1);
        m["architecture"]["hidden"] = json!(hidden);
        m["architecture"]["heads"] = json!(heads);
        m["architecture"]["intermediate"] = json!(2 * hidden);
        m["embed"]["dim"] = json!(hidden);
        let mut f = Fixture::new(&format!("metal-ln-{hidden}"), m);
        f.weights("weights/model.safetensors", &bert_weights(1, hidden, 2 * hidden, 3, 0.05));
        f.write();
        let (g, c) = (on_metal(&f.dir), Loaded::load(&f.dir).unwrap());
        let rows: Vec<Vec<i32>> =
            (0..3).map(|r| (0..40 - 9 * r).map(|p| 1000 + (r * 131 + p * 17) as i32).collect()).collect();
        let t = Tokens::new(&rows, 0);
        let cs = Session::create(c.m, None).unwrap();
        cs.write_tokens(&t.batch(), None).unwrap();
        let want = cs.run().unwrap().rows();
        for (precision, floor) in [(TURBO_PRECISION_MODEL, 0.99999), (TURBO_PRECISION_FASTEST, 0.999)] {
            let gs = Session::create(g.m, Some(&session_desc(0, 0, precision))).unwrap();
            gs.write_tokens(&t.batch(), None).unwrap();
            for (r, (a, b)) in gs.run().unwrap().rows().iter().zip(&want).enumerate() {
                let cos = cosine(a, b);
                assert!(cos >= floor, "hidden {hidden} precision {precision} row {r}: cosine {cos} with the cpu");
            }
        }
    }
}

/// The bundle's reference cases on the Metal device at each precision,
/// against the upstream vectors, within the tolerance of the compute
/// dtype the session reports (tests/common/conformance.rs).
fn reference_cases_match_upstream(dir: &std::path::Path) {
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
        conformance::check(dir, metal, p);
    }
}

#[test]
fn the_reference_cases_match_upstream() {
    let _t = turn();
    let Some(_) = metal_device("the_reference_cases_match_upstream") else { return };
    reference_cases_match_upstream(&tiny_bundle());
}

#[test]
#[ignore = "needs a real bundle directory in TURBO_TEST_BUNDLE"]
fn the_reference_cases_of_a_real_bundle_match_upstream() {
    let _t = turn();
    let dir = named_bundle().expect("TURBO_TEST_BUNDLE is not set");
    let Some(_) = metal_device("the_reference_cases_of_a_real_bundle_match_upstream") else { return };
    reference_cases_match_upstream(&dir);
}
