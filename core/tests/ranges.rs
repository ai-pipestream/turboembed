//! Late pooling through the C interface on the CPU backend this build
//! links: turbo_embed_write_ranges over the small BERT, with a range of a
//! whole row against that row's own vector, ranges of every kind against
//! the encoder written plainly in f64, the refusals the header lists, the
//! shape of a range run's result, the session's max_ranges and cpu_threads
//! fields, and the same write on a static model.

mod common;

use std::fs;
use std::path::Path;
use std::ptr;

use common::*;
use serde_json::json;
use turbo::safetensors::{Dtype, File};
use turbo::status::*;
use turbo::*;

fn tiny() -> Loaded {
    Loaded::load(&tiny_bundle()).unwrap_or_else(|e| panic!("{e:?}"))
}

/// A session of the model's size that takes `max_ranges` ranges.
fn session_of(l: &Loaded, max_ranges: u32) -> Session {
    let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
    d.max_ranges = max_ranges;
    Session::create(l.m, Some(&d)).unwrap_or_else(|e| panic!("{e:?}"))
}

fn opts(edit: impl FnOnce(&mut turbo_embed_options)) -> turbo_embed_options {
    let mut o = embed_options();
    edit(&mut o);
    o
}

fn normalized(normalize: u32) -> turbo_embed_options {
    opts(|o| o.normalize = normalize)
}

const TEXTS: [&str; 3] =
    ["The quick brown fox jumps over the lazy dog. It was a sunny day.", "how do I reset a password", "a b c"];

/// `texts` tokenized with the bundle in `dir` and padded on the right, and
/// each row's length.
fn tokens_of(dir: &Path, texts: &[&str]) -> (Tokens, Vec<usize>) {
    let tok = Tok::create(dir).unwrap();
    let rows: Vec<Vec<i32>> = texts.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let lens = rows.iter().map(Vec::len).collect();
    (Tokens::new(&rows, 0), lens)
}

/// Row `r` of `t`: its ids, its mask and its types (zero when `t` has none).
fn row_of(t: &Tokens, r: usize) -> (&[i32], &[i32], Vec<i32>) {
    let seq = t.seq as usize;
    let at = r * seq..(r + 1) * seq;
    let types = t.types.as_ref().map_or(vec![0; seq], |v| v[at.clone()].to_vec());
    (&t.ids[at.clone()], &t.mask[at], types)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn run_ranges(s: &Session, t: &Tokens, ranges: &[turbo_token_range], o: &turbo_embed_options) -> Vec<Vec<f32>> {
    s.write_ranges(&t.batch(), ranges, Some(o)).unwrap_or_else(|e| panic!("{e:?}"));
    s.run().unwrap_or_else(|e| panic!("{e:?}")).rows()
}

// ---- A range over a whole row -------------------------------------------------------

#[test]
fn a_range_over_a_whole_row_is_the_rows_mean_vector() {
    let l = tiny();
    let s = session_of(&l, 0);
    let (t, lens) = tokens_of(&tiny_bundle(), &TEXTS);
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

// ---- Ranges against the arithmetic written plainly ----------------------------------

/// Ranges of every kind in three rows of `TEXTS`, run on the model in `dir`
/// and held to PlainBert; the largest difference seen.
fn check_against_plain(dir: &Path, l: &Loaded, dim: usize, output_dim: u32) -> f64 {
    let plain = PlainBert::new(dir);
    let s = session_of(l, 0);
    let (mut t, lens) = tokens_of(dir, &TEXTS);
    let (seq, n) = (t.seq, t.seq as usize);
    // Row 2 has a masked token inside its text, and types of 1 after its
    // first token.
    t.mask[2 * n + 3] = 0;
    let mut types = vec![0; 3 * n];
    for p in 1..lens[2] {
        types[2 * n + p] = 1;
    }
    t.types = Some(types.clone());
    let (l0, l1, l2) = (lens[0] as u32, lens[1] as u32, lens[2] as u32);
    assert!(l0 > l1 && l1 > 4 && l2 > 4, "{lens:?}");
    let ranges = [
        // Row 0 as three sentence-like pieces, and one token alone.
        range(0, 0, l0 / 3),
        range(0, l0 / 3, 2 * l0 / 3),
        range(0, 2 * l0 / 3, l0),
        range(0, 2, 3),
        // Row 1: the last two live tokens and the padding after them, and
        // the whole row.
        range(1, l1 - 2, seq),
        range(1, 0, l1),
        // Row 2: ranges over the masked token, and one that ends in padding.
        range(2, 0, 5),
        range(2, 2, 4),
        range(2, 3, seq),
    ];
    let mut worst = 0f64;
    for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
        let o = opts(|o| {
            o.normalize = normalize;
            o.output_dim = output_dim;
        });
        let got = run_ranges(&s, &t, &ranges, &o);
        assert_eq!(got.len(), ranges.len());
        for (i, (g, rg)) in got.iter().zip(&ranges).enumerate() {
            let (ids, mask, ty) = row_of(&t, rg.row as usize);
            let want = plain.embed_range(
                ids,
                mask,
                &ty,
                rg.start as usize,
                rg.end as usize,
                dim,
                normalize == TURBO_NORMALIZE_L2,
            );
            assert_eq!(g.len(), dim, "range {i}");
            for (a, b) in g.iter().zip(&want) {
                let d = (*a as f64 - b).abs();
                worst = worst.max(d);
                assert!(d < 1e-5 * (1.0 + b.abs()), "normalize {normalize} range {i}: {a} vs {b}");
            }
        }
    }
    worst
}

#[test]
fn ranges_match_the_arithmetic_written_plainly() {
    let l = tiny();
    let mut worst = check_against_plain(&tiny_bundle(), &l, 32, 0);
    worst = worst.max(check_against_plain(&tiny_bundle(), &l, 32, 32));

    // A bundle that lists an output width: the vector is cut, then
    // normalized.
    let mut f = Fixture::new("range-output-dims", {
        let mut m = model_manifest();
        m["embed"]["output_dims"] = json!([4]);
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let cut = f.load().unwrap();
    worst = worst.max(check_against_plain(&f.dir, &cut, 4, 4));
    worst = worst.max(check_against_plain(&f.dir, &cut, 8, 0));
    println!("largest difference from the f64 encoder: {worst:.3e}");
}

// ---- Overlap, repeats, order and skipped rows ---------------------------------------

#[test]
fn ranges_may_overlap_repeat_and_skip_rows() {
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let l = tiny();
    let s = session_of(&l, 5);
    let (t, lens) = tokens_of(&dir, &TEXTS);
    let (l0, l2) = (lens[0] as u32, lens[2] as u32);
    // Row 1 has no range; row 2 comes before row 0; the first and third
    // ranges are the same; the fourth overlaps the second.
    let ranges = [range(2, 1, l2), range(0, 0, l0), range(2, 1, l2), range(0, 3, l0 - 1), range(2, 0, 2)];
    let o = normalized(TURBO_NORMALIZE_L2);
    let got = run_ranges(&s, &t, &ranges, &o);
    assert_eq!(got.len(), 5);
    assert_eq!(bits(&got[0]), bits(&got[2]), "the same range twice");
    for (i, (g, rg)) in got.iter().zip(&ranges).enumerate() {
        let (ids, mask, ty) = row_of(&t, rg.row as usize);
        let want = plain.embed_range(ids, mask, &ty, rg.start as usize, rg.end as usize, 32, true);
        for (a, b) in g.iter().zip(&want) {
            assert!((*a as f64 - b).abs() < 1e-5 * (1.0 + b.abs()), "range {i}: {a} vs {b}");
        }
    }

    // The order of the ranges is the order of the vectors, and nothing else
    // about a vector depends on it.
    let reversed: Vec<_> = ranges.iter().rev().copied().collect();
    let back = run_ranges(&s, &t, &reversed, &o);
    for (i, g) in got.iter().enumerate() {
        assert_eq!(bits(g), bits(&back[4 - i]), "range {i}");
    }
}

/// `t` laid out with `gap` more columns per row than seq, the gap filled
/// with live tokens the batch must not reach.
fn strided(t: &Tokens, gap: u32) -> Tokens {
    let (seq, stride) = (t.seq as usize, (t.seq + gap) as usize);
    let mut s = Tokens {
        ids: vec![1000; t.batch as usize * stride],
        mask: vec![1; t.batch as usize * stride],
        types: None,
        batch: t.batch,
        seq: t.seq,
        stride: stride as u32,
    };
    for r in 0..t.batch as usize {
        s.ids[r * stride..r * stride + seq].copy_from_slice(&t.ids[r * seq..(r + 1) * seq]);
        s.mask[r * stride..r * stride + seq].copy_from_slice(&t.mask[r * seq..(r + 1) * seq]);
    }
    s
}

#[test]
fn a_range_reads_its_row_through_the_row_stride() {
    let dir = tiny_bundle();
    let l = tiny();
    let s = session_of(&l, 5);
    let (t, lens) = tokens_of(&dir, &TEXTS);
    let (l0, l1, l2) = (lens[0] as u32, lens[1] as u32, lens[2] as u32);
    let ranges = [range(0, 0, l0), range(1, 1, l1), range(2, 0, l2), range(0, 2, 5), range(1, 0, t.seq)];
    let o = normalized(TURBO_NORMALIZE_L2);
    let packed = run_ranges(&s, &t, &ranges, &o);
    let wide = strided(&t, 3);
    let got = run_ranges(&s, &wide, &ranges, &o);
    for (i, (a, b)) in packed.iter().zip(&got).enumerate() {
        assert_eq!(bits(a), bits(b), "range {i}: the gap past seq is not read");
    }
    // A range holding no live token pools those mask-0 columns through
    // the stride exactly as packed: the same bits either way.
    let masked = [range(2, l2, t.seq)];
    let packed = run_ranges(&s, &t, &masked, &o);
    let got = run_ranges(&s, &wide, &masked, &o);
    assert_eq!(bits(&packed[0]), bits(&got[0]), "a mask-0 range reads its row through the stride");
    assert!(packed[0].iter().all(|v| v.is_finite()));
}

// ---- Refusals -----------------------------------------------------------------------

#[test]
fn a_range_write_refuses_what_it_cannot_take() {
    let l = tiny();
    let s = session_of(&l, 4);
    let (t, lens) = tokens_of(&tiny_bundle(), &TEXTS);
    let (seq, l1) = (t.seq, lens[1] as u32);
    assert!(l1 < seq);
    let good = [range(0, 0, 2), range(1, 0, 3)];
    let with_third = |third: turbo_token_range| [good[0], good[1], third, range(2, 0, 2)];

    // After a write the run takes, a refused write leaves nothing to run.
    let refused = |ranges: &[turbo_token_range], o: Option<&turbo_embed_options>| -> Failure {
        s.write_ranges(&t.batch(), &good, None).unwrap();
        let e = s.write_ranges(&t.batch(), ranges, o).expect_err("the write is refused");
        assert_eq!(s.run().err().unwrap().code, INVALID_STATE, "{e:?}");
        e
    };

    let e = refused(&[], None);
    assert_eq!(e.code, INVALID_ARGUMENT, "count 0: {e:?}");
    let five = [good[0], good[1], good[0], good[1], good[0]];
    let e = refused(&five, None);
    assert_eq!(e.code, CAPACITY, "count over max_ranges: {e:?}");

    for pooling in [TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        let e = refused(&good, Some(&opts(|o| o.pooling = pooling)));
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 5), "pooling {pooling}: {e:?}");
    }
    let e = refused(&good, Some(&opts(|o| o.pooling = 99)));
    assert_eq!(e.code, INVALID_ENUM, "pooling 99 is no TURBO_POOLING_* value: {e:?}");
    let e = refused(&good, Some(&opts(|o| o.truncate = TURBO_TRUNCATE_RIGHT)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 1), "{e:?}");
    let e = refused(&good, Some(&opts(|o| o.prompt_role = TURBO_PROMPT_QUERY)));
    assert_eq!((e.code, e.field), (INVALID_ARGUMENT, 3), "{e:?}");

    for (what, bad) in [
        ("row 3 of 3", range(3, 0, 1)),
        ("start 2 is not under end 2", range(0, 2, 2)),
        ("start 3 is not under end 2", range(0, 3, 2)),
        ("end past seq", range(0, 0, seq + 1)),
    ] {
        let e = refused(&with_third(bad), None);
        assert_eq!(e.code, INVALID_ARGUMENT, "{what}: {e:?}");
        assert!(e.message.contains("ranges[2]"), "{what}: {e:?}");
    }
    // No live token is not a refusal on cpu BERT: the mask-0 columns
    // pool their own states (turbo.h).
    s.write_ranges(&t.batch(), &with_third(range(1, l1, seq)), None).unwrap();
    assert_eq!(s.run().unwrap().info().batch, 4);

    let mut err = new_error();
    s.write_ranges(&t.batch(), &good, None).unwrap();
    let rc = unsafe { turbo_embed_write_ranges(s.0, &t.batch(), ptr::null(), 1, ptr::null(), &mut err) };
    assert_eq!(rc, INVALID_ARGUMENT, "{:?}", failure(rc, &err));
    assert_eq!(s.run().err().unwrap().code, INVALID_STATE);

    // A good write after the refusals runs.
    s.write_ranges(&t.batch(), &good, None).unwrap();
    assert_eq!(s.run().unwrap().info().batch, 2);
}

// ---- The result ---------------------------------------------------------------------

#[test]
fn a_range_run_reports_its_shape() {
    let l = tiny();
    let s = session_of(&l, 6);
    let (t, lens) = tokens_of(&tiny_bundle(), &TEXTS);
    let ranges = [range(0, 0, 4), range(0, 4, lens[0] as u32), range(1, 0, 2), range(2, 1, 3)];
    let count = ranges.len();
    for (normalize, pool_then_normalize) in
        [(TURBO_NORMALIZE_L2, TURBO_STAGE_HOST), (TURBO_NORMALIZE_NONE, TURBO_STAGE_UNUSED)]
    {
        s.write_ranges(&t.batch(), &ranges, Some(&normalized(normalize))).unwrap();
        let r = s.run().unwrap();
        let i = r.info();
        assert_eq!((i.batch as usize, i.dim), (count, 32));
        assert_eq!(i.bytes, (count * 32 * 4) as u64);
        assert_eq!((i.dtype, i.placement), (TURBO_DTYPE_F32, TURBO_PLACE_HOST));
        assert_eq!(i.stage[TURBO_EMBED_STAGE_POOL], TURBO_STAGE_HOST);
        assert_eq!(i.stage[TURBO_EMBED_STAGE_NORMALIZE], pool_then_normalize, "normalize {normalize}");

        let mut flat = vec![f32::NAN; count * 32];
        let mut written = 0u64;
        let rc = unsafe {
            turbo_result_read(r.0, flat.as_mut_ptr().cast(), (count * 32 * 4) as u64, &mut written, ptr::null_mut())
        };
        assert_eq!((rc, written), (0, (count * 32 * 4) as u64));
        assert!(flat.iter().all(|v| v.is_finite()));

        let mut buf = ptr::null_mut();
        assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buf, ptr::null_mut()) }, 0);
        let mut d: turbo_buffer_desc = unsafe { std::mem::zeroed() };
        d.struct_size = size_of::<turbo_buffer_desc>() as u32;
        let mut host = ptr::null_mut();
        unsafe {
            assert_eq!(turbo_buffer_get_desc(buf, &mut d, ptr::null_mut()), 0);
            assert_eq!(turbo_buffer_host_ptr(buf, &mut host, ptr::null_mut()), 0);
        }
        assert_eq!((d.ndim, d.shape, d.bytes), (2, [count as u64, 32], (count * 32 * 4) as u64));
        let in_place = unsafe { std::slice::from_raw_parts(host as *const f32, count * 32) };
        assert_eq!(bits(in_place), bits(&flat), "the buffer is the vectors where they are");
        unsafe { turbo_buffer_release(buf) };
    }
}

// ---- max_ranges and cpu_threads -----------------------------------------------------

#[test]
fn max_ranges_defaults_to_max_batch_and_is_bounded() {
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let l = tiny();
    let (t, lens) = tokens_of(&dir, &TEXTS[1..]);
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
    let (one, _) = tokens_of(&dir, &TEXTS[..1]);
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
    let (t, lens) = tokens_of(&tiny_bundle(), &TEXTS);
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

// ---- A static model -----------------------------------------------------------------

/// The static bundle's table, token weights and token mapping, widened.
struct Table {
    values: Vec<f64>,
    weights: Option<Vec<f64>>,
    mapping: Option<Vec<usize>>,
    dim: usize,
}

impl Table {
    fn read(dir: &Path) -> Table {
        let bytes = fs::read(dir.join("weights/static.safetensors")).unwrap();
        let f = File::parse("static.safetensors", &bytes).unwrap();
        let wide = |name: &str| -> Option<(Vec<f64>, Vec<u64>)> {
            let t = f.tensor(name)?;
            let v = match t.dtype {
                Dtype::F16 => t.data.chunks(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64).collect(),
                Dtype::F32 => t.f32s().into_iter().map(f64::from).collect(),
                Dtype::F64 => t.data.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
                Dtype::I64 => t.data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
                other => panic!("{other:?}"),
            };
            Some((v, t.shape.clone()))
        };
        let (values, shape) = wide("embeddings").unwrap();
        Table {
            values,
            weights: wide("weights").map(|w| w.0),
            mapping: wide("mapping").map(|m| m.0.iter().map(|&v| v as usize).collect()),
            dim: shape[1] as usize,
        }
    }

    /// The vector token `id` contributes: its table row times its weight.
    fn token(&self, id: i32) -> Vec<f64> {
        let id = id as usize;
        let row = self.mapping.as_ref().map_or(id, |m| m[id]);
        let w = self.weights.as_ref().map_or(1.0, |w| w[id]);
        self.values[row * self.dim..(row + 1) * self.dim].iter().map(|v| v * w).collect()
    }
}

/// An F16 value widened; the table of the bundle holds normal numbers and 0.
fn half_to_f32(h: u16) -> f32 {
    let (sign, exp, man) = ((h as u32 & 0x8000) << 16, (h >> 10) & 0x1f, h as u32 & 0x3ff);
    if exp == 0 {
        let v = man as f32 * 2f32.powi(-24);
        return if sign != 0 { -v } else { v };
    }
    f32::from_bits(sign | ((exp as u32 + 112) << 23) | (man << 13))
}

#[test]
fn a_static_models_range_is_the_mean_of_its_tokens_rows() {
    let dir = testdata().join("tiny-static-bundle");
    let table = Table::read(&dir);
    let l = Loaded::load(&dir).unwrap_or_else(|e| panic!("{e:?}"));
    let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
    d.max_ranges = 64;
    let s = Session::create(l.m, Some(&d)).unwrap();
    let texts = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];
    let (t, lens) = tokens_of(&dir, &texts);
    assert!(lens.iter().all(|&n| n >= 1));
    let o = normalized(TURBO_NORMALIZE_NONE);

    s.write_tokens(&t.batch(), Some(&o)).unwrap();
    let rows = s.run().unwrap().rows();
    let whole: Vec<_> = (0..3).map(|r| range(r, 0, t.seq)).collect();
    let got = run_ranges(&s, &t, &whole, &o);
    for (r, (g, w)) in got.iter().zip(&rows).enumerate() {
        assert_eq!(g.len(), table.dim);
        assert!(max_abs_diff(g, w) <= 1e-6, "row {r}");
    }

    // A range of one token is that token's table row, times its weight.
    let mut singles = Vec::new();
    for (r, &n) in lens.iter().enumerate() {
        for p in 0..n {
            singles.push((r, p, range(r as u32, p as u32, p as u32 + 1)));
        }
    }
    assert!(singles.len() <= 64);
    let ranges: Vec<_> = singles.iter().map(|s| s.2).collect();
    let got = run_ranges(&s, &t, &ranges, &o);
    for ((r, p, _), g) in singles.iter().zip(&got) {
        let want = table.token(t.ids[r * t.seq as usize + p]);
        for (a, b) in g.iter().zip(&want) {
            assert!((*a as f64 - b).abs() <= 1e-6, "row {r} token {p}: {a} vs {b}");
        }
    }
}
