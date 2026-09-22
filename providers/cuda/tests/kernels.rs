//! Device-kernel correctness for the CUDA provider (`src/kernels.cu`).
//!
//! Every kernel is run on random inputs through `cuda::DeviceMem`,
//! `PinnedMem`, and `Stream`, and compared against a sequential f32 CPU
//! reference computed in the same order the kernel accumulates. Argument
//! validation is checked to reject before any launch: the destination keeps
//! its sentinel and the next valid launch still succeeds, which it would not
//! if the rejected call had left a sticky CUDA error behind.
//!
//! Each test prints a reason and returns when the machine has no CUDA
//! device, so the file is safe to run anywhere.

use std::ffi::c_void;

use turbo_provider_cuda::cuda::{self, DeviceMem, PinnedMem, Stream};

/// Sentinel written into an output buffer before a rejected launch.
const SENTINEL: f32 = -12345.0;

/// A stream on device 0, or `None` with a printed reason when the CUDA
/// runtime reports no usable device.
fn stream() -> Option<Stream> {
    match cuda::devices() {
        Ok(devices) if devices.is_empty() => {
            eprintln!("skipping: the CUDA runtime reports no devices");
            None
        }
        Ok(devices) => Some(Stream::new(devices[0].index).expect("create a stream on CUDA device 0")),
        Err(e) => {
            eprintln!("skipping: the CUDA runtime is unavailable: {e}");
            None
        }
    }
}

/// Deterministic xorshift generator, so every run checks the same numbers.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 8_388_608.0 - 1.0
    }

    /// Uniform in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Copy `src` to a fresh device allocation through pinned staging and wait
/// for the copy, so the staging buffer may be dropped on return.
fn upload<T: Copy>(src: &[T], stream: &Stream) -> DeviceMem {
    let bytes = std::mem::size_of_val(src);
    let staging = PinnedMem::new(bytes).expect("pinned staging for the upload");
    // SAFETY: staging is `bytes` long and src holds exactly `bytes` bytes.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), staging.ptr(), bytes) };
    let device = DeviceMem::new(stream.device(), bytes).expect("device allocation for the upload");
    // SAFETY: staging outlives the copy because the stream is synchronized below.
    unsafe { cuda::copy_h2d(&device, staging.ptr(), bytes, stream) }.expect("H2D copy");
    stream.synchronize().expect("synchronize the upload");
    device
}

/// Read `len` floats back from a device allocation.
fn download(src: &DeviceMem, len: usize, stream: &Stream) -> Vec<f32> {
    let bytes = len * std::mem::size_of::<f32>();
    let staging = PinnedMem::new(bytes).expect("pinned staging for the download");
    // SAFETY: src holds at least `bytes` bytes and staging is `bytes` long.
    unsafe { cuda::copy_d2h(staging.ptr(), src.ptr(), bytes, stream) }.expect("D2H copy");
    stream.synchronize().expect("synchronize the download");
    let mut out = vec![0f32; len];
    // SAFETY: staging holds `len` f32 written by the copy above.
    unsafe { std::ptr::copy_nonoverlapping(staging.ptr().cast::<f32>(), out.as_mut_ptr(), len) };
    out
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length {} but the reference has {}", got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol, "{what}: element {i} is {g} but the CPU reference says {w} (tolerance {tol})");
    }
}

/// `[batch, seq, dim]` hidden states, uniform in `[-1, 1)`.
fn hidden_states(batch: usize, seq: usize, dim: usize, rng: &mut Rng) -> Vec<f32> {
    (0..batch * seq * dim).map(|_| rng.unit()).collect()
}

/// Masks covering the shapes the kernel must handle: a fully live row, a row
/// with a single live token, a fully masked row, and random prefixes.
fn masks(batch: usize, seq: usize, rng: &mut Rng) -> Vec<i32> {
    assert!(batch >= 3, "the mask fixture needs at least three rows");
    let mut mask = vec![0i32; batch * seq];
    for b in 0..batch {
        let live = match b {
            0 => seq,
            1 => 1,
            2 => 0,
            _ => 1 + rng.below(seq),
        };
        for c in 0..live {
            mask[b * seq + c] = 1;
        }
    }
    mask
}

/// Pooling problem shape: `[batch, seq, dim]` in, `[batch, out_dim]` out.
#[derive(Clone, Copy)]
struct Shape {
    batch: usize,
    seq: usize,
    dim: usize,
    out_dim: usize,
}

impl Shape {
    /// A shape that keeps every dimension (`out_dim == dim`).
    fn full(batch: usize, seq: usize, dim: usize) -> Self {
        Self { batch, seq, dim, out_dim: dim }
    }
}

/// Sequential f32 reference for `turbo_cuda_pool`: pool, truncate to
/// `out_dim`, then L2-normalize, in that order.
fn pool_reference(hidden: &[f32], mask: &[i32], shape: Shape, mode: i32, normalize: bool) -> Vec<f32> {
    let Shape { batch, seq, dim, out_dim } = shape;
    let mut out = vec![0f32; batch * out_dim];
    for b in 0..batch {
        let row = &mask[b * seq..(b + 1) * seq];
        let count = row.iter().filter(|&&m| m != 0).count();
        let last = row.iter().rposition(|&m| m != 0).unwrap_or(0);
        let dst = &mut out[b * out_dim..(b + 1) * out_dim];
        for (d, o) in dst.iter_mut().enumerate() {
            *o = match mode {
                0 if count == 0 => 0.0,
                0 => {
                    let mut acc = 0f32;
                    for (s, &m) in row.iter().enumerate() {
                        if m != 0 {
                            acc += hidden[(b * seq + s) * dim + d];
                        }
                    }
                    acc / count as f32
                }
                1 => hidden[b * seq * dim + d],
                _ => hidden[(b * seq + last) * dim + d],
            };
        }
        if normalize {
            let norm = dst.iter().map(|v| v * v).sum::<f32>().sqrt();
            let inv = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
            for v in dst.iter_mut() {
                *v *= inv;
            }
        }
    }
    out
}

/// Pool on the device with an i32 or an i64 mask and return the result rows.
fn pool_on_device(
    stream: &Stream,
    hidden: &[f32],
    mask: &[i32],
    shape: Shape,
    mode: i32,
    normalize: bool,
    mask_width: i32,
) -> Vec<f32> {
    let Shape { batch, seq, dim, out_dim } = shape;
    let d_hidden = upload(hidden, stream);
    let d_mask = match mask_width {
        4 => upload(mask, stream),
        8 => upload(&mask.iter().map(|&m| m as i64).collect::<Vec<i64>>(), stream),
        other => panic!("mask width {other} is not 4 or 8"),
    };
    let d_out = upload(&vec![SENTINEL; batch * out_dim], stream);
    // SAFETY: every pointer is a live device allocation of the stated extent.
    let rc = unsafe {
        cuda::turbo_cuda_pool(
            d_hidden.ptr().cast::<f32>(),
            d_mask.ptr().cast_const(),
            mask_width,
            d_out.ptr().cast::<f32>(),
            batch as i32,
            seq as i32,
            dim as i32,
            out_dim as i32,
            mode,
            normalize as i32,
            stream.raw(),
        )
    };
    assert_eq!(rc, 0, "turbo_cuda_pool(mode {mode}, mask width {mask_width}) returned {rc}");
    stream.synchronize().expect("synchronize the pooling kernel");
    download(&d_out, batch * out_dim, stream)
}

#[test]
fn pool_mean_matches_the_cpu_reference_for_i32_and_i64_masks() {
    let Some(stream) = stream() else { return };
    let (batch, seq, dim) = (6, 17, 40);
    let mut rng = Rng::new(0x5eed_1234);
    let hidden = hidden_states(batch, seq, dim, &mut rng);
    let mask = masks(batch, seq, &mut rng);
    let want = pool_reference(&hidden, &mask, Shape::full(batch, seq, dim), 0, false);
    for width in [4, 8] {
        let got = pool_on_device(&stream, &hidden, &mask, Shape::full(batch, seq, dim), 0, false, width);
        assert_close(&got, &want, 1e-5, &format!("mean pooling with a {}-bit mask", width * 8));
    }
}

#[test]
fn pool_mean_of_a_fully_masked_row_is_zero() {
    let Some(stream) = stream() else { return };
    let (batch, seq, dim) = (4, 11, 32);
    let mut rng = Rng::new(0xdead_beef);
    let hidden = hidden_states(batch, seq, dim, &mut rng);
    let mask = masks(batch, seq, &mut rng);
    assert!(mask[2 * seq..3 * seq].iter().all(|&m| m == 0), "row 2 of the fixture is the fully masked row");
    for normalize in [false, true] {
        for width in [4, 8] {
            let got = pool_on_device(&stream, &hidden, &mask, Shape::full(batch, seq, dim), 0, normalize, width);
            let row = &got[2 * dim..3 * dim];
            assert!(
                row.iter().all(|&v| v == 0.0),
                "a fully masked row must mean-pool to zeros (normalize {normalize}, mask width {width}), got {row:?}"
            );
        }
    }
}

#[test]
fn pool_mean_of_a_single_live_token_is_that_token() {
    let Some(stream) = stream() else { return };
    let (batch, seq, dim) = (4, 11, 32);
    let mut rng = Rng::new(0x0123_4567);
    let hidden = hidden_states(batch, seq, dim, &mut rng);
    let mask = masks(batch, seq, &mut rng);
    assert_eq!(mask[seq..2 * seq].iter().sum::<i32>(), 1, "row 1 of the fixture has a single live token");
    let got = pool_on_device(&stream, &hidden, &mask, Shape::full(batch, seq, dim), 0, false, 4);
    assert_close(&got[dim..2 * dim], &hidden[seq * dim..seq * dim + dim], 0.0, "mean over one live token");
}

#[test]
fn pool_cls_reads_the_first_column_and_last_reads_the_final_live_token() {
    let Some(stream) = stream() else { return };
    let (batch, seq, dim) = (5, 13, 48);
    let mut rng = Rng::new(0xabcd_ef01);
    let hidden = hidden_states(batch, seq, dim, &mut rng);
    let mask = masks(batch, seq, &mut rng);
    for (mode, name) in [(1, "cls"), (2, "last")] {
        for width in [4, 8] {
            let want = pool_reference(&hidden, &mask, Shape::full(batch, seq, dim), mode, false);
            let got = pool_on_device(&stream, &hidden, &mask, Shape::full(batch, seq, dim), mode, false, width);
            assert_close(&got, &want, 0.0, &format!("{name} pooling with a {}-bit mask", width * 8));
        }
    }
    // The reference and the kernel must both name the same token, not just
    // agree with each other: row 3 has a random prefix length.
    let live = mask[3 * seq..4 * seq].iter().filter(|&&m| m != 0).count();
    let got = pool_on_device(&stream, &hidden, &mask, Shape::full(batch, seq, dim), 2, false, 4);
    let want = &hidden[(3 * seq + live - 1) * dim..(3 * seq + live) * dim];
    assert_close(&got[3 * dim..4 * dim], want, 0.0, "last pooling picks the final live token");
}

#[test]
fn pool_l2_normalization_yields_unit_rows_for_every_live_row() {
    let Some(stream) = stream() else { return };
    let (batch, seq, dim) = (6, 19, 64);
    let mut rng = Rng::new(0x7777_1111);
    let hidden = hidden_states(batch, seq, dim, &mut rng);
    let mask = masks(batch, seq, &mut rng);
    for mode in [0, 1, 2] {
        let want = pool_reference(&hidden, &mask, Shape::full(batch, seq, dim), mode, true);
        let got = pool_on_device(&stream, &hidden, &mask, Shape::full(batch, seq, dim), mode, true, 4);
        assert_close(&got, &want, 1e-5, &format!("L2-normalized pooling in mode {mode}"));
        for b in 0..batch {
            let row = &got[b * dim..(b + 1) * dim];
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            let expected = if mode == 0 && mask[b * seq..(b + 1) * seq].iter().all(|&m| m == 0) { 0.0 } else { 1.0 };
            assert!((norm - expected).abs() < 1e-4, "mode {mode} row {b}: L2 norm is {norm}, expected {expected}");
        }
    }
}

#[test]
fn pool_truncates_to_out_dim_before_normalizing() {
    let Some(stream) = stream() else { return };
    let (batch, seq, dim, out_dim) = (4, 9, 40, 12);
    let mut rng = Rng::new(0x1357_9bdf);
    let hidden = hidden_states(batch, seq, dim, &mut rng);
    let mask = masks(batch, seq, &mut rng);
    let want = pool_reference(&hidden, &mask, Shape { batch, seq, dim, out_dim }, 0, true);
    let got = pool_on_device(&stream, &hidden, &mask, Shape { batch, seq, dim, out_dim }, 0, true, 4);
    assert_eq!(got.len(), batch * out_dim, "the result is [batch, out_dim], not [batch, dim]");
    assert_close(&got, &want, 1e-5, "Matryoshka truncation followed by L2");
    // Truncating first is what makes the short row a unit vector; the first
    // out_dim components of the full-width normalized row are not.
    let full = pool_reference(&hidden, &mask, Shape::full(batch, seq, dim), 0, true);
    for b in 0..batch {
        if mask[b * seq..(b + 1) * seq].iter().all(|&m| m == 0) {
            continue;
        }
        let norm = got[b * out_dim..(b + 1) * out_dim].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "row {b}: truncated row has L2 norm {norm}, expected 1");
        let prefix = full[b * dim..b * dim + out_dim].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(prefix < 0.999, "row {b}: the prefix of a full-width unit row has norm {prefix}, so the order matters");
    }
    // Truncation without normalization is a plain prefix of the full result.
    let plain = pool_on_device(&stream, &hidden, &mask, Shape { batch, seq, dim, out_dim }, 0, false, 4);
    let plain_full = pool_reference(&hidden, &mask, Shape::full(batch, seq, dim), 0, false);
    for b in 0..batch {
        assert_close(
            &plain[b * out_dim..(b + 1) * out_dim],
            &plain_full[b * dim..b * dim + out_dim],
            1e-5,
            &format!("row {b}: unnormalized truncation is a prefix"),
        );
    }
}

/// One `turbo_cuda_pool` call, so a table of rejected calls reads as a list
/// of the single argument each one gets wrong.
#[derive(Clone, Copy)]
struct PoolCall {
    what: &'static str,
    hidden: *const f32,
    mask: *const c_void,
    out: *mut f32,
    mask_width: i32,
    batch: i32,
    seq: i32,
    dim: i32,
    out_dim: i32,
    mode: i32,
}

#[test]
fn pool_rejects_invalid_arguments_without_launching() {
    let Some(stream) = stream() else { return };
    let (batch, seq, dim) = (3, 5, 8);
    let mut rng = Rng::new(0x2468_ace0);
    let hidden = hidden_states(batch, seq, dim, &mut rng);
    let mask = masks(batch, seq, &mut rng);
    let d_hidden = upload(&hidden, &stream);
    let d_mask = upload(&mask, &stream);
    let d_out = upload(&vec![SENTINEL; batch * dim], &stream);
    let h = d_hidden.ptr().cast::<f32>();
    let m = d_mask.ptr().cast_const();
    let o = d_out.ptr().cast::<f32>();
    let valid = PoolCall {
        what: "",
        hidden: h,
        mask: m,
        out: o,
        mask_width: 4,
        batch: batch as i32,
        seq: seq as i32,
        dim: dim as i32,
        out_dim: dim as i32,
        mode: 0,
    };
    let cases = [
        PoolCall { what: "null hidden", hidden: std::ptr::null(), ..valid },
        PoolCall { what: "null mask", mask: std::ptr::null(), ..valid },
        PoolCall { what: "null out", out: std::ptr::null_mut(), ..valid },
        PoolCall { what: "out_dim above dim", out_dim: dim as i32 + 1, ..valid },
        PoolCall { what: "zero out_dim", out_dim: 0, ..valid },
        PoolCall { what: "mode above last", mode: 3, ..valid },
        PoolCall { what: "negative mode", mode: -1, ..valid },
        PoolCall { what: "mask width 2", mask_width: 2, ..valid },
        PoolCall { what: "zero batch", batch: 0, ..valid },
        PoolCall { what: "zero seq", seq: 0, ..valid },
    ];
    for c in cases {
        let what = c.what;
        // SAFETY: the kernel validates its arguments before dereferencing them.
        let rc = unsafe {
            cuda::turbo_cuda_pool(
                c.hidden,
                c.mask,
                c.mask_width,
                c.out,
                c.batch,
                c.seq,
                c.dim,
                c.out_dim,
                c.mode,
                0,
                stream.raw(),
            )
        };
        assert_ne!(rc, 0, "turbo_cuda_pool must reject {what}");
        stream.synchronize().expect("the stream stays healthy after a rejected call");
        let out = download(&d_out, batch * dim, &stream);
        assert!(out.iter().all(|&v| v == SENTINEL), "a rejected call ({what}) must not launch and write the output");
    }
    // A rejected call leaves no sticky error behind.
    let want = pool_reference(&hidden, &mask, Shape::full(batch, seq, dim), 0, false);
    let got = pool_on_device(&stream, &hidden, &mask, Shape::full(batch, seq, dim), 0, false, 4);
    assert_close(&got, &want, 1e-5, "mean pooling after ten rejected calls");
}

#[test]
fn sigmoid_matches_the_cpu_reference() {
    let Some(stream) = stream() else { return };
    let n = 1000;
    let mut rng = Rng::new(0xfeed_face);
    let input: Vec<f32> = (0..n).map(|_| rng.unit() * 8.0).collect();
    let d_in = upload(&input, &stream);
    let d_out = upload(&vec![SENTINEL; n], &stream);
    // SAFETY: both allocations hold n f32.
    let rc = unsafe { cuda::turbo_cuda_sigmoid(d_in.ptr().cast(), d_out.ptr().cast(), n as i32, stream.raw()) };
    assert_eq!(rc, 0, "turbo_cuda_sigmoid returned {rc}");
    stream.synchronize().expect("synchronize the sigmoid kernel");
    let got = download(&d_out, n, &stream);
    let want: Vec<f32> = input.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    assert_close(&got, &want, 1e-6, "sigmoid");
    assert!(got.iter().all(|&v| (0.0..=1.0).contains(&v)), "sigmoid outputs are probabilities");
}

#[test]
fn sigmoid_rejects_invalid_arguments_without_launching() {
    let Some(stream) = stream() else { return };
    let n = 16;
    let d_in = upload(&vec![0.5f32; n], &stream);
    let d_out = upload(&vec![SENTINEL; n], &stream);
    let cases: [(&str, *const f32, *mut f32, i32); 4] = [
        ("null input", std::ptr::null(), d_out.ptr().cast(), n as i32),
        ("null output", d_in.ptr().cast(), std::ptr::null_mut(), n as i32),
        ("zero length", d_in.ptr().cast(), d_out.ptr().cast(), 0),
        ("negative length", d_in.ptr().cast(), d_out.ptr().cast(), -1),
    ];
    for (what, input, output, len) in cases {
        // SAFETY: the kernel validates its arguments before dereferencing them.
        let rc = unsafe { cuda::turbo_cuda_sigmoid(input, output, len, stream.raw()) };
        assert_ne!(rc, 0, "turbo_cuda_sigmoid must reject {what}");
        stream.synchronize().expect("the stream stays healthy after a rejected call");
        let out = download(&d_out, n, &stream);
        assert!(out.iter().all(|&v| v == SENTINEL), "a rejected call ({what}) must not launch and write the output");
    }
    // SAFETY: both allocations hold n f32.
    let rc = unsafe { cuda::turbo_cuda_sigmoid(d_in.ptr().cast(), d_out.ptr().cast(), n as i32, stream.raw()) };
    assert_eq!(rc, 0, "a valid launch still succeeds after rejected calls");
    stream.synchronize().expect("synchronize the sigmoid kernel");
}

/// Row softmax on the device for `rows` rows of `width` logits.
fn softmax_on_device(stream: &Stream, input: &[f32], rows: usize, width: usize) -> Vec<f32> {
    let d_in = upload(input, stream);
    let d_out = upload(&vec![SENTINEL; rows * width], stream);
    // SAFETY: both allocations hold rows*width f32.
    let rc = unsafe {
        cuda::turbo_cuda_softmax_rows(d_in.ptr().cast(), d_out.ptr().cast(), rows as i32, width as i32, stream.raw())
    };
    assert_eq!(rc, 0, "turbo_cuda_softmax_rows(rows {rows}, width {width}) returned {rc}");
    stream.synchronize().expect("synchronize the softmax kernel");
    download(&d_out, rows * width, stream)
}

#[test]
fn softmax_rows_sum_to_one_and_preserve_the_argmax() {
    let Some(stream) = stream() else { return };
    let rows = 5;
    for width in [2usize, 9, 300, 4096] {
        let mut rng = Rng::new(0x9e37_79b9 ^ width as u64);
        let input: Vec<f32> = (0..rows * width).map(|_| rng.unit() * 12.0).collect();
        let got = softmax_on_device(&stream, &input, rows, width);
        for r in 0..rows {
            let x = &input[r * width..(r + 1) * width];
            let y = &got[r * width..(r + 1) * width];
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let want: Vec<f32> = exps.iter().map(|e| e / denom).collect();
            assert_close(y, &want, 1e-6, &format!("softmax width {width} row {r}"));
            let sum: f32 = y.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "softmax width {width} row {r} sums to {sum}, not 1");
            let arg_in =
                x.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
            let arg_out =
                y.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
            assert_eq!(arg_out.0, arg_in.0, "softmax width {width} row {r} moved the argmax");
            assert!(y.iter().all(|&v| (0.0..=1.0).contains(&v)), "softmax width {width} row {r} is not a distribution");
        }
    }
}

#[test]
fn softmax_rejects_a_width_above_the_block_limit() {
    let Some(stream) = stream() else { return };
    let (rows, width) = (2usize, 4097usize);
    let d_in = upload(&vec![1.0f32; rows * width], &stream);
    let d_out = upload(&vec![SENTINEL; rows * width], &stream);
    // SAFETY: both allocations hold rows*width f32; the kernel rejects the width.
    let rc = unsafe {
        cuda::turbo_cuda_softmax_rows(d_in.ptr().cast(), d_out.ptr().cast(), rows as i32, width as i32, stream.raw())
    };
    assert_ne!(rc, 0, "a row wider than 4096 must be rejected, not silently truncated");
    stream.synchronize().expect("the stream stays healthy after a rejected call");
    let out = download(&d_out, rows * width, &stream);
    assert!(out.iter().all(|&v| v == SENTINEL), "the rejected softmax must not launch and write the output");
    // 4096 is the last accepted width, and a launch after the rejection works.
    let got = softmax_on_device(&stream, &vec![0.25f32; rows * 4096], rows, 4096);
    for r in 0..rows {
        let sum: f32 = got[r * 4096..(r + 1) * 4096].iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "width 4096 row {r} sums to {sum}, not 1");
    }
}

#[test]
fn softmax_rejects_invalid_arguments_without_launching() {
    let Some(stream) = stream() else { return };
    let (rows, width) = (2usize, 8usize);
    let n = rows * width;
    let d_in = upload(&vec![0.5f32; n], &stream);
    let d_out = upload(&vec![SENTINEL; n], &stream);
    let cases: [(&str, *const f32, *mut f32, i32, i32); 5] = [
        ("null input", std::ptr::null(), d_out.ptr().cast(), rows as i32, width as i32),
        ("null output", d_in.ptr().cast(), std::ptr::null_mut(), rows as i32, width as i32),
        ("zero rows", d_in.ptr().cast(), d_out.ptr().cast(), 0, width as i32),
        ("zero width", d_in.ptr().cast(), d_out.ptr().cast(), rows as i32, 0),
        ("negative width", d_in.ptr().cast(), d_out.ptr().cast(), rows as i32, -1),
    ];
    for (what, input, output, r, w) in cases {
        // SAFETY: the kernel validates its arguments before dereferencing them.
        let rc = unsafe { cuda::turbo_cuda_softmax_rows(input, output, r, w, stream.raw()) };
        assert_ne!(rc, 0, "turbo_cuda_softmax_rows must reject {what}");
        stream.synchronize().expect("the stream stays healthy after a rejected call");
        let out = download(&d_out, n, &stream);
        assert!(out.iter().all(|&v| v == SENTINEL), "a rejected call ({what}) must not launch and write the output");
    }
}
