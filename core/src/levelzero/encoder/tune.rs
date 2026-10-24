//! The tuner: which kernels run a session's linear layers.

use std::ffi::c_void;

use super::session::*;
use super::*;
use crate::backend::{turbo_backend_embed_rows, turbo_backend_tuning};
use crate::levelzero::gpu::{LOG_INFO, Res, fail, guarded};
use crate::status::INVALID_STATE;
use crate::{
    TURBO_AUTOTUNE_ON, TURBO_AUTOTUNE_RETUNE, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN, TURBO_TUNED_CACHE,
    TURBO_TUNED_DEFAULT, TURBO_TUNED_FORCED, TURBO_TUNED_MEASURED, turbo_error,
};

/// The tuner times a session's linear layers on rows of max_seq tokens,
/// this many tokens in all at most (fewer when the session holds fewer).
pub(super) const TUNE_TOKENS: u32 = 8192;
/// Timed runs of each variant, after an untimed one.
pub(super) const TUNE_RUNS: usize = 5;
/// A variant replaces the incumbent only when this much faster.
pub(super) const TUNE_MARGIN: f64 = 0.05;
/// Times whose median is more than this far above their least are apart:
/// the clocks still coming up, or a device shared or throttling.
pub(super) const TUNE_APART: f64 = 0.25;
/// How many times a variant whose times are apart is timed in all.
pub(super) const TUNE_TRIES: usize = 3;

/// As turbo_backend.h says for session_create_tuned: the session
/// session_create makes, then its linear layers' kernels chosen as
/// docs/levelzero.md says (Kernel choices), and the choice reported.
///
/// # Safety
/// As turbo_backend.h says for session_create_tuned.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn session_create_tuned(
    model: *mut c_void,
    task: u32,
    max_batch: u32,
    max_seq: u32,
    precision: u32,
    tuning: *mut turbo_backend_tuning,
    compute_dtype: *mut u32,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let shape = Shape { task, max_batch, max_seq, precision, max_ranges: max_batch };
    unsafe { create_tuned(model, shape, tuning, compute_dtype, out, err) }
}

/// The session `create` makes, then tuned as session_create_tuned says.
pub(super) unsafe fn create_tuned(
    model: *mut c_void,
    shape: Shape,
    tuning: *mut turbo_backend_tuning,
    compute_dtype: *mut u32,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let rc = unsafe { create(model, shape, compute_dtype, out, err) };
    if rc != 0 || tuning.is_null() {
        return rc;
    }
    let rc = unsafe { guarded(err, || tune(&mut *(*out as *mut Session), &mut *tuning)) };
    if rc != 0 {
        unsafe {
            session_release(*out);
            *out = std::ptr::null_mut();
        }
    }
    rc
}

pub(super) fn write_choices(dst: &mut [std::ffi::c_char; crate::TURBO_CHOICES_LEN], text: &str) {
    dst.fill(0);
    for (d, b) in dst.iter_mut().zip(text.bytes().take(crate::TURBO_CHOICES_LEN - 1)) {
        *d = b as std::ffi::c_char;
    }
}

/// The least and the median of `times`.
pub(super) fn least_and_median(times: &[f64]) -> (f64, f64) {
    let mut t = times.to_vec();
    t.sort_by(f64::total_cmp);
    (t[0], t[t.len() / 2])
}

/// The session's linear layers chosen as the tuning struct asks, with
/// session_create's choice (forced, else the default) as the default.
pub(super) fn tune(s: &mut Session, t: &mut turbo_backend_tuning) -> Res<()> {
    t.tuned = TURBO_TUNED_DEFAULT;
    t.tune_ms = 0;
    t.numerics_used = t.numerics_allowed;
    // The context outlives its sessions.
    let c = unsafe { &*s.ctx };
    let line = |l: Linear, forced: bool| format!("linear={};forced={}", l.name(), if forced { "linear" } else { "" });
    if let Some(l) = forced_linear()? {
        write_choices(&mut t.choices, &line(l, true));
        t.tuned = TURBO_TUNED_FORCED;
        return Ok(());
    }
    // One path: nothing to choose, nothing reported.
    if !onednn_runs(c, s.half.as_ref()) {
        write_choices(&mut t.choices, "");
        return Ok(());
    }
    let default = s.linear;
    let cached = if t.cached.is_null() {
        None
    } else {
        let text = unsafe { std::ffi::CStr::from_ptr(t.cached) }.to_string_lossy();
        parse_linear(&text).ok().flatten()
    };
    match (t.mode, cached) {
        (TURBO_AUTOTUNE_ON, Some(l)) => {
            s.linear = l;
            write_choices(&mut t.choices, &line(l, false));
            t.tuned = TURBO_TUNED_CACHE;
            return Ok(());
        }
        (TURBO_AUTOTUNE_ON | TURBO_AUTOTUNE_RETUNE, _) => {}
        _ => {
            write_choices(&mut t.choices, &line(default, false));
            return Ok(());
        }
    }
    let incumbent = cached.unwrap_or(default);
    let challenger = if incumbent == Linear::Own { Linear::Onednn } else { Linear::Own };
    let started = std::time::Instant::now();
    let measured = measure(s, [incumbent, challenger], t.budget_ms);
    // The session runs the incumbent unless the challenger was timed and
    // beat it by the margin.
    s.linear = incumbent;
    let measured = measured?;
    let tune_ms = started.elapsed().as_millis().min(u32::MAX as u128) as u32;
    let (bin, times) = match measured {
        Measured::Timed(bin, times) => (bin, times),
        Measured::Apart(l) => {
            c.say(
                LOG_INFO,
                &format!(
                    "levelzero device {}: not tuned: linear={}'s times stayed more than {:.0}% apart, so the device \
                     is shared or throttling; the next session measures again",
                    c.ordinal,
                    l.name(),
                    TUNE_APART * 100.0
                ),
            );
            write_choices(&mut t.choices, &line(incumbent, false));
            return Ok(());
        }
    };
    let least = |l: Linear| times.iter().find(|(v, _)| *v == l).map(|(_, ms)| least_and_median(ms).0);
    let (Some(inc), challenged) = (least(incumbent), least(challenger)) else {
        return Err(fail(INVALID_STATE, "levelzero: the tuner timed no incumbent"));
    };
    if let Some(ch) = challenged
        && ch < inc * (1.0 - TUNE_MARGIN)
    {
        s.linear = challenger;
    }
    let lines: Vec<String> =
        times.iter().map(|(l, ms)| format!("{bin}/linear/{}={:.4}", l.name(), least_and_median(ms).0)).collect();
    if !t.timings.is_null() && t.timings_len > 0 {
        let text = lines.join("\n");
        let n = text.len().min(t.timings_len as usize - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(text.as_ptr(), t.timings as *mut u8, n);
            *t.timings.add(n) = 0;
        }
    }
    c.say(
        LOG_INFO,
        &format!(
            "levelzero device {}: linear layers chosen in {tune_ms} ms: {} ({bin}, a layer: {}{})",
            c.ordinal,
            s.linear.name(),
            times
                .iter()
                .map(|(l, ms)| format!("{} {:.3} ms", l.name(), least_and_median(ms).0))
                .collect::<Vec<_>>()
                .join(", "),
            if challenged.is_none() { "; the challenger not timed, past the budget" } else { "" }
        ),
    );
    write_choices(&mut t.choices, &line(s.linear, false));
    t.tune_ms = tune_ms;
    t.tuned = TURBO_TUNED_MEASURED;
    Ok(())
}

pub(super) enum Measured {
    /// The token bin timed at, and each variant timed with its times in
    /// ms; a variant past the budget is left out.
    Timed(&'static str, Vec<(Linear, Vec<f64>)>),
    /// This variant's times stayed apart.
    Apart(Linear),
}

/// Times one layer of the encoder with each variant's linear layers, in
/// order, on rows of max_seq tokens with ids 1 (TUNE_TOKENS in all at
/// most): one untimed run, then TUNE_RUNS timed ones while the budget
/// lasts, which the untimed runs (where a variant's kernels are built)
/// do not count against. A variant whose times are apart is timed again,
/// TUNE_TRIES times in all.
pub(super) fn measure(s: &mut Session, order: [Linear; 2], budget_ms: u32) -> Res<Measured> {
    let d = &s.model().desc;
    let batch = (TUNE_TOKENS / s.max_seq).clamp(1, s.max_batch);
    let n = batch as usize * s.max_seq as usize;
    let (ids, mask) = (vec![1i32; n], vec![1i32; n]);
    let rows = turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch,
        seq: s.max_seq,
        row_stride: s.max_seq,
        ids: ids.as_ptr(),
        mask: mask.as_ptr(),
        types: std::ptr::null(),
        pooling: TURBO_POOLING_MEAN,
        normalize: TURBO_NORMALIZE_NONE,
        output_dim: d.hidden,
        head: 0,
    };
    let (tokens, longest, _) = unsafe { s.pack(&rows)? };
    s.has_types = false;
    s.batch = batch;
    s.ranges = 0;
    s.tokens = tokens;
    s.longest = longest;
    s.pooling = rows.pooling;
    s.normalize = rows.normalize;
    s.output_dim = rows.output_dim;
    s.written = false;
    let bin = ["le256", "le1k", "le4k", "le16k", "gt16k"][crate::tuning::tcap_bin(tokens) as usize];
    let budget = std::time::Duration::from_millis(budget_ms as u64);
    let mut spent = std::time::Duration::ZERO;
    let mut timed = Vec::new();
    for l in order {
        s.linear = l;
        s.encode_and_wait(1)?;
        let mut tries = 0;
        let times = loop {
            let mut times = Vec::with_capacity(TUNE_RUNS);
            while times.len() < TUNE_RUNS && spent < budget {
                let t0 = std::time::Instant::now();
                s.encode_and_wait(1)?;
                let e = t0.elapsed();
                spent += e;
                times.push(e.as_secs_f64() * 1e3);
            }
            tries += 1;
            if times.len() < 2 {
                break times;
            }
            let (least, median) = least_and_median(&times);
            if median <= least * (1.0 + TUNE_APART) {
                break times;
            }
            if tries == TUNE_TRIES {
                return Ok(Measured::Apart(l));
            }
        };
        if !times.is_empty() {
            timed.push((l, times));
        }
    }
    Ok(Measured::Timed(bin, timed))
}
