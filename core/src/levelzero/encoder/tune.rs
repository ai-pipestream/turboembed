//! The tuner: which kernels run a session's linear layers.

use std::ffi::c_void;

use super::linears::*;
use super::session::*;
use super::*;
use crate::backend::{turbo_backend_embed_rows, turbo_backend_tuning};
use crate::levelzero::gpu::{LOG_INFO, Res, fail, guarded};
use crate::status::INVALID_STATE;
use crate::tuning::tcap_bin;
use crate::{
    TURBO_AUTOTUNE_ON, TURBO_AUTOTUNE_RETUNE, TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN, TURBO_TUNED_CACHE,
    TURBO_TUNED_DEFAULT, TURBO_TUNED_FORCED, TURBO_TUNED_MEASURED, turbo_error,
};

/// The packed-token counts the tuner times at, smallest first: one in
/// each bin up to le16k. A session that holds fewer is timed at its
/// capacity as well, the largest count it runs.
pub(super) const TUNE_COUNTS: [u32; 4] = [256, 1024, 4096, 8192];
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

/// The session's linear layers chosen as the tuning struct asks, per
/// token bin, with session_create's choice (forced, else the default) as
/// the default.
pub(super) fn tune(s: &mut Session, t: &mut turbo_backend_tuning) -> Res<()> {
    t.tuned = TURBO_TUNED_DEFAULT;
    t.tune_ms = 0;
    t.numerics_used = t.numerics_allowed;
    // The context outlives its sessions.
    let c = unsafe { &*s.ctx };
    let top = tcap_bin(s.max_batch.saturating_mul(s.max_seq));
    // session_create took what the environment forces.
    if forced_linear()?.is_some() {
        write_choices(&mut t.choices, &s.linear.line(top, true));
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
        parse_linears(&text).ok().flatten().map(|p| p.or(default))
    };
    match (t.mode, cached) {
        (TURBO_AUTOTUNE_ON, Some(l)) => {
            s.linear = l;
            write_choices(&mut t.choices, &l.line(top, false));
            t.tuned = TURBO_TUNED_CACHE;
            return Ok(());
        }
        (TURBO_AUTOTUNE_ON | TURBO_AUTOTUNE_RETUNE, _) => {}
        _ => {
            write_choices(&mut t.choices, &default.line(top, false));
            return Ok(());
        }
    }
    let incumbent = cached.unwrap_or(default);
    let started = std::time::Instant::now();
    let measured = measure(s, incumbent, t.budget_ms);
    // The session runs the incumbent at each bin unless the challenger was
    // timed there and beat it by the margin.
    s.linear = incumbent;
    let measured = measured?;
    let tune_ms = started.elapsed().as_millis().min(u32::MAX as u128) as u32;
    let (counts, cut) = match measured {
        Measured::Timed(counts, cut) => (counts, cut),
        Measured::Apart(bin, l) => {
            c.say(
                LOG_INFO,
                &format!(
                    "levelzero device {}: not tuned: linear={}'s times at {bin} stayed more than {:.0}% apart, so \
                     the device is shared or throttling; the next session measures again",
                    c.ordinal,
                    l.name(),
                    TUNE_APART * 100.0
                ),
            );
            write_choices(&mut t.choices, &incumbent.line(top, false));
            return Ok(());
        }
    };
    let mut chosen = incumbent;
    for n in &counts {
        let inc = incumbent.0[n.bin];
        let Some(least) = n.least(inc) else {
            return Err(fail(INVALID_STATE, "levelzero: the tuner timed no incumbent"));
        };
        if let Some(ch) = n.least(inc.other())
            && ch < least * (1.0 - TUNE_MARGIN)
        {
            chosen.0[n.bin] = inc.other();
        }
    }
    // The tuner stops short of gt16k, which runs as le16k does.
    if counts.iter().any(|n| n.bin == 3) {
        chosen.0[4] = chosen.0[3];
    }
    s.linear = chosen;
    let lines: Vec<String> = counts
        .iter()
        .flat_map(|n| {
            n.times.iter().map(|(l, ms)| format!("{}/linear/{}={:.4}", BINS[n.bin], l.name(), least_and_median(ms).0))
        })
        .collect();
    if !t.timings.is_null() && t.timings_len > 0 {
        let text = lines.join("\n");
        let n = text.len().min(t.timings_len as usize - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(text.as_ptr(), t.timings as *mut u8, n);
            *t.timings.add(n) = 0;
        }
    }
    let line = chosen.line(top, false);
    let each: Vec<String> = counts
        .iter()
        .map(|n| {
            let times: Vec<String> =
                n.times.iter().map(|(l, ms)| format!("{} {:.3} ms", l.name(), least_and_median(ms).0)).collect();
            format!("{}: {}", BINS[n.bin], times.join(", "))
        })
        .collect();
    c.say(
        LOG_INFO,
        &format!(
            "levelzero device {}: linear layers chosen in {tune_ms} ms: {line} (a layer at {}{})",
            c.ordinal,
            each.join("; "),
            if cut { "; past the budget, what was left was not timed" } else { "" }
        ),
    );
    write_choices(&mut t.choices, &line);
    t.tune_ms = tune_ms;
    t.tuned = TURBO_TUNED_MEASURED;
    Ok(())
}

/// One count's times: its token bin, and each variant timed with its times
/// in ms, the incumbent first.
pub(super) struct Count {
    pub(super) bin: usize,
    pub(super) times: Vec<(Linear, Vec<f64>)>,
}

impl Count {
    fn least(&self, l: Linear) -> Option<f64> {
        self.times.iter().find(|(v, _)| *v == l).map(|(_, ms)| least_and_median(ms).0)
    }
}

pub(super) enum Measured {
    /// Each count whose incumbent was timed, smallest first, and whether
    /// the budget left a count or a challenger untimed.
    Timed(Vec<Count>, bool),
    /// This variant's times at this bin stayed apart.
    Apart(&'static str, Linear),
}

/// Times one layer of the encoder at each of TUNE_COUNTS packed tokens the
/// session holds, and at its capacity when that is past the last of them:
/// rows of max_seq tokens or fewer, ids 1. At each, the bin's incumbent
/// then the challenger run once untimed, then TUNE_RUNS timed while the
/// budget lasts, which the untimed runs (where a variant's kernels are
/// built) do not count against; past it, no variant runs at all. A variant whose times are apart is timed
/// again, TUNE_TRIES times in all.
pub(super) fn measure(s: &mut Session, incumbent: Linears, budget_ms: u32) -> Res<Measured> {
    let cap = s.max_batch.saturating_mul(s.max_seq).min(TUNE_COUNTS[3]);
    let mut counts: Vec<u32> = TUNE_COUNTS.into_iter().filter(|&n| n <= cap).collect();
    if counts.last().is_none_or(|&n| n < cap) {
        counts.push(cap);
    }
    let budget = std::time::Duration::from_millis(budget_ms as u64);
    let mut spent = std::time::Duration::ZERO;
    let (mut timed, mut cut) = (Vec::new(), false);
    for n in counts {
        let bin = tcap_bin(write_tuning_rows(s, n)?) as usize;
        let inc = incumbent.0[bin];
        let mut count = Count { bin, times: Vec::new() };
        for l in [inc, inc.other()] {
            // Past the budget nothing more is built or timed.
            if spent >= budget {
                cut = true;
                continue;
            }
            s.linear = Linears::all(l);
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
                    return Ok(Measured::Apart(BINS[bin], l));
                }
            };
            if times.is_empty() {
                cut = true;
            } else {
                count.times.push((l, times));
            }
        }
        if count.times.first().is_some_and(|(l, _)| *l == inc) {
            timed.push(count);
        }
    }
    Ok(Measured::Timed(timed, cut))
}

/// Writes rows of `n` packed tokens in all, ids 1, into the session's
/// staging as embed_write would: rows of max_seq tokens, the last holding
/// what is left. Returns the tokens packed.
fn write_tuning_rows(s: &mut Session, n: u32) -> Res<u32> {
    let hidden = s.model().desc.hidden;
    let seq = s.max_seq.min(n);
    let batch = n.div_ceil(seq);
    let len = batch as usize * seq as usize;
    let ids = vec![1i32; len];
    let mut mask = vec![1i32; len];
    mask[(n as usize)..].fill(0);
    let rows = turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch,
        seq,
        row_stride: seq,
        ids: ids.as_ptr(),
        mask: mask.as_ptr(),
        types: std::ptr::null(),
        pooling: TURBO_POOLING_MEAN,
        normalize: TURBO_NORMALIZE_NONE,
        output_dim: hidden,
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
    Ok(tokens)
}
