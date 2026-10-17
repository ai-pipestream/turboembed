use crate::measure::percentile;

/// The headers TEI 1.8.3 answers /embed with that time the request, as
/// router/src/lib.rs names them (`impl From<ResponseMetadata> for
/// HeaderMap`), each whole milliseconds. x-total-time runs from the
/// handler's start, the request body already parsed, to the response
/// headers being built, so it leaves out HTTP, parsing the body and
/// writing the JSON. The others are, for a request of several inputs,
/// the mean over its inputs (router/src/http/server.rs, embed) of: the
/// time from the input's own start to its entering the queue, waiting
/// behind the request's other inputs included (x-tokenization-time); its
/// time in the queue (x-queue-time); and the duration of the backend
/// batch it ran in (x-inference-time), core/src/infer.rs.
pub const TIMING_HEADERS: [&str; 4] = ["x-total-time", "x-tokenization-time", "x-queue-time", "x-inference-time"];

/// The histograms TEI's batcher records each time it polls its queue
/// (core/src/queue.rs, as of v1.8.3): the inputs and the tokens of the
/// batch it takes, both 0 when the queue was empty and it takes none. So
/// each drain of the queue ends with an empty poll or two. Their buckets
/// are powers of two, from 1 to 4096 inputs and from 1 to 2^20 tokens
/// (router/src/prometheus.rs).
pub const BATCH_SIZE_METRIC: &str = "te_batch_next_size";
pub const BATCH_TOKENS_METRIC: &str = "te_batch_next_tokens";

/// One answer's TIMING_HEADERS, in their order, in milliseconds; None
/// where the header is missing or is not a whole number.
pub fn timing_headers(headers: &ureq::http::HeaderMap) -> [Option<u64>; 4] {
    TIMING_HEADERS.map(|h| headers.get(h).and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse().ok()))
}

/// The timed requests' round trip and TEI's headers, as the procedure
/// gives them: each header's p50 and p99 over the requests, or that TEI
/// did not send it with every answer.
pub fn timing_text(round_trip_sorted_ms: &[f64], headers: &[[Option<u64>; 4]]) -> String {
    let mut out = format!(
        "round trip (measured) p50 {:.3} ms, p99 {:.3} ms; TEI's own headers, whole ms, over the same requests:",
        percentile(round_trip_sorted_ms, 50.0),
        percentile(round_trip_sorted_ms, 99.0)
    );
    for (i, name) in TIMING_HEADERS.iter().enumerate() {
        let got: Option<Vec<f64>> = headers.iter().map(|h| h[i].map(|v| v as f64)).collect();
        let sep = if i == 0 { " " } else { ", " };
        match got {
            Some(mut v) if !v.is_empty() => {
                v.sort_by(f64::total_cmp);
                out += &format!("{sep}{name} p50 {} p99 {}", percentile(&v, 50.0), percentile(&v, 99.0));
            }
            _ => out += &format!("{sep}{name} not sent"),
        }
    }
    out
}

/// The p50 of x-total-time, in milliseconds, as timing_text wrote it
/// into a procedure; None when it is not there.
pub fn total_time_p50(procedure: &str) -> Option<f64> {
    let key = format!("{} p50 ", TIMING_HEADERS[0]);
    let rest = &procedure[procedure.find(&key)? + key.len()..];
    rest.split(' ').next()?.parse().ok()
}

/// A Prometheus histogram's samples at one reading, from the text
/// exposition format: `_count`, `_sum`, and each `_bucket`'s upper bound
/// (`le`, +Inf as infinity) with its cumulative count, in the order
/// given. A summary has no buckets.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    pub count: f64,
    pub sum: f64,
    pub buckets: Vec<(f64, f64)>,
}

/// One sample line: its metric name, its labels (the text between the
/// braces, empty when there are none), and its value.
fn sample(line: &str) -> Option<(&str, &str, f64)> {
    let line = line.trim();
    let name_end = line.find(|c: char| c == '{' || c.is_whitespace())?;
    let (name, rest) = line.split_at(name_end);
    let (labels, rest) = match rest.strip_prefix('{') {
        Some(r) => r.split_once('}')?,
        None => ("", rest),
    };
    Some((name, labels, prom_float(rest.split_whitespace().next()?)?))
}

fn prom_float(v: &str) -> Option<f64> {
    match v {
        "+Inf" => Some(f64::INFINITY),
        "-Inf" => Some(f64::NEG_INFINITY),
        v => v.parse().ok(),
    }
}

/// The histogram `name` in a /metrics body; None when it has no `_count`
/// and `_sum` (TEI has not recorded it, or the body is something else).
/// Series of `name` with other labels than `le` are added together.
pub fn parse_histogram(text: &str, name: &str) -> Option<Histogram> {
    let (mut count, mut sum, mut buckets) = (None, None, Vec::<(f64, f64)>::new());
    for line in text.lines().filter(|l| !l.trim_start().starts_with('#')) {
        let Some((series, labels, value)) = sample(line) else { continue };
        let Some(suffix) = series.strip_prefix(name) else { continue };
        match suffix {
            "_count" => *count.get_or_insert(0.0) += value,
            "_sum" => *sum.get_or_insert(0.0) += value,
            "_bucket" => {
                let Some(le) = labels
                    .split(',')
                    .find_map(|l| l.trim().strip_prefix("le=\""))
                    .and_then(|v| v.strip_suffix('"'))
                    .and_then(prom_float)
                else {
                    continue;
                };
                match buckets.iter_mut().find(|(b, _)| *b == le) {
                    Some((_, c)) => *c += value,
                    None => buckets.push((le, value)),
                }
            }
            _ => {}
        }
    }
    buckets.sort_by(|a, b| a.0.total_cmp(&b.0));
    Some(Histogram { count: count?, sum: sum?, buckets })
}

impl Histogram {
    /// What was recorded between `before` and this reading; with no
    /// `before`, everything this one holds.
    pub fn since(&self, before: Option<&Histogram>) -> Histogram {
        let Some(b) = before else { return self.clone() };
        let earlier = |le: f64| b.buckets.iter().find(|(x, _)| *x == le).map_or(0.0, |(_, c)| *c);
        Histogram {
            count: self.count - b.count,
            sum: self.sum - b.sum,
            buckets: self.buckets.iter().map(|&(le, c)| (le, c - earlier(le))).collect(),
        }
    }
}

/// TEI's batch histograms at one reading of /metrics: None when it has
/// no BATCH_SIZE_METRIC; `tokens` None when it has no
/// BATCH_TOKENS_METRIC.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchMetrics {
    pub size: Histogram,
    pub tokens: Option<Histogram>,
}

pub fn parse_batch_metrics(text: &str) -> Option<BatchMetrics> {
    Some(BatchMetrics {
        size: parse_histogram(text, BATCH_SIZE_METRIC)?,
        tokens: parse_histogram(text, BATCH_TOKENS_METRIC),
    })
}

/// A histogram's cumulative count at the bucket `le`, if it has that one.
fn at(h: &Histogram, le: f64) -> Option<f64> {
    h.buckets.iter().find(|(b, _)| *b == le).map(|(_, c)| *c)
}

/// How TEI split the timed requests into backend batches, for the
/// procedure, from /metrics read before (`before`) and after (`after`)
/// them. The batcher's empty polls are in te_batch_next_size as batches
/// of 0 inputs; te_batch_next_tokens has them as 0 tokens, and, when
/// every row has at least 2 tokens (`min_row_tokens`), no batch of inputs
/// has fewer, so its le="1" bucket counts exactly the empty polls. The
/// figures: the inputs, the batches without the empty polls, batches per
/// request, their mean size, and how many fell in each size bucket. What
/// cannot be told is said to be unknown: the batches without the tokens
/// histogram or with a row of 1 token, and everything when /metrics was
/// not read before the timed requests though warmup requests ran.
pub fn batches_text(
    before: Option<&BatchMetrics>,
    after: Option<&BatchMetrics>,
    requests: u32,
    warmup: u32,
    min_row_tokens: usize,
) -> String {
    let Some(after) = after else {
        return format!(
            "TEI's /metrics gave no {BATCH_SIZE_METRIC} after the timed requests, so how many backend batches they \
             ran as is not known"
        );
    };
    if before.is_none() && warmup > 0 {
        return format!(
            "TEI's /metrics gave no {BATCH_SIZE_METRIC} before the timed requests, after {warmup} warmup requests, so \
             how many backend batches the timed requests ran as is not known"
        );
    };
    let size = after.size.since(before.map(|b| &b.size));
    let tokens = match (&after.tokens, before) {
        (Some(t), None) => Some(t.clone()),
        (Some(t), Some(b)) => b.tokens.as_ref().map(|bt| t.since(Some(bt))),
        (None, _) => None,
    };
    let inputs = size.sum;
    let mut out = format!("TEI's /metrics over the timed requests: {inputs} inputs in {requests} requests");
    let empty = match tokens.as_ref().and_then(|t| at(t, 1.0)) {
        None => {
            return out
                + &format!(
                    "; how many backend batches they ran as is not known: {BATCH_SIZE_METRIC} also counts the \
                     batcher's empty polls, and {BATCH_TOKENS_METRIC}, which tells them apart, was not given"
                );
        }
        Some(_) if min_row_tokens < 2 => {
            return out
                + &format!(
                    "; how many backend batches they ran as is not known: {BATCH_SIZE_METRIC} also counts the \
                     batcher's empty polls, and with a row of {min_row_tokens} token {BATCH_TOKENS_METRIC} does not \
                     tell them apart"
                );
        }
        Some(z) => z,
    };
    let batches = size.count - empty;
    if batches <= 0.0 || requests == 0 {
        return out + &format!("; {BATCH_SIZE_METRIC} recorded no backend batch");
    }
    out += &format!(
        ", {batches} backend batches ({BATCH_SIZE_METRIC}'s {} samples less the {empty} empty polls \
         {BATCH_TOKENS_METRIC} shows), {:.2} batches per request, mean batch size {:.2} inputs",
        size.count,
        batches / requests as f64,
        inputs / batches
    );
    let mut parts = Vec::new();
    let (mut prev_le, mut prev_c) = (0.0_f64, 0.0);
    for &(le, c) in &size.buckets {
        // Every bucket is cumulative, so each holds the empty polls.
        let c = c - empty;
        let n = c - prev_c;
        if n > 0.0 {
            let low = prev_le.floor() + 1.0;
            parts.push(if le.is_infinite() {
                format!("above {prev_le}: {n}")
            } else if low >= le {
                format!("{le}: {n}")
            } else {
                format!("{low} to {le}: {n}")
            });
        }
        (prev_le, prev_c) = (le, c);
    }
    if !parts.is_empty() {
        out += &format!(" (batches by size, {})", parts.join(", "));
    }
    out
}
