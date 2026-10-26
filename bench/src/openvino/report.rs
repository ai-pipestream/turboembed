use turbo::manifest::{Format, GraphInput, Manifest};
use turbo::record::{Measured, ReferenceRun};

use super::{DEVICE_NPU, NAME};
use crate::docker::Log;
use crate::measure::Rows;
use crate::{Result, onnx};

/// What benchmark_app's report says.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// The OpenVINO build, from the `Build` line under `OpenVINO:`.
    pub version: String,
    /// Inferences timed; each is the whole batch.
    pub count: u64,
    /// Their wall time, in ms.
    pub duration_ms: f64,
    /// The latency at the run's percentile, in ms.
    pub latency_ms: f64,
    pub average_ms: f64,
    /// As benchmark_app computes it: frames of its batch dimension a
    /// second.
    pub throughput_fps: f64,
}

/// A line's text after benchmark_app's `[ INFO ] ` tag, trimmed.
fn info_lines(out: &str) -> impl Iterator<Item = &str> {
    out.lines().filter_map(|l| l.trim_start().strip_prefix("[ INFO ]")).map(str::trim)
}

/// `<value> ms` or `<value> us`, in ms: benchmark_app prints latencies in
/// microseconds when their average is under a millisecond.
fn latency(text: &str) -> Option<f64> {
    let mut it = text.split_whitespace();
    let v: f64 = it.next()?.parse().ok()?;
    match it.next()? {
        "ms" => Some(v),
        "us" => Some(v / 1000.0),
        _ => None,
    }
}

/// Parse benchmark_app's report of a run with `-latency_percentile
/// <percentile>`: the `Build` line under `OpenVINO:`, `Count:`,
/// `Duration:`, the `Latency:` block's percentile line (`Median:` for 50)
/// and `Average:`, and `Throughput:`. benchmark_app's Python and C++
/// forms both print these, each after an `[ INFO ]` tag, except the C++
/// form's lines of the version block.
pub fn parse(out: &str, percentile: u32) -> Result<Report> {
    let missing = |what: &str| format!("benchmark_app output has no {what}");
    let lines: Vec<&str> = info_lines(out).collect();
    let after = |tag: &str| lines.iter().find_map(|l| l.strip_prefix(tag)).map(str::trim);
    // The C++ form prints the version block untagged after its first line.
    let version = out
        .lines()
        .map(|l| l.trim_start().strip_prefix("[ INFO ]").unwrap_or(l).trim())
        .skip_while(|l| *l != "OpenVINO:")
        .find_map(|l| l.strip_prefix("Build"))
        .map(|v| v.trim_start_matches(['.', ' ', ':']).trim().to_owned())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| missing("Build line under OpenVINO:"))?;
    let count = after("Count:")
        .and_then(|v| v.strip_suffix("iterations")?.trim().parse().ok())
        .ok_or_else(|| missing("Count line"))?;
    let duration_ms = after("Duration:")
        .and_then(|v| v.strip_suffix("ms")?.trim().parse().ok())
        .ok_or_else(|| missing("Duration line"))?;
    let block: Vec<&str> = lines.iter().skip_while(|l| **l != "Latency:").skip(1).take(4).copied().collect();
    let tag = if percentile == 50 { "Median:".to_owned() } else { format!("{percentile} percentile:") };
    let in_block = |tag: &str| block.iter().find_map(|l| l.strip_prefix(tag)).and_then(latency);
    let latency_ms = in_block(&tag).ok_or_else(|| missing(&format!("{tag} line in its Latency block")))?;
    let average_ms = in_block("Average:").ok_or_else(|| missing("Average: line in its Latency block"))?;
    let throughput_fps = after("Throughput:")
        .and_then(|v| v.strip_suffix("FPS")?.trim().parse().ok())
        .ok_or_else(|| missing("Throughput line"))?;
    if count == 0 {
        return Err("benchmark_app timed no inference".into());
    }
    Ok(Report { version, count, duration_ms, latency_ms, average_ms, throughput_fps })
}

/// A home directory in a program's own error, written as `<home>` so a
/// record does not name the account that ran it. Commands are already
/// recorded with placeholders; this is the failure text.
pub fn redact_user_paths(text: &str) -> String {
    const PREFIXES: [&str; 6] = ["/home/", "/root", "/var/home/", "/Users/", "C:\\Users\\", "C:/Users/"];
    const END_CHARS: [char; 13] = ['/', '\\', ' ', '\t', '\n', '\r', ':', ';', ',', '"', '\'', ')', ']'];
    let mut out = text.to_owned();
    let mut i = 0;
    while i < out.len() {
        let rest_lower = out[i..].to_ascii_lowercase();
        let found = PREFIXES
            .iter()
            .filter_map(|&p| {
                let p_lower = p.to_ascii_lowercase();
                if p == "/root" {
                    rest_lower
                        .match_indices(&p_lower)
                        .find(|&(at, _)| {
                            let after = at + p.len();
                            after == rest_lower.len()
                                || rest_lower[after..].chars().next().is_some_and(|c| END_CHARS.contains(&c))
                        })
                        .map(|(at, _)| (at, p))
                } else {
                    rest_lower.find(&p_lower).map(|at| (at, p))
                }
            })
            .min_by_key(|(at, _)| *at);
        let Some((at, prefix)) = found else { break };
        let start = i + at;
        if prefix == "/root" {
            let root_end = start + "/root".len();
            out.replace_range(start..root_end, "<home>");
            i = start + "<home>".len();
            continue;
        }
        let after = start + prefix.len();
        let user_len = out[after..]
            .char_indices()
            .find(|(_, c)| END_CHARS.contains(c))
            .map(|(n, _)| n)
            .unwrap_or(out.len() - after);
        if user_len == 0 {
            out.replace_range(start..after, "<home>/");
            i = start + "<home>/".len();
        } else {
            out.replace_range(start..after + user_len, "<home>");
            i = start + "<home>".len();
        }
    }
    out
}

pub(super) fn not_run(image: &str, log: Log, procedure: &str, why: String) -> ReferenceRun {
    ReferenceRun {
        name: NAME.into(),
        role: "kernel".into(),
        pinned: image.into(),
        version: String::new(),
        commands: log.commands,
        procedure: procedure.into(),
        measured: None,
        not_run: Some(redact_user_paths(&why)),
        kernel_time: None,
    }
}

/// The file `benchmark_app -m` is given. GPU compiles the upstream ONNX
/// file. NPU compiles the static token-id IR (the xml; the bin sits
/// beside it). The dynamic ONNX file is not the NPU input.
pub fn model_file(device: &str, manifest: &Manifest) -> std::result::Result<String, String> {
    if device == DEVICE_NPU {
        let ir = manifest.artifacts.iter().find(|a| {
            a.format == Format::OpenvinoIr
                && a.graph_input == GraphInput::TokenIds
                && a.backends.iter().any(|b| b == "npu")
        });
        match ir.map(|a| a.files.as_slice()) {
            Some([xml, ..]) if xml.ends_with(".xml") => Ok(xml.clone()),
            Some(_) => Err("the npu OpenVINO IR's first file is not the xml benchmark_app -d NPU compiles".into()),
            None => {
                Err("the bundle carries no static OpenVINO IR (FORMAT_OPENVINO_IR, INPUT_TOKEN_IDS, backend npu) for \
                 benchmark_app -d NPU. The NPU plugin is not given onnx/model.onnx"
                    .into())
            }
        }
    } else {
        onnx::file(manifest, None, "for benchmark_app to compile")
    }
}

/// The measured reference from the median run and the 99th percentile
/// run of the same arguments, on `rows` at their static shape.
pub fn measured(image: &str, log: Log, procedure: &str, p50: Report, p99: Report, rows: &Rows) -> Result<ReferenceRun> {
    if p50.version != p99.version || p50.count != p99.count {
        return Err(format!(
            "benchmark_app's two runs differ: OpenVINO {} and {}, {} and {} iterations",
            p50.version, p99.version, p50.count, p99.count
        ));
    }
    if p99.latency_ms < p50.latency_ms {
        return Err(format!(
            "benchmark_app's 99th percentile run gave {} ms, under its median run's {} ms: the two runs are not \
             comparable; run again",
            p99.latency_ms, p50.latency_ms
        ));
    }
    Ok(ReferenceRun {
        name: NAME.into(),
        role: "kernel".into(),
        pinned: image.into(),
        version: p50.version,
        commands: log.commands,
        procedure: format!(
            "{procedure}; the median run took {} ms for {} inferences, average {} ms, and reported {} FPS; \
             the 99th percentile run averaged {} ms and reported {} FPS",
            p50.duration_ms, p50.count, p50.average_ms, p50.throughput_fps, p99.average_ms, p99.throughput_fps
        ),
        measured: Some(Measured {
            iterations: p50.count,
            p50_ms: p50.latency_ms,
            p99_ms: p99.latency_ms,
            rows_per_second: p50.count as f64 * rows.row_count() as f64 / (p50.duration_ms / 1000.0),
            min_cosine: None,
            computed_tokens: Some(rows.padded_tokens()),
        }),
        not_run: None,
        kernel_time: None,
    })
}

/// One pass over a batch-1 cycle: each pair is one case's median run and
/// its 99th percentile run, in case order. p50 and p99 are the sums of
/// those latencies. A single pair is `measured` unchanged.
pub fn measured_frames(
    image: &str,
    log: Log,
    procedure: &str,
    pairs: &[(Report, Report)],
    rows: &Rows,
) -> Result<ReferenceRun> {
    if pairs.len() != rows.frames() || pairs.is_empty() {
        return Err(format!("benchmark_app ran {} frames and the rows have {}", pairs.len(), rows.frames()));
    }
    let (first, _) = &pairs[0];
    for (p50, p99) in pairs {
        if p50.version != first.version
            || p99.version != first.version
            || p50.count != first.count
            || p99.count != first.count
        {
            return Err(format!(
                "benchmark_app's frames differ: OpenVINO {} and {}, {} and {} iterations",
                p50.version, p99.version, p50.count, p99.count
            ));
        }
        if p99.latency_ms < p50.latency_ms {
            return Err(format!(
                "benchmark_app's 99th percentile run gave {} ms, under its median run's {} ms: the two runs are not \
                 comparable; run again",
                p99.latency_ms, p50.latency_ms
            ));
        }
    }
    if pairs.len() == 1 {
        return measured(image, log, procedure, pairs[0].0.clone(), pairs[0].1.clone(), rows);
    }
    let p50_ms: f64 = pairs.iter().map(|(p, _)| p.latency_ms).sum();
    let p99_ms: f64 = pairs.iter().map(|(_, p)| p.latency_ms).sum();
    let duration_ms: f64 = pairs.iter().map(|(p, _)| p.duration_ms).sum();
    let count = first.count;
    let n = rows.row_count() as f64;
    Ok(ReferenceRun {
        name: NAME.into(),
        role: "kernel".into(),
        pinned: image.into(),
        version: first.version.clone(),
        commands: log.commands,
        procedure: format!(
            "{procedure}; {n} cases, each its own [{}, {}] request: median latencies sum to {p50_ms} ms over {count} \
             inferences a case (durations sum to {duration_ms} ms), and the 99th percentile latencies sum to {p99_ms} ms",
            rows.batch, rows.seq
        ),
        measured: Some(Measured {
            iterations: count,
            p50_ms,
            p99_ms,
            rows_per_second: count as f64 * n / (duration_ms / 1000.0),
            min_cosine: None,
            computed_tokens: Some(rows.padded_tokens()),
        }),
        not_run: None,
        kernel_time: None,
    })
}
