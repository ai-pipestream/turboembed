//! TEI's kernel time on a GPU, through Nsight Systems (#978): none of
//! TEI's headers is its compute, so its requests are traced instead.
//!
//! The image has no profiler, so the nsys command line's target directory
//! (`target-linux-x64` of the machine's Nsight Systems) is mounted into
//! the container read-only, with a directory of libraries it needs that
//! the image lacks where there are any (`libelf.so.1` built for the
//! image's glibc, say), and TEI's router is started under `nsys launch`
//! with CUDA tracing, in a container of its own: tracing slows every
//! request, so the end-to-end timing is of TEI without it, in the first
//! container, which is gone before this one starts. After its warm-up the
//! tool starts the collection (`nsys start` in the container), sends `requests` /embed
//! requests one at a time with the GPU left idle for IDLE after each,
//! stops it, and reads the trace on the host (`nsys stats`, the
//! cuda_gpu_trace report as CSV, after the host's QdstrmImporter where
//! the in-container nsys leaves only the raw stream). A request's kernels
//! are the activity between two gaps of at least SPLIT; the trace must
//! fall into exactly `requests` such runs, or nothing is recorded. The
//! GPU's SM clock is read after each chunk's requests, and a kernel time
//! longer than the end-to-end time it is part of is refused (check).

use std::path::{Path, PathBuf};
use std::time::Duration;

use turbo::record::KernelTime;

use crate::Result;
use crate::docker::{Log, argv};
use crate::measure::percentile;

/// The profiler's target directory, as a recorded command names it.
pub const NSYS: &str = "<tei-nsys>";
/// Its extra libraries.
pub const NSYS_LIBS: &str = "<tei-nsys-libs>";
/// The host's QdstrmImporter.
pub const IMPORTER: &str = "<nsys-importer>";

/// The GPU's idle time after each profiled request: short enough that the
/// GPU keeps its clocks between them (at 250 ms an RTX 4080 SUPER's
/// dropped, and the kernels ran slower than TEI's whole unprofiled
/// request).
pub const IDLE: Duration = Duration::from_millis(50);
/// The least gap between two requests' activity: a request's work is done
/// when its answer arrives, and the next one is sent IDLE after that, so
/// every gap between requests is at least IDLE. A gap as long can fall
/// inside a request too (TEI's host work between its batches, on a loaded
/// machine), so a trace of n requests is cut at its n - 1 longest gaps,
/// each at least this, and every request's kernel count must be within
/// SPREAD percent of their median.
pub const SPLIT_NS: u64 = IDLE.as_nanos() as u64;

/// How far, in percent, a request's kernel count may be from the median.
pub const SPREAD: u64 = 25;

const SESSION: &str = "turbo-bench";

/// Where the profiler comes from (`--tei-nsys`, `--tei-nsys-libs`,
/// `--nsys-importer`). The host's `nsys` on the PATH reads the trace.
#[derive(Debug, Clone)]
pub struct Nsys {
    pub target: PathBuf,
    pub libs: Option<PathBuf>,
    pub importer: Option<PathBuf>,
}

/// What goes before the image in TEI's `docker run`: the profiler's
/// mounts, the output directory and the entry point. `shown` is whether
/// the paths are written as placeholders.
pub fn mounts(n: &Nsys, out: &Path, shown: bool) -> Vec<String> {
    let path = |p: &Path, placeholder: &str| if shown { placeholder.to_owned() } else { p.display().to_string() };
    let mut a = argv(&["--mount"]);
    a.push(format!("type=bind,src={},dst=/opt/nsys/target-linux-x64,readonly", path(&n.target, NSYS)));
    if let Some(libs) = &n.libs {
        a.push("--mount".into());
        a.push(format!("type=bind,src={},dst=/opt/nsys/libs,readonly", path(libs, NSYS_LIBS)));
    }
    a.push("--mount".into());
    a.push(format!("type=bind,src={},dst=/out", path(out, crate::docker::WORK)));
    a.extend(argv(&["--entrypoint", "/bin/sh"]));
    a
}

/// What goes after the image in place of the router's arguments: a shell
/// that links the nsys command line where it expects to be, and runs the
/// router under it with the image's own library path; the router's
/// arguments follow.
pub fn launch() -> Vec<String> {
    let script = format!(
        "mkdir -p /opt/nsys/bin && ln -sf ../target-linux-x64/nsys /opt/nsys/bin/nsys && own=\"$LD_LIBRARY_PATH\" && \
         LD_LIBRARY_PATH=/opt/nsys/libs${{own:+:$own}} exec /opt/nsys/bin/nsys launch --trace=cuda \
         --session-new={SESSION} --env-var=LD_LIBRARY_PATH=\"$own\" text-embeddings-router \"$@\""
    );
    argv(&["-c", &script, "sh"])
}

/// `nsys <args>` run in the container.
pub fn exec(container: &str, args: &[&str]) -> Vec<String> {
    let mut a = argv(&["docker", "exec", "--env", "LD_LIBRARY_PATH=/opt/nsys/libs", container, "/opt/nsys/bin/nsys"]);
    a.extend(argv(args));
    a
}

/// The start and stop of the collection, as run in the container.
pub fn start(container: &str) -> Vec<String> {
    exec(container, &["start", &format!("--session={SESSION}"), "--output=/out/tei", "--force-overwrite=true"])
}

pub fn stop(container: &str) -> Vec<String> {
    exec(container, &["stop", &format!("--session={SESSION}")])
}

/// TEI's `docker run` (`run_argv`) with the profiler: its mounts and
/// entry point before the image, and the launch script after it.
pub fn profiled(a: Vec<String>, image: &str, n: &Nsys, out: &Path, shown: bool) -> Vec<String> {
    let at = a.iter().position(|x| x == image).unwrap_or(a.len());
    let (before, after) = a.split_at(at);
    let mut p = before.to_vec();
    p.extend(mounts(n, out, shown));
    p.extend(after.iter().take(1).cloned());
    p.extend(launch());
    p.extend(after.iter().skip(1).cloned());
    p
}

/// A profiled TEI: its container, the profiler, and the directory the
/// trace goes to, mounted at /out.
pub struct Profile<'a> {
    pub container: String,
    pub nsys: &'a Nsys,
    pub out: PathBuf,
    /// The GPU's ordinal, whose SM clock is read.
    pub gpu: u32,
}

/// What the procedure says of the profiled requests, with the SM clocks
/// (MHz) read after each chunk.
pub fn procedure(requests: u32, warmup: u32, clocks: &[u32]) -> String {
    let mut sorted = clocks.to_vec();
    sorted.sort_unstable();
    let clock = match (sorted.first(), sorted.get(sorted.len().saturating_sub(1) / 2)) {
        (Some(min), Some(median)) => {
            format!("the SM clock after each chunk's requests min {min} MHz, median {median} MHz")
        }
        _ => "no SM clock read".to_owned(),
    };
    format!(
        "then, with that server stopped, TEI started again in another container under Nsight Systems tracing CUDA \
         (nsys launch --trace=cuda, collected only around the profiled requests), {warmup} untimed /embed requests, \
         then {requests} more of the same rows one at a time, the GPU left idle {} ms after each, each request's \
         kernel time the summed durations of its kernels, a trace of n requests cut at its n - 1 longest gaps, each \
         at least {} ms, and every request's kernel count within {SPREAD}% of their median; collected \
         {CHUNK} requests at a time, and a collection whose trace the host's tools refuse is taken again, up to \
         {ATTEMPTS} times, as the commands show; {clock}; a kernel p50 longer than the end-to-end p50 is refused",
        IDLE.as_millis(),
        SPLIT_NS / 1_000_000
    )
}

/// Refuse a kernel time that cannot be: the kernels are part of the
/// request the end-to-end time covers, so a p50 longer than that one's
/// says the profiler, or the GPU under it, slowed them.
pub fn check(k: &KernelTime, e2e_p50_ms: f64) -> Result<()> {
    if k.p50_ms > e2e_p50_ms {
        return Err(format!(
            "TEI's kernel time: a kernel p50 of {:.3} ms is longer than the end-to-end p50 of {e2e_p50_ms:.3} ms \
             it is part of; the profiled requests did not run as the timed ones did, and nothing is recorded",
            k.p50_ms
        ));
    }
    Ok(())
}

/// Requests in one collection: Nsight Systems 2023.4's importer refuses
/// longer traces of TEI as events out of order (every one of three
/// collections of 200 requests on an RTX 4080 SUPER host), and 30 import.
pub const CHUNK: u32 = 25;

/// Collections of one chunk tried before the tool gives up: the importer
/// still refuses a short trace now and then, and a trace can hold a gap
/// as long as IDLE inside a request; a new collection of the same
/// requests is taken.
pub const ATTEMPTS: u32 = 3;

/// Collect around `requests` calls of `send`, each followed by IDLE, in
/// collections of at most CHUNK, and read their kernel time; a trace the
/// host's tools cannot read is collected again, up to ATTEMPTS times.
/// With it, the SM clock (MHz) read after each collection's requests.
pub fn measure(
    log: &mut Log,
    p: &Profile,
    requests: u32,
    mut send: impl FnMut() -> Result<()>,
) -> Result<(KernelTime, Vec<u32>)> {
    let profiler = log.run(&exec(&p.container, &["--version"]))?;
    let mut all = Vec::with_capacity(requests as usize);
    let mut clocks = Vec::new();
    let mut left = requests;
    while left > 0 {
        let n = left.min(CHUNK);
        all.extend(collect(log, p, n, &mut send, &mut clocks)?);
        left -= n;
    }
    let k = kernel_time(&profiler, &all).map_err(|why| format!("TEI's kernel time: {why}"))?;
    Ok((k, clocks))
}

/// The GPU's SM clock as `nvidia-smi` reads it.
pub fn sm_clock(gpu: u32) -> Vec<String> {
    let gpu = gpu.to_string();
    argv(&["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits", "-i", &gpu])
}

/// One chunk of `n` requests, collected until its trace reads.
fn collect(
    log: &mut Log,
    p: &Profile,
    n: u32,
    send: &mut impl FnMut() -> Result<()>,
    clocks: &mut Vec<u32>,
) -> Result<Vec<Request>> {
    let mut failed = Vec::new();
    for _ in 0..ATTEMPTS {
        for f in ["tei.qdstrm", "tei.nsys-rep", "tei.sqlite", "tei_cuda_gpu_trace.csv"] {
            let _ = std::fs::remove_file(p.out.join(f));
        }
        log.run(&start(&p.container))?;
        let sent = (0..n).try_for_each(|_| {
            send()?;
            std::thread::sleep(IDLE);
            Ok::<(), String>(())
        });
        let clock = sent.as_ref().ok().map(|_| log.run(&sm_clock(p.gpu)));
        log.run(&stop(&p.container))?;
        sent?;
        if let Some(c) = clock {
            let c = c?;
            clocks.push(c.trim().parse().map_err(|_| format!("nvidia-smi read the SM clock as {:?}", c.trim()))?);
        }
        match read_trace(log, p.nsys, &p.out) {
            Ok(csv) => match parse_trace(&csv).and_then(|a| requests(a, n as u64)) {
                Ok(r) => return Ok(r),
                Err(why) => failed.push(why),
            },
            Err(e) => failed.push(e.lines().find(|l| l.contains("failed")).unwrap_or(&e).to_owned()),
        }
    }
    Err(format!("TEI's kernel time: no collection of {ATTEMPTS} could be read as {n} requests: {}", failed.join("; ")))
}

/// One traced GPU activity: its start and duration in ns, and whether it
/// is a memory copy or set rather than a kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Activity {
    pub start: u64,
    pub duration: u64,
    pub copy: bool,
}

/// The cuda_gpu_trace report as `nsys stats --format csv` writes it: a
/// header naming `Start (ns)`, `Duration (ns)` and, last, `Name`. A name
/// may hold commas; it is the last column, so the line is cut into at
/// most as many fields as the header has.
pub fn parse_trace(csv: &str) -> Result<Vec<Activity>> {
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().ok_or("an empty cuda_gpu_trace report")?.split(',').collect();
    let col = |name: &str| header.iter().position(|h| h.trim() == name).ok_or(format!("no {name:?} column"));
    let (start, duration, name) = (col("Start (ns)")?, col("Duration (ns)")?, col("Name")?);
    if name + 1 != header.len() {
        return Err("the cuda_gpu_trace report's Name is not its last column".into());
    }
    lines
        .map(|l| {
            let f: Vec<&str> = l.splitn(header.len(), ',').collect();
            let n = |i: usize| f.get(i).and_then(|v| v.trim().parse::<u64>().ok());
            match (n(start), n(duration), f.get(name)) {
                (Some(s), Some(d), Some(nm)) => {
                    Ok(Activity { start: s, duration: d, copy: nm.trim_matches('"').starts_with("[CUDA mem") })
                }
                _ => Err(format!("a cuda_gpu_trace line not read: {l:?}")),
            }
        })
        .collect()
}

/// The activities as `n` runs, cut at the n - 1 longest gaps from one
/// run's last end to the next one's first start, each at least `least`
/// ns; none when there are fewer such gaps, or fewer activities.
pub fn split(mut a: Vec<Activity>, n: usize, least: u64) -> Option<Vec<Vec<Activity>>> {
    a.sort_by_key(|x| x.start);
    if n == 0 || a.len() < n {
        return None;
    }
    // The gap before each activity after the first, as (gap, index).
    let mut end = a[0].start + a[0].duration;
    let mut gaps = Vec::with_capacity(a.len());
    for (i, x) in a.iter().enumerate().skip(1) {
        gaps.push((x.start.saturating_sub(end), i));
        end = end.max(x.start + x.duration);
    }
    gaps.sort_unstable_by(|p, q| q.cmp(p));
    let mut cuts: Vec<usize> = gaps[..n - 1].iter().map(|&(_, i)| i).collect();
    if gaps[..n - 1].iter().any(|&(g, _)| g < least) {
        return None;
    }
    cuts.sort_unstable();
    let mut runs = Vec::with_capacity(n);
    let mut rest = a;
    for &c in cuts.iter().rev() {
        runs.push(rest.split_off(c));
    }
    runs.push(rest);
    runs.reverse();
    Some(runs)
}

/// One profiled request: its kernels' and copies' summed ns, and the
/// kernels it launched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Request {
    pub kernel_ns: u64,
    pub copy_ns: u64,
    pub kernels: u64,
}

/// The `requests` requests of one collection's trace, or why it does not
/// hold them.
pub fn requests(a: Vec<Activity>, requests: u64) -> std::result::Result<Vec<Request>, String> {
    let ms = SPLIT_NS / 1_000_000;
    let Some(runs) = split(a, requests as usize, SPLIT_NS) else {
        return Err(format!(
            "the trace has fewer than the {} idle gaps of {ms} ms {requests} requests leave",
            requests - 1
        ));
    };
    let sum = |r: &[Activity], copy: bool| r.iter().filter(|x| x.copy == copy).map(|x| x.duration).sum::<u64>();
    let kernels = |r: &[Activity]| r.iter().filter(|x| !x.copy).count() as u64;
    let out: Vec<Request> =
        runs.iter().map(|r| Request { kernel_ns: sum(r, false), copy_ns: sum(r, true), kernels: kernels(r) }).collect();
    if out.iter().any(|r| r.kernels == 0) {
        return Err("a profiled request ran no kernel".into());
    }
    // Every request is of the same rows, though TEI's kernel count still
    // varies a little between them (557 to 593 at [32, 512]). A cut in the
    // wrong place leaves one cut out too, which merges two requests into
    // one of about twice the kernels.
    let mut counts: Vec<u64> = out.iter().map(|r| r.kernels).collect();
    counts.sort_unstable();
    let median = counts[(counts.len() - 1) / 2];
    if counts.iter().any(|&k| k.abs_diff(median) * 100 > median * SPREAD) {
        counts.dedup();
        return Err(format!(
            "cut at its longest gaps, the trace's requests launch {counts:?} kernels, some more than {SPREAD}% from \
             their median of {median}"
        ));
    }
    Ok(out)
}

/// The kernel time of the profiled requests, of every collection; none
/// of no requests.
pub fn kernel_time(profiler: &str, r: &[Request]) -> std::result::Result<KernelTime, String> {
    if r.is_empty() {
        return Err("no request was profiled".into());
    }
    let ms = |f: fn(&Request) -> u64| {
        let mut v: Vec<f64> = r.iter().map(|x| f(x) as f64 / 1e6).collect();
        v.sort_by(f64::total_cmp);
        v
    };
    let (kernels, copies) = (ms(|x| x.kernel_ns), ms(|x| x.copy_ns));
    let mut launches: Vec<u64> = r.iter().map(|x| x.kernels).collect();
    launches.sort_unstable();
    Ok(KernelTime {
        profiler: profiler.trim().to_owned(),
        requests: r.len() as u64,
        p50_ms: percentile(&kernels, 50.0),
        p99_ms: percentile(&kernels, 99.0),
        copy_p50_ms: percentile(&copies, 50.0),
        kernels_p50: launches[(launches.len() - 1) / 2],
    })
}

/// The trace the collection left in `out` as CSV, through the host's
/// tools: the raw stream imported first when no report is there.
pub fn read_trace(log: &mut Log, n: &Nsys, out: &Path) -> Result<String> {
    let (rep, raw) = (out.join("tei.nsys-rep"), out.join("tei.qdstrm"));
    let shown = |f: &str| format!("{}/{f}", crate::docker::WORK);
    if !rep.exists() {
        let importer = n.importer.as_ref().ok_or(
            "nsys left only its raw stream (tei.qdstrm) and no --nsys-importer was given to import it on the host",
        )?;
        let run = vec![importer.display().to_string(), "-i".into(), raw.display().to_string()];
        log.run_as(&run, argv(&[IMPORTER, "-i", &shown("tei.qdstrm")]))?;
    }
    let prefix = out.join("tei");
    let stats = |rep: &str, prefix: &str| {
        argv(&["nsys", "stats", "--quiet", "--report", "cuda_gpu_trace", "--format", "csv", "--force-export=true"])
            .into_iter()
            .chain([format!("--output={prefix}"), rep.to_owned()])
            .collect::<Vec<_>>()
    };
    log.run_as(
        &stats(&rep.display().to_string(), &prefix.display().to_string()),
        stats(&shown("tei.nsys-rep"), &shown("tei")),
    )?;
    let csv = out.join("tei_cuda_gpu_trace.csv");
    std::fs::read_to_string(&csv).map_err(|e| format!("{}: {e}", shown("tei_cuda_gpu_trace.csv")))
}

#[cfg(test)]
#[path = "kernels_tests.rs"]
mod tests;
