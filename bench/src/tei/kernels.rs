//! TEI's kernel time on a GPU, through Nsight Systems (#978): none of
//! TEI's headers is its compute, so its requests are traced instead.
//!
//! The image has no profiler, so the nsys command line's target directory
//! (`target-linux-x64` of the machine's Nsight Systems) is mounted into
//! the container read-only, with a directory of libraries it needs that
//! the image lacks where there are any (`libelf.so.1` built for the
//! image's glibc, say), and TEI's router is started under `nsys launch`
//! with CUDA tracing. After the end-to-end timing, the tool starts the
//! collection (`nsys start` in the container), sends `requests` /embed
//! requests one at a time with the GPU left idle for IDLE after each,
//! stops it, and reads the trace on the host (`nsys stats`, the
//! cuda_gpu_trace report as CSV, after the host's QdstrmImporter where
//! the in-container nsys leaves only the raw stream). A request's kernels
//! are the activity between two gaps of at least SPLIT; the trace must
//! fall into exactly `requests` such runs, or nothing is recorded.

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

/// The GPU's idle time after each profiled request.
pub const IDLE: Duration = Duration::from_millis(100);
/// The least gap between two requests' activity, and more than any gap
/// inside one request's.
pub const SPLIT_NS: u64 = 50_000_000;

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
}

/// What the procedure says of the profiled requests.
pub fn procedure(requests: u32) -> String {
    format!(
        "then, with Nsight Systems tracing CUDA in the container (nsys launch --trace=cuda, collected only around \
         these), {requests} more /embed requests of the same rows one at a time, the GPU left idle {} ms after \
         each, each request's kernel time the summed durations of the trace's kernels between idle gaps of at least \
         {} ms; collected {CHUNK} requests at a time, and a collection whose trace the host's tools refuse is taken \
         again, up to {ATTEMPTS} times, as the commands show",
        IDLE.as_millis(),
        SPLIT_NS / 1_000_000
    )
}

/// Requests in one collection: Nsight Systems 2023.4's importer refuses
/// longer traces of TEI as events out of order (every one of three
/// collections of 200 requests on an RTX 4080 SUPER host), and 30 import.
pub const CHUNK: u32 = 25;

/// Collections of one chunk tried before the tool gives up: the importer
/// still refuses a short trace now and then, and a new collection of the
/// same requests imports.
pub const ATTEMPTS: u32 = 3;

/// Collect around `requests` calls of `send`, each followed by IDLE, in
/// collections of at most CHUNK, and read their kernel time; a trace the
/// host's tools cannot read is collected again, up to ATTEMPTS times.
pub fn measure(log: &mut Log, p: &Profile, requests: u32, mut send: impl FnMut() -> Result<()>) -> Result<KernelTime> {
    let profiler = log.run(&exec(&p.container, &["--version"]))?;
    let mut all = Vec::with_capacity(requests as usize);
    let mut left = requests;
    while left > 0 {
        let n = left.min(CHUNK);
        all.extend(collect(log, p, n, &mut send)?);
        left -= n;
    }
    Ok(kernel_time(&profiler, &all))
}

/// One chunk of `n` requests, collected until its trace reads.
fn collect(log: &mut Log, p: &Profile, n: u32, send: &mut impl FnMut() -> Result<()>) -> Result<Vec<Request>> {
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
        log.run(&stop(&p.container))?;
        sent?;
        match read_trace(log, p.nsys, &p.out) {
            Ok(csv) => {
                return requests(parse_trace(&csv)?, n as u64).map_err(|why| format!("TEI's kernel time: {why}"));
            }
            Err(e) => failed.push(e.lines().find(|l| l.contains("failed")).unwrap_or(&e).to_owned()),
        }
    }
    Err(format!("TEI's trace could not be read in {ATTEMPTS} collections: {}", failed.join("; ")))
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

/// The activities as runs separated by gaps of at least `split` ns, from
/// one run's last end to the next one's first start.
pub fn split(mut a: Vec<Activity>, split: u64) -> Vec<Vec<Activity>> {
    a.sort_by_key(|x| x.start);
    let mut runs: Vec<Vec<Activity>> = Vec::new();
    let mut end = 0u64;
    for x in a {
        match runs.last_mut() {
            Some(run) if x.start < end.saturating_add(split) => run.push(x),
            _ => runs.push(vec![x]),
        }
        end = end.max(x.start + x.duration);
    }
    runs
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
    let runs = split(a, SPLIT_NS);
    if runs.len() as u64 != requests {
        return Err(format!(
            "the trace falls into {} runs of activity between idle gaps of {} ms, not the {requests} requests sent",
            runs.len(),
            SPLIT_NS / 1_000_000
        ));
    }
    let sum = |r: &[Activity], copy: bool| r.iter().filter(|x| x.copy == copy).map(|x| x.duration).sum::<u64>();
    let kernels = |r: &[Activity]| r.iter().filter(|x| !x.copy).count() as u64;
    let out: Vec<Request> =
        runs.iter().map(|r| Request { kernel_ns: sum(r, false), copy_ns: sum(r, true), kernels: kernels(r) }).collect();
    if out.iter().any(|r| r.kernels == 0) {
        return Err("a profiled request ran no kernel".into());
    }
    Ok(out)
}

/// The kernel time of the profiled requests, of every collection.
pub fn kernel_time(profiler: &str, r: &[Request]) -> KernelTime {
    let ms = |f: fn(&Request) -> u64| {
        let mut v: Vec<f64> = r.iter().map(|x| f(x) as f64 / 1e6).collect();
        v.sort_by(f64::total_cmp);
        v
    };
    let (kernels, copies) = (ms(|x| x.kernel_ns), ms(|x| x.copy_ns));
    let mut launches: Vec<u64> = r.iter().map(|x| x.kernels).collect();
    launches.sort_unstable();
    KernelTime {
        profiler: profiler.trim().to_owned(),
        requests: r.len() as u64,
        p50_ms: percentile(&kernels, 50.0),
        p99_ms: percentile(&kernels, 99.0),
        copy_p50_ms: percentile(&copies, 50.0),
        kernels_p50: launches[(launches.len() - 1) / 2],
    }
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
mod tests {
    use super::*;

    const TRACE: &str = "Start (ns),Duration (ns),CorrId,GrdX,Name
100,10,1,1,\"void gemm<float, (int)128>(params)\"
120,5,2,,[CUDA memcpy Host-to-Device]
140,20,3,4,ucopy_f32
60000200,30,4,1,\"void gemm<float, (int)128>(params)\"
60000300,3,5,,[CUDA memset]
";

    #[test]
    fn a_trace_reads_with_commas_in_its_names() {
        let a = parse_trace(TRACE).unwrap();
        assert_eq!(a.len(), 5);
        assert_eq!(a[0], Activity { start: 100, duration: 10, copy: false });
        assert!(a[1].copy && !a[2].copy && a[4].copy);
        assert!(parse_trace("Name,Start (ns),Duration (ns)\n").is_err(), "Name must be last");
        assert!(parse_trace("Start (ns),Duration (ns),Name\nx,1,k\n").is_err());
    }

    #[test]
    fn requests_are_the_runs_between_idle_gaps() {
        let a = parse_trace(TRACE).unwrap();
        let runs = split(a.clone(), SPLIT_NS);
        assert_eq!(runs.iter().map(Vec::len).collect::<Vec<_>>(), [3, 2]);
        let r = requests(a.clone(), 2).unwrap();
        assert_eq!(r[0], Request { kernel_ns: 30, copy_ns: 5, kernels: 2 });
        assert_eq!(r[1], Request { kernel_ns: 30, copy_ns: 3, kernels: 1 });
        let k = kernel_time("nsys 2023.4\n", &r);
        assert_eq!((k.profiler.as_str(), k.requests, k.kernels_p50), ("nsys 2023.4", 2, 1));
        assert_eq!((k.p50_ms, k.p99_ms), (30.0 / 1e6, 30.0 / 1e6));
        assert_eq!(k.copy_p50_ms, 3.0 / 1e6);
        let wrong = requests(a, 3).unwrap_err();
        assert!(wrong.contains("2 runs") && wrong.contains("the 3 requests"), "{wrong}");
    }

    #[test]
    fn a_gap_is_measured_from_the_last_end() {
        // A long activity covers the next start: one run.
        let a = [(0, 70_000_000), (60_000_000, 10)].map(|(s, d)| Activity { start: s, duration: d, copy: false });
        assert_eq!(split(a.to_vec(), SPLIT_NS).len(), 1);
    }

    #[test]
    fn the_launch_script_keeps_the_images_library_path_for_the_router() {
        let l = launch();
        assert_eq!((l[0].as_str(), l[2].as_str()), ("-c", "sh"));
        assert!(l[1].contains("--env-var=LD_LIBRARY_PATH=\"$own\"") && l[1].ends_with("text-embeddings-router \"$@\""));
        let n = Nsys { target: "/x/t".into(), libs: Some("/x/l".into()), importer: None };
        let shown = mounts(&n, Path::new("/w"), true).join(" ");
        assert!(
            shown.contains("src=<tei-nsys>,")
                && shown.contains("src=<tei-nsys-libs>,")
                && shown.contains("src=<work>,")
        );
        assert!(!shown.contains("/x/"));
    }
}
