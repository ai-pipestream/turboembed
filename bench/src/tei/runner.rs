use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::json;
use turbo::record::{Measured, ReferenceRun};

use super::*;
use crate::Result;
use crate::cpus;
use crate::docker::{self, Log, Running, argv};
use crate::measure::{Measurement, cosine, percentile};

/// How long the server may take to load the model and answer /health.
const START_TIMEOUT: Duration = Duration::from_secs(600);

/// A native router started by the tool, and the file its output goes
/// to: the router is stopped and the file removed when this goes.
struct Native(std::process::Child, PathBuf);

impl Drop for Native {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
        let _ = fs::remove_file(&self.1);
    }
}

/// What TEI's vectors are compared with, in a clause.
fn compared(m: &Measurement) -> &'static str {
    if (0..m.rows.row_count()).all(|r| m.rows.whole(r, &m.reference)) {
        "min_cosine is against the bundle's reference vectors"
    } else {
        "min_cosine is against the bundle's reference vector for a row that is its whole case, and the library's \
         vector of the row alone for a row cut to seq"
    }
}

fn not_run(image: &str, log: Log, procedure: &str, why: String) -> ReferenceRun {
    ReferenceRun {
        name: NAME.into(),
        role: "end_to_end".into(),
        pinned: image.into(),
        version: String::new(),
        commands: log.commands,
        procedure: procedure.into(),
        measured: None,
        not_run: Some(crate::openvino::redact_user_paths(&why)),
        kernel_time: None,
    }
}

fn post(url: &str, body: &str) -> Result<String> {
    post_timed(url, body).map(|(text, _)| text)
}

/// The answer's body and its TIMING_HEADERS.
fn post_timed(url: &str, body: &str) -> Result<(String, [Option<u64>; 4])> {
    let mut resp = ureq::post(url)
        .header("content-type", "application/json")
        .send(body)
        .map_err(|e| format!("POST {url}: {e}"))?;
    let timing = timing_headers(resp.headers());
    let text =
        resp.body_mut().with_config().limit(u64::MAX).read_to_string().map_err(|e| format!("POST {url}: {e}"))?;
    Ok((text, timing))
}

fn get(url: &str) -> Result<String> {
    let mut resp = ureq::get(url).call().map_err(|e| format!("GET {url}: {e}"))?;
    resp.body_mut().read_to_string().map_err(|e| format!("GET {url}: {e}"))
}

/// Start the server, check it runs the same rows, time `iterations`
/// /embed requests of the whole batch after `warmup` untimed ones, and
/// stop it. A thing TEI cannot do for this bundle is a record that says
/// so; a failure of docker or the server is an error.
///
/// `library` is the library's thread count when the device measured is
/// the CPU, for the procedure, which says what each side ran on.
pub fn run(
    tei: &Tei,
    m: &Measurement,
    gpu: Option<u32>,
    library: Option<usize>,
    warmup: u32,
    iterations: u32,
) -> Result<ReferenceRun> {
    // What names the program: the image's digest, or the native
    // router's SHA-256.
    let pinned = match &tei.binary {
        Some(bin) => native_pin(bin)?,
        None => docker::check_pinned("--tei-image", &tei.image)?.to_owned(),
    };
    let image = pinned.as_str();
    let procedure = format!(
        "POST /decode then /tokenize to check the rows survive TEI's re-tokenization; then POST /embed with the \
         batch's {} rows as token ids, {warmup} untimed then {iterations} timed, each timed from sending the \
         request to reading the whole response, the p50 and p99 of TEI's {} headers beside it, and TEI's /metrics read \
         before and after the timed requests for its {BATCH_SIZE_METRIC} and {BATCH_TOKENS_METRIC} histograms; {}",
        m.rows.row_count(),
        TIMING_HEADERS.join(", "),
        compared(m)
    );
    let mut log = Log::default();
    let dtype = match dtype(m.compute_dtype, gpu.is_some() || tei.binary.is_some()) {
        Ok(d) => d,
        Err(why) => return Ok(not_run(image, log, &procedure, why)),
    };
    let dir = fs::canonicalize(&tei.model_dir).map_err(|e| format!("{}: {e}", tei.model_dir.display()))?;
    if let Err(why) = check_model_dir(&dir, m) {
        return Ok(not_run(image, log, &procedure, why));
    }
    if let Some(bin) = &tei.binary {
        return run_native(bin, image, &dir, m, dtype, procedure, library, warmup, iterations, log);
    }
    docker::require_image(&mut log, image)?;
    let env =
        parse_env(&log.run(&argv(&["docker", "image", "inspect", "--format", "{{json .Config.Env}}", image]))?)?;
    let (what, threads) = (procedure, cpus::procedure(tei.cpus.as_ref(), library, &env));

    let container = format!("turbo-bench-tei-{}", std::process::id());
    let profile = match (&tei.nsys, gpu) {
        (Some(n), Some(_)) => {
            let out = std::env::temp_dir().join(format!("turbo-bench-tei-nsys-{}", std::process::id()));
            fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
            Some(kernels::Profile { container: container.clone(), nsys: n, out })
        }
        (Some(_), None) => return Err("--tei-nsys profiles TEI's kernels on a GPU; this device is not one".into()),
        (None, _) => None,
    };
    let _trace = profile.as_ref().map(|p| Scratch(p.out.clone()));
    let start_argv = |dir: &Path, shown: bool| {
        let a = run_argv(
            image,
            dir,
            &container,
            gpu,
            tei.cpus.as_ref(),
            dtype,
            pooling(m.pooling()),
            m.rows.row_count() as u32,
            m.rows.seq,
        );
        match &profile {
            Some(p) => kernels::profiled(a, image, p.nsys, &p.out, shown),
            None => a,
        }
    };
    log.run_as(&start_argv(&dir, false), start_argv(Path::new(docker::TEI_MODEL), true))?;
    let _running = Running { name: container.clone() };
    let port = docker::parse_port(&log.run(&argv(&["docker", "port", &container, "80/tcp"]))?)?;
    let base = format!("http://127.0.0.1:{port}");

    let start = Instant::now();
    while get(&format!("{base}/health")).is_err() {
        let state = log.run(&argv(&["docker", "inspect", "--format", "{{.State.Running}}", &container]));
        if state.as_deref().map(str::trim) != Ok("true") || start.elapsed() > START_TIMEOUT {
            let logs = log.run(&argv(&["docker", "logs", &container])).unwrap_or_default();
            return Err(format!("TEI did not become healthy within {START_TIMEOUT:?}:\n{logs}"));
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    against(&base, image, log, what, threads, m, warmup, iterations, profile.as_ref())
}

/// A directory the tool made, removed when this goes.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// TEI's router run natively on the model directory, as run() runs the
/// image: `what` is the procedure so far.
#[allow(clippy::too_many_arguments)]
fn run_native(
    bin: &Path,
    pinned: &str,
    dir: &Path,
    m: &Measurement,
    dtype: &str,
    what: String,
    library: Option<usize>,
    warmup: u32,
    iterations: u32,
    log: Log,
) -> Result<ReferenceRun> {
    let bin = fs::canonicalize(bin).map_err(|e| format!("--tei-bin {}: {e}", bin.display()))?;
    let threads = cpus::procedure(None, library, &[]);
    let what = format!("{what}; TEI's router built natively and run on this machine, not in a container");
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map_err(|e| format!("a loopback port for TEI: {e}"))?
        .port();
    let (pool, batch, seq) = (pooling(m.pooling()), m.rows.row_count() as u32, m.rows.seq);
    let run = native_argv(&bin.to_string_lossy(), &dir.to_string_lossy(), port, dtype, pool, batch, seq);
    let mut log = log;
    log.commands.push(native_argv(TEI_BIN, docker::TEI_MODEL, port, dtype, pool, batch, seq));
    let out = std::env::temp_dir().join(format!("turbo-bench-tei-{}.log", std::process::id()));
    let file = fs::File::create(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let err = file.try_clone().map_err(|e| format!("{}: {e}", out.display()))?;
    let child = std::process::Command::new(&run[0])
        .args(&run[1..])
        .env("HF_HUB_OFFLINE", "1")
        .stdout(file)
        .stderr(err)
        .spawn()
        .map_err(|e| {
            let _ = fs::remove_file(&out);
            format!("{}: {e}", run.join(" "))
        })?;
    let mut server = Native(child, out.clone());
    let base = format!("http://127.0.0.1:{port}");
    let start = Instant::now();
    while get(&format!("{base}/health")).is_err() {
        let exited = server.0.try_wait().map_err(|e| format!("TEI: {e}"))?;
        if exited.is_some() || start.elapsed() > START_TIMEOUT {
            let logs = fs::read_to_string(&out).unwrap_or_default();
            return Err(format!("TEI did not become healthy within {START_TIMEOUT:?}:\n{logs}"));
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    against(&base, pinned, log, what, threads, m, warmup, iterations, None)
}

/// Check, warm up and time the server at `base`, as the procedure says.
#[allow(clippy::too_many_arguments)]
fn against(
    base: &str,
    image: &str,
    log: Log,
    what: String,
    threads: String,
    m: &Measurement,
    warmup: u32,
    iterations: u32,
    profile: Option<&kernels::Profile>,
) -> Result<ReferenceRun> {
    let procedure = format!("{what}; {threads}");
    let mut log = log;
    let info = parse_info(&get(&format!("{base}/info"))?)?;

    // The rows as TEI will see them.
    let distinct: Vec<usize> = {
        let mut seen = std::collections::BTreeSet::new();
        (0..m.rows.row_count()).filter(|&r| seen.insert(m.rows.cases[r])).collect()
    };
    let sent: Vec<Vec<i32>> = distinct.iter().map(|&r| m.rows.live(r).to_vec()).collect();
    let texts: Vec<String> = serde_json::from_str(&post(
        &format!("{base}/decode"),
        &json!({ "ids": sent, "skip_special_tokens": false }).to_string(),
    )?)
    .map_err(|e| format!("TEI /decode: {e}"))?;
    let back = parse_tokenize(&post(
        &format!("{base}/tokenize"),
        &json!({ "inputs": texts, "add_special_tokens": false }).to_string(),
    )?)?;
    if let Some((i, a, b)) = first_changed(&sent, &back) {
        let mut run = not_run(
            image,
            log,
            &procedure,
            format!(
                "TEI re-tokenizes token ids, and case {} comes back as other ids ({a:?} became {b:?}): \
                 it would not run the same rows",
                m.rows.cases[distinct[i]]
            ),
        );
        run.version = info.version;
        return Ok(run);
    }

    let body = embed_body(&m.rows, m.normalize());
    let url = format!("{base}/embed");
    for _ in 0..warmup {
        post(&url, &body)?;
    }
    let metrics = || get(&format!("{base}/metrics")).ok().and_then(|t| parse_batch_metrics(&t));
    let before = metrics();
    let mut ms = Vec::with_capacity(iterations as usize);
    let mut own = Vec::with_capacity(iterations as usize);
    let mut last = String::new();
    let started = Instant::now();
    for _ in 0..iterations {
        let t = Instant::now();
        let (text, timing) = post_timed(&url, &body)?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        last = text;
        own.push(timing);
    }
    let total = started.elapsed().as_secs_f64();
    let kernel_time = match profile {
        Some(p) => Some(kernels::measure(&mut log, p, iterations, || post(&url, &body).map(|_| ()))?),
        None => None,
    };
    let min_row_tokens = (0..m.rows.row_count()).map(|r| m.rows.live(r).len()).min().unwrap_or(0);
    let batches = batches_text(before.as_ref(), metrics().as_ref(), iterations, warmup, min_row_tokens);
    let vectors = parse_embed(&last, m.rows.row_count(), m.model.dim as usize)?;
    let min_cosine = vectors.iter().zip(&m.expected).map(|(v, want)| cosine(v, want)).fold(1.0, f64::min);
    ms.sort_by(f64::total_cmp);
    let mut procedure = format!("{what}; {}; {batches}; {threads}", timing_text(&ms, &own));
    if profile.is_some() {
        procedure = format!("{procedure}; {}", kernels::procedure(iterations));
    }
    Ok(ReferenceRun {
        name: NAME.into(),
        role: "end_to_end".into(),
        pinned: image.into(),
        version: format!("{} ({})", info.version, info.model_dtype),
        commands: log.commands,
        procedure,
        measured: Some(Measured {
            iterations: iterations as u64,
            p50_ms: percentile(&ms, 50.0),
            p99_ms: percentile(&ms, 99.0),
            rows_per_second: m.rows.row_count() as f64 * iterations as f64 / total,
            min_cosine: Some(min_cosine),
            computed_tokens: computed_tokens(&m.rows),
        }),
        not_run: None,
        kernel_time,
    })
}
