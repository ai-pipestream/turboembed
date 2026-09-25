//! `--cpus`: the processors a CPU measurement runs on, the same for the
//! library and for TEI's container, and the thread counts both are given.
//!
//! The tool pins its own process to the list (sched_setaffinity, which
//! the session's threads inherit) and sets TURBO_CPU_THREADS to its
//! length (docs/cpu.md): one thread per logical CPU, since the library's
//! kernels are written to share a core's two hardware threads. TEI is
//! started with `--cpuset-cpus` and every thread setting its CPU image
//! reads: MKL's (MKL_VARS) at the list's physical cores, since MKL's
//! threads contend for a core's vector units and MKL itself defaults to
//! one per core, and RAYON_NUM_THREADS at the list's length.

use std::path::Path;

use crate::Result;

/// The environment variables MKL reads for its thread count in TEI's CPU
/// image, which links MKL's OpenMP threading for its matrix products.
/// Each is set to the list's physical cores.
pub const MKL_VARS: [&str; 2] = ["OMP_NUM_THREADS", "MKL_NUM_THREADS"];

/// The variable candle's other operators read in TEI's CPU image (its
/// Dockerfile sets it to 8), set to the list's length. ONNX Runtime's
/// intra-op threads and the router's tokenization workers are counted
/// from the processors the container may use, which `--cpuset-cpus`
/// sets.
pub const RAYON_VAR: &str = "RAYON_NUM_THREADS";

/// Every thread setting TEI's CPU image reads.
pub const THREAD_VARS: [&str; 3] = [MKL_VARS[0], MKL_VARS[1], RAYON_VAR];

/// The library's thread count variable.
pub const LIBRARY_VAR: &str = "TURBO_CPU_THREADS";

/// Where Linux describes each processor's place: cpuN/topology/.
pub const SYSFS_CPUS: &str = "/sys/devices/system/cpu";

/// The most processors a list may name: the size of glibc's cpu_set_t.
const MAX_CPU: usize = 1024;

/// A list of processors, as `0-15` or `0-7,16-23` names them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cpus {
    /// As given, which is how docker's --cpuset-cpus takes it too.
    pub list: String,
    /// Each processor once, in increasing order.
    pub ids: Vec<usize>,
    /// The physical cores they are on.
    pub cores: usize,
}

impl Cpus {
    /// Ranges and single numbers, comma-separated, no processor twice;
    /// their cores read from `sysfs`, a tree laid out as SYSFS_CPUS.
    pub fn parse(list: &str, sysfs: &Path) -> Result<Cpus> {
        let ids = parse_ids(list)?;
        let cores = physical_cores(&ids, sysfs)?;
        Ok(Cpus { list: list.to_owned(), ids, cores })
    }

    /// The library's thread count, and TEI's RAYON_NUM_THREADS: one per
    /// logical CPU.
    pub fn threads(&self) -> usize {
        self.ids.len()
    }

    /// The `docker run` options that give the container these processors
    /// and its thread counts.
    pub fn docker_args(&self) -> Vec<String> {
        let mut a = vec!["--cpuset-cpus".to_owned(), self.list.clone()];
        for (var, n) in self.tei_threads() {
            a.push("--env".into());
            a.push(format!("{var}={n}"));
        }
        a
    }

    /// Each of THREAD_VARS with the count TEI is given.
    pub fn tei_threads(&self) -> [(&'static str, usize); 3] {
        [(MKL_VARS[0], self.cores), (MKL_VARS[1], self.cores), (RAYON_VAR, self.threads())]
    }
}

/// The processors a list names, each once, in increasing order.
fn parse_ids(list: &str) -> Result<Vec<usize>> {
    let bad = |why: &str| format!("--cpus {list:?}: {why}; give processors as 0-15 or 0-7,16-23");
    let mut ids = Vec::new();
    for part in list.split(',') {
        let num = |s: &str| s.parse::<usize>().map_err(|_| bad(&format!("{s:?} is not a processor number")));
        let (lo, hi) = match part.split_once('-') {
            Some((a, b)) => (num(a)?, num(b)?),
            None => (num(part)?, num(part)?),
        };
        if lo > hi {
            return Err(bad(&format!("{part} runs backwards")));
        }
        if hi >= MAX_CPU {
            return Err(bad(&format!("processor {hi} is past {}", MAX_CPU - 1)));
        }
        ids.extend(lo..=hi);
    }
    ids.sort_unstable();
    if ids.windows(2).any(|w| w[0] == w[1]) {
        return Err(bad("a processor is named twice"));
    }
    Ok(ids)
}

/// The distinct (package, core) pairs of `ids`, from each processor's
/// topology/physical_package_id and topology/core_id under `sysfs`.
pub fn physical_cores(ids: &[usize], sysfs: &Path) -> Result<usize> {
    let mut cores = std::collections::BTreeSet::new();
    for &id in ids {
        let read = |f: &str| {
            let p = sysfs.join(format!("cpu{id}/topology/{f}"));
            let v = std::fs::read_to_string(&p).map_err(|e| format!("--cpus: processor {id}: {}: {e}", p.display()))?;
            v.trim().parse::<i64>().map_err(|_| format!("--cpus: processor {id}: {} is {v:?}", p.display()))
        };
        cores.insert((read("physical_package_id")?, read("core_id")?));
    }
    Ok(cores.len())
}

/// The library's thread count, as session_create will read it: `var`
/// (TURBO_CPU_THREADS) when set, else the processors this process may
/// run on.
pub fn library_threads(var: Option<&str>) -> Result<usize> {
    match var {
        None => Ok(std::thread::available_parallelism().map_or(1, std::num::NonZero::get)),
        Some(v) => v
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&n| n >= 1)
            .ok_or_else(|| format!("{LIBRARY_VAR} {v:?} is not a count")),
    }
}

/// Pin this process to `cpus` and set TURBO_CPU_THREADS to their count,
/// then check the pin took: the processors this process may run on are
/// exactly the list. Call it before any thread is started, which the
/// session's threads then inherit it from.
#[cfg(target_os = "linux")]
pub fn pin(cpus: &Cpus) -> Result<()> {
    if let Ok(v) = std::env::var(LIBRARY_VAR)
        && library_threads(Some(&v)).ok() != Some(cpus.threads())
    {
        return Err(format!("{LIBRARY_VAR}={v} is set, and --cpus {} names {}", cpus.list, cpus.threads()));
    }
    let mut set = [0u64; MAX_CPU / 64];
    for &c in &cpus.ids {
        set[c / 64] |= 1 << (c % 64);
    }
    // SAFETY: `set` is a cpu_set_t of glibc's size (1024 bits), passed
    // with that size; pid 0 is the calling thread, the only one there is.
    let rc = unsafe { sched_setaffinity(0, size_of_val(&set), set.as_ptr()) };
    if rc != 0 {
        return Err(format!("--cpus {}: sched_setaffinity failed: {}", cpus.list, std::io::Error::last_os_error()));
    }
    let mut got = [0u64; MAX_CPU / 64];
    // SAFETY: as above, written by the call.
    let rc = unsafe { sched_getaffinity(0, size_of_val(&got), got.as_mut_ptr()) };
    if rc != 0 || got != set {
        return Err(format!(
            "--cpus {}: this process may not run on all of them (its cgroup's cpuset leaves some out)",
            cpus.list
        ));
    }
    // SAFETY: no other thread runs yet to read the environment meanwhile.
    unsafe { std::env::set_var(LIBRARY_VAR, cpus.threads().to_string()) };
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn pin(cpus: &Cpus) -> Result<()> {
    Err(format!("--cpus {}: pinning is done on Linux only", cpus.list))
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
    fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
}

/// The value `var` has in an image's environment (`KEY=value` entries,
/// as `docker image inspect` gives Config.Env), if it sets one.
pub fn image_value<'a>(env: &'a [String], var: &str) -> Option<&'a str> {
    env.iter().find_map(|e| e.strip_prefix(var).and_then(|r| r.strip_prefix('=')))
}

/// What a record's procedure says of the processors and threads each
/// side ran with. `library` is the library's thread count, None when the
/// device measured is not the CPU; `image_env` is the TEI image's
/// environment.
pub fn procedure(cpus: Option<&Cpus>, library: Option<usize>, image_env: &[String]) -> String {
    let ours = match (cpus, library) {
        (Some(c), Some(n)) => {
            format!("the library ran pinned to CPUs {} with {LIBRARY_VAR}={n} threads, one per logical CPU", c.list)
        }
        (Some(c), None) => format!("the tool ran pinned to CPUs {}", c.list),
        (None, Some(n)) => format!("the library ran unpinned with {n} threads"),
        (None, None) => "the tool ran unpinned".to_owned(),
    };
    let tei = match cpus {
        Some(c) => {
            let set: Vec<String> = c.tei_threads().iter().map(|(v, n)| format!("{v}={n}")).collect();
            let mut s = format!(
                "TEI ran with --cpuset-cpus {} and {} (MKL's threads one per physical core of the list, as MKL \
                 itself defaults, since two on a core contend for its vector units; candle's rayon threads one per \
                 logical CPU)",
                c.list,
                set.join(", ")
            );
            let over: Vec<String> = c
                .tei_threads()
                .iter()
                .filter_map(|(v, n)| {
                    image_value(image_env, v).filter(|x| *x != n.to_string()).map(|x| format!("{v}={x}"))
                })
                .collect();
            if !over.is_empty() {
                s += &format!(", over the image's {}", over.join(", "));
            }
            s + "; its ONNX Runtime and tokenizer threads counted from those CPUs"
        }
        None => {
            let vars = THREAD_VARS.map(|v| format!("{v}={}", image_value(image_env, v).unwrap_or("unset")));
            format!("TEI ran unpinned with the image's thread settings: {}", vars.join(", "))
        }
    };
    format!("{ours}; {tei}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_names_each_processor_once() {
        assert_eq!(parse_ids("0-3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_ids("16-17,0-1,8").unwrap(), vec![0, 1, 8, 16, 17]);
        for bad in ["", "3-1", "0-3,2", "a", "0-", "-3", "0,,1", "1024", "0-1024", " 1"] {
            assert!(parse_ids(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_library_count_is_the_variable_or_the_processors() {
        assert_eq!(library_threads(Some("12")), Ok(12));
        assert!(library_threads(Some("0")).is_err());
        assert!(library_threads(None).unwrap() >= 1);
    }
}
