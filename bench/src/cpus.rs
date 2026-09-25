//! `--cpus`: the processors a CPU measurement runs on, the same for the
//! library and for TEI's container, and the thread counts both are given.
//!
//! The tool pins its own process to the list (sched_setaffinity, which
//! the session's threads inherit) and sets TURBO_CPU_THREADS to its
//! length (docs/cpu.md). TEI is started with `--cpuset-cpus` and every
//! thread setting its CPU image reads (THREAD_VARS) at the same count.

use crate::Result;

/// The environment variables TEI's CPU image reads for its thread counts:
/// OMP_NUM_THREADS and MKL_NUM_THREADS for MKL's matrix products (it
/// links MKL's OpenMP threading), RAYON_NUM_THREADS for candle's other
/// operators (the image's Dockerfile sets it to 8). ONNX Runtime's
/// intra-op threads and the router's tokenization workers are counted
/// from the processors the container may use, which `--cpuset-cpus`
/// sets.
pub const THREAD_VARS: [&str; 3] = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"];

/// The library's thread count variable.
pub const LIBRARY_VAR: &str = "TURBO_CPU_THREADS";

/// The most processors a list may name: the size of glibc's cpu_set_t.
const MAX_CPU: usize = 1024;

/// A list of processors, as `0-15` or `0-7,16-23` names them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cpus {
    /// As given, which is how docker's --cpuset-cpus takes it too.
    pub list: String,
    /// Each processor once, in increasing order.
    pub ids: Vec<usize>,
}

impl Cpus {
    /// Ranges and single numbers, comma-separated; no processor twice.
    pub fn parse(list: &str) -> Result<Cpus> {
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
        Ok(Cpus { list: list.to_owned(), ids })
    }

    /// The thread count both sides are given: one per processor.
    pub fn threads(&self) -> usize {
        self.ids.len()
    }

    /// The `docker run` options that give the container these processors
    /// and the same thread count.
    pub fn docker_args(&self) -> Vec<String> {
        let mut a = vec!["--cpuset-cpus".to_owned(), self.list.clone()];
        for var in THREAD_VARS {
            a.push("--env".into());
            a.push(format!("{var}={}", self.threads()));
        }
        a
    }
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
        (Some(c), Some(n)) => format!("the library ran pinned to CPUs {} with {LIBRARY_VAR}={n} threads", c.list),
        (Some(c), None) => format!("the tool ran pinned to CPUs {}", c.list),
        (None, Some(n)) => format!("the library ran unpinned with {n} threads"),
        (None, None) => "the tool ran unpinned".to_owned(),
    };
    let tei = match cpus {
        Some(c) => {
            let n = c.threads();
            let mut s = format!("TEI ran with --cpuset-cpus {} and ", c.list);
            s += &THREAD_VARS.map(|v| format!("{v}={n}")).join(", ");
            let over: Vec<String> = THREAD_VARS
                .iter()
                .filter_map(|v| image_value(image_env, v).filter(|x| *x != n.to_string()).map(|x| format!("{v}={x}")))
                .collect();
            if !over.is_empty() {
                s += &format!(" (over the image's {})", over.join(", "));
            }
            s + ", its ONNX Runtime and tokenizer threads counted from those CPUs"
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
        assert_eq!(Cpus::parse("0-3").unwrap().ids, vec![0, 1, 2, 3]);
        assert_eq!(Cpus::parse("16-17,0-1,8").unwrap().ids, vec![0, 1, 8, 16, 17]);
        assert_eq!(Cpus::parse("5").unwrap().threads(), 1);
        for bad in ["", "3-1", "0-3,2", "a", "0-", "-3", "0,,1", "1024", "0-1024", " 1"] {
            assert!(Cpus::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_library_count_is_the_variable_or_the_processors() {
        assert_eq!(library_threads(Some("12")), Ok(12));
        assert!(library_threads(Some("0")).is_err());
        assert!(library_threads(None).unwrap() >= 1);
    }
}
