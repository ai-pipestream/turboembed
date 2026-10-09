//! The threads a session computes on: made when the session is, parked
//! between runs, and joined when it is released.
//!
//! A run hands the pool one job at a time: a function of a task index and
//! a thread index, and how many tasks there are. The caller's thread is
//! thread 0 and takes tasks too. Each task is taken by exactly one thread,
//! so which thread computes a task varies from run to run; the encoder
//! splits work so that a task's result does not depend on that
//! (encoder.rs).
//!
//! Tasks are claimed from one atomic word that holds the job's number, its
//! task count and the next task, so a claim is either of the current job
//! or fails: a worker that was descheduled over the end of a job cannot
//! take a task of the next one with the old one's function, and the caller
//! waits only for the job's tasks, not for every worker to look in.
//!
//! A worker that waited past its spin parks on its own thread, with a
//! flag saying so. A job wakes parked workers as a tree: the caller wakes
//! a few, and each worker woken wakes a few more while the job has units
//! left, before it takes any. No lock is on the way: a pool of many
//! threads that all woke through one lock woke them one at a time, each
//! after the last had taken and left it.
//!
//! Nothing here allocates after `new`: a job is published through
//! atomics.

use std::hint::spin_loop;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{JoinHandle, Thread};
use std::time::{Duration, Instant};

/// A job's function.
type Task<'a> = dyn Fn(usize, usize) + Sync + 'a;

/// How long a worker of an encoder's pool waits for the next job before
/// it parks: an encoder's jobs follow each other in microseconds, and a
/// worker woken from a park costs one of them tens of microseconds.
pub(crate) const SPIN: Duration = Duration::from_micros(200);

/// The same for a pool that runs one job per batch, as the static table's
/// and the tokenizer's do: the next job is a batch away, so a worker parks
/// almost at once rather than burn a processor waiting for it.
pub(crate) const SPIN_BATCH: Duration = Duration::from_micros(20);

/// The claim word: the job's number in the top 32 bits, its units in the
/// next 16, the next unit to take in the low 16. A job of more tasks than
/// a unit count holds gives each unit several.
const UNITS: usize = 0xffff;

/// Parked workers each thread wakes for a job: the caller first, then
/// each worker it wakes, so a pool of n wakes in about log(n) steps.
const FAN: usize = 2;

fn word(job: u64, units: usize, next: usize) -> u64 {
    (job << 32) | ((units as u64) << 16) | next as u64
}

struct Shared {
    /// See UNITS.
    claim: AtomicU64,
    /// The current job's function: a pointer to the caller's reference to
    /// it, on the caller's stack for as long as the job runs.
    job: AtomicPtr<&'static Task<'static>>,
    /// Tasks per unit of the current job, and its tasks.
    per: AtomicUsize,
    tasks: AtomicUsize,
    /// Units of the current job done.
    done: AtomicUsize,
    stop: AtomicBool,
    /// How long a worker spins before it parks.
    spin: Duration,
    /// Each thread's flag, set while it is parked or about to be; the
    /// caller's, at 0, is never set. Whoever clears a set flag unparks
    /// that thread.
    parked: Box<[AtomicBool]>,
    /// The workers' threads, worker 1 first, from when `new` has made
    /// them.
    threads: OnceLock<Box<[Thread]>>,
}

pub(crate) struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// The number of the last job published.
    jobs: u64,
}

/// Aborts the process if a task panics: on a worker, the caller would
/// otherwise wait for its task forever; on the caller, it would leave
/// run() while workers may still use the job it borrowed.
struct AbortOnUnwind;

impl Drop for AbortOnUnwind {
    fn drop(&mut self) {
        if std::thread::panicking() {
            std::process::abort();
        }
    }
}

impl Pool {
    /// A pool of `threads` threads, the caller's among them, so
    /// `threads - 1` are spawned. Fewer are when the system will not
    /// start more; the pool computes the same with any number. Its
    /// workers spin for SPIN before they park.
    pub(crate) fn new(threads: usize) -> Pool {
        Pool::with_spin(threads, SPIN)
    }

    /// The same, with workers that spin for `spin` before they park.
    pub(crate) fn with_spin(threads: usize, spin: Duration) -> Pool {
        let shared = Arc::new(Shared {
            claim: AtomicU64::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            per: AtomicUsize::new(1),
            tasks: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            spin,
            parked: (0..threads.max(1)).map(|_| AtomicBool::new(false)).collect(),
            threads: OnceLock::new(),
        });
        let mut workers = Vec::with_capacity(threads.saturating_sub(1));
        for id in 1..threads.max(1) {
            let s = Arc::clone(&shared);
            let spawned = std::thread::Builder::new().name(format!("turbo-cpu-{id}")).spawn(move || worker(&s, id));
            match spawned {
                Ok(h) => workers.push(h),
                Err(_) => break,
            }
        }
        let _ = shared.threads.set(workers.iter().map(|w| w.thread().clone()).collect());
        Pool { shared, workers, jobs: 0 }
    }

    /// The threads a job runs on, the caller's included.
    pub(crate) fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// f(task, thread) for every task in 0..tasks, spread over the pool's
    /// threads; returns when all are done. `thread` is under threads().
    pub(crate) fn run(&mut self, tasks: usize, f: &Task<'_>) {
        // A task that panics on this thread must not unwind out of here
        // while workers may still call f, which borrows this frame.
        let _abort = AbortOnUnwind;
        if self.workers.is_empty() || tasks <= 1 {
            (0..tasks).for_each(|t| f(t, 0));
            return;
        }
        let s = &*self.shared;
        let per = tasks.div_ceil(UNITS);
        let units = tasks.div_ceil(per);
        self.jobs += 1;
        let mut r: &Task<'_> = f;
        // SAFETY: only the lifetime is erased. A worker dereferences this
        // pointer, and calls f, only after it claims a unit of this job,
        // and every unit is done before this function returns (below),
        // while `r` and `f` are still live.
        let job = unsafe { std::mem::transmute::<*mut &Task<'_>, *mut &'static Task<'static>>(&mut r) };
        s.job.store(job, Ordering::Relaxed);
        s.per.store(per, Ordering::Relaxed);
        s.tasks.store(tasks, Ordering::Relaxed);
        s.done.store(0, Ordering::Relaxed);
        // SeqCst, as the parked flags: a worker that sets its flag and
        // then reads the word either sees this job or is seen parked.
        s.claim.store(word(self.jobs, units, 0), Ordering::SeqCst);
        wake(s);
        take_units(s, 0);
        let mut spins = 0u32;
        while s.done.load(Ordering::Acquire) != units {
            spins += 1;
            if spins < 1 << 12 {
                spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// Unparks up to FAN parked workers while the current job has units no
/// thread has taken; a worker spinning takes its own.
fn wake(s: &Shared) {
    let Some(threads) = s.threads.get() else { return };
    let mut woken = 0;
    for (flag, t) in s.parked[1..].iter().zip(threads.iter()) {
        if woken == FAN || !has_units(s) {
            return;
        }
        if flag.load(Ordering::Relaxed) && flag.swap(false, Ordering::SeqCst) {
            t.unpark();
            woken += 1;
        }
    }
}

/// Whether the current job has a unit no thread has taken.
fn has_units(s: &Shared) -> bool {
    let w = s.claim.load(Ordering::Acquire);
    (w as usize & UNITS) < ((w >> 16) as usize & UNITS)
}

/// Claims and runs units of the current job until none is left. Returns
/// the claim word it saw last.
fn take_units(s: &Shared, thread: usize) -> u64 {
    let mut w = s.claim.load(Ordering::Acquire);
    loop {
        let (units, next) = ((w >> 16) as usize & UNITS, w as usize & UNITS);
        if next >= units {
            return w;
        }
        match s.claim.compare_exchange_weak(w, w + 1, Ordering::AcqRel, Ordering::Acquire) {
            Err(now) => w = now,
            Ok(_) => {
                // SAFETY: the claim succeeded, so the word was this job's
                // and the job was not done: the caller stored its function,
                // per and tasks before the word (Release, which the
                // exchange acquired) and keeps them until every unit is
                // done, this one included.
                let (f, per, tasks) = unsafe {
                    (*s.job.load(Ordering::Relaxed), s.per.load(Ordering::Relaxed), s.tasks.load(Ordering::Relaxed))
                };
                for t in next * per..((next + 1) * per).min(tasks) {
                    f(t, thread);
                }
                s.done.fetch_add(1, Ordering::Release);
                w += 1;
            }
        }
    }
}

fn worker(s: &Shared, id: usize) {
    let _abort = AbortOnUnwind;
    // The word the pool was made with: a worker that starts after the
    // first job, or after the pool stopped, sees the word moved.
    let mut seen = 0;
    loop {
        let start = Instant::now();
        let mut spins = 0u32;
        let moved = |s: &Shared| s.claim.load(Ordering::SeqCst) != seen || s.stop.load(Ordering::SeqCst);
        while !moved(s) {
            spins = spins.wrapping_add(1);
            if !spins.is_multiple_of(256) || start.elapsed() < s.spin {
                spin_loop();
                continue;
            }
            let flag = &s.parked[id];
            flag.store(true, Ordering::SeqCst);
            // A job published before the flag was set is seen here; one
            // published after it finds the flag and unparks this thread.
            while flag.load(Ordering::SeqCst) && !moved(s) {
                std::thread::park();
            }
            // Woken by a job, or saw it first: either way not parked. A
            // waker that cleared the flag meanwhile left an unpark token,
            // which at worst ends a later park early.
            flag.store(false, Ordering::SeqCst);
        }
        if s.stop.load(Ordering::Acquire) {
            return;
        }
        wake(s);
        seen = take_units(s, id);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let s = &*self.shared;
        s.stop.store(true, Ordering::Release);
        // A job of no units: every worker sees the word move, and stops.
        s.claim.store(word(self.jobs + 1, 0, 0), Ordering::SeqCst);
        if let Some(threads) = s.threads.get() {
            for (flag, t) in s.parked[1..].iter().zip(threads.iter()) {
                flag.store(false, Ordering::SeqCst);
                t.unpark();
            }
        }
        for w in self.workers.drain(..) {
            // A worker that panicked aborted the process; there is no
            // error to carry.
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_once_on_a_thread_of_the_pool() {
        for threads in [1, 2, 3, 8] {
            let mut p = Pool::new(threads);
            assert_eq!(p.threads(), threads);
            for tasks in [0, 1, 2, 7, 1000, 3 * UNITS + 5] {
                let hits: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
                p.run(tasks, &|t, th| {
                    assert!(th < threads);
                    hits[t].fetch_add(1, Ordering::Relaxed);
                });
                assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "{threads} threads, {tasks} tasks");
            }
        }
    }

    /// Workers that parked between jobs wake for the next one.
    #[test]
    fn a_parked_pool_wakes() {
        let mut p = Pool::new(4);
        for _ in 0..3 {
            std::thread::sleep(SPIN * 5);
            let n = AtomicUsize::new(0);
            p.run(64, &|_, _| {
                n.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(n.load(Ordering::Relaxed), 64);
        }
    }

    /// The same for a pool that parks at once, with fewer units than
    /// workers: those not woken still are when the next job wants them.
    #[test]
    fn a_batch_pool_wakes_the_workers_a_job_wants() {
        let mut p = Pool::with_spin(8, SPIN_BATCH);
        for tasks in [2, 3, 64, 2, 1000] {
            std::thread::sleep(SPIN * 5);
            let n = AtomicUsize::new(0);
            p.run(tasks, &|_, _| {
                n.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(n.load(Ordering::Relaxed), tasks);
        }
    }

    /// More workers than processors, parked between jobs or caught just
    /// as they park: every job's tasks run once, whatever the timing, and
    /// a job of more units than the first wake reaches still has every
    /// worker woken in turn.
    #[test]
    fn a_pool_of_many_parked_workers_wakes_them_as_a_tree() {
        let mut p = Pool::with_spin(32, SPIN_BATCH);
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        for _ in 0..300 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            std::thread::sleep(Duration::from_micros(x % 60));
            let tasks = 1 + (x >> 20) as usize % 200;
            let hits: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
            let threads: Vec<AtomicBool> = (0..32).map(|_| AtomicBool::new(false)).collect();
            p.run(tasks, &|t, th| {
                hits[t].fetch_add(1, Ordering::Relaxed);
                threads[th].store(true, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "{tasks} tasks");
        }
        // A long job: workers beyond the caller's FAN take part.
        std::thread::sleep(SPIN * 5);
        let threads: Vec<AtomicBool> = (0..32).map(|_| AtomicBool::new(false)).collect();
        p.run(64, &|_, th| {
            threads[th].store(true, Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(20));
        });
        let used = threads.iter().filter(|t| t.load(Ordering::Relaxed)).count();
        assert!(used > 1 + FAN, "{used} threads took tasks");
    }

    /// Many short jobs back to back, as a run's steps are.
    #[test]
    fn short_jobs_follow_each_other() {
        let mut p = Pool::new(4);
        let n = AtomicUsize::new(0);
        for j in 0..20_000 {
            p.run(2 + j % 5, &|_, _| {
                n.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(n.load(Ordering::Relaxed), (0..20_000).map(|j| 2 + j % 5).sum::<usize>());
    }
}
