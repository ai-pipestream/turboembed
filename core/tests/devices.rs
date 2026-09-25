//! Devices, the capability matrix and selection, through the C interface,
//! with the backends this build links. The CPU backend lists the host
//! processor, which is real hardware on every machine the tests run on.

mod common;

use std::ffi::c_char;

use common::new_error;
use turbo::*;

struct Rt(*mut turbo_runtime);

impl Rt {
    fn new() -> Rt {
        let mut rt = std::ptr::null_mut();
        let mut err = new_error();
        assert_eq!(unsafe { turbo_runtime_create(std::ptr::null(), &mut rt, &mut err) }, 0);
        Rt(rt)
    }

    fn count(&self) -> u32 {
        let mut n = 0;
        assert_eq!(unsafe { turbo_runtime_device_count(self.0, &mut n, std::ptr::null_mut()) }, 0);
        n
    }

    fn info(&self, i: u32) -> Result<turbo_device_info, (i32, String)> {
        let mut out: turbo_device_info = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_device_info>() as u32;
        let mut err = new_error();
        match unsafe { turbo_runtime_device_info(self.0, i, &mut out, &mut err) } {
            0 => Ok(out),
            rc => Err((rc, s(&err.message))),
        }
    }

    fn capability(&self, i: u32, task: u32, precision: u32) -> Result<turbo_capability, (i32, String)> {
        let mut out: turbo_capability = unsafe { std::mem::zeroed() };
        out.struct_size = size_of::<turbo_capability>() as u32;
        let mut err = new_error();
        match unsafe { turbo_runtime_capability(self.0, i, task, precision, &mut out, &mut err) } {
            0 => Ok(out),
            rc => Err((rc, s(&err.message))),
        }
    }
}

impl Rt {
    /// The first device that is not a CPU and runs embed: what selection
    /// picks while no cell has a benchmark record.
    fn first_gpu_offering_embed(&self) -> Option<u32> {
        (0..self.count()).find(|&i| {
            self.info(i).unwrap().kind != TURBO_DEVICE_CPU
                && self.capability(i, TURBO_TASK_EMBED, TURBO_PRECISION_MODEL).unwrap().status != 0
        })
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

fn s(b: &[c_char]) -> String {
    b.iter().take_while(|&&c| c != 0).map(|&c| char::from(u8::from_ne_bytes(c.to_ne_bytes()))).collect()
}

fn cpu(rt: &Rt) -> u32 {
    (0..rt.count()).find(|&i| rt.info(i).unwrap().kind == TURBO_DEVICE_CPU).expect("the cpu backend lists the host")
}

#[test]
fn the_host_processor_is_listed() {
    let rt = Rt::new();
    let info = rt.info(cpu(&rt)).unwrap();
    assert_eq!(s(&info.backend), "cpu");
    assert_eq!(s(&info.arch), std::env::consts::ARCH);
    assert_eq!(info.unified_memory, 1);
    assert_eq!(s(&info.runtime_version), "", "the cpu backend links no vendor runtime");
    if cfg!(target_os = "linux") {
        assert!(!s(&info.name).is_empty(), "/proc/cpuinfo names the processor");
        assert!(
            info.memory_total > 0 && info.memory_free <= info.memory_total,
            "{} {}",
            info.memory_total,
            info.memory_free
        );
    }
}

#[test]
fn the_version_lists_the_linked_backends() {
    let v = unsafe { std::ffi::CStr::from_ptr(turbo_version()) }.to_str().unwrap();
    // In device order: a GPU backend's devices come before the CPU's.
    let mut want = String::from("0.1.0");
    for (on, name) in [
        (cfg!(feature = "cuda"), "cuda"),
        (cfg!(feature = "levelzero"), "levelzero"),
        (cfg!(feature = "metal"), "metal"),
        (cfg!(feature = "hailo"), "hailo"),
        (true, "cpu"),
    ] {
        if on {
            want = want + " " + name;
        }
    }
    assert_eq!(v, want);
}

#[test]
fn a_device_index_past_the_list_is_refused() {
    let rt = Rt::new();
    let (rc, msg) = rt.info(rt.count()).unwrap_err();
    assert_eq!(rc, status::INVALID_ARGUMENT, "{msg}");
    let (rc, _) = rt.capability(rt.count(), TURBO_TASK_EMBED, TURBO_PRECISION_MODEL).unwrap_err();
    assert_eq!(rc, status::INVALID_ARGUMENT);
}

#[test]
fn a_wrong_struct_size_is_refused() {
    let rt = Rt::new();
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = 8;
    let rc = unsafe { turbo_runtime_device_info(rt.0, 0, &mut info, std::ptr::null_mut()) };
    assert_eq!(rc, status::INVALID_STRUCT_SIZE);
    let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
    cap.struct_size = 8;
    let rc = unsafe { turbo_runtime_capability(rt.0, 0, TURBO_TASK_EMBED, 0, &mut cap, std::ptr::null_mut()) };
    assert_eq!(rc, status::INVALID_STRUCT_SIZE);
}

#[test]
fn an_unknown_task_or_precision_is_refused() {
    let rt = Rt::new();
    let i = cpu(&rt);
    assert_eq!(rt.capability(i, 9, TURBO_PRECISION_MODEL).unwrap_err().0, status::INVALID_ENUM);
    assert_eq!(rt.capability(i, 0, TURBO_PRECISION_MODEL).unwrap_err().0, status::INVALID_ENUM);
    assert_eq!(rt.capability(i, TURBO_TASK_EMBED, 3).unwrap_err().0, status::INVALID_ENUM);
}

#[test]
fn the_cpu_runs_embed_unmeasured_in_f32() {
    let rt = Rt::new();
    let i = cpu(&rt);
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let cap = rt.capability(i, TURBO_TASK_EMBED, p).unwrap();
        assert_eq!(cap.status, backend::TURBO_CAP_EXPERIMENTAL, "it runs; no record measures it");
        assert_eq!(s(&cap.reason), "no benchmark record for this cell");
        assert_eq!(s(&cap.benchmark), "", "no record backs it");
        assert_eq!((cap.cosine_floor, cap.speed_ratio), (0.0, 0.0), "no numbers without a record");
        assert_eq!(cap.dtype, TURBO_DTYPE_F32, "FASTEST too: F32 is the one dtype it computes in");
        assert_eq!(cap.options_honored, 0b111111, "every field of turbo_embed_options");
    }
}

#[test]
fn selection_never_picks_a_cpu_and_says_why() {
    let rt = Rt::new();
    let mut pick = 99u32;
    let mut reason = [0 as c_char; 256];
    let mut err = new_error();
    let rc = unsafe {
        turbo_runtime_select(rt.0, TURBO_TASK_EMBED, &mut pick, reason.as_mut_ptr(), reason.len() as u32, &mut err)
    };
    // A GPU that runs the task is picked, the first in device order.
    if let Some(first) = rt.first_gpu_offering_embed() {
        assert_eq!(rc, 0, "{}", s(&err.message));
        assert_eq!(pick, first, "{}", s(&reason));
        return;
    }
    // Else nothing can be selected, though the cpu runs the task.
    assert_eq!(rc, status::DEVICE_NOT_FOUND);
    assert_eq!(pick, 99, "out is untouched on failure");
    assert!(s(&reason).contains("a CPU is never selected"), "{}", s(&reason));
    assert_eq!(s(&reason), s(&err.message));
    let rc = unsafe { turbo_runtime_select(rt.0, 7, &mut pick, std::ptr::null_mut(), 0, std::ptr::null_mut()) };
    assert_eq!(rc, status::INVALID_ENUM, "an unknown task is refused even when no device would be tried");
}

#[test]
fn device_info_is_read_at_query_time() {
    // The host's free memory moves between calls; everything else is the
    // same device.
    let rt = Rt::new();
    let i = cpu(&rt);
    let (mut a, mut b) = (rt.info(i).unwrap(), rt.info(i).unwrap());
    let mut free = Vec::new();
    for _ in 0..50 {
        let _held = std::hint::black_box(vec![1u8; 64 << 20]);
        free.push(rt.info(i).unwrap().memory_free);
    }
    if cfg!(target_os = "linux") {
        free.sort();
        free.dedup();
        assert!(free.len() > 1, "memory_free is read again on each call: {free:?}");
    }
    a.memory_free = 0;
    b.memory_free = 0;
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

#[test]
fn null_handles_and_outputs_are_refused() {
    let rt = Rt::new();
    let null = std::ptr::null_mut::<turbo_runtime>();
    let mut n = 0u32;
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
    cap.struct_size = size_of::<turbo_capability>() as u32;
    let p = TURBO_PRECISION_MODEL;
    unsafe {
        assert_eq!(turbo_runtime_device_count(null, &mut n, null_err()), status::INVALID_HANDLE);
        assert_eq!(turbo_runtime_device_info(null, 0, &mut info, null_err()), status::INVALID_HANDLE);
        assert_eq!(
            turbo_runtime_capability(null, 0, TURBO_TASK_EMBED, p, &mut cap, null_err()),
            status::INVALID_HANDLE
        );
        let rc = turbo_runtime_select(null, TURBO_TASK_EMBED, &mut n, std::ptr::null_mut(), 0, null_err());
        assert_eq!(rc, status::INVALID_HANDLE);

        assert_eq!(turbo_runtime_device_count(rt.0, std::ptr::null_mut(), null_err()), status::INVALID_ARGUMENT);
        assert_eq!(turbo_runtime_device_info(rt.0, 0, std::ptr::null_mut(), null_err()), status::INVALID_ARGUMENT);
        let rc = turbo_runtime_capability(rt.0, 0, TURBO_TASK_EMBED, p, std::ptr::null_mut(), null_err());
        assert_eq!(rc, status::INVALID_ARGUMENT);
        let rc =
            turbo_runtime_select(rt.0, TURBO_TASK_EMBED, std::ptr::null_mut(), std::ptr::null_mut(), 0, null_err());
        assert_eq!(rc, status::INVALID_ARGUMENT);
    }
}

fn null_err() -> *mut turbo_error {
    std::ptr::null_mut()
}

#[test]
fn the_select_reason_fits_the_buffer_it_is_given() {
    let rt = Rt::new();
    let mut pick = 0u32;
    let want = if rt.first_gpu_offering_embed().is_some() { 0 } else { status::DEVICE_NOT_FOUND };
    let mut reason = [b'x' as c_char; 16];
    let rc = unsafe { turbo_runtime_select(rt.0, TURBO_TASK_EMBED, &mut pick, reason.as_mut_ptr(), 8, null_err()) };
    assert_eq!(rc, want);
    assert_eq!(reason[7], 0, "cut and terminated within 8 bytes");
    assert_eq!(s(&reason).len(), 7);
    assert!(reason[8..].iter().all(|&c| c == b'x' as c_char), "nothing written past reason_len");

    let mut reason = [b'x' as c_char; 16];
    let rc = unsafe { turbo_runtime_select(rt.0, TURBO_TASK_EMBED, &mut pick, reason.as_mut_ptr(), 0, null_err()) };
    assert_eq!(rc, want);
    assert!(reason.iter().all(|&c| c == b'x' as c_char), "reason_len 0 writes nothing");
}

#[test]
fn the_device_index_is_checked_before_the_task() {
    let rt = Rt::new();
    assert_eq!(rt.capability(rt.count(), 9, 7).unwrap_err().0, status::INVALID_ARGUMENT);
}

#[test]
fn selection_ranks_by_status_then_by_speed_ratio() {
    let cell = |status: u32, speed_ratio: f32| {
        let mut c: turbo_capability = unsafe { std::mem::zeroed() };
        c.status = status;
        c.speed_ratio = speed_ratio;
        c
    };
    use backend::{TURBO_CAP_EXPERIMENTAL as EXP, TURBO_CAP_SUPPORTED as SUP};
    assert!(ranks_above(&cell(SUP, 0.0), &cell(EXP, 0.5)), "a higher status wins whatever the speed");
    assert!(!ranks_above(&cell(EXP, 0.5), &cell(SUP, 0.0)));
    assert!(ranks_above(&cell(SUP, 0.8), &cell(SUP, 1.2)), "among equals the lower ratio wins");
    assert!(!ranks_above(&cell(SUP, 1.2), &cell(SUP, 0.8)));
    assert!(ranks_above(&cell(SUP, 3.0), &cell(SUP, 0.0)), "0 is no record and loses to any record");
    assert!(!ranks_above(&cell(SUP, 0.0), &cell(SUP, 3.0)));
    for c in [cell(EXP, 0.0), cell(SUP, 1.0)] {
        assert!(!ranks_above(&c, &c) && !ranks_above(&c, &c), "equal cells tie, and the first is kept");
    }
}
