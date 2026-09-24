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

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe { turbo_runtime_release(self.0) };
    }
}

fn s(b: &[c_char]) -> String {
    b.iter().take_while(|&&c| c != 0).map(|&c| c as u8 as char).collect()
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
    assert_eq!(v, "0.1.0 cpu");
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
fn the_cpu_says_embed_is_not_built_and_why() {
    let rt = Rt::new();
    let i = cpu(&rt);
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let cap = rt.capability(i, TURBO_TASK_EMBED, p).unwrap();
        assert_eq!(cap.status, backend::TURBO_CAP_UNSUPPORTED);
        assert!(!s(&cap.reason).is_empty(), "an unsupported cell says why");
        assert_eq!(s(&cap.benchmark), "", "no record backs it");
        assert_eq!((cap.cosine_floor, cap.speed_ratio), (0.0, 0.0), "no numbers without a record");
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
    // This build links only the cpu backend, so nothing can be selected.
    assert_eq!(rc, status::DEVICE_NOT_FOUND);
    assert_eq!(pick, 99, "out is untouched on failure");
    assert!(s(&reason).contains("a CPU is never selected"), "{}", s(&reason));
    assert_eq!(s(&reason), s(&err.message));
    let rc = unsafe { turbo_runtime_select(rt.0, 7, &mut pick, std::ptr::null_mut(), 0, std::ptr::null_mut()) };
    assert_eq!(rc, status::INVALID_ENUM, "an unknown task is refused even when no device would be tried");
}
