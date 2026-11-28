//! Exercise the public reporting ABI without running inference. Only the
//! enumerated device and its backend's advertised encoder cell are fixtures.
use super::*;

const ARCH: &str = "rtx4080super";
const GPU: &str = "NVIDIA GeForce RTX 4080 SUPER";
#[cfg(target_os = "linux")]
const BGE: &str = "rtx4080super.cuda.rerank.exact.bge-reranker-base-dce186f3.66bbe5e2ff6e.json";

static CUDA: backend::turbo_backend = backend::turbo_backend { name: c"cuda".as_ptr(), ..crate::cpu::BACKEND };

fn report(
    table: &'static backend::turbo_backend,
    arch: &str,
    name: &str,
    task: u32,
    precision: u32,
) -> turbo_capability {
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.kind = TURBO_DEVICE_GPU;
    write_str(&mut info.arch, arch);
    write_str(&mut info.name, name);
    write_str(&mut info.vendor, "NVIDIA");
    let device = Device { backend: table, ordinal: 0, info };
    let inner = Runtime { log: None, log_user_data: 0, devices: vec![device], tune_cache: tuning::Cache::default() };
    let mut rt = turbo_runtime { magic: RUNTIME_MAGIC, inner: Arc::new(inner) };
    let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
    cap.struct_size = size_of::<turbo_capability>() as u32;
    assert_eq!(unsafe { turbo_runtime_capability(&mut rt, 0, task, precision, &mut cap, std::ptr::null_mut()) }, 0);
    cap
}

#[test]
#[cfg(target_os = "linux")]
fn exact_rerank_reports_the_newest_reference_backed_record() {
    let cap = report(&CUDA, ARCH, GPU, TURBO_TASK_RERANK, TURBO_PRECISION_EXACT);
    assert_eq!(cap.status, backend::TURBO_CAP_SUPPORTED, "{}", cstr(&cap.reason));
    assert_eq!(cap.dtype, TURBO_DTYPE_F32);
    assert_eq!(cap.options_honored, 0b111);
    assert_eq!(cstr(&cap.benchmark), BGE);
    assert_eq!(cap.speed_ratio, (1542.372519_f64 / 2392.982597) as f32);
    assert_eq!(cap.cosine_floor, 0.0);
    assert!(cstr(&cap.reason).is_empty());
}

#[test]
fn another_device_precision_or_task_cannot_borrow_rerank_evidence() {
    for (arch, name, task, precision) in [
        ("other", GPU, TURBO_TASK_RERANK, TURBO_PRECISION_EXACT),
        (ARCH, "different GPU", TURBO_TASK_RERANK, TURBO_PRECISION_EXACT),
        (ARCH, GPU, TURBO_TASK_RERANK, TURBO_PRECISION_MODEL),
        (ARCH, GPU, TURBO_TASK_RERANK, TURBO_PRECISION_FASTEST),
        (ARCH, GPU, TURBO_TASK_EXTRACTIVE_QA, TURBO_PRECISION_EXACT),
    ] {
        let cap = report(&CUDA, arch, name, task, precision);
        assert_eq!(cap.status, backend::TURBO_CAP_EXPERIMENTAL);
        assert_eq!(cap.speed_ratio, 0.0);
        assert_eq!(cap.cosine_floor, 0.0);
        assert!(cstr(&cap.benchmark).is_empty());
        assert!(!cstr(&cap.reason).is_empty());
    }
}

#[test]
#[cfg(target_os = "linux")]
fn qa_names_the_timing_without_a_reference() {
    let cap = report(&CUDA, ARCH, GPU, TURBO_TASK_EXTRACTIVE_QA, TURBO_PRECISION_EXACT);
    assert_eq!(
        cstr(&cap.reason),
        "rtx4080super.cuda.qa.exact.roberta-base-squad2-da039877.ceae91f1cd5d.json: no reference program measured"
    );
    let absent = report(&CUDA, ARCH, GPU, TURBO_TASK_EXTRACTIVE_QA, TURBO_PRECISION_MODEL);
    assert_eq!(cstr(&absent.reason), record::NO_RECORD);
}

#[test]
fn a_backend_without_the_task_stays_unsupported() {
    static WITHOUT_TASKS: backend::turbo_backend =
        backend::turbo_backend { name: c"other".as_ptr(), embed_write_ranges: None, ..crate::cpu::BACKEND };
    for task in [TURBO_TASK_RERANK, TURBO_TASK_EXTRACTIVE_QA] {
        let cap = report(&WITHOUT_TASKS, ARCH, GPU, task, TURBO_PRECISION_EXACT);
        assert_eq!(cap.status, backend::TURBO_CAP_UNSUPPORTED);
        assert_eq!((cap.dtype, cap.options_honored), (0, 0));
        assert_eq!((cap.speed_ratio, cap.cosine_floor), (0.0, 0.0));
        assert!(cstr(&cap.benchmark).is_empty());
    }
}

#[test]
fn an_unavailable_encoder_cannot_inherit_a_supported_rerank_cell() {
    unsafe extern "C" fn unavailable(
        _: u32,
        _: u32,
        _: u32,
        status: *mut u32,
        _: *mut u32,
        _: *mut u32,
        _: *mut c_char,
        _: u32,
        _: *mut turbo_error,
    ) -> i32 {
        unsafe { *status = backend::TURBO_CAP_UNSUPPORTED };
        0
    }
    static UNAVAILABLE: backend::turbo_backend = backend::turbo_backend { capability: unavailable, ..CUDA };
    let cap = report(&UNAVAILABLE, ARCH, GPU, TURBO_TASK_RERANK, TURBO_PRECISION_EXACT);
    assert_eq!(cap.status, backend::TURBO_CAP_UNSUPPORTED);
    assert_eq!(cap.dtype, 0);
    assert_eq!(cap.speed_ratio, 0.0);
    assert!(cstr(&cap.benchmark).is_empty());
}
