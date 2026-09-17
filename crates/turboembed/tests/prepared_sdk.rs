#![cfg(feature = "prepared")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
use turboembed::prepared::{Context, Device};

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
struct TrackingAllocator;
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.try_with(Cell::get).unwrap_or(false) {
            let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if TRACK.try_with(Cell::get).unwrap_or(false) {
            let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn bundle() -> String {
    std::env::var("TURBOEMBED_PREPARED_BUNDLE")
        .expect("set TURBOEMBED_PREPARED_BUNDLE to the pinned MiniLM bundle")
}
fn parity(expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    let mut max = 0f32;
    let mut sum = 0f64;
    for (&a, &b) in expected.iter().zip(actual) {
        assert!(a.is_finite() && b.is_finite());
        let delta = (a - b).abs();
        max = max.max(delta);
        sum += f64::from(delta).powi(2);
    }
    assert!(max <= 5e-4 && (sum / expected.len() as f64).sqrt() <= 1e-4);
}

#[test]
#[ignore = "requires the installed prepared SDK (runs on any OpenVINO host, CPU-only included)"]
fn device_discovery_and_explicit_selection() {
    use turboembed::prepared;

    let devices = prepared::devices().unwrap();
    assert!(!devices.is_empty(), "no selectable device was enumerated");
    // GPUs precede a single trailing CPU entry, in ascending ordinal order.
    let cpus: Vec<usize> = devices
        .iter()
        .enumerate()
        .filter(|(_, d)| d.device == prepared::DEVICE_OPENVINO_CPU)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(cpus, [devices.len() - 1], "expected one trailing CPU entry");
    let ordinals: Vec<u32> = devices[..devices.len() - 1]
        .iter()
        .map(|d| {
            assert_eq!(d.device, prepared::DEVICE_OPENVINO_GPU);
            d.ordinal
        })
        .collect();
    assert!(ordinals.is_sorted(), "GPU ordinals must be ascending");
    for info in &devices {
        let gpu = info.device == prepared::DEVICE_OPENVINO_GPU;
        assert!(!info.device_name.is_empty() && !info.runtime_version.is_empty());
        assert_eq!(!info.driver_version.is_empty(), gpu);
        let expected = prepared::CAP_TEXT
            | prepared::CAP_PREPARED_I32
            | prepared::CAP_HOST_READ
            | if gpu { prepared::CAP_OPENCL_RESULT } else { 0 };
        assert_eq!(info.capabilities, expected);
        // Explicit selection of a discovered device resolves the same identity.
        let context = Context::new(info.selector()).unwrap();
        let resolved = context.info().unwrap();
        assert_eq!(
            (resolved.device, resolved.ordinal, resolved.capabilities),
            (info.device, info.ordinal, info.capabilities)
        );
        assert_eq!(resolved.device_name, info.device_name);
        assert_eq!(resolved.runtime_version, info.runtime_version);
        assert_eq!(resolved.driver_version, info.driver_version);
    }
    // Selecting a device that discovery did not list fails loud; there is no
    // silent CPU fallback for an absent GPU.
    let absent = ordinals.iter().max().map_or(0, |max| max + 1_000_000);
    let error = match Context::new(Device::Gpu { ordinal: absent }) {
        Err(error) => error,
        Ok(_) => panic!("selecting an absent GPU must fail"),
    };
    assert_eq!(error.code, 4, "absent GPU must return UNAVAILABLE: {error}");
}

/// Machine B (krick-1, Intel Battlemage G31) receipt gate. The discovery API
/// was added after the 2026-09-14 receipts and is hardware-unverified on an
/// Intel GPU until this test is re-run there. See docs/native-sdk.md.
#[test]
#[ignore = "requires installed prepared SDK and Intel GPU; re-run on Machine B (krick-1) to refresh the receipt"]
fn machine_b_gpu_discovery_receipt() {
    use turboembed::prepared;

    let devices = prepared::devices().unwrap();
    let gpu = devices
        .iter()
        .find(|d| d.device == prepared::DEVICE_OPENVINO_GPU)
        .expect("Machine B must enumerate its Intel GPU");
    assert_ne!(gpu.capabilities & prepared::CAP_OPENCL_RESULT, 0);
    assert!(!gpu.driver_version.is_empty() && !gpu.runtime_version.is_empty());
    let context = Context::new(gpu.selector()).unwrap();
    let info = context.info().unwrap();
    assert_eq!((info.device, info.ordinal), (gpu.device, gpu.ordinal));
    assert_eq!(info.device_name, gpu.device_name);
    assert!(!info.driver_version.is_empty());
}

/// Explicit-CPU execution against the repository MiniLM reference fixture.
/// Runs on hosts without a GPU; GPU coverage lives in the tests below.
#[test]
#[ignore = "requires installed prepared SDK and pinned model bundle (CPU only)"]
fn prepared_cpu_reference_execution() {
    let path = bundle();
    let cpu = Context::new(Device::Cpu).unwrap();
    let info = cpu.info().unwrap();
    assert_eq!(info.device, turboembed::prepared::DEVICE_OPENVINO_CPU);
    assert_eq!(
        info.capabilities & turboembed::prepared::CAP_OPENCL_RESULT,
        0,
        "CPU must not advertise a device-result capability"
    );
    let model = cpu.load_model(&path).unwrap();
    let details = model.info().unwrap();
    assert_eq!(
        (
            details.dimension,
            details.pooling.as_str(),
            details.normalized
        ),
        (384, "mean", true)
    );
    // Unsupported shapes fail loud instead of clamping or recompiling.
    assert!(model.slot(1, details.max_sequence_length + 1).is_err());
    assert!(model.slot(details.max_batch_size + 1, 32).is_err());

    let mut slot = model.slot(1, 32).unwrap();
    slot.write_text(&["hello world"]).unwrap();
    let single = slot.execute().unwrap().to_vec().unwrap();
    // Cross-runtime reference recorded from ORT CUDA fp32 with identical
    // pooling/normalization; gates match the repository parity tolerances.
    let fixture: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/reference_embeddings/ort_cuda_minilm_short.json"
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(fixture["text"], "hello world");
    let reference: Vec<f32> = fixture["vector"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    parity(&reference, &single);

    // A padded mixed-length batch reproduces the single-row outputs.
    slot.write_text(&[""]).unwrap();
    let empty = slot.execute().unwrap().to_vec().unwrap();
    let mut mixed = model.slot(2, 32).unwrap();
    mixed.write_text(&["hello world", ""]).unwrap();
    let rows = mixed.execute().unwrap().to_vec().unwrap();
    assert_eq!(rows.len(), 768);
    parity(&single, &rows[..384]);
    parity(&empty, &rows[384..]);
}

#[test]
#[ignore = "requires installed prepared SDK, pinned model bundle, CPU and Intel GPU"]
fn prepared_lifetimes_inputs_and_cpu_gpu_parity() {
    let path = bundle();
    let cpu = Context::new(Device::Cpu).unwrap();
    let model = cpu.load_model(&path).unwrap();
    assert_eq!(model.info().unwrap().dimension, 384);
    let mut cpu_slot = model.slot(1, 32).unwrap();
    // Safe wrappers use independent native retention for these parents.
    drop(model);
    drop(cpu);
    cpu_slot.write_text(&["hello world"]).unwrap();
    let expected = cpu_slot.execute().unwrap().to_vec().unwrap();

    let gpu = Context::new(Device::Gpu { ordinal: 0 }).unwrap();
    assert_eq!(gpu.info().unwrap().device, 1);
    let model = gpu.load_model(&path).unwrap();
    let mut slot = model.slot(1, 32).unwrap();
    let tokenizer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{path}/tokenizer.json")).unwrap()).unwrap();
    let vocab = &tokenizer["model"]["vocab"];
    let mut ids = vec![vocab["[PAD]"].as_i64().unwrap() as i32; 32];
    let mut mask = vec![0; 32];
    for (i, token) in ["[CLS]", "hello", "world", "[SEP]"].iter().enumerate() {
        ids[i] = vocab[token].as_i64().unwrap() as i32;
        mask[i] = 1;
    }
    slot.write_tokens(&ids, &mask, None).unwrap();
    {
        let mut result = slot.execute().unwrap();
        assert_eq!((result.batch(), result.dimension()), (1, 384));
        let mut output = [987.0f32; 386];
        result.read_into(&mut output).unwrap();
        parity(&expected, &output[..384]);
        assert_eq!(&output[384..], &[987.0, 987.0]);
        {
            let view = result.opencl().unwrap();
            // No submission or escaped use; the enclosing result remains alive.
            let raw = unsafe { view.raw_handles() };
            assert!(raw.context != 0 && raw.queue != 0 && raw.buffer != 0);
            assert_eq!(raw.byte_size, 1536);
        }
        result.close().unwrap();
    }
    // FFI never reads beyond a shorter mask/types slice; failed writes invalidate.
    assert_eq!(
        slot.write_tokens(&ids, &mask[..1], None).unwrap_err().code,
        1
    );
    assert!(slot.execute().is_err());
    assert_eq!(
        slot.write_tokens(&ids, &mask, Some(&[0])).unwrap_err().code,
        1
    );
    assert!(slot.execute().is_err());
    assert_eq!(slot.write_text(&[]).unwrap_err().code, 1);
    assert!(slot.execute().is_err());
    for text in ["", "Café 日本語", "hello\0world"] {
        cpu_slot.write_text(&[text]).unwrap();
        slot.write_text(&[text]).unwrap();
        parity(
            &cpu_slot.execute().unwrap().to_vec().unwrap(),
            &slot.execute().unwrap().to_vec().unwrap(),
        );
    }
    slot.write_text(&["hello world"]).unwrap();
    // Counters are thread-local, so parallel test allocation cannot contaminate
    // this claim. Native C++ and driver allocations are outside Rust's allocator.
    {
        let _ = slot.execute().unwrap();
    }
    ALLOCATIONS.with(|n| n.set(0));
    TRACK.with(|flag| flag.set(true));
    for _ in 0..32 {
        slot.execute().unwrap().close().unwrap();
    }
    TRACK.with(|flag| flag.set(false));
    assert_eq!(
        ALLOCATIONS.with(Cell::get),
        0,
        "reused execute/release allocated in Rust"
    );
}

#[test]
#[ignore = "requires installed prepared SDK, pinned model bundle and Intel GPU"]
fn prepared_model_sharing_and_slot_transfer() {
    let context = Context::new(Device::Gpu { ordinal: 0 }).unwrap();
    let model = context.load_model(&bundle()).unwrap();
    let (mut first, mut second) = std::thread::scope(|scope| {
        let one = scope.spawn(|| model.slot(1, 32).unwrap());
        let two = scope.spawn(|| model.slot(1, 32).unwrap());
        (one.join().unwrap(), two.join().unwrap())
    });
    drop(context);
    drop(model);
    first.write_text(&["hello world"]).unwrap();
    second.write_text(&["hello world"]).unwrap();
    let one = std::thread::spawn(move || {
        let result = first.execute().unwrap();
        std::thread::scope(|scope| {
            scope
                .spawn(move || result.to_vec().unwrap())
                .join()
                .unwrap()
        })
    });
    let two = std::thread::spawn(move || second.execute().unwrap().to_vec().unwrap());
    parity(&one.join().unwrap(), &two.join().unwrap());
}
