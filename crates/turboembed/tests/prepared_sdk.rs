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
