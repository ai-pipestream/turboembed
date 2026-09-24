//! CUDA-specific provider behavior through the safe Rust API.
//!
//! The tests load the built `libturbo_provider_cuda.so` the way an
//! application does (`turbo_core::create_runtime` with `provider_paths`) and
//! check what only this provider can be asked about: its device record, its
//! capability matrix, its `cuda_lib_dir` context option, its buffer
//! placements and native handle kinds, and its session counters. Contract
//! behavior every provider must share lives in
//! `crates/turbo-conformance/tests/live_*.rs` instead.
//!
//! The runtime comes from `turbo_core::create_runtime` rather than
//! `turbo::create_runtime`: the two are the same call, but `turbo` links
//! `turbo-provider-static`, whose `turbo_provider_get` export collides with
//! this crate's when both are linked into one test binary.
//!
//! Every test prints a reason and returns when the provider library, a CUDA
//! device, or a bundle is missing, so the file is safe to run anywhere.
//! `TURBO_CUDA_LIB_DIR` names the CUDA user-space libraries (see
//! `providers/cuda/README.md`); `TURBO_LIVE_BUNDLE` names a MiniLM bundle.

use std::path::PathBuf;
use std::sync::Arc;

use turbo_core::abi;
use turbo_core::{
    BufferDesc, CapStatus, Context, ContextDesc, DType, DeviceInfo, DeviceKind, DeviceSelector, EmbedOptions,
    HandleKind, Modality, ModelDesc, NativeHandle, Options, Placement, RuntimeDesc, SelectPolicy, SessionDesc, Task,
};
use turbo_provider_cuda::cuda::{self, DeviceMem, PinnedMem, Stream};
use turbo_provider_cuda::{CUDA_CAPS, CUDA_PROVIDER_ID, LIB_DIR_OPTION};

/// The `cdylib` this crate builds, next to the test binary in
/// `target/<profile>/`, or `None` with a printed reason.
fn provider_lib() -> Option<PathBuf> {
    let exe = std::env::current_exe().expect("the test binary has a path");
    // target/<profile>/deps/<test binary> -> target/<profile>/
    let dir = exe.parent().and_then(|p| p.parent()).expect("target/<profile>");
    let path = dir.join("libturbo_provider_cuda.so");
    if path.is_file() {
        return Some(path);
    }
    eprintln!("skipping: `{}` is not built; run `cargo build -p turbo-provider-cuda`", path.display());
    None
}

/// A runtime holding only the CUDA provider library, or `None` with a
/// printed reason when the library is absent or enumerates no device.
fn runtime() -> Option<Arc<turbo_core::Runtime>> {
    let lib = provider_lib()?;
    let rt = turbo_core::create_runtime(RuntimeDesc {
        no_default_providers: true,
        provider_paths: vec![lib.to_string_lossy().into_owned()],
        ..Default::default()
    })
    .unwrap_or_else(|e| panic!("load `{}`: {e}", lib.display()));
    assert!(rt.failures().is_empty(), "the CUDA provider library loaded with failures: {:?}", rt.failures());
    if rt.devices().is_empty() {
        eprintln!("skipping: the CUDA provider enumerated no devices on this machine");
        return None;
    }
    Some(rt)
}

/// The CUDA device at ordinal 0 and its info.
fn device_zero(rt: &Arc<turbo_core::Runtime>) -> (u32, DeviceInfo) {
    let index = rt
        .select(&DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: CUDA_PROVIDER_ID.to_string(),
            ordinal: 0,
            ..Default::default()
        })
        .expect("select CUDA ordinal 0");
    (index, rt.device(index).expect("the selected device").info)
}

/// A context on CUDA ordinal 0, or `None` with a printed reason.
fn context() -> Option<(Arc<Context>, DeviceInfo)> {
    let rt = runtime()?;
    let (index, info) = device_zero(&rt);
    match Context::create(rt, index, &ContextDesc::default()) {
        Ok(ctx) => Some((ctx, info)),
        Err(e) if e.code() == abi::TURBO_E_DEVICE_UNAVAILABLE => {
            eprintln!("skipping: the CUDA user-space libraries are not loadable ({e}); set TURBO_CUDA_LIB_DIR");
            None
        }
        Err(e) => panic!("create a CUDA context: {e}"),
    }
}

/// The MiniLM bundle from `TURBO_LIVE_BUNDLE`, or `None` with a reason.
fn minilm() -> Option<PathBuf> {
    match std::env::var_os("TURBO_LIVE_BUNDLE") {
        Some(v) => Some(PathBuf::from(v)),
        None => {
            eprintln!("skipping: TURBO_LIVE_BUNDLE does not name a MiniLM bundle");
            None
        }
    }
}

#[test]
fn cuda_devices_are_nvidia_gpus_with_memory_and_the_declared_capability_bits() {
    let Some(rt) = runtime() else { return };
    for entry in rt.devices() {
        let d = &entry.info;
        assert_eq!(d.provider_id, CUDA_PROVIDER_ID, "every device of this library belongs to the cuda provider");
        assert_eq!(d.kind, DeviceKind::Gpu, "`{}` must report kind GPU, not {:?}", d.name, d.kind);
        assert_eq!(d.vendor, "NVIDIA", "`{}` must report vendor NVIDIA, not `{}`", d.name, d.vendor);
        assert_eq!(d.vendor_id, 0x10DE, "`{}` must report the PCI vendor id of NVIDIA", d.name);
        assert!(d.memory_total > 0, "`{}` reports {} bytes of total memory", d.name, d.memory_total);
        assert!(
            d.memory_free > 0 && d.memory_free <= d.memory_total,
            "`{}` reports {} free of {} total",
            d.name,
            d.memory_free,
            d.memory_total
        );
        assert_eq!(d.caps, CUDA_CAPS, "`{}` must report exactly CUDA_CAPS across the plugin ABI", d.name);
        // The bits the provider's data path is built on, spelled out so a
        // silent edit to CUDA_CAPS cannot pass this test unnoticed.
        for (bit, name) in [
            (abi::TURBO_CAP_DEVICE_RESULT, "DEVICE_RESULT"),
            (abi::TURBO_CAP_DEVICE_POSTPROCESS, "DEVICE_POSTPROCESS"),
            (abi::TURBO_CAP_HOST_PTR_IMPORT, "HOST_PTR_IMPORT"),
            (abi::TURBO_CAP_DYNAMIC_SHAPE, "DYNAMIC_SHAPE"),
        ] {
            assert_eq!(d.caps & bit, bit, "`{}` must advertise TURBO_CAP_{name}", d.name);
        }
        assert!(!d.runtime_version.is_empty(), "`{}` reports no runtime version", d.name);
        assert!(!d.driver_version.is_empty(), "`{}` reports no driver version", d.name);
    }
}

#[test]
fn cuda_capability_cells_follow_the_receipts() {
    let Some(rt) = runtime() else { return };
    let (index, info) = device_zero(&rt);
    // Embeddings are SUPPORTED on a discrete GPU (precision receipt plus the
    // RTX 4080 SUPER matched-native pair) and EXPERIMENTAL on an integrated one (the
    // Orin Nano pair is under the 0.95 line at 1x32); the other text tasks have
    // no matched-native benchmark and stay EXPERIMENTAL everywhere.
    let integrated = cuda::devices().expect("enumerate CUDA devices")[0].integrated;
    let offered = [Task::Embed, Task::Rerank, Task::Classify, Task::TokenClassify];
    for &task in Task::ALL {
        for &modality in Modality::ALL {
            let cap = rt.capability(index, task, modality).expect("query the capability cell");
            let want = if modality != Modality::Text || !offered.contains(&task) {
                CapStatus::Unsupported
            } else if task == Task::Embed && !integrated {
                CapStatus::Supported
            } else {
                CapStatus::Experimental
            };
            assert_eq!(
                cap.status, want,
                "`{}` (integrated {integrated}) reports {:?} for {task:?} x {modality:?}, expected {want:?}",
                info.name, cap.status
            );
            match want {
                CapStatus::Supported => {
                    assert_eq!(cap.dtype, Some(DType::F32), "{task:?}: the CUDA path runs in f32");
                    assert!(cap.cosine_floor >= 0.999, "{task:?}: a SUPPORTED cell states its cosine floor: {cap:?}");
                    assert!(
                        cap.notes.contains("cuda-2026-09-21") && cap.notes.contains("compare-cuda-rtx4080-embed"),
                        "{task:?}: a SUPPORTED cell names its receipts: {}",
                        cap.notes
                    );
                }
                CapStatus::Experimental => {
                    assert_eq!(cap.dtype, Some(DType::F32), "{task:?}: the CUDA path runs in f32");
                    assert_eq!(cap.cosine_floor, 0.0, "{task:?}: no floor is claimed without a benchmark");
                    assert!(
                        cap.notes.contains("no matched-native benchmark") || cap.notes.contains("integrated GPU"),
                        "{task:?}: an EXPERIMENTAL cell must say why it is not SUPPORTED: {}",
                        cap.notes
                    );
                }
                _ => {}
            }
        }
    }
}

#[test]
fn cuda_context_rejects_an_unknown_option_naming_its_one_based_field_index() {
    let Some(rt) = runtime() else { return };
    let (index, _) = device_zero(&rt);
    let desc = ContextDesc { options: Options(vec![("not_a_cuda_option".into(), "1".into())]) };
    let e = Context::create(rt.clone(), index, &desc).expect_err("an unknown context option must be rejected");
    assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "unknown option: {e}");
    assert_eq!(e.field(), 1, "the first option is field 1: {e}");
    // The index is the option's position, not a constant.
    let desc = ContextDesc {
        options: Options(vec![(LIB_DIR_OPTION.into(), lib_dir()), ("still_not_an_option".into(), "1".into())]),
    };
    let e = Context::create(rt, index, &desc).expect_err("an unknown context option must be rejected");
    assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "unknown second option: {e}");
    assert_eq!(e.field(), 2, "the second option is field 2: {e}");
}

/// `TURBO_CUDA_LIB_DIR`, or an empty string when it is unset (the provider
/// then relies on the loader's own search path).
fn lib_dir() -> String {
    std::env::var("TURBO_CUDA_LIB_DIR").unwrap_or_default()
}

#[test]
fn cuda_lib_dir_without_libraries_is_device_unavailable_not_a_cpu_fallback() {
    let Some(rt) = runtime() else { return };
    let (index, _) = device_zero(&rt);
    let empty = tempfile::tempdir().expect("a temporary directory");
    std::fs::write(empty.path().join("readme.txt"), b"no shared libraries here").expect("write a decoy file");
    let desc =
        ContextDesc { options: Options(vec![(LIB_DIR_OPTION.into(), empty.path().to_string_lossy().into_owned())]) };
    let e = Context::create(rt.clone(), index, &desc).expect_err("a library-free cuda_lib_dir must fail");
    assert_eq!(e.code(), abi::TURBO_E_DEVICE_UNAVAILABLE, "a missing CUDA library is never a fallback: {e}");
    let missing = empty.path().join("does-not-exist");
    let desc = ContextDesc { options: Options(vec![(LIB_DIR_OPTION.into(), missing.to_string_lossy().into_owned())]) };
    let e = Context::create(rt, index, &desc).expect_err("a cuda_lib_dir that does not exist must fail");
    assert_eq!(e.code(), abi::TURBO_E_DEVICE_UNAVAILABLE, "an unreadable cuda_lib_dir: {e}");
}

#[test]
fn cuda_allocates_device_pinned_and_host_buffers_and_rejects_shared() {
    let Some((ctx, _)) = context() else { return };
    let shape = [64u64];
    for placement in [Placement::Device, Placement::Pinned, Placement::Host] {
        let desc = BufferDesc::packed(placement, DType::F32, &shape).expect("buffer description");
        let buffer = ctx.alloc(&desc).unwrap_or_else(|e| panic!("allocate a {placement:?} buffer: {e}"));
        assert_eq!(buffer.desc().placement, placement, "the buffer keeps the requested placement");
        assert_eq!(buffer.desc().bytes, 256, "64 f32 is 256 bytes");
        assert_eq!(
            buffer.host_ptr().is_some(),
            placement.host_visible(),
            "{placement:?}: host visibility must match the placement"
        );
    }
    let desc = BufferDesc::packed(Placement::Shared, DType::F32, &shape).expect("buffer description");
    let e = ctx.alloc(&desc).expect_err("the CUDA provider does not offer managed memory");
    assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_PLACEMENT, "SHARED must be rejected as a placement: {e}");
}

#[test]
fn cuda_device_buffers_export_a_cuda_pointer_and_pinned_buffers_a_host_pointer() {
    let Some((ctx, info)) = context() else { return };
    let shape = [16u64];
    let device = ctx.alloc(&BufferDesc::packed(Placement::Device, DType::F32, &shape).unwrap()).expect("device buffer");
    let handle = device.export(HandleKind::CudaPtr).expect("a device buffer exports a CUDA pointer");
    assert_eq!(handle.kind, HandleKind::CudaPtr, "the exported handle names its own kind");
    assert_ne!(handle.handle, 0, "the exported CUDA pointer is not NULL");
    assert_eq!(handle.aux, info.ordinal as u64, "the handle carries the owning device ordinal");
    assert_eq!(handle.offset, 0, "the CUDA provider exports whole allocations");
    for &kind in HandleKind::ALL.iter().filter(|&&k| k != HandleKind::CudaPtr) {
        let e = device.export(kind).expect_err("a device buffer exports CUDA_PTR only");
        assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED, "device buffer exported as {kind:?}: {e}");
    }

    let pinned = ctx.alloc(&BufferDesc::packed(Placement::Pinned, DType::F32, &shape).unwrap()).expect("pinned buffer");
    let handle = pinned.export(HandleKind::HostPtr).expect("a pinned buffer exports a host pointer");
    assert_eq!(handle.kind, HandleKind::HostPtr, "the exported handle names its own kind");
    assert_eq!(
        handle.handle,
        pinned.host_ptr().expect("pinned memory is host visible").as_ptr() as u64,
        "the exported host pointer is the buffer's own mapping"
    );
    for &kind in HandleKind::ALL.iter().filter(|&&k| k != HandleKind::HostPtr) {
        let e = pinned.export(kind).expect_err("a pinned buffer exports HOST_PTR only");
        assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED, "pinned buffer exported as {kind:?}: {e}");
    }
}

#[test]
fn cuda_reads_back_an_imported_device_pointer_it_did_not_allocate() {
    let Some((ctx, info)) = context() else { return };
    let ordinal = info.ordinal as i32;
    let values: Vec<f32> = (0..96).map(|i| i as f32 * 0.25 - 12.0).collect();
    let bytes = std::mem::size_of_val(&values[..]);
    // Allocate and fill with cudart directly, outside the provider.
    let stream = Stream::new(ordinal).expect("a stream on the selected device");
    let staging = PinnedMem::new(bytes).expect("pinned staging");
    // SAFETY: staging is `bytes` long and values holds exactly `bytes` bytes.
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), staging.ptr(), bytes) };
    let mem = DeviceMem::new(ordinal, bytes).expect("device allocation");
    // SAFETY: staging outlives the copy because the stream is synchronized next.
    unsafe { cuda::copy_h2d(&mem, staging.ptr(), bytes, &stream) }.expect("H2D copy");
    stream.synchronize().expect("synchronize the upload");

    let desc = BufferDesc::packed(Placement::Device, DType::F32, &[values.len() as u64]).expect("descriptor");
    let handle = NativeHandle { kind: HandleKind::CudaPtr, handle: mem.ptr() as u64, aux: ordinal as u64, offset: 0 };
    let imported = ctx.import(&desc, &handle).expect("import a CUDA pointer allocated outside the provider");
    let mut read = vec![0u8; bytes];
    imported.read_to_host(&mut read).expect("read the imported pointer back to the host");
    let got: Vec<f32> = read.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(got, values, "read_to_host must return what was written through the imported pointer");
    assert_eq!(
        imported.export(HandleKind::CudaPtr).expect("an imported device buffer re-exports its pointer").handle,
        mem.ptr() as u64,
        "the re-exported pointer is the one that was imported"
    );
    assert!(imported.host_ptr().is_none(), "an imported device pointer is not host visible");
    // A pointer tagged with another device belongs to another context.
    let other = NativeHandle { aux: ordinal as u64 + 7, ..handle };
    let e = ctx.import(&desc, &other).expect_err("a pointer from another device must be rejected");
    assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "cross-device import: {e}");
    let null = NativeHandle { handle: 0, ..handle };
    let e = ctx.import(&desc, &null).expect_err("a NULL CUDA pointer must be rejected");
    assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "NULL import: {e}");
    drop(imported);
}

#[test]
fn cuda_session_stats_count_uploads_per_run_and_never_read_embeddings_back() {
    let Some((ctx, _)) = context() else { return };
    let Some(bundle) = minilm() else { return };
    let model = ctx.load_model(&bundle, &ModelDesc::default()).expect("load the MiniLM bundle");
    let session = model.create_session(&SessionDesc { max_batch: 4, max_seq: 64, ..Default::default() }).unwrap();

    let before = session.stats().expect("stats before the first run");
    assert_eq!(before.runs, 0, "a fresh session has run nothing");
    assert_eq!(before.h2d_bytes, 0, "a fresh session has uploaded nothing");
    assert_eq!(before.d2h_bytes, 0, "a fresh session has read nothing back");
    assert!(before.input_bytes > 0, "the session reports the size of its input staging");

    session.write_text(&["the first sentence"], &EmbedOptions::default()).expect("write");
    let result = session.run(&Default::default()).expect("run");
    let output = result.output(0).expect("the embedding output");
    assert_eq!(output.shape, vec![1, 384], "MiniLM embeds to 384 dimensions");
    let logical = output.logical_bytes().expect("the output has a logical size");
    drop(result);

    let after_one = session.stats().expect("stats after one run");
    assert_eq!(after_one.runs, 1, "one run counted");
    assert!(after_one.h2d_bytes > 0, "the token rows are uploaded: {after_one:?}");
    assert_eq!(after_one.d2h_bytes, 0, "an embedding run leaves the result on the device: {after_one:?}");
    assert_eq!(
        after_one.output_bytes,
        session_output_bytes(&session),
        "output_bytes is the session's result buffer, not the logical result"
    );
    assert!(
        after_one.output_bytes >= logical,
        "the result buffer ({} bytes) holds the logical result ({logical} bytes)",
        after_one.output_bytes
    );

    session.write_text(&["the second sentence"], &EmbedOptions::default()).expect("write");
    let result = session.run(&Default::default()).expect("run");
    drop(result);
    let after_two = session.stats().expect("stats after two runs");
    assert_eq!(after_two.runs, 2, "two runs counted");
    assert!(
        after_two.h2d_bytes > after_one.h2d_bytes,
        "each run uploads its rows again: {} then {}",
        after_one.h2d_bytes,
        after_two.h2d_bytes
    );
    assert_eq!(after_two.d2h_bytes, 0, "still no device-to-host traffic for embeddings: {after_two:?}");
    assert_eq!(after_two.output_bytes, after_one.output_bytes, "the result buffer is allocated once per session");
}

/// The session's result buffer size: `[max_batch, dim]` f32.
fn session_output_bytes(session: &Arc<turbo_core::Session>) -> u64 {
    session.desc().max_batch as u64 * session.model().info().dim as u64 * 4
}
