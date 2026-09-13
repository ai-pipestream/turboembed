//! ABI smoke: create / list / load / embed (single + batch) / stream /
//! free through the safe wrapper, plus a raw FFI path.

use std::os::raw::c_void;
use std::ptr;

use turboembed::ffi::{
    turboembed_abi_version, turboembed_device, turboembed_embed, turboembed_embed_result,
    turboembed_embed_result_free, turboembed_embed_stream, turboembed_engine,
    turboembed_engine_create, turboembed_engine_destroy, turboembed_load_model, turboembed_status,
    turboembed_str,
};
use turboembed::{abi_version, register_provider_stub, Device, EmbedOptions, Engine, Error};

#[test]
fn create_list_load_embed_free() {
    assert_eq!(abi_version(), 1);

    let engine = Engine::create(Device::Mock).expect("create mock engine");
    let models = engine.list_models().expect("list");
    assert_eq!(models.len(), 1);
    let info = models.get(0).expect("mock-embed row");
    assert_eq!(info.alias, "mock-embed");
    assert_eq!(info.dim, 8);
    assert!(info.ready);

    engine.load_model("mock-embed").expect("load mock");

    let opts = EmbedOptions::default();
    let one = engine
        .embed_one("mock-embed", "hello world", &opts)
        .expect("embed_one");
    assert_eq!(one.dim(), 8);
    assert_eq!(one.count(), 1);
    assert_eq!(one.values().len(), 8);
    assert_eq!(one.packed().len(), 8 * 4);
    assert!(!one.values().iter().all(|v| *v == 0.0));

    let batch = engine
        .embed(
            "mock-embed",
            &["hello world", "other", "hello world"],
            &opts,
        )
        .expect("embed batch");
    assert_eq!(batch.count(), 3);
    assert_eq!(batch.row(0).unwrap(), one.values());
    assert_ne!(batch.row(0).unwrap(), batch.row(1).unwrap());
    assert_eq!(batch.row(0).unwrap(), batch.row(2).unwrap());
}

#[test]
#[cfg(not(any(feature = "ort-cuda", feature = "genai")))]
fn catalog_alias_is_not_implemented() {
    let engine = Engine::create(Device::Mock).expect("create mock engine");
    let err = engine
        .load_model("minilm")
        .expect_err("mock must not fake MiniLM");
    assert!(
        matches!(
            err,
            Error::NotImplemented(_) | Error::NotFound(_) | Error::Unavailable(_)
        ),
        "catalog alias on Mock must fail, got {err:?}"
    );
}

#[test]
fn gpu_without_gpu_fails_loud_never_cpu() {
    // CUDA create is allowed only when --features ort-cuda (load then
    // fails loud if the EP is missing). Same for OpenVINO GPU + genai.
    let mut devices = vec![Device::TensorRt, Device::OpenVinoNpu];
    if cfg!(not(feature = "ort-cuda")) {
        devices.push(Device::Cuda);
    }
    if cfg!(not(feature = "genai")) {
        devices.push(Device::OpenVinoGpu);
    }
    for device in devices {
        let err = match Engine::create(device) {
            Ok(_) => panic!("{device:?} must fail when that GPU is missing — never CPU/mock"),
            Err(e) => e,
        };
        assert!(
            matches!(err, Error::Unavailable(_) | Error::UnsupportedDevice(_)),
            "{device:?}: {err:?}"
        );
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("refusing cpu fallback") || msg.contains("refusing cpu"),
            "{device:?} error must say it refused CPU, got: {err}"
        );
    }
}

#[test]
fn register_provider_is_reserved() {
    let err = register_provider_stub().expect_err("plugin registration reserved");
    assert!(matches!(err, Error::NotImplemented(_)));
}

#[test]
fn cpu_only_when_explicit() {
    let engine = Engine::create(Device::Cpu).expect("explicit CPU is allowed");
    engine.load_model("mock-embed").expect("mock on CPU");
    let models = engine.list_models().expect("list");
    assert!(
        models.iter().all(|m| m.alias == "mock-embed" && m.device != Device::Metal),
        "explicit CPU must not advertise Metal MiniLM: {:?}",
        models.iter().map(|m| (m.alias.clone(), m.device)).collect::<Vec<_>>()
    );
    #[cfg(not(feature = "genai"))]
    {
        let err = engine
            .load_model("minilm")
            .expect_err("CPU stub must not silently serve MiniLM");
        assert!(matches!(err, Error::NotImplemented(_) | Error::NotFound(_)));
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_fails_on_linux() {
    let err = match Engine::create(Device::Metal) {
        Ok(_) => panic!("Metal must fail on Linux — refusing CPU fallback"),
        Err(e) => e,
    };
    assert!(matches!(
        err,
        Error::Unavailable(_) | Error::UnsupportedDevice(_)
    ));
    assert!(
        err.to_string().to_lowercase().contains("refusing cpu"),
        "Metal: {err}"
    );
}

#[cfg(all(not(target_os = "macos"), not(any(feature = "ort-cuda", feature = "genai"))))]
#[test]
fn auto_fails_on_linux_stub_without_accelerator() {
    let err = match Engine::create(Device::Auto) {
        Ok(_) => panic!("AUTO must fail when this stub has no GPU — never CPU"),
        Err(e) => e,
    };
    assert!(matches!(
        err,
        Error::Unavailable(_) | Error::UnsupportedDevice(_)
    ));
    assert!(
        err.to_string().to_lowercase().contains("refusing cpu"),
        "AUTO: {err}"
    );
}

#[test]
fn stream_callback_sees_each_row() {
    let engine = Engine::create(Device::Mock).unwrap();
    engine.load_model("mock").unwrap();

    let mut seen = Vec::new();
    let result = engine
        .embed_stream(
            "mock-embed",
            &["a", "b"],
            &EmbedOptions::default(),
            |index, row, is_final| {
                seen.push((index, row.to_vec(), is_final));
            },
        )
        .unwrap();

    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].0, 0);
    assert!(!seen[0].2);
    assert_eq!(seen[1].0, 1);
    assert!(seen[1].2);
    assert_eq!(seen[0].1, result.row(0).unwrap());
    assert_eq!(seen[1].1, result.row(1).unwrap());
}

#[test]
fn raw_ffi_embed_and_free() {
    unsafe {
        assert_eq!(turboembed_abi_version(), 1);
        let mut engine: *mut turboembed_engine = ptr::null_mut();
        let st = turboembed_engine_create(
            turboembed_device::TURBOEMBED_DEVICE_MOCK,
            ptr::null(),
            &mut engine,
        );
        assert_eq!(st, turboembed_status::TURBOEMBED_OK);
        assert!(!engine.is_null());

        let alias = b"mock-embed";
        let st = turboembed_load_model(engine, alias.as_ptr().cast(), alias.len());
        assert_eq!(st, turboembed_status::TURBOEMBED_OK);

        let text = b"ffi";
        let view = turboembed_str {
            ptr: text.as_ptr().cast(),
            len: text.len(),
        };
        let mut out: *mut turboembed_embed_result = ptr::null_mut();
        let st = turboembed_embed(
            engine,
            alias.as_ptr().cast(),
            alias.len(),
            &view,
            1,
            ptr::null(),
            &mut out,
        );
        assert_eq!(st, turboembed_status::TURBOEMBED_OK);
        assert!(!out.is_null());
        assert_eq!((*out).dim, 8);
        assert_eq!((*out).count, 1);
        turboembed_embed_result_free(out);

        let mut streamed: *mut turboembed_embed_result = ptr::null_mut();
        let st = turboembed_embed_stream(
            engine,
            alias.as_ptr().cast(),
            alias.len(),
            &view,
            1,
            ptr::null(),
            None,
            ptr::null_mut::<c_void>(),
            &mut streamed,
        );
        assert_eq!(st, turboembed_status::TURBOEMBED_OK);
        turboembed_embed_result_free(streamed);

        turboembed_engine_destroy(engine);
    }
}

#[test]
fn empty_batch_is_invalid() {
    let engine = Engine::create(Device::Mock).unwrap();
    engine.load_model("mock-embed").unwrap();
    let err = engine
        .embed("mock-embed", &[], &EmbedOptions::default())
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));
}
