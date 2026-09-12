//! Mock-as-default kill: catalog aliases must never be served by the
//! 8-d FNV mock. GPU / AUTO / METAL / CUDA / OPENVINO_* either hit a
//! real provider or fail loud. Explicit `Device::Mock` keeps the smoke
//! path for `mock-embed` only.

use turboembed::{Device, EmbedOptions, Engine, Error};

const CATALOG_ALIASES: &[&str] = &[
    "minilm",
    "minilm-l12",
    "bge-small",
    "bge-base",
    "bge-large",
    "bge-m3",
    "e5-small",
    "mpnet",
    "gte-small",
    "nomic-embed-text",
];

fn refuse_cpu(err: &Error) {
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("refusing cpu") || msg.contains("refusing cpu fallback"),
        "GPU-missing error must refuse CPU, got: {err}"
    );
}

fn fail_if_fnv_mock(alias: &str, dim: usize) {
    assert_ne!(
        dim, 8,
        "FAKE: catalog alias {alias} returned dim=8 (FNV mock). \
         Real MiniLM/BGE/E5 are 384/768/1024. Mock is smoke-only."
    );
}

/// `Device::Mock` may embed `mock-embed` as 8-d FNV. Catalog aliases
/// on that engine must fail — never return dim 8 as if they were MiniLM.
#[test]
fn catalog_alias_on_mock_device_never_returns_fnv8() {
    let engine = Engine::create(Device::Mock).expect("explicit Mock is allowed");
    engine
        .load_model("mock-embed")
        .expect("mock-embed smoke path stays");
    let opts = EmbedOptions::default();
    let smoke = engine
        .embed_one("mock-embed", "hello world", &opts)
        .expect("mock-embed embed");
    assert_eq!(smoke.dim(), 8, "mock-embed is the 8-d FNV smoke path");

    for alias in CATALOG_ALIASES {
        match engine.load_model(alias) {
            Ok(()) => {
                match engine.embed_one(alias, "hello world", &opts) {
                    Ok(emb) => {
                        fail_if_fnv_mock(alias, emb.dim());
                        panic!(
                            "FAKE: {alias} loaded on Device::Mock and embedded dim={}",
                            emb.dim()
                        );
                    }
                    Err(err) => {
                        assert!(
                            matches!(
                                err,
                                Error::NotImplemented(_)
                                    | Error::NotFound(_)
                                    | Error::Unavailable(_)
                                    | Error::UnsupportedDevice(_)
                            ),
                            "{alias} on Mock must fail, got {err:?}"
                        );
                    }
                }
            }
            Err(err) => {
                assert!(
                    matches!(
                        err,
                        Error::NotImplemented(_)
                            | Error::NotFound(_)
                            | Error::Unavailable(_)
                            | Error::UnsupportedDevice(_)
                    ),
                    "{alias} on Mock must fail, got {err:?}"
                );
            }
        }
    }
}

/// AUTO / METAL / CUDA / OPENVINO_* must not serve catalog aliases via
/// mock. Create either fails (no accelerator) or load/embed of `minilm`
/// is not dim 8.
#[test]
fn catalog_aliases_never_fnv8_on_accelerator_devices() {
    let mut devices = vec![
        Device::Cuda,
        Device::TensorRt,
        Device::OpenVinoGpu,
        Device::OpenVinoNpu,
        Device::Metal,
        Device::Auto,
    ];
    // On this Mac, Metal/AUTO are the live GPU — still must not be mock.
    for device in devices.drain(..) {
        match Engine::create(device) {
            Err(err) => {
                assert!(
                    matches!(err, Error::Unavailable(_) | Error::UnsupportedDevice(_)),
                    "{device:?} create must fail loud or succeed on a real GPU, got {err:?}"
                );
                refuse_cpu(&err);
            }
            Ok(engine) => {
                // Do not load MiniLM here — that is the ignored live
                // receipt suite. Listing must not advertise catalog
                // aliases as 8-d FNV.
                let models = engine.list_models().expect("list");
                for m in models.iter() {
                    if m.alias != "mock-embed" && m.alias != "mock" {
                        fail_if_fnv_mock(&m.alias, m.dim as usize);
                    }
                }
            }
        }
    }
}

#[cfg(all(
    not(target_os = "macos"),
    not(any(feature = "ort-cuda", feature = "genai"))
))]
#[test]
fn auto_without_accelerator_fails_loud() {
    let err = match Engine::create(Device::Auto) {
        Ok(_) => panic!("AUTO must fail when this stub has no GPU — never CPU/mock"),
        Err(e) => e,
    };
    assert!(matches!(
        err,
        Error::Unavailable(_) | Error::UnsupportedDevice(_)
    ));
    refuse_cpu(&err);
}

#[test]
fn mock_embed_rejected_on_gpu_create_paths() {
    // If CUDA/Metal create is refused, that is already loud-fail. If it
    // succeeds (real GPU), loading mock-embed on that engine must fail.
    for device in [Device::Cuda, Device::Metal, Device::Auto] {
        if let Ok(engine) = Engine::create(device) {
            let err = engine
                .load_model("mock-embed")
                .expect_err("GPU/AUTO must not load the FNV mock alias");
            assert!(
                matches!(
                    err,
                    Error::NotImplemented(_) | Error::NotFound(_) | Error::Unavailable(_)
                ),
                "{device:?}: {err:?}"
            );
        }
    }
}
