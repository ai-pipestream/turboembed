//! A model served on a device runs there (#2514): for each GPU backend in
//! this build, the tiny bundle is loaded on that backend's first device,
//! and both the model's metadata and a result's own record, read back from
//! the runtime, say that backend and no CPU.
//!
//! A backend in the build that lists no device fails the test: a server
//! test that asked for index 0 and got the CPU used to pass. A host with
//! no such device sets `TURBO_TEST_REQUIRE_<BACKEND>=0`, and the test then
//! says what it skipped. A build with the cpu backend alone has nothing
//! to place and says so.

#[path = "kserve/mod.rs"]
mod kserve;

use kserve::*;

/// The backends whose devices run the tiny BERT bundle's embed task.
const GPU_BACKENDS: [&str; 3] = ["cuda", "levelzero", "metal"];

/// The runtime index of the first device `backend` lists.
fn first_of(rt: &Runtime, backend: &str) -> Option<u32> {
    (0..64).find(|&i| rt.device_info(i).is_ok_and(|d| field(&d.backend) == backend))
}

fn text(p: &HashMap<String, InferParameter>, key: &str) -> String {
    match p.get(key).and_then(|v| v.parameter_choice.clone()) {
        Some(ParameterChoice::StringParam(s)) => s,
        other => panic!("{key}: {other:?}"),
    }
}

#[tokio::test]
async fn a_model_loaded_on_a_gpu_runs_on_that_gpu() {
    let linked = version();
    let backends: Vec<&str> =
        GPU_BACKENDS.into_iter().filter(|b| linked.split_whitespace().skip(1).any(|l| l == *b)).collect();
    if backends.is_empty() {
        println!("no GPU backend is in this build ({linked}): nothing to place");
        return;
    }
    let rt = Runtime::create().unwrap();
    let cpu_rows = {
        let (s, mut c) = serve(1).await;
        let rows = embed(&mut c, texts(&["a model runs where it was loaded"])).await;
        s.stop().await;
        rows
    };
    for backend in backends {
        let Some(device) = first_of(&rt, backend) else {
            assert!(
                turbo_pins::device_skip_allowed(backend),
                "the {backend} backend is in this build and lists no device; TURBO_TEST_REQUIRE_{}=0 lets the test skip",
                backend.to_uppercase()
            );
            println!("{backend}: skipped: the backend lists no device");
            continue;
        };
        let on_device = ModelConfig { device: Device::Index(device), ..model(&tiny(), 1) };
        let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![on_device], MAX, Serve::default()).await.unwrap();
        s.load().await.unwrap();
        let mut c = client(&s).await;
        let meta = c.model_metadata(ModelMetadataRequest { name: NAME.into(), version: String::new() }).await.unwrap();
        let meta = meta.into_inner().properties;
        assert_eq!(meta["device_info.backend"], backend);
        assert_ne!(meta["device_info.kind"], "DEVICE_CPU", "{backend}: {meta:?}");
        let got = c.model_infer(texts(&["a model runs where it was loaded"])).await.unwrap().into_inner();
        assert_eq!(text(&got.parameters, "backend"), backend, "{:?}", got.parameters);
        assert_eq!(text(&got.parameters, "stage.EMBED_STAGE_ENCODE"), "STAGE_DEVICE", "{:?}", got.parameters);
        // The same model, so the same vectors as the CPU's within the
        // devices' tolerance.
        let rows = vectors(&got);
        assert_eq!(rows.len(), cpu_rows.len());
        let cos = cosine(&rows[0], &cpu_rows[0]);
        assert!(cos > 0.999, "{backend}: cosine {cos} against the CPU");
        println!("{backend}: device {device} ran the encode stage");
        s.stop().await;
    }
}
