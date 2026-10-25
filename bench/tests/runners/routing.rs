use super::*;

#[test]
fn each_backend_gets_its_reference_programs() {
    use turbo_bench::{applies, wanted};
    assert_eq!(applies("cuda"), [TEI, TRT]);
    assert_eq!(applies("cpu"), [TEI]);
    assert_eq!(applies("levelzero"), [TEI, openvino::NAME], "OpenVINO, and TEI's CPU image as the end-to-end baseline");
    assert_eq!(
        applies("npu"),
        [TEI, openvino::NAME],
        "OpenVINO on the NPU, and TEI's CPU image as the end-to-end baseline"
    );
    assert_eq!(applies("metal"), [TEI], "TEI's router built natively with Metal");
    for backend in ["cuda", "cpu", "levelzero", "npu", "metal", "hailo"] {
        for name in applies(backend) {
            assert!(turbo::record::REFERENCES.iter().any(|r| r.0 == *name), "{name}");
        }
    }
    wanted("levelzero", |n| n == TEI || n == openvino::NAME).unwrap();
    let e = wanted("levelzero", |n| n == TEI).unwrap_err();
    assert_eq!(e, "levelzero: OpenVINO is a reference here: give --openvino-image or --openvino-bin, or --no-openvino");
    wanted("npu", |n| n == TEI || n == openvino::NAME).unwrap();
    let e = wanted("npu", |n| n == TEI).unwrap_err();
    assert_eq!(e, "npu: OpenVINO is a reference here: give --openvino-image or --openvino-bin, or --no-openvino");
    assert_eq!(wanted("npu", |_| true).unwrap_err(), "npu: TensorRT is not a reference for this backend");
    let e = wanted("levelzero", |_| true).unwrap_err();
    assert_eq!(e, "levelzero: TensorRT is not a reference for this backend");
    let e = wanted("cuda", |n| n == TEI || n == TRT).map(|_| ());
    assert!(e.is_ok());
    let e = wanted("cuda", |_| true).unwrap_err();
    assert_eq!(e, "cuda: OpenVINO is not a reference for this backend");
    wanted("metal", |n| n == TEI).unwrap();
    let e = wanted("metal", |_| false).unwrap_err();
    assert_eq!(e, "metal: TEI is a reference here: give --tei-image (or --tei-bin) and --tei-model, or --no-tei");
    assert_eq!(wanted("metal", |_| true).unwrap_err(), "metal: TensorRT is not a reference for this backend");
}

#[test]
fn hailo_requires_its_native_reference_or_an_explicit_disable() {
    assert_eq!(turbo_bench::applies("hailo"), ["hailort"]);
    turbo_bench::wanted("hailo", |n| n == "hailort").unwrap();
    assert!(turbo_bench::wanted("hailo", |_| false).unwrap_err().contains("--hailort-bin"));
    assert!(turbo_bench::wanted("cpu", |n| n == TEI || n == "hailort").unwrap_err().contains("not a reference"));
}

#[test]
fn f16_is_offered_to_a_router_on_a_gpu_only() {
    assert_eq!(tei::dtype(turbo::TURBO_DTYPE_F16, true), Ok("float16"));
    assert!(tei::dtype(turbo::TURBO_DTYPE_F16, false).is_err());
    assert_eq!(tei::dtype(turbo::TURBO_DTYPE_F32, false), Ok("float32"));
}

// ---- TensorRT ----
