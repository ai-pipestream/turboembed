//! Always-on ABI tests. No model weights required.

use turborerank::{
    abi_version, device_name, Activation, Device, Engine, Error, TokenBuffer, Truncation,
};

#[test]
fn abi_version_is_one() {
    assert_eq!(abi_version(), 1);
    assert_eq!(device_name(Device::Cpu), "CPU");
}

#[test]
fn buffer_is_64_byte_aligned_and_caller_writable() {
    let mut buf = TokenBuffer::alloc(Device::Cpu, 2, 32).unwrap();
    assert!(buf.ptr_aligned(), "CPU buffers must be 64-byte aligned");
    assert_eq!(buf.batch(), 2);
    assert_eq!(buf.seq(), 32);
    buf.input_ids_mut()[0] = 101;
    buf.input_ids_mut()[1] = 7592;
    assert_eq!(buf.input_ids()[0], 101);
    assert_eq!(buf.input_ids()[1], 7592);
}

#[cfg(not(turborerank_cuda))]
#[test]
fn cuda_buffer_fails_loud_without_cuda() {
    let err = TokenBuffer::alloc(Device::Cuda, 1, 16).unwrap_err();
    assert!(
        matches!(err, Error::NotImplemented(_) | Error::Unavailable(_)),
        "{err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("refus")
            || msg.to_lowercase().contains("not implemented")
            || msg.to_lowercase().contains("without cuda"),
        "{msg}"
    );
}

#[cfg(turborerank_cuda)]
#[test]
fn cuda_buffer_is_pinned_and_caller_writable() {
    let mut buf = TokenBuffer::alloc(Device::Cuda, 2, 32).expect("cudaHostAlloc");
    assert_eq!(buf.device(), Device::Cuda);
    assert!(buf.ptr_aligned(), "pinned buffers must be 64-byte aligned");
    buf.input_ids_mut()[0] = 101;
    buf.input_ids_mut()[1] = 7592;
    assert_eq!(buf.input_ids()[0], 101);
    assert_eq!(buf.input_ids()[1], 7592);
    let auto_buf = TokenBuffer::alloc(Device::Auto, 1, 16).expect("AUTO→CUDA buffer");
    assert_eq!(auto_buf.device(), Device::Cuda);
}

#[test]
fn pack_writes_cls_sep_types() {
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 16).unwrap();
    buf.pack_ids(0, &[10, 11], &[20, 21, 22], Truncation::LongestFirst, 16)
        .unwrap();
    let ids = buf.input_ids();
    assert_eq!(&ids[..8], &[101, 10, 11, 102, 20, 21, 22, 102]);
    assert_eq!(&buf.attention_mask()[..8], &[1, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(buf.attention_mask()[8], 0);
    assert_eq!(&buf.token_type_ids()[..8], &[0, 0, 0, 0, 1, 1, 1, 1]);
}

#[test]
fn pack_empty_query_or_doc() {
    let mut buf = TokenBuffer::alloc(Device::Cpu, 2, 8).unwrap();
    buf.pack_ids(0, &[5], &[], Truncation::LongestFirst, 8)
        .unwrap();
    assert_eq!(&buf.input_ids()[..4], &[101, 5, 102, 102]);
    buf.pack_ids(1, &[], &[9, 8], Truncation::LongestFirst, 8)
        .unwrap();
    assert_eq!(&buf.input_ids()[8..13], &[101, 102, 9, 8, 102]);
}

#[test]
fn pack_longest_first_and_query_priority() {
    let mut buf = TokenBuffer::alloc(Device::Cpu, 2, 16).unwrap();
    let q = [1, 2, 3, 4, 5];
    let d = [6, 7, 8, 9, 10, 11, 12];
    buf.pack_ids(0, &q, &d, Truncation::LongestFirst, 8)
        .unwrap();
    let ones: i32 = buf.attention_mask()[..16].iter().sum();
    assert_eq!(ones, 8);
    assert_eq!(buf.input_ids()[0], 101);
    assert_eq!(buf.input_ids()[7], 102);

    buf.pack_ids(1, &q, &d, Truncation::QueryPriority, 8)
        .unwrap();
    let row = &buf.input_ids()[16..24];
    assert_eq!(row[0], 101);
    assert_eq!(&row[1..6], &[1, 2, 3, 4, 5]);
    assert_eq!(row[6], 102);
    assert_eq!(row[7], 102);
}

#[test]
fn pack_error_and_max_len_boundary() {
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 16).unwrap();
    let err = buf
        .pack_ids(
            0,
            &[1, 2, 3, 4, 5],
            &[6, 7, 8, 9],
            Truncation::Error,
            8,
        )
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));

    buf.pack_ids(0, &[1, 2, 3], &[4], Truncation::LongestFirst, 3)
        .unwrap();
    assert_eq!(&buf.input_ids()[..3], &[101, 102, 102]);

    let err = buf
        .pack_ids(0, &[1], &[2], Truncation::LongestFirst, 2)
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)));
}

#[test]
fn remaining_accelerators_fail_loud_no_cpu_fallback() {
    for device in [
        Device::TensorRt,
        Device::OpenVinoGpu,
        Device::OpenVinoNpu,
        Device::Metal,
    ] {
        let err = Engine::create(device).unwrap_err();
        assert!(
            matches!(err, Error::Unavailable(_) | Error::UnsupportedDevice(_)),
            "{device:?}: {err:?}"
        );
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("refus") || msg.contains("cpu fallback"),
            "{device:?}: {err}"
        );
    }
    let err = Engine::create(Device::OpenVinoCpu).unwrap_err();
    assert!(matches!(err, Error::NotImplemented(_)), "{err:?}");
}

#[cfg(not(turborerank_cuda))]
#[test]
fn cuda_and_auto_create_fail_loud_without_cuda() {
    for device in [Device::Auto, Device::Cuda] {
        let err = Engine::create(device).unwrap_err();
        assert!(
            matches!(err, Error::Unavailable(_) | Error::UnsupportedDevice(_)),
            "{device:?}: {err:?}"
        );
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("refus") || msg.contains("cpu fallback"),
            "{device:?}: {err}"
        );
    }
}

#[cfg(turborerank_cuda)]
#[test]
fn cuda_and_auto_create_succeed_when_compiled() {
    let cuda = Engine::create(Device::Cuda).expect("CUDA engine");
    drop(cuda);
    let auto = Engine::create(Device::Auto).expect("AUTO→CUDA engine");
    drop(auto);
}

#[test]
fn mock_refuses_catalog_ce() {
    let engine = Engine::create(Device::Mock).unwrap();
    let err = engine.load_model("ms-marco-minilm-l6").unwrap_err();
    assert!(matches!(err, Error::NotImplemented(_)), "{err:?}");
    assert!(
        err.to_string().to_lowercase().contains("mock"),
        "{}",
        err
    );
}

#[test]
fn missing_weights_fails_loud() {
    let engine = Engine::create_with_config(Device::Cpu, Some("/no/such/turborerank".as_ref()))
        .unwrap();
    let err = engine.load_model("not-a-real-ce").unwrap_err();
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
    let msg = err.to_string().to_lowercase();
    assert!(msg.contains("missing") || msg.contains("not found") || msg.contains("unavailable"));
}

#[test]
fn cpu_engine_lists_catalog_alias() {
    let engine = Engine::create(Device::Cpu).unwrap();
    let models = engine.list_models().unwrap();
    assert!(
        models.iter().any(|m| m.alias == "ms-marco-minilm-l6"),
        "{models:?}"
    );
    assert!(models.iter().all(|m| !m.ready || m.hidden_size == 384));
}

#[test]
fn forward_without_load_fails() {
    let engine = Engine::create(Device::Cpu).unwrap();
    let buf = TokenBuffer::alloc(Device::Cpu, 1, 16).unwrap();
    let err = engine.forward(&buf, 1, Activation::Sigmoid).unwrap_err();
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
}
