use super::*;

fn info(batch: u32, dim: u32) -> turbo_result_info {
    let mut i: turbo_result_info = unsafe { std::mem::zeroed() };
    i.batch = batch;
    i.dim = dim;
    i
}

#[test]
fn a_result_of_another_shape_is_refused() {
    assert!(result_shape(&info(3, 32), 3, 32).is_ok());
    let e = result_shape(&info(2, 32), 3, 32).unwrap_err();
    assert_eq!(e.code, INTERNAL);
    assert_eq!(e.message, "a result of 2 vectors of 32, for 3 vectors of 32");
    assert!(result_shape(&info(3, 16), 3, 32).is_err());
    assert!(result_shape(&info(4, 32), 3, 32).is_err());
}

/// The core's two refusals of a range write, as the core words them
/// (session.rs and backend.rs), name the backend and the write the
/// vector steps need; other errors pass through.
#[test]
fn a_backend_without_the_range_write_is_named() {
    let session = Error::field(UNSUPPORTED_OPTION, 7, "max_ranges 128: the hailo backend pools no ranges");
    let write = Error::new(UNSUPPORTED, "the hailo backend does not offer embed_write_ranges");
    for e in [session, write] {
        let m = needs_range_write(e, "hailo");
        assert_eq!(m.code, UNSUPPORTED);
        assert!(m.message.starts_with("the hailo backend pools no ranges: sentence and chunk vectors need"), "{m}");
        assert!(m.message.contains("(embed_write_ranges)"), "{m}");
    }
    let other = Error::field(UNSUPPORTED_OPTION, 7, "max_ranges 900 is over max_batch 8 times max_seq 64, 512");
    assert_eq!(needs_range_write(other.clone(), "cpu"), other);
    let other = Error::new(UNSUPPORTED, "the x backend does not offer embed_write_tokens");
    assert_eq!(needs_range_write(other.clone(), "x"), other);
}

#[test]
fn zero_stride_rows_are_refused() {
    assert!(std::panic::catch_unwind(|| Rows::new(0, 0)).is_err());
}
