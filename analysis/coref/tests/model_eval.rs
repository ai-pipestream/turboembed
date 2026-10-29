//! The retired archive boundary refuses input before model evaluation.
//! GIS/QN link-probability tests belonged to the removed archive path;
//! this refusal test does not claim replacement ranker accuracy.

use turbo_coref::CorefModel;

#[test]
fn coreference_archive_loading_is_disabled() {
    for bytes in [Vec::new(), b"PK\x03\x04".to_vec(), b"not a model".to_vec()] {
        let error = CorefModel::read(bytes).unwrap_err();
        assert!(error.contains("disabled"), "{error}");
    }
}
