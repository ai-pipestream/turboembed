use std::io::Cursor;
use turbo_postag::neural::{BilstmPosModel, FeedforwardPosModel};

#[test]
fn retired_neural_streams_reject_before_decoding() {
    for result in [
        BilstmPosModel::load(Cursor::new(b"not a model")).map(|_| ()),
        FeedforwardPosModel::load(Cursor::new(b"not a model")).map(|_| ()),
    ] {
        assert!(result.unwrap_err().to_string().contains("OpenNLP model loading is disabled"));
    }
}

#[test]
fn retired_neural_paths_reject_before_file_access() {
    for result in [
        BilstmPosModel::load_from_path("missing-retired-model.bin").map(|_| ()),
        FeedforwardPosModel::load_from_path("missing-retired-model.bin").map(|_| ()),
    ] {
        assert!(result.unwrap_err().to_string().contains("OpenNLP model loading is disabled"));
    }
}
