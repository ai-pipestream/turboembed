use turbo_maxent::{Error, ModelFile};

#[test]
fn archive_reader_refuses_inputs_before_decoding() {
    for bytes in [b"".as_slice(), b"not an archive", b"PK\x03\x04"] {
        let error = ModelFile::open(bytes.to_vec()).expect_err("OpenNLP loading must be disabled");
        assert!(matches!(error, Error::Disabled), "{error}");
    }
}

#[test]
fn disabled_loader_does_not_read_the_requested_path() {
    let error = ModelFile::read("model-loading-disabled-do-not-open.bin").unwrap_err();
    assert!(matches!(error, Error::Disabled), "{error}");
}

#[test]
fn raw_classical_model_reader_cannot_bypass_archive_retirement() {
    let error = turbo_maxent::MaxentModel::read(b"not a model").unwrap_err();
    assert!(matches!(error, Error::Disabled), "{error}");
}
