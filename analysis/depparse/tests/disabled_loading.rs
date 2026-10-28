use turbo_depparse::{DependencyModel, FeedforwardDependencyModel};

#[test]
fn every_retired_dependency_reader_rejects_before_decoding() {
    for result in [
        DependencyModel::from_raw(b"not a model").map(|_| ()),
        DependencyModel::open(b"not a model".to_vec()).map(|_| ()),
        FeedforwardDependencyModel::from_bytes(b"not a model").map(|_| ()),
        FeedforwardDependencyModel::from_java_bytes(b"not a model").map(|_| ()),
        FeedforwardDependencyModel::from_apache_bytes(b"not a model").map(|_| ()),
    ] {
        assert!(result.unwrap_err().to_string().contains("OpenNLP model loading is disabled"));
    }
}
