//! Verifies that all first-party TurboEmbed protobuf contracts live in
//! versioned `turbo.<name>.v1` packages and register under `.v1` gRPC service
//! names while `inference` and `grpc.health.v1` keep their standard upstream
//! package names (issue #1296).

use std::collections::BTreeMap;
use std::path::Path;

#[test]
fn all_turbo_protos_use_versioned_v1_packages_and_service_names() {
    assert_eq!(turbo_kserve::proto::grpc_inference_service_server::SERVICE_NAME, "inference.GRPCInferenceService");
    assert_eq!(turbo_kserve::proto::health::health_server::SERVICE_NAME, "grpc.health.v1.Health");
    assert_eq!(turbo_kserve::proto::result::analysis_service_server::SERVICE_NAME, "turbo.result.v1.AnalysisService");
    assert_eq!(
        turbo_kserve::proto::search::turbo_search_service_server::SERVICE_NAME,
        "turbo.search.v1.TurboSearchService"
    );
    assert_eq!(turbo_kserve::proto::sink::turbo_sink_service_server::SERVICE_NAME, "turbo.sink.v1.TurboSinkService");
    assert_eq!(
        turbo_kserve::proto::training::turbo_catalog_service_server::SERVICE_NAME,
        "turbo.training.v1.TurboCatalogService"
    );
    assert_eq!(
        turbo_kserve::proto::training::turbo_training_service_server::SERVICE_NAME,
        "turbo.training.v1.TurboTrainingService"
    );
    assert_eq!(
        turbo_kserve::proto::vocabulary::turbo_vocabulary_service_server::SERVICE_NAME,
        "turbo.vocabulary.v1.TurboVocabularyService"
    );

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let proto_dir = manifest_dir.join("proto");

    let expected_packages = BTreeMap::from([
        ("health.proto", "grpc.health.v1"),
        ("open_inference_grpc.proto", "inference"),
        ("turbo_analysis.proto", "turbo.result.v1"),
        ("turbo_search.proto", "turbo.search.v1"),
        ("turbo_sink.proto", "turbo.sink.v1"),
        ("turbo_training.proto", "turbo.training.v1"),
        ("turbo_typed_annotations.proto", "turbo.result.v1"),
        ("turbo_vocabulary.proto", "turbo.vocabulary.v1"),
    ]);

    let mut actual_packages = BTreeMap::new();
    for entry in std::fs::read_dir(&proto_dir).expect("read server/proto") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|s| s.to_str()) != Some("proto") {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        let content = std::fs::read_to_string(&path).expect("read proto file");
        let pkg = content
            .lines()
            .find_map(|line| line.trim().strip_prefix("package ")?.strip_suffix(';'))
            .expect("package declaration")
            .trim()
            .to_owned();
        if name.starts_with("turbo_") {
            assert!(
                pkg.starts_with("turbo.") && pkg.ends_with(".v1"),
                "{name} must declare a turbo.<name>.v1 package, got {pkg:?}"
            );
        }
        actual_packages.insert(name, pkg);
    }

    let actual_refs: BTreeMap<&str, &str> = actual_packages.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    assert_eq!(actual_refs, expected_packages);

    let buf_yaml = std::fs::read_to_string(proto_dir.join("buf.yaml")).expect("server/proto/buf.yaml exists");
    assert!(buf_yaml.contains("PACKAGE_VERSION_SUFFIX"));
    assert!(buf_yaml.contains("FILE"));
    assert!(buf_yaml.contains("WIRE_JSON"));

    let grpc_doc = std::fs::read_to_string(manifest_dir.join("../docs/grpc.md")).expect("docs/grpc.md exists");
    assert!(grpc_doc.contains("### Proto compatibility policy"));
    assert!(grpc_doc.contains("What counts as breaking"));
    assert!(grpc_doc.contains("How `v2` is introduced"));
    assert!(grpc_doc.contains("How long `v1` stays served"));
}
