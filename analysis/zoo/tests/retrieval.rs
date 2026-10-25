//! The retrieval checkpoints in the built-in index (#913).

use turbo_zoo::{Format, Index, Kind, Licence, LicenceClass, Policy};

const RETRIEVAL_CHECKPOINTS: [&str; 2] = ["opensearch-neural-sparse-encoding-v1", "colbert-ir-colbertv2.0"];

#[test]
fn the_retrieval_checkpoints_are_pinned_local_only_through_ms_marco() {
    let index = Index::builtin();
    for (id, repo, revision, licence, files) in [
        (
            RETRIEVAL_CHECKPOINTS[0],
            "opensearch-project/opensearch-neural-sparse-encoding-v1",
            "708f7e68f4aada15c2d59144a5187c521da6d8db",
            Licence::Apache2,
            &["model.safetensors", "config.json", "tokenizer.json", "1_SpladePooling/config.json"][..],
        ),
        (
            RETRIEVAL_CHECKPOINTS[1],
            "colbert-ir/colbertv2.0",
            "c1e84128e85ef755c096a95bdb06b47793b13acf",
            Licence::Mit,
            &["model.safetensors", "config.json", "tokenizer.json"][..],
        ),
    ] {
        let a = index.get(id).unwrap();
        assert_eq!(
            (a.kind, a.format, a.license, a.class),
            (Kind::Model, Format::Safetensors, licence, LicenceClass::LocalOnly),
            "{id}"
        );
        assert_eq!(a.version, revision, "{id}");
        let origin = format!("https://huggingface.co/{repo}");
        assert_eq!(a.origin.as_ref().map(|o| o.as_str()), Some(origin.as_str()), "{id}");
        let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, files, "{id}");
        for f in &a.files {
            assert_eq!(f.url.as_str(), format!("{origin}/resolve/{revision}/{}", f.path.as_str()), "{id}");
        }
        let data: Vec<Licence> = a.trained_on.iter().map(|t| t.license).collect();
        assert_eq!(data, [Licence::Apache2, Licence::RestrictedPublic], "{id}");
        let notice = a.attribution.as_deref().unwrap();
        assert!(notice.contains(revision), "{id}");
        for source in &a.trained_on {
            assert!(notice.contains(&source.name), "{id}");
            assert!(notice.contains(source.license_terms.as_str()), "{id}");
        }
        assert_eq!(
            a.licences_to_accept(Policy::default()),
            [turbo_zoo::Requirement {
                licence: Licence::RestrictedPublic,
                text: "https://microsoft.github.io/msmarco/".into(),
            }],
            "{id}"
        );
    }
}
