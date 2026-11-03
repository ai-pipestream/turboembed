//! The LitBank coreference head in the built-in index (#1813).

use turbo_zoo::{Component, Format, Index, Kind, Licence, LicenceClass, Policy};

use super::COREF_HEAD;

const RELEASE: &str = "https://github.com/ai-pipestream/turboembed/releases/download/models-coref-litbank-1";

#[test]
fn the_litbank_coreference_head_is_a_permissive_default_tier_bundle() {
    let index = Index::builtin();
    let a = index.get(COREF_HEAD).unwrap();
    assert_eq!(
        (a.kind, a.format, a.license, a.class, a.component),
        (
            Kind::Head,
            Format::Safetensors,
            Licence::Apache2,
            LicenceClass::Permissive,
            Some(Component::CoreferenceResolver)
        )
    );
    assert_eq!(a.language.as_ref().map(|l| l.as_str()), Some("en"));
    assert_eq!(a.version, "1");
    assert!(a.origin.is_none());
    let data: Vec<(&str, Licence)> = a.trained_on.iter().map(|t| (t.name.as_str(), t.license)).collect();
    assert_eq!(data, [("LitBank", Licence::CcBy40)]);
    let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        ["manifest.json", "reference/reference.safetensors", "tokenizer.json", "weights/model.safetensors"]
    );
    // Release assets carry the artifact's id and the path with `/` as `-`.
    for f in &a.files {
        let asset = f.path.as_str().replace('/', "-");
        assert_eq!(f.url.as_str(), format!("{RELEASE}/{COREF_HEAD}-{asset}"));
    }
    let notice = a.attribution.as_deref().unwrap();
    assert!(notice.contains("LitBank") && notice.contains("conll_f1"), "{notice}");
    assert!(a.licences_to_accept(Policy::default()).is_empty());
}
