//! Two neural checkpoints' pins, licences and attributions: the NLI
//! cross-encoder and SaT's sat-3l.

use turbo_zoo::{Format, Index, Kind, Licence, LicenceClass, Policy};

use super::{CHECKPOINTS, SHARE_ALIKE_CHECKPOINTS};

#[test]
fn the_nli_checkpoint_is_pinned_and_share_alike() {
    let index = Index::builtin();
    let a = index.get(SHARE_ALIKE_CHECKPOINTS[0]).unwrap();
    assert_eq!(
        (a.kind, a.format, a.license, a.class),
        (Kind::Model, Format::Safetensors, Licence::Apache2, LicenceClass::ShareAlike)
    );
    let revision = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d";
    assert_eq!(a.version, revision);
    let notice = a.attribution.as_deref().unwrap();
    assert!(notice.contains(revision));
    for source in &a.trained_on {
        assert!(notice.contains(&source.name));
        assert!(notice.contains(source.license_terms.as_str()));
    }

    let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt"
        ]
    );
    for f in &a.files {
        let url =
            format!("https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768/resolve/{revision}/{}", f.path.as_str());
        assert_eq!(f.url.as_str(), url);
    }
    assert_eq!(
        a.licences_to_accept(Policy::default()),
        [
            turbo_zoo::Requirement::of(Licence::CcBySa30).unwrap(),
            turbo_zoo::Requirement::of(Licence::CcBySa40).unwrap(),
        ]
    );
}

/// SaT's sat-3l, trained on mC4 alone with limited lookahead (#887).
#[test]
fn sat_3l_is_pinned_permissive_and_attributed() {
    let a = Index::builtin().get(CHECKPOINTS[6]).unwrap().clone();
    let k = (a.kind, a.format, a.license, a.class);
    assert_eq!(k, (Kind::Model, Format::Pytorch, Licence::Mit, LicenceClass::Permissive));
    let revision = "a3b7ba6b61881d619989757aea971cde7d3d6a98";
    assert_eq!(a.version, revision);
    let licences: Vec<Licence> = a.trained_on.iter().map(|t| t.license).collect();
    assert_eq!(licences, [Licence::Mit, Licence::OdcBy10]);
    assert!(a.attribution.as_deref().unwrap().contains("segment-any-text/sat-3l at a3b7ba6b"));
    for f in &a.files {
        let url = format!("https://huggingface.co/segment-any-text/sat-3l/resolve/{revision}/{}", f.path.as_str());
        assert_eq!(f.url.as_str(), url);
    }
}
