//! The neural annotator checkpoints in the built-in index.

use turbo_zoo::{Component, Format, Index, Kind, Licence, LicenceClass, Policy};

use super::{
    CHECKPOINTS, LOCAL_CHECKPOINTS, RETRIEVAL_CHECKPOINTS, SHARE_ALIKE_CHECKPOINTS, UNRELEASED_DATA_CHECKPOINTS,
};

#[test]
fn the_squad2_checkpoint_is_pinned_and_share_alike_through_its_data() {
    let index = Index::builtin();
    let a = index.get(SHARE_ALIKE_CHECKPOINTS[1]).unwrap();
    assert_eq!(
        (a.kind, a.format, a.license, a.class),
        (Kind::Model, Format::Safetensors, Licence::CcBy40, LicenceClass::ShareAlike)
    );
    assert!(a.trained_on.iter().any(|t| t.license == Licence::CcBySa40));
    let revision = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4";
    assert_eq!(a.version, revision);
    let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "config.json",
            "model.safetensors",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer_config.json"
        ]
    );
    for f in &a.files {
        let url = format!("https://huggingface.co/deepset/roberta-base-squad2/resolve/{revision}/{}", f.path.as_str());
        assert_eq!(f.url.as_str(), url);
    }
}

#[test]
fn the_glirel_tokenizer_is_pinned_and_permissive() {
    let index = Index::builtin();
    let a = index.get(CHECKPOINTS[0]).unwrap();
    assert_eq!(
        (a.kind, a.format, a.license, a.class),
        (Kind::Model, Format::Sentencepiece, Licence::Mit, LicenceClass::Permissive)
    );
    let revision = "64a8c8eab3e352a784c658aef62be1662607476f";
    assert_eq!(a.version, revision);
    let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(paths, ["spm.model", "config.json"]);
    for f in &a.files {
        let url = format!("https://huggingface.co/microsoft/deberta-v3-large/resolve/{revision}/{}", f.path.as_str());
        assert_eq!(f.url.as_str(), url);
    }
    // The same SentencePiece model GLiNER2's checkpoint carries.
    let gliner2 = index.get(UNRELEASED_DATA_CHECKPOINTS[0].0).unwrap();
    let spm =
        |a: &turbo_zoo::Artifact| a.files.iter().find(|f| f.path.as_str() == "spm.model").unwrap().sha256.to_string();
    assert_eq!(spm(a), spm(gliner2));
}

#[test]
fn the_glirel_checkpoint_is_pinned_local_only_and_needs_its_licence() {
    let index = Index::builtin();
    let a = index.get(LOCAL_CHECKPOINTS[0]).unwrap();
    assert_eq!(
        (a.kind, a.format, a.license, a.class),
        (Kind::Model, Format::Pytorch, Licence::CcByNcSa40, LicenceClass::LocalOnly)
    );
    let revision = "40a523e12a8432d6da364cf2a195a28755ff04d3";
    assert_eq!(a.version, revision);
    let origin = "https://huggingface.co/jackboyla/glirel-large-v0";
    assert_eq!(a.origin.as_ref().map(|o| o.as_str()), Some(origin));
    let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(paths, ["pytorch_model.bin", "glirel_config.json"]);
    for f in &a.files {
        assert_eq!(f.url.as_str(), format!("{origin}/resolve/{revision}/{}", f.path.as_str()));
    }
    assert_eq!(a.licences_to_accept(Policy::default()), [turbo_zoo::Requirement::of(Licence::CcByNcSa40).unwrap()]);
}

#[test]
fn the_gliner2_checkpoint_is_pinned_local_only_and_needs_its_card_terms() {
    let index = Index::builtin();
    let (id, terms) = UNRELEASED_DATA_CHECKPOINTS[0];
    let a = index.get(id).unwrap();
    assert_eq!(
        (a.kind, a.format, a.license, a.class),
        (Kind::Model, Format::Safetensors, Licence::Apache2, LicenceClass::LocalOnly)
    );
    assert_eq!(a.origin.as_ref().map(|o| o.as_str()), Some("https://huggingface.co/fastino/gliner2-base-v1"));
    assert_eq!(
        a.licences_to_accept(Policy::default()),
        [turbo_zoo::Requirement { licence: Licence::RestrictedPublic, text: terms.into() }]
    );
    let revision = "f9634218e53580c56edf0de97ca1a7d3f1c2354e";
    assert_eq!(a.version, revision);
    let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        ["model.safetensors", "config.json", "added_tokens.json", "spm.model", "encoder_config/config.json"]
    );
    for f in &a.files {
        let url = format!("https://huggingface.co/fastino/gliner2-base-v1/resolve/{revision}/{}", f.path.as_str());
        assert_eq!(f.url.as_str(), url);
    }
}

#[test]
fn the_gliner_v2_1_checkpoints_are_local_only_and_their_tokenizers_permissive() {
    let index = Index::builtin();
    for (id, repo, rev, format, licence, lang, files) in [
        (
            "urchade-gliner-small-v2.1",
            "urchade/gliner_small-v2.1",
            "4e091416cf7c3481db542c2a3d26156916f3a47f",
            Format::Pytorch,
            Licence::Apache2,
            "en",
            &["pytorch_model.bin", "gliner_config.json"][..],
        ),
        (
            "microsoft-deberta-v3-small-tokenizer",
            "microsoft/deberta-v3-small",
            "a36c739020e01763fe789b4b85e2df55d6180012",
            Format::Sentencepiece,
            Licence::Mit,
            "en",
            &["spm.model", "config.json"][..],
        ),
        (
            "urchade-gliner-multi-v2.1",
            "urchade/gliner_multi-v2.1",
            "443d26d654e0324125a96bebd8e796c14ff2efe6",
            Format::Safetensors,
            Licence::Apache2,
            "mul",
            &["model.safetensors", "gliner_config.json"][..],
        ),
        (
            "microsoft-mdeberta-v3-base-tokenizer",
            "microsoft/mdeberta-v3-base",
            "a0484667b22365f84929a935b5e50a51f71f159d",
            Format::Sentencepiece,
            Licence::Mit,
            "mul",
            &["spm.model", "config.json"][..],
        ),
    ] {
        let a = index.get(id).unwrap();
        // The checkpoints carry The Pile's text through their training
        // data (Kristian, 2026-10-28); the tokenizers are MIT alone.
        let unreleased = UNRELEASED_DATA_CHECKPOINTS.iter().find(|(u, _)| *u == id);
        let class = if unreleased.is_some() { LicenceClass::LocalOnly } else { LicenceClass::Permissive };
        assert_eq!((a.kind, a.format, a.license, a.class), (Kind::Model, format, licence, class));
        assert_eq!(a.language.as_ref().map(|l| l.as_str()), Some(lang));
        assert_eq!(a.version, rev);
        let notice = a.attribution.as_deref().unwrap();
        assert!(notice.contains(repo) && notice.contains(rev));
        for source in &a.trained_on {
            assert!(notice.contains(&source.name));
            assert!(notice.contains(source.license_terms.as_str()));
        }
        let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, files);
        for f in &a.files {
            assert_eq!(f.url.as_str(), format!("https://huggingface.co/{repo}/resolve/{rev}/{}", f.path.as_str()));
        }
        match unreleased {
            Some((_, terms)) => {
                assert_eq!(
                    a.origin.as_ref().map(|o| o.as_str()),
                    Some(format!("https://huggingface.co/{repo}").as_str())
                );
                assert_eq!(
                    a.licences_to_accept(Policy::default()),
                    [turbo_zoo::Requirement { licence: Licence::RestrictedPublic, text: (*terms).into() }]
                );
            }
            None => assert!(a.licences_to_accept(Policy::default()).is_empty()),
        }
    }
}

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
    let a = Index::builtin().get(CHECKPOINTS[3]).unwrap().clone();
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

#[test]
fn the_bge_dense_encoders_are_pinned_local_only_and_fetched_from_their_publisher() {
    let index = Index::builtin();
    for (id, repo, revision) in [
        ("baai-bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
        ("baai-bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
        ("baai-bge-large-en-v1.5", "BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09"),
    ] {
        assert!(RETRIEVAL_CHECKPOINTS.contains(&id));
        let a = index.get(id).unwrap();
        assert_eq!(
            (a.kind, a.format, a.license, a.class, a.component),
            (Kind::Model, Format::Safetensors, Licence::Mit, LicenceClass::LocalOnly, Some(Component::DenseEncoder)),
            "{id}"
        );
        assert_eq!(a.language.as_ref().map(|l| l.as_str()), Some("en"), "{id}");
        assert_eq!(a.version, revision, "{id}");
        let origin = format!("https://huggingface.co/{repo}");
        assert_eq!(a.origin.as_ref().map(|o| o.as_str()), Some(origin.as_str()), "{id}");
        let paths: Vec<&str> = a.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["model.safetensors", "config.json", "tokenizer.json"], "{id}");
        for f in &a.files {
            assert_eq!(f.url.as_str(), format!("{origin}/resolve/{revision}/{}", f.path.as_str()), "{id}");
        }
        // The same bert-base-uncased tokenizer the sparse encoder carries.
        let tokenizer = |a: &turbo_zoo::Artifact| {
            a.files.iter().find(|f| f.path.as_str() == "tokenizer.json").unwrap().sha256.to_string()
        };
        assert_eq!(tokenizer(a), tokenizer(index.get(RETRIEVAL_CHECKPOINTS[0]).unwrap()), "{id}");
        assert_eq!(
            a.licences_to_accept(Policy::default()),
            [turbo_zoo::Requirement {
                licence: Licence::RestrictedPublic,
                text: "https://microsoft.github.io/msmarco/".into()
            }],
            "{id}"
        );
    }
}
