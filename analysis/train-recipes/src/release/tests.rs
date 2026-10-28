use std::collections::BTreeSet;

use super::*;
use crate::treebank::TextRights;

#[test]
fn every_release_is_tagged_once_and_names_recipes_in_their_order() {
    let tags: BTreeSet<&str> = MODELS_RELEASES.iter().map(|r| r.tag).collect();
    assert_eq!(tags.len(), MODELS_RELEASES.len());
    for release in MODELS_RELEASES {
        let tag = release.tag;
        assert!(tag.starts_with(|c: char| c.is_ascii_alphanumeric()), "{tag}");
        assert!(tag.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')), "{tag}");
        assert!(!tag.contains("..") && !tag.ends_with('.') && !tag.ends_with(".lock"), "{tag}");
        let masc = release.recipes().into_iter().map(|r| r.model_id());
        let ud = release.ud_recipes().into_iter().map(|r| r.model_id);
        let language = release.language_recipes().into_iter().map(|r| r.model_id);
        let web = release.web_language_recipes().into_iter().map(|r| r.model_id);
        let depparse = release.depparse_recipes().into_iter().map(|r| r.model_id);
        let ids: Vec<&str> = masc.chain(ud).chain(language).chain(web).chain(depparse).collect();
        assert_eq!(ids, release.models, "{tag}: a model no recipe trains, or out of order");
        assert_eq!(ModelsRelease::tagged(tag).unwrap(), release);
    }
    assert_eq!(
        MASC_MODELS.url("turbo-en-masc-3.0.0-pos-1.bin"),
        "https://github.com/ai-pipestream/turboembed/releases/download/models-masc-3.0.0-1/\
         turbo-en-masc-3.0.0-pos-1.bin"
    );
    for unknown in ["", "--check", "models-masc-3.0.0-2", "../models-masc-3.0.0-1"] {
        match ModelsRelease::tagged(unknown) {
            Err(Error::UnknownRelease { tag, known }) => {
                assert_eq!(tag, unknown);
                assert_eq!(
                    known,
                    [MASC_MODELS.tag, UD_MODELS.tag, TATOEBA_MODELS.tag, FINEWEB_MODELS.tag, UD_DEPPARSE_MODELS.tag]
                );
            }
            other => panic!("{unknown:?}: {other:?}"),
        }
    }
}

/// The UD release holds every UD recipe's model but those of a
/// treebank whose text rights are not settled and those that failed
/// their gates, the Italian ones (analysis/train-recipes/README.md).
#[test]
fn the_ud_release_leaves_out_porttinari_and_markit() {
    for recipe in UD_RECIPES {
        let released = UD_MODELS.models.contains(&recipe.model_id);
        let settled = !matches!(recipe.treebank.text_rights, TextRights::NotSettled { .. });
        let italian = recipe.treebank.language == "it";
        assert_eq!(released, settled && !italian, "{}", recipe.model_id);
    }
}

/// The MASC gates read no reference: the UD 1.3 models built with
/// OpenNLP 2.5.4 are not fetched or pinned (#1509), so the release lists
/// none and masc-references.txt keeps only its header.
#[test]
fn masc_references_list_no_model() {
    assert_eq!(references(&MASC_MODELS).unwrap(), Vec::new());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let listed = std::fs::read_to_string(root.join("masc-references.txt")).unwrap();
    let references: Vec<_> = listed.lines().filter(|line| !line.is_empty() && !line.starts_with('#')).collect();
    assert_eq!(references, Vec::<&str>::new());
}

/// No OpenNLP 1.5 model is fetched or scored (#1509): the MASC tagger,
/// chunker and name finders have no reference model, no reference a
/// release reads is one, and masc-references.txt lists none.
#[test]
fn no_masc_recipe_or_listed_reference_is_an_opennlp_1_5_model() {
    for recipe in crate::RECIPES {
        match recipe {
            crate::Recipe::Segmentation(_) => assert!(recipe.reference().is_ok(), "{}", recipe.model_id()),
            _ => match recipe.reference() {
                Err(Error::NoReference { recipe: id }) => assert_eq!(id, recipe.model_id()),
                other => panic!("{}: {other:?}", recipe.model_id()),
            },
        }
    }
    let retired = ["models-1.5", "en-pos-maxent", "en-chunker", "en-ner-", "-1.5"];
    for release in MODELS_RELEASES {
        for reference in release.recipe_references() {
            assert!(!retired.iter().any(|r| reference.id().contains(r)), "{}: {}", release.tag, reference.id());
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let listed = std::fs::read_to_string(root.join("masc-references.txt")).unwrap();
    for line in listed.lines().filter(|l| !l.starts_with('#')) {
        assert!(!retired.iter().any(|r| line.contains(r)), "{line}");
    }
}
