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

/// Preserve the corrected reference metadata from #1567 without restoring
/// retired training workflows or runtime model loading.
#[test]
fn masc_references_do_not_use_the_retired_download_host() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let listed = std::fs::read_to_string(root.join("masc-references.txt")).unwrap();
    let references: Vec<_> = listed.lines().filter(|line| !line.is_empty() && !line.starts_with('#')).collect();
    assert!(!references.is_empty());
    for line in references {
        assert!(!line.contains("://opennlp.sourceforge.net/"), "{line}");
    }
}
