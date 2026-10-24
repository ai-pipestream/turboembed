//! The split: which documents each side gets, in what order, and the
//! documents it leaves out or refuses.

use std::path::{Path, PathBuf};

use crate::common;
use crate::common::{GATED_DOCUMENTS, SENTENCES, nest_token, set_text_class, temp_dir, write_corpus};
use crate::split_of;
use turbo_formats::masc::{MascCorpus, MascTokenReader};
use turbo_train_components::{TokSpanEventStream, TokenizerFactory};
use turbo_train_recipes::masc::{ExcludedDocument, Exclusion, SPLIT};
use turbo_train_recipes::postag::{self, POS, ReferenceTags, Scored};
use turbo_train_recipes::segmentation::{SENTENCE, TOKENS, in_text_order, read_token_samples};
use turbo_train_recipes::{Error, Recipe, run};

/// `path` under `root` with its components joined by `/`, as the split
/// records it on every OS.
fn slashed(root: &Path, path: &Path) -> String {
    let inside = path.strip_prefix(root).unwrap();
    inside.components().map(|c| c.as_os_str().to_str().unwrap()).collect::<Vec<_>>().join("/")
}

#[test]
fn the_split_holds_out_every_tenth_document_in_read_order() {
    let root = temp_dir("split");
    write_corpus(&root, 25, SENTENCES);
    let all: Vec<PathBuf> = MascCorpus::read(&root).unwrap().documents().iter().map(|d| d.path().to_owned()).collect();
    let (split, digest) = split_of(&root);
    // A header, the text, and the quark, sentence, Penn, named entity,
    // noun chunk and verb chunk files.
    assert_eq!(digest.files, 25 * 8);
    assert_eq!(split.train.documents().len(), 23);
    let held: Vec<String> = [9, 19].iter().map(|&i| slashed(&root, &all[i])).collect();
    assert_eq!(split.held_out_paths, held);
    assert!(
        split.held_out_paths.iter().all(|p| !p.starts_with('/') && !p.contains('\\')),
        "{:?}",
        split.held_out_paths
    );
    // The read order is the reader's: each directory's entries by name,
    // a directory's subdirectories after its files, the last pushed first.
    assert!(split.held_out_paths[0].starts_with("written/news/"), "{:?}", split.held_out_paths);

    // A document the read skips (here, its header names no sentence
    // file) would move the documents after it: the split refuses the
    // corpus.
    let header = root.join(&split.held_out_paths[1]);
    let xml = std::fs::read_to_string(&header).unwrap();
    let without = xml.lines().filter(|l| !l.contains("f.id=\"f.s\"")).collect::<Vec<_>>().join("\n");
    assert_ne!(without, xml);
    std::fs::write(&header, without).unwrap();
    let corpus = MascCorpus::read(&root).unwrap();
    assert_eq!(corpus.skipped().len(), 1);
    let err = SPLIT.apply(corpus, &root).unwrap_err();
    assert!(matches!(&err, Error::SkippedDocuments { documents } if documents.len() == 1), "{err:?}");
    assert!(err.to_string().contains(&split.held_out_paths[1]), "{err}");
    std::fs::remove_dir_all(&root).unwrap();
}

/// The documents of MASC's Twitter genre, those whose header's
/// `textClass` names the category `TW`, are left out of both sides
/// wherever the split puts them; every other document keeps its side,
/// and the card and the gate record say what was left out.
#[test]
fn the_twitter_genre_is_left_out_of_both_sides_and_moves_no_other_document() {
    let root = temp_dir("twitter");
    // 90 documents: 9 held out, 8 once one is left out, 32 sentence
    // detector samples.
    write_corpus(&root, 90, SENTENCES);
    let corpus = || MascCorpus::read(&root).unwrap();
    let rel = |p: &Path| slashed(&root, p);
    let all: Vec<String> = corpus().documents().iter().map(|d| rel(d.path())).collect();
    let (tweets_trained, tweets_held_out) = (all[0].clone(), all[19].clone());
    set_text_class(&root, &tweets_trained, "WR TW");
    set_text_class(&root, &tweets_held_out, "WR TW");
    set_text_class(&root, &all[1], "WR BL");
    set_text_class(&root, &all[9], "WR TWX");

    let kept = SPLIT.apply_excluding(corpus(), &root, &[]).unwrap();
    assert!(kept.excluded.is_empty());
    let (split, digest) = split_of(&root);
    let excluded = |path: &str| ExcludedDocument { path: path.to_owned(), exclusion: Exclusion::TwitterGenre };
    assert_eq!(split.excluded, [excluded(&tweets_trained), excluded(&tweets_held_out)]);
    let paths = |c: &MascCorpus| c.documents().iter().map(|d| rel(d.path())).collect::<Vec<_>>();
    let without = |paths: Vec<String>, left_out: &str| paths.into_iter().filter(|p| p != left_out).collect::<Vec<_>>();
    assert_eq!(paths(&split.train), without(paths(&kept.train), &tweets_trained));
    assert_eq!(split.held_out_paths, without(kept.held_out_paths.clone(), &tweets_held_out));
    assert_eq!((split.train.documents().len(), split.held_out.documents().len()), (80, 8));

    let trained = SENTENCE.train(&split.train).unwrap();
    let outcome = run(&Recipe::Segmentation(SENTENCE), &split, &digest, false, "itself", &trained.bytes).unwrap();
    assert_eq!(outcome.card.corpus.excluded_documents, split.excluded);
    assert_eq!(outcome.card.corpus.held_out_documents, split.held_out_paths);
    assert_eq!(
        outcome.gate.set.name,
        "MASC 3.0.0 held out (every 10th document from the 10th, the Twitter genre left out)"
    );
    let json = outcome.card.to_json();
    assert!(
        json.contains(&format!("\"path\": \"{tweets_held_out}\",\n        \"exclusion\": \"twitter_genre\"")),
        "{json}"
    );
    std::fs::remove_dir_all(&root).unwrap();
}

/// A sentence whose tokens are out of text order (a token inside the one
/// before it) is skipped alone by the token recipe's reader, on both
/// sides, and counted with the other skipped sentences, where
/// `TokSpanEventStream` would refuse it; the tagger's reader reads it.
#[test]
fn a_sentence_with_a_token_inside_another_is_skipped_by_the_token_recipe() {
    // 250 documents of 40 sentences: 1,000 held-out sentences.
    let root = temp_dir("nested");
    write_corpus(&root, 250, SENTENCES);
    let (split, _) = split_of(&root);
    nest_token(&root, &split.held_out_paths[0], 0);
    let (split, _) = split_of(&root);
    let tokens = TOKENS.held_out(&split.held_out).unwrap();
    assert_eq!((tokens.len(), tokens.skipped()), (999, 1));
    let tags = POS.held_out(&split.held_out).unwrap();
    assert_eq!((tags.len(), tags.skipped()), (1000, 0));

    let mut refused = 0;
    let skipped = read_token_samples(&split.held_out, |sample| {
        assert!(in_text_order(sample.tokens()));
        Ok(())
    })
    .unwrap();
    assert_eq!(skipped, 1);
    let factory = TokenizerFactory::new("en", None, false, None).unwrap();
    let mut stream = TokSpanEventStream::new(&factory, turbo_train_recipes::segmentation::WHITESPACE);
    for read in MascTokenReader::new(&split.held_out).unwrap() {
        let sample = read.unwrap().sample;
        if !in_text_order(sample.tokens()) {
            let err = stream.events(&sample, &mut Vec::new()).unwrap_err();
            assert!(matches!(err, turbo_train_components::Error::TokensOutOfOrder { token: 1 }), "{err}");
            refused += 1;
        }
    }
    assert_eq!(refused, 1);

    // A training sentence is skipped the same way, and training goes on.
    let small = temp_dir("nested-train");
    write_corpus(&small, GATED_DOCUMENTS, SENTENCES);
    let (split, _) = split_of(&small);
    let first = split.train.documents()[0].path().strip_prefix(&small).unwrap().to_string_lossy().into_owned();
    let all = split.train.documents().len() * common::SENTENCES_PER_DOCUMENT;
    nest_token(&small, &first, 0);
    let (split, _) = split_of(&small);
    let trained = TOKENS.train(&split.train).unwrap();
    assert_eq!((trained.samples, trained.skipped_sentences), (all - 1, 1));
    for dir in [root, small] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn a_held_out_set_of_another_component_is_refused() {
    let root = temp_dir("component");
    write_corpus(&root, 20, SENTENCES);
    let (split, _) = split_of(&root);
    let model = SENTENCE.train(&split.train).unwrap();
    let tokens = TOKENS.held_out(&split.held_out).unwrap();
    assert!(SENTENCE.score(&model.bytes, &tokens).is_err());
    // A sentence model is not a tokenizer model.
    let sentences = SENTENCE.held_out(&split.held_out).unwrap();
    assert!(TOKENS.score(&model.bytes, &sentences).is_err());
    let tokens_again = TOKENS.held_out(&split.held_out).unwrap();
    assert!(TOKENS.score(&model.bytes, &tokens_again).is_err());
    // A sentence model is not a part-of-speech model, in any format, and
    // a part-of-speech model is not a tokenizer model.
    let tags = POS.held_out(&split.held_out).unwrap();
    let penn = ReferenceTags::of(&turbo_analysis::TagSet::new("penn", &["NN", "."]), &postag::REFERENCE_MAPPINGS);
    assert!(matches!(POS.score(&model.bytes, Scored::Candidate, &tags, &penn), Err(Error::Model(_))));
    assert!(matches!(POS.score(&model.bytes, Scored::Reference, &tags, &penn), Err(Error::Model(_))));
    let tagger = POS.train(&split.train).unwrap();
    assert!(TOKENS.score(&tagger.bytes, &tokens_again).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}
