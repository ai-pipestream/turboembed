//! Sentences the recipes' readers skip, and how many they may.

use crate::common;
use crate::common::{GATED_DOCUMENTS, SENTENCES, add_chunk, rewrite_document, temp_dir, write_corpus};
use crate::{sentences_after, split_of, with_skipped_sentences};
use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::card::ChunksCard;
use turbo_train_recipes::chunk::{self, CHUNKS, Skipped};
use turbo_train_recipes::masc::SPLIT;
use turbo_train_recipes::postag::POS;
use turbo_train_recipes::segmentation::{SENTENCE, TOKENS};
use turbo_train_recipes::{Error, LeftOut, RECIPES, Recipe, held_out_name, run};

fn too_many(err: Option<Error>, side: LeftOut, skipped: u64, of: u64) {
    match err {
        Some(Error::TooManyLeftOut { what, count, of: total, limit })
            if (what, count, total) == (side, skipped, of) =>
        {
            assert_eq!(limit, turbo_train_recipes::MAX_SKIPPED_SENTENCES);
        }
        other => panic!("{side:?}, {skipped} of {of}: {other:?}"),
    }
}

/// A recipe's reader skips held-out sentences as it skips training ones;
/// the gate run and the card record how many, and a recipe fails when
/// either side skips more than one sentence in a thousand.
#[test]
fn skipped_sentences_are_recorded_and_bounded() {
    let pool = with_skipped_sentences();
    let (untagged, no_tokens) = (SENTENCES.len(), SENTENCES.len() + 1);
    // 250 documents of 40 sentences: 1,000 held-out sentences.
    let root = temp_dir("skips");
    write_corpus(&root, 250, SENTENCES);
    let (split, _) = split_of(&root);
    rewrite_document(&root, &split.held_out_paths[0], &pool, &sentences_after(&[no_tokens]));
    let (split, digest) = split_of(&root);
    assert_eq!(split.held_out.documents().len(), 25);

    // One sentence in a thousand is within the limit.
    let tags = POS.held_out(&split.held_out).unwrap();
    assert_eq!((tags.len(), tags.skipped()), (999, 1));
    let tokens = TOKENS.held_out(&split.held_out).unwrap();
    assert_eq!((tokens.len(), tokens.skipped()), (999, 1));
    assert_eq!(SENTENCE.held_out(&split.held_out).unwrap().skipped(), 0);

    // The gate run reports the count, which `run` puts on the card.
    let small = temp_dir("skips-train");
    write_corpus(&small, GATED_DOCUMENTS, SENTENCES);
    let (small_split, _) = split_of(&small);
    for recipe in [Recipe::Segmentation(TOKENS), Recipe::Pos(POS)] {
        let trained = recipe.train(&small_split.train).unwrap();
        let gated = recipe
            .gate(&trained.bytes, "itself", &trained.bytes, &split.held_out, &held_out_name(), &digest.sha256)
            .unwrap();
        assert_eq!((gated.held_out.samples, gated.held_out.skipped_sentences), (999, 1), "{}", recipe.model_id());
        assert_eq!(gated.record.set.skipped_sentences, 1);
        assert_eq!(gated.record.units.len(), 999);
    }

    // Two in a thousand are not: a sentence without a tag for the tagger,
    rewrite_document(&root, &split.held_out_paths[1], &pool, &sentences_after(&[untagged]));
    let (split, _) = split_of(&root);
    too_many(POS.held_out(&split.held_out).err(), LeftOut::HeldOutSentences, 2, 1000);
    let message = POS.held_out(&split.held_out).unwrap_err().to_string();
    assert_eq!(message, "2 of 1000 held-out sentences skipped, more than the limit of 1 in 1000");
    assert_eq!(TOKENS.held_out(&split.held_out).unwrap().skipped(), 1);
    // and another sentence without tokens for both.
    rewrite_document(&root, &split.held_out_paths[1], &pool, &sentences_after(&[no_tokens]));
    let (split, _) = split_of(&root);
    too_many(POS.held_out(&split.held_out).err(), LeftOut::HeldOutSentences, 2, 1000);
    too_many(TOKENS.held_out(&split.held_out).err(), LeftOut::HeldOutSentences, 2, 1000);

    // Training sentences are bounded the same way, before any training.
    let first = split.train.documents()[0].path().strip_prefix(&root).unwrap().to_string_lossy().into_owned();
    rewrite_document(&root, &first, &pool, &[[untagged; 20], [no_tokens; 20]].concat());
    let (split, _) = split_of(&root);
    too_many(POS.train(&split.train).err(), LeftOut::TrainingSentences, 40, 9000);
    too_many(TOKENS.train(&split.train).err(), LeftOut::TrainingSentences, 20, 9000);
    for dir in [root, small] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}

/// The Penn number of the first token of each sentence of a document
/// written of `sentences` of `pool`, as `write_document` numbers them.
fn sentence_starts(pool: &[&str], sentences: &[usize]) -> Vec<usize> {
    let mut starts = Vec::with_capacity(sentences.len());
    let mut q = 0;
    for &i in sentences {
        starts.push(q);
        q += common::tokens(pool[i]).len();
    }
    starts
}

/// The chunker's reader skips a sentence with chunks that chunk tags
/// cannot express: one with part of a chunk that crosses into another sentence,
/// or with two chunks over a token. The same chunk given twice is one
/// chunk. The gate run records the skipped sentences and the chunks in no
/// sentence, and a recipe fails when either side skips more than one in
/// ten, before any training, or on the recorded release skips other
/// counts than the recorded ones.
#[test]
fn sentences_whose_chunks_cross_or_overlap_are_skipped_by_the_chunker_and_bounded() {
    let root = temp_dir("chunk-skips");
    write_corpus(&root, GATED_DOCUMENTS, SENTENCES);
    let (split, _) = split_of(&root);
    let held = split.held_out_paths[0].clone();
    let sentences = sentences_after(&[0, 1]);
    rewrite_document(&root, &held, SENTENCES, &sentences);
    let starts = sentence_starts(SENTENCES, &sentences);
    assert_eq!(&starts[..3], [0, 15, 20]);
    // `Dr. Smith` again as a noun chunk
    add_chunk(&root, &held, "nc", 0, &[0, 1]);
    let (split, _) = split_of(&root);
    let chunks = CHUNKS.held_out(&split.held_out).unwrap();
    assert_eq!((chunks.len(), chunks.skipped().total()), (320, 0));
    let read = chunks.chunks().chunks;
    assert_eq!(chunks.chunks(), &ChunksCard { chunks: read, ..ChunksCard::default() });

    // a noun chunk from the first sentence's last token into the second,
    // and a verb chunk over `Smith said` in the third
    add_chunk(&root, &held, "nc", 1, &[14, 15]);
    add_chunk(&root, &held, "vc", 2, &[21, 22]);
    // a noun chunk of a Penn token no sentence has, and a verb chunk of
    // no token: in no sentence, so no sentence is skipped for them
    add_chunk(&root, &held, "nc", 3, &[99_999]);
    add_chunk(&root, &held, "vc", 4, &[]);
    let (split, digest) = split_of(&root);
    let chunks = CHUNKS.held_out(&split.held_out).unwrap();
    assert_eq!(chunks.skipped(), Skipped { partial_chunk: 2, overlapping_chunks: 1, ..Skipped::default() });
    assert_eq!(chunks.len(), 317);
    let left_out = ChunksCard { chunks: read + 4, outside_sentences: 1, across_sentences: 1, without_tokens: 1 };
    assert_eq!(chunks.chunks(), &left_out);
    let trained = CHUNKS.train(&split.train).unwrap();
    assert_eq!(trained.chunks, Some(chunk::chunk_counts(&split.train)));
    let gated = Recipe::Chunk(CHUNKS)
        .gate(&trained.bytes, "itself", &trained.bytes, &split.held_out, &held_out_name(), &digest.sha256)
        .unwrap();
    assert_eq!((gated.held_out.samples, gated.held_out.skipped_sentences), (317, 3));
    assert_eq!((gated.record.set.skipped_sentences, gated.record.units.len()), (3, 317));
    assert_eq!(gated.held_out.chunks, Some(left_out));

    // On the recorded release the sentences skipped must be the counts
    // the recipe records (1,872 training, 349 held out), checked before
    // training and after it; on another corpus only the bound applies.
    let e = Recipe::Chunk(CHUNKS).check_data(&split, true, "reference", None).unwrap_err();
    assert!(
        matches!(e, Error::SkippedNotRecorded { what: LeftOut::TrainingSentences, count: 0, recorded: 1872 }),
        "{e:?}"
    );
    assert_eq!(e.to_string(), "0 training sentences skipped, not the recorded release's 1872");
    Recipe::Chunk(CHUNKS).check_data(&split, false, "reference", None).unwrap();
    let e = run(&Recipe::Chunk(CHUNKS), &split, &digest, true, "itself", &trained.bytes).unwrap_err();
    assert!(
        matches!(e, Error::SkippedNotRecorded { what: LeftOut::TrainingSentences, count: 0, recorded: 1872 }),
        "{e:?}"
    );
    chunk::check_recorded_skips(LeftOut::TrainingSentences, 1872).unwrap();
    chunk::check_recorded_skips(LeftOut::HeldOutSentences, 349).unwrap();
    let e = chunk::check_recorded_skips(LeftOut::HeldOutSentences, 350).unwrap_err();
    assert!(
        matches!(e, Error::SkippedNotRecorded { what: LeftOut::HeldOutSentences, count: 350, recorded: 349 }),
        "{e:?}"
    );
    assert_eq!(e.to_string(), "350 held-out sentences skipped, not the recorded release's 349");
    assert!(chunk::check_recorded_skips(LeftOut::TrainingSentences, 1871).is_err());
    Recipe::Chunk(CHUNKS).check_recorded_skips(false, LeftOut::TrainingSentences, 0).unwrap();
    for recipe in RECIPES.into_iter().filter(|r| !matches!(r, Recipe::Chunk(_))) {
        recipe.check_recorded_skips(true, LeftOut::TrainingSentences, 0).unwrap();
    }

    // A verb chunk over the first two tokens of each sentence of another
    // held-out document: 43 of 320 is more than one in ten.
    let other = split.held_out_paths[1].clone();
    let doc = split.held_out.documents()[1].path().to_owned();
    assert!(doc.ends_with(&other));
    let written: Vec<usize> = (0..common::SENTENCES_PER_DOCUMENT).map(|k| k % SENTENCES.len()).collect();
    rewrite_document(&root, &other, SENTENCES, &written);
    for (k, &q) in sentence_starts(SENTENCES, &written).iter().enumerate() {
        add_chunk(&root, &other, "vc", 10 + k, &[q, q + 1]);
    }
    let (split, _) = split_of(&root);
    let limited = |err: Option<Error>, side: LeftOut, count: u64, of: u64| match err {
        Some(Error::TooManyLeftOut { what, count: c, of: o, limit }) => {
            assert_eq!((what, c, o, limit), (side, count, of, chunk::MAX_SKIPPED_SENTENCES));
        }
        other => panic!("{other:?}"),
    };
    limited(CHUNKS.held_out(&split.held_out).err(), LeftOut::HeldOutSentences, 43, 320);
    let message = CHUNKS.held_out(&split.held_out).unwrap_err().to_string();
    assert_eq!(message, "43 of 320 held-out sentences skipped, more than the limit of 1 in 10");
    // The check made before training finds it too.
    let checked = Recipe::Chunk(CHUNKS).check_data(&split, false, "reference", None).unwrap_err();
    assert_eq!(checked.to_string(), message);

    // Training sentences are bounded the same way, before any training:
    // every sentence of eight of the 72 training documents is skipped.
    let training: Vec<String> = split.train.documents()[..8]
        .iter()
        .map(|d| d.path().strip_prefix(&root).unwrap().to_string_lossy().into_owned())
        .collect();
    for doc in &training {
        rewrite_document(&root, doc, SENTENCES, &written);
        for (k, &q) in sentence_starts(SENTENCES, &written).iter().enumerate() {
            add_chunk(&root, doc, "vc", 10 + k, &[q, q + 1]);
        }
    }
    let (split, _) = split_of(&root);
    assert_eq!(split.train.documents().len() * common::SENTENCES_PER_DOCUMENT, 2880);
    limited(CHUNKS.train(&split.train).err(), LeftOut::TrainingSentences, 320, 2880);
    limited(CHUNKS.check_training(&split.train).err(), LeftOut::TrainingSentences, 320, 2880);
    std::fs::remove_dir_all(&root).unwrap();
}

/// The chunker's reference gives chunk types MASC does not annotate; its
/// tags of those types are made `O` before it is scored, so its `NP` and
/// `VP` chunks are scored as they are and the others count neither for
/// nor against it. Here the reference is the recipe trained on the same
/// samples with every `IN` token a `PP` chunk, as CoNLL-2000 has them:
/// scored, it is right on every chunk, as the candidate is, where its
/// `PP` chunks would count as wrong predictions if they were kept.
#[test]
fn the_reference_s_chunks_of_types_the_gate_does_not_score_are_left_out() {
    use turbo_formats::evaluate::ChunkerEvaluator;
    use turbo_formats::sample::ChunkSample;
    use turbo_formats::source::Field;

    let root = temp_dir("chunk-types");
    write_corpus(&root, GATED_DOCUMENTS, SENTENCES);
    let (split, digest) = split_of(&root);
    let candidate = CHUNKS.train(&split.train).unwrap();
    let mut samples = Vec::new();
    chunk::read_samples(&split.train, |s| samples.push(s)).unwrap();
    let with_pp: Vec<ChunkSample<'_>> = samples
        .iter()
        .map(|s| {
            let chunks = s
                .tags()
                .iter()
                .zip(s.chunks())
                .map(|(tag, c)| if tag.as_str() == "IN" { Field::made("B-PP".to_owned(), c.span) } else { c.clone() })
                .collect();
            ChunkSample::new(s.tokens().to_vec(), s.tags().to_vec(), chunks, s.source()).unwrap()
        })
        .collect();
    let factory = turbo_train_components::ChunkerFactory::new("en");
    let mut config = turbo_train::TrainingConfiguration::default();
    let params = chunk::training_parameters();
    let stamp = turbo_train_components::Stamp::at(turbo_train_recipes::segmentation::TIMESTAMP_MS);
    let reference = turbo_train_components::chunk::train_chunker_model(&with_pp, &factory, &params, &mut config, stamp)
        .unwrap()
        .package
        .to_bytes();

    let run = Recipe::Chunk(CHUNKS)
        .gate(&candidate.bytes, "with-pp", &reference, &split.held_out, &held_out_name(), &digest.sha256)
        .unwrap();
    assert_eq!((run.record.result.candidate.score, run.record.result.reference.score), (100.0, 100.0));
    let selected = |counts: &[turbo_eval::model_gate::Counts]| counts.iter().map(|c| c.selected).sum::<u64>();
    let candidate_selected = selected(&run.record.units.iter().map(|u| u.candidate).collect::<Vec<_>>());
    let reference_selected = selected(&run.record.units.iter().map(|u| u.reference).collect::<Vec<_>>());
    assert_eq!(candidate_selected, reference_selected);

    // Kept, its `PP` chunks are predictions no held-out chunk matches.
    let held_out = CHUNKS.held_out(&split.held_out).unwrap();
    let chunker = turbo_phrase_chunker::PhraseChunker::new(
        turbo_phrase_chunker::ChunkerModel::open(turbo_maxent::ModelFile::open(reference).unwrap()).unwrap(),
    );
    let mut evaluator = ChunkerEvaluator::default();
    let mut pp = 0;
    for sample in held_out.samples() {
        let words: Vec<turbo_sequence::chunker::TokenTag<'_>> = sample
            .tokens()
            .iter()
            .zip(sample.tags())
            .map(|(t, tag)| turbo_sequence::chunker::TokenTag { token: t.as_str(), tag: tag.as_str() })
            .collect();
        let (mut outcomes, mut probs) = (Vec::new(), Vec::new());
        assert!(chunker.outcomes(&words, &mut outcomes, &mut probs));
        pp += outcomes.iter().filter(|o| **o == "B-PP").count() as u64;
        evaluator.evaluate(sample, &outcomes).unwrap();
    }
    assert!(pp > 0);
    assert_eq!(evaluator.fmeasure().selected(), reference_selected + pp);
    assert!(evaluator.fmeasure().f_measure() < 1.0);
    std::fs::remove_dir_all(&root).unwrap();
}

/// A corpus read without its chunk files has no chunk sample: the
/// chunker's reader refuses it, and so does the check made before
/// training.
#[test]
fn the_chunker_refuses_a_corpus_read_without_its_chunk_files() {
    let root = temp_dir("chunk-files");
    write_corpus(&root, 20, SENTENCES);
    let split = SPLIT.apply(MascCorpus::read(&root).unwrap(), &root).unwrap();
    let err = CHUNKS.held_out(&split.held_out).unwrap_err();
    assert!(matches!(err, Error::Masc(turbo_formats::masc::MascError::NoChunks)), "{err:?}");
    assert!(Recipe::Chunk(CHUNKS).check_data(&split, false, "reference", None).is_err());
    assert!(CHUNKS.train(&split.train).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}
