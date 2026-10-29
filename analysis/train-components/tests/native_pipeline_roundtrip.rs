//! End-to-end round-trip test for native Rust training -> archive deserialization
//! -> `Analyzer` pipeline producing offset-tracked `Document` layers.

use std::sync::Arc;
use turbo_analysis::{Analyzer, Document};
use turbo_analysis_layers::{ChunkTags, Lemmas, PhraseChunks, PosTags, Sentences, Tokens};
use turbo_formats::native::{ChunkReader, LemmaReader, SentenceReader, TokenReader, WordTagReader};
use turbo_lemmatizer::{LemmatizerModel, MaxentLemmatizer};
use turbo_phrase_chunker::{ChunkerModel, PhraseChunker};
use turbo_postag::{PosModel, PosTagFormat, PosTagger};
use turbo_sentdetect::me::{SentenceDetector, SentenceModel};
use turbo_tokenize::TokenizerOptions;
use turbo_tokenize::me::{TokenizerMe, TokenizerModel};
use turbo_train::{Key, TrainingConfiguration, TrainingParameters};
use turbo_train_components::{
    ChunkerFactory, LemmatizerFactory, PosTaggerFactory, SentenceDetectorFactory, Stamp, TokenizerFactory,
    train_chunker_model, train_lemmatizer_model, train_pos_model, train_sentence_model, train_tokenizer_model,
};

#[test]
fn native_trained_models_roundtrip_into_offset_tracked_pipeline() {
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 20);
    params.put(Key::CUTOFF, 0);

    // 1. Train SentenceDetectorME
    let sd_samples: Vec<_> = SentenceReader::new(
        "Dr. Smith cats barked loudly.\nDogs ran fast.\n\nDr. Jones cats purred softly.\nDogs slept well.\n",
    )
    .unwrap()
    .map(Result::unwrap)
    .collect();
    let sd_factory = SentenceDetectorFactory::new("en", true, None, None).unwrap();
    let trained_sd =
        train_sentence_model(&sd_samples, &sd_factory, &params, &mut TrainingConfiguration::default(), Stamp::at(0))
            .unwrap();
    let sd_model = Arc::new(SentenceModel::from_trained_archive(trained_sd.package.to_bytes()).unwrap());
    let sd_annotator = SentenceDetector::new(sd_model);

    // 2. Train TokenizerME
    let tok_samples: Vec<_> =
        TokenReader::new("Cats barked<SPLIT>.\nDogs ran<SPLIT>.\nCats purred<SPLIT>.\nDogs slept<SPLIT>.\n")
            .unwrap()
            .map(Result::unwrap)
            .collect();
    let tok_factory = TokenizerFactory::new("en", None, false, None).unwrap();
    let trained_tok =
        train_tokenizer_model(&tok_samples, &tok_factory, &params, &mut TrainingConfiguration::default(), Stamp::at(0))
            .unwrap();
    let tok_model = TokenizerModel::from_trained_archive(trained_tok.package.to_bytes()).unwrap();
    let tok_annotator = TokenizerMe::new(tok_model, TokenizerOptions::default());

    // 3. Train PosTagger
    let pos_samples: Vec<_> = WordTagReader::new("The_DT cats_NNS barked_VBD ._.\nThe_DT dogs_NNS ran_VBD ._.\n")
        .unwrap()
        .map(Result::unwrap)
        .collect();
    let pos_factory = PosTaggerFactory::new("en", None, None).unwrap();
    let trained_pos =
        train_pos_model(&pos_samples, &pos_factory, &params, &mut TrainingConfiguration::default(), Stamp::at(0))
            .unwrap();
    let pos_model = Arc::new(
        PosModel::from_trained_archive_with_format(trained_pos.package.to_bytes(), PosTagFormat::Penn).unwrap(),
    );
    let pos_annotator = PosTagger::new(pos_model);

    // 4. Train MaxentLemmatizer
    let lemma_samples: Vec<_> = LemmaReader::new(
        "The\tDT\tthe\ncats\tNNS\tcat\nbarked\tVBD\tbark\n.\t.\t.\n\n\
         The\tDT\tthe\ndogs\tNNS\tdog\nran\tVBD\trun\n.\t.\t.\n",
    )
    .unwrap()
    .map(Result::unwrap)
    .collect();
    let lemma_factory = LemmatizerFactory::new("en");
    let trained_lemma = train_lemmatizer_model(
        &lemma_samples,
        &lemma_factory,
        &params,
        &mut TrainingConfiguration::default(),
        Stamp::at(0),
    )
    .unwrap();
    let lemma_model = Arc::new(LemmatizerModel::from_trained_archive(trained_lemma.package.to_bytes()).unwrap());
    let lemma_annotator = MaxentLemmatizer::new(lemma_model);

    // 5. Train PhraseChunker
    let chunk_samples: Vec<_> = ChunkReader::new(
        "The DT B-NP\ncats NNS I-NP\nbarked VBD B-VP\n. . O\n\n\
         The DT B-NP\ndogs NNS I-NP\nran VBD B-VP\n. . O\n",
    )
    .unwrap()
    .map(Result::unwrap)
    .collect();
    let chunk_factory = ChunkerFactory::new("en");
    let trained_chunk = train_chunker_model(
        &chunk_samples,
        &chunk_factory,
        &params,
        &mut TrainingConfiguration::default(),
        Stamp::at(0),
    )
    .unwrap();
    let chunk_model = Arc::new(ChunkerModel::from_trained_archive(trained_chunk.package.to_bytes()).unwrap());
    let chunk_annotator = PhraseChunker::new(chunk_model);

    // Build and run full Analyzer pipeline on a Document
    let analyzer = Analyzer::builder()
        .add(sd_annotator)
        .add(tok_annotator)
        .add(pos_annotator)
        .add(lemma_annotator)
        .add(chunk_annotator)
        .build()
        .unwrap();

    let text = "The cats barked. The dogs ran.";
    let mut doc = Document::new(text);
    assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);

    // Verify bidirectional offset-tracked layers
    let sentences = doc.get::<Sentences>().expect("Sentences layer present");
    assert_eq!(sentences.spans.len(), 2);
    let (s_start, s_end) = sentences.spans.columns();
    assert_eq!(&text[s_start[0] as usize..s_end[0] as usize], "The cats barked.");
    assert_eq!(&text[s_start[1] as usize..s_end[1] as usize], "The dogs ran.");

    let tokens = doc.get::<Tokens>().expect("Tokens layer present");
    let (t_start, t_end) = tokens.spans.columns();
    let token_texts: Vec<&str> =
        (0..tokens.spans.len()).map(|i| &text[t_start[i] as usize..t_end[i] as usize]).collect();
    assert_eq!(token_texts, ["The", "cats", "barked", ".", "The", "dogs", "ran", "."]);

    let pos_tags = doc.get::<PosTags>().expect("PosTags layer present");
    let pos_values: Vec<&str> = pos_tags.tags.iter().map(|&t| pos_tags.tagset.label(t)).collect();
    assert_eq!(pos_values, ["DT", "NNS", "VBD", ".", "DT", "NNS", "VBD", "."]);

    let lemmas = doc.get::<Lemmas>().expect("Lemmas layer present");
    let lemma_values: Vec<&str> = lemmas.lemma.iter().collect();
    assert_eq!(lemma_values, ["the", "cat", "bark", ".", "the", "dog", "run", "."]);

    let chunk_tags = doc.get::<ChunkTags>().expect("ChunkTags layer present");
    assert_eq!(chunk_tags.tags.len(), 8);

    let phrase_chunks = doc.get::<PhraseChunks>().expect("PhraseChunks layer present");
    let (c_start, c_end) = phrase_chunks.spans.columns();
    let phrases: Vec<(&str, &str)> = (0..phrase_chunks.spans.len())
        .map(|i| (phrase_chunks.kinds.label(phrase_chunks.kind[i]), &text[c_start[i] as usize..c_end[i] as usize]))
        .collect();
    assert_eq!(phrases, [("NP", "The cats"), ("VP", "barked"), ("NP", "The dogs"), ("VP", "ran")]);
}
