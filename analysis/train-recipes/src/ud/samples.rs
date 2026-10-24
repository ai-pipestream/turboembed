use turbo_eval::model_gate::{Counts, HeldOutSet, ModelGateRecord, ModelId, PairedUnit, Scoring, TagScoring, Task};
use turbo_formats::FormatError;
use turbo_formats::conllu::{ConlluReader, ConlluSentenceReader, ConlluTokenReader, Sentence, WordLine};
use turbo_formats::sample::{LemmaSample, PosSample};
use turbo_train::TrainingConfiguration;
use turbo_train_components::ud::{self as published, Component};
use turbo_train_components::{
    LemmaEventStream, LemmatizerFactory, PosEventStream, PosTaggerFactory, SdEventStream, SentenceDetectorFactory,
    Stamp, TokSpanEventStream, TokenizerFactory, lemmatize, postag, sentdetect, tokenize,
};
use turbo_zoo::Sha256;

use super::lemma::{ReferenceClasses, lemma_samples_sha256, lemma_scores, lemma_tokens, score_lemmas};
use super::{
    MAX_TOKENS_LEFT_OUT, ReadSamples, SENTENCES_PER_SAMPLE, TAG_FORMAT, TAGSET, UD_RECIPES, UdRecipe, UdSamples,
    component_name,
};
use crate::card::{HeldOutCard, TagCount, TokenCoverage};
use crate::error::{Error, LeftOut, Result};
use crate::postag::{HeldOutTags, PosRecipe, Scored};
use crate::segmentation::{self, HeldOut, SegmentationRecipe, TIMESTAMP_MS, WHITESPACE};
use crate::treebank::{RELEASE, TreebankText};
use crate::{GateRun, Trained, check_skipped};

/// The words of a sentence the tagger and the lemmatizer read: its word
/// lines without the empty nodes.
fn words<'s, 'a>(sentence: &'s Sentence<'a>) -> impl Iterator<Item = &'s WordLine<'a>> {
    sentence.word_lines.iter().filter(|l| !l.id.as_str().contains('.'))
}

/// Reads the part-of-speech samples of a CoNLL-U text: one per sentence,
/// its words' forms and universal tags, empty nodes left out. A sentence
/// with no word is skipped and counted.
///
/// # Errors
///
/// A sentence the CoNLL-U reader refuses.
pub fn pos_samples(conllu: &str) -> std::result::Result<(Vec<PosSample<'_>>, usize), FormatError> {
    let mut samples = Vec::new();
    let mut skipped = 0;
    for sentence in ConlluReader::new(conllu)? {
        let sentence = sentence?;
        let tokens: Vec<_> = words(&sentence).map(|l| l.form.clone()).collect();
        if tokens.is_empty() {
            skipped += 1;
            continue;
        }
        let tags = words(&sentence).map(|l| l.tag(TAGSET).clone()).collect();
        samples.push(PosSample::new(tokens, tags, sentence.source)?);
    }
    Ok((samples, skipped))
}

/// Reads the lemma samples of a CoNLL-U text as [`pos_samples`] reads the
/// part-of-speech samples, with each word's lemma.
///
/// # Errors
///
/// A sentence the CoNLL-U reader refuses.
pub fn lemma_samples(conllu: &str) -> std::result::Result<(Vec<LemmaSample<'_>>, usize), FormatError> {
    let mut samples = Vec::new();
    let mut skipped = 0;
    for sentence in ConlluReader::new(conllu)? {
        let sentence = sentence?;
        let tokens: Vec<_> = words(&sentence).map(|l| l.form.clone()).collect();
        if tokens.is_empty() {
            skipped += 1;
            continue;
        }
        let tags = words(&sentence).map(|l| l.tag(TAGSET).clone()).collect();
        let lemmas = words(&sentence).map(|l| l.lemma.clone()).collect();
        samples.push(LemmaSample::new(tokens, tags, lemmas, sentence.source)?);
    }
    Ok((samples, skipped))
}

impl UdRecipe {
    /// The recipe of [`UD_RECIPES`] that trains the model `id`.
    pub fn trains(id: &str) -> Option<UdRecipe> {
        UD_RECIPES.into_iter().find(|r| r.model_id == id)
    }

    /// The gate's task.
    pub fn task(&self) -> Task {
        match self.component {
            Component::Sentence => Task::SentenceDetector,
            Component::Tokens => Task::Tokenizer,
            Component::Pos => Task::PartOfSpeechTagger,
            Component::Lemmas => Task::Lemmatizer,
        }
    }

    /// The component the zoo lists the trained model under.
    pub fn zoo_component(&self) -> turbo_zoo::Component {
        match self.component {
            Component::Sentence => turbo_zoo::Component::SentenceDetector,
            Component::Tokens => turbo_zoo::Component::Tokenizer,
            Component::Pos => turbo_zoo::Component::Pos,
            Component::Lemmas => turbo_zoo::Component::Lemmatizer,
        }
    }

    /// For a sentence detector or a tokenizer, the MASC recipes' gate for
    /// it: a [`SegmentationRecipe`] of the model and its reference.
    pub fn segmentation(&self) -> Option<SegmentationRecipe> {
        let component = match self.component {
            Component::Sentence => segmentation::Component::Sentence,
            Component::Tokens => segmentation::Component::Tokens,
            Component::Pos | Component::Lemmas => return None,
        };
        Some(SegmentationRecipe { component, model_id: self.model_id, reference_id: self.reference.id() })
    }

    /// For a tagger, the MASC tagger's gate for it: a [`PosRecipe`] of the
    /// model and its reference, the reference asked for [`TAG_FORMAT`], no
    /// mapping, and [`MAX_TOKENS_LEFT_OUT`].
    pub fn tagger(&self) -> Option<PosRecipe> {
        (self.component == Component::Pos).then_some(PosRecipe {
            model_id: self.model_id,
            reference: self.reference,
            reference_format: TAG_FORMAT,
            reference_mappings: &[],
            max_left_out: MAX_TOKENS_LEFT_OUT,
        })
    }

    /// The name of the held-out set in the records.
    pub fn held_out_name(&self) -> String {
        format!("{} {RELEASE} test ({})", self.treebank.repository, self.treebank.test.name)
    }

    /// Reads the recipe's samples from a CoNLL-U text, as the trainer
    /// tool reads them (with the change the module documentation gives).
    ///
    /// # Errors
    ///
    /// A sentence the reader refuses ([`Error::Conllu`], naming `file`),
    /// or no sample ([`Error::EmptySplit`]).
    pub fn samples<'c>(&self, conllu: &'c str, file: &'static str) -> Result<ReadSamples<'c>> {
        let conllu_error =
            |error: FormatError| Error::Conllu { treebank: self.treebank.repository, file, error: Box::new(error) };
        let (samples, skipped) = match self.component {
            Component::Sentence => (
                UdSamples::Segments(HeldOut::Sentences(
                    ConlluSentenceReader::new(conllu, SENTENCES_PER_SAMPLE)
                        .map_err(conllu_error)?
                        .collect::<std::result::Result<_, _>>()
                        .map_err(conllu_error)?,
                )),
                0,
            ),
            Component::Tokens => (
                UdSamples::Segments(HeldOut::Tokens {
                    samples: ConlluTokenReader::new(conllu, WHITESPACE)
                        .map_err(conllu_error)?
                        .collect::<std::result::Result<_, _>>()
                        .map_err(conllu_error)?,
                    skipped: 0,
                }),
                0,
            ),
            Component::Pos => {
                let (s, skipped) = pos_samples(conllu).map_err(conllu_error)?;
                (UdSamples::Tags(s), skipped)
            }
            Component::Lemmas => {
                let (s, skipped) = lemma_samples(conllu).map_err(conllu_error)?;
                (UdSamples::Lemmas(s), skipped)
            }
        };
        if samples.is_empty() {
            return Err(Error::EmptySplit { what: "sample of the treebank file" });
        }
        Ok(ReadSamples { samples, skipped })
    }

    /// The held-out samples of the `test` file, checked as
    /// [`run_ud`](super::run_ud) checks them: at most
    /// [`crate::MAX_SKIPPED_SENTENCES`] skipped.
    ///
    /// # Errors
    ///
    /// What [`UdRecipe::samples`] fails on, or [`Error::TooManyLeftOut`].
    pub fn held_out<'c>(&self, text: &'c TreebankText) -> Result<ReadSamples<'c>> {
        let held_out = self.samples(&text.test, self.treebank.test.name)?;
        check_skipped(LeftOut::HeldOutSentences, held_out.skipped, held_out.samples.len())?;
        Ok(held_out)
    }

    /// Reads the training and held-out samples of `text` without training
    /// or scoring, and checks what [`run_ud`](super::run_ud) checks of
    /// them: each file has a sample, the reader skips at most
    /// [`crate::MAX_SKIPPED_SENTENCES`] of the sentences of either, that
    /// `reference` is a valid model of the component, and for a tagger or a
    /// lemmatizer, that the reference can give the tag, or the lemma the
    /// gold lemma class gives, of all but [`MAX_TOKENS_LEFT_OUT`] of the
    /// held-out tokens ([`ReferenceClasses::scores`]). The binary runs it
    /// for every recipe before it trains any, as it runs
    /// [`crate::Recipe::check_data`] for the MASC recipes.
    ///
    /// # Errors
    ///
    /// What reading the samples fails on, [`Error::TooManyLeftOut`],
    /// [`Error::TagsOutsideReference`], [`Error::ClassesOutsideReference`],
    /// or a reference that is not a model of the component.
    pub fn check_data(&self, text: &TreebankText, reference: &[u8]) -> Result<()> {
        let train = self.samples(&text.train, self.treebank.train.name)?;
        check_skipped(LeftOut::TrainingSentences, train.skipped, train.samples.len())?;
        let held_out = self.held_out(text)?;
        match (self.segmentation(), self.tagger(), held_out.samples) {
            (Some(seg), _, UdSamples::Segments(samples)) => {
                let empty = match samples {
                    HeldOut::Sentences(_) => HeldOut::Sentences(Vec::new()),
                    HeldOut::Tokens { skipped, .. } => HeldOut::Tokens { samples: Vec::new(), skipped },
                };
                seg.score(reference, &empty)?;
            }
            (_, Some(tagger), UdSamples::Tags(samples)) => {
                let tags = tagger.reference_tags(reference)?;
                tagger.check_coverage(self.reference.id(), &tags, &HeldOutTags::new(samples, held_out.skipped))?;
            }
            (None, None, UdSamples::Lemmas(samples)) => {
                self.check_classes(&ReferenceClasses::of_model(reference)?, &samples)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// The held-out tokens whose gold lemma class gives their lemma and no
    /// class of `classes`, a reference's, does, by tag
    /// ([`ReferenceClasses::left_out`]).
    ///
    /// # Errors
    ///
    /// More of them than [`MAX_TOKENS_LEFT_OUT`] allows
    /// ([`Error::ClassesOutsideReference`]).
    pub fn check_classes(&self, classes: &ReferenceClasses, samples: &[LemmaSample<'_>]) -> Result<Vec<TagCount>> {
        let left_out = classes.left_out(samples);
        let tokens = lemma_tokens(samples);
        if MAX_TOKENS_LEFT_OUT.allows(left_out.iter().map(|t| t.tokens).sum(), tokens) {
            Ok(left_out)
        } else {
            Err(Error::ClassesOutsideReference {
                reference: self.reference.id().to_owned(),
                left_out,
                of: tokens,
                limit: MAX_TOKENS_LEFT_OUT,
            })
        }
    }

    /// Trains the model from the treebank's `train` file.
    ///
    /// # Errors
    ///
    /// What reading the samples fails on, more skipped sentences than
    /// [`crate::MAX_SKIPPED_SENTENCES`] allows ([`Error::TooManyLeftOut`]),
    /// or what training fails on.
    pub fn train(&self, train: &str) -> Result<Trained> {
        self.train_with_lemma_format(train, false)
    }

    pub(super) fn train_with_lemma_format(&self, train: &str, native_lemmas: bool) -> Result<Trained> {
        let read = self.samples(train, self.treebank.train.name)?;
        check_skipped(LeftOut::TrainingSentences, read.skipped, read.samples.len())?;
        let language = self.treebank.language;
        let stamp = Stamp::at(TIMESTAMP_MS);
        let params =
            if native_lemmas { super::native::training_parameters() } else { published::training_parameters() };
        let mut config = TrainingConfiguration::default();
        let mut events = Vec::new();
        let (bytes, report) = match &read.samples {
            UdSamples::Segments(HeldOut::Sentences(samples)) => {
                let factory = SentenceDetectorFactory::new(language, true, None, None)?;
                let mut stream = SdEventStream::new(&factory, WHITESPACE);
                samples.iter().for_each(|s| stream.events(s, &mut events));
                let t = sentdetect::train_sentence_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
            UdSamples::Segments(HeldOut::Tokens { samples, .. }) => {
                let factory = TokenizerFactory::new(language, None, false, None)?;
                let mut stream = TokSpanEventStream::new(&factory, WHITESPACE);
                for s in samples {
                    stream.events(s, &mut events)?;
                }
                let t = tokenize::train_tokenizer_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
            UdSamples::Tags(samples) => {
                let factory = PosTaggerFactory::new(language, None, None)?;
                let mut stream = PosEventStream::new(&factory)?;
                samples.iter().for_each(|s| stream.events(s, &mut events));
                let t = postag::train_pos_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
            UdSamples::Lemmas(samples) => {
                let factory = if native_lemmas {
                    LemmatizerFactory::new(language)
                } else {
                    // Existing release ids keep their published class format.
                    LemmatizerFactory::opennlp(language)
                };
                let mut stream = LemmaEventStream::new(&factory);
                samples.iter().for_each(|s| stream.events(s, &mut events));
                let t = lemmatize::train_lemmatizer_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
        };
        Ok(Trained {
            bytes,
            samples: read.samples.len(),
            skipped_sentences: read.skipped,
            events: events.len(),
            event_hash: report.event_hash.expect("trained from events"),
            names: None,
            chunks: None,
        })
    }

    /// Each held-out sample's counts for the model in `bytes`, scored with
    /// OpenNLP's evaluator for the component: for a tagger, as the trained
    /// model is scored ([`Scored::Candidate`]) over the tokens whose tag the
    /// model in `bytes` can give, and for a lemmatizer, over the tokens it
    /// scores as the reference ([`ReferenceClasses::scores`]).
    ///
    /// # Errors
    ///
    /// A model file that is not a model of the component, samples of
    /// another component, or a prediction the evaluator throws on.
    pub fn score(&self, bytes: &[u8], held_out: &ReadSamples<'_>) -> Result<Vec<Counts>> {
        match (self.segmentation(), self.tagger(), &held_out.samples) {
            (Some(s), _, UdSamples::Segments(samples)) => s.score(bytes, samples),
            (_, Some(t), UdSamples::Tags(samples)) => {
                let tags = HeldOutTags::new(samples.clone(), held_out.skipped);
                t.score(bytes, Scored::Candidate, &tags, &t.reference_tags(bytes)?)
            }
            (None, None, UdSamples::Lemmas(samples)) => {
                score_lemmas(bytes, samples, &ReferenceClasses::of_model(bytes)?)
            }
            (_, _, samples) => Err(self.mismatch(samples)),
        }
    }

    fn mismatch(&self, samples: &UdSamples<'_>) -> Error {
        Error::ComponentMismatch { recipe: component_name(self.component), samples: samples.kind() }
    }

    /// Scores the trained model and the reference (its file's bytes in
    /// `reference`) on the held-out samples and runs the gate, as the
    /// MASC recipes' gates do (the module documentation says how), with
    /// the `test` file's name and SHA-256 naming the held-out set.
    ///
    /// # Errors
    ///
    /// A tagger's reference that cannot give the tags of more than
    /// [`MAX_TOKENS_LEFT_OUT`] of the held-out tokens
    /// ([`Error::TagsOutsideReference`]), a lemmatizer's reference that
    /// cannot give the lemmas of more than that
    /// ([`Error::ClassesOutsideReference`]), samples of another component,
    /// what scoring fails on, or the gate's errors.
    pub fn gate(&self, trained: &[u8], reference: &[u8], held_out: ReadSamples<'_>) -> Result<GateRun> {
        let reference_id = self.reference.id();
        let (name, corpus_sha256) = (self.held_out_name(), self.treebank.test.sha256);
        match (self.segmentation(), self.tagger(), held_out.samples) {
            (Some(s), _, UdSamples::Segments(samples)) => {
                s.gate(trained, reference_id, reference, &samples, &name, corpus_sha256)
            }
            (_, Some(t), UdSamples::Tags(samples)) => {
                let samples = HeldOutTags::new(samples, held_out.skipped);
                t.gate(trained, reference_id, reference, &samples, &name, corpus_sha256)
            }
            (None, None, UdSamples::Lemmas(samples)) => {
                let classes = ReferenceClasses::of_model(reference)?;
                let left_out = self.check_classes(&classes, &samples)?;
                let candidate = score_lemmas(trained, &samples, &classes)?;
                let baseline = lemma_scores(reference, &samples, &classes)?;
                let units = candidate
                    .into_iter()
                    .zip(baseline.units)
                    .map(|(candidate, reference)| PairedUnit { candidate, reference })
                    .collect();
                let tokens = lemma_tokens(&samples);
                let set = HeldOutSet {
                    name,
                    corpus_sha256: corpus_sha256.to_owned(),
                    samples_sha256: lemma_samples_sha256(&samples, &classes),
                    skipped_sentences: held_out.skipped as u64,
                    scoring: Scoring::Tags(TagScoring {
                        tokens,
                        left_out: left_out.iter().map(|t| t.tokens).sum(),
                        reference_mappings: Vec::new(),
                    }),
                };
                let id = |id: &str, bytes: &[u8]| ModelId { id: id.to_owned(), sha256: Sha256::of(bytes).to_string() };
                let record = ModelGateRecord::new(
                    self.task(),
                    set,
                    id(self.model_id, trained),
                    id(reference_id, reference),
                    units,
                )
                .map_err(Error::Gate)?;
                let held_out = HeldOutCard {
                    samples: samples.len(),
                    skipped_sentences: held_out.skipped,
                    tokens: Some(TokenCoverage {
                        tokens,
                        reference_mappings: Vec::new(),
                        left_out,
                        reference_mismatches: baseline.mismatches,
                    }),
                    names: None,
                    chunks: None,
                };
                Ok(GateRun { record, held_out })
            }
            (_, _, samples) => Err(self.mismatch(&samples)),
        }
    }
}
