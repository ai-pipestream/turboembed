use std::sync::Arc;

use turbo_tokenize::detok::{Detokenizer, DetokenizerDictionary};

use super::{
    CONLLU_FORMAT, DependencySamples, Format, LemmaSamples, NameSamples, NativeFormat, OPENNLP_FORMAT, PosSamples,
    ReadOptions, Samples, SentenceSamples, TokenSamples,
};
use crate::conll::{
    CONLL02_FORMAT, CONLL03_FORMAT, CONLLX_FORMAT, Conll02Language, Conll02Reader, Conll03Language, Conll03Reader,
    ConllXPosReader, EntityTypes,
};
use crate::conllu::{
    ConlluDependencyReader, ConlluLemmaReader, ConlluPosReader, ConlluSentenceReader, ConlluTokenReader, Tagset,
};
use crate::convert::{ChunkSize, ToSentenceSamples, ToTokenSamples};
use crate::error::FormatError;
use crate::native::{NameReader, WordTagReader};

/// Which samples a conversion format reads: the name format (OpenNLP's
/// `namefinder` conversion formats) or the word and tag format (`pos`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConvertFrom {
    Names,
    Pos,
}

impl ConvertFrom {
    /// The name OpenNLP registers the conversion format under.
    pub fn format_name(self) -> &'static str {
        match self {
            ConvertFrom::Names => "namefinder",
            ConvertFrom::Pos => "pos",
        }
    }
}

/// Token samples converted from name or word and tag samples with a
/// detokenizer dictionary: `NameToTokenSampleStreamFactory` and
/// `POSToTokenSampleStreamFactory`.
#[derive(Clone, Debug)]
pub struct ConvertedTokens {
    pub from: ConvertFrom,
    pub dictionary: Arc<DetokenizerDictionary>,
}

impl Format<TokenSamples> for ConvertedTokens {
    fn name(&self) -> &str {
        self.from.format_name()
    }

    fn open<'a>(&'a self, source: &'a str, options: ReadOptions) -> Result<Samples<'a, TokenSamples>, FormatError> {
        let detokenizer = Detokenizer::new(&self.dictionary);
        Ok(match self.from {
            ConvertFrom::Names => {
                Box::new(ToTokenSamples::new(NameReader::new(source, options.whitespace)?, detokenizer))
            }
            ConvertFrom::Pos => {
                Box::new(ToTokenSamples::new(WordTagReader::new(source, options.whitespace)?, detokenizer))
            }
        })
    }
}

/// Sentence samples converted from name or word and tag samples with a
/// detokenizer dictionary: `NameToSentenceSampleStreamFactory` and
/// `POSToSentenceSampleStreamFactory`, which use a chunk size of 30
/// ([`ChunkSize::COMMAND_LINE`]).
#[derive(Clone, Debug)]
pub struct ConvertedSentences {
    pub from: ConvertFrom,
    pub dictionary: Arc<DetokenizerDictionary>,
    pub chunk: ChunkSize,
}

impl Format<SentenceSamples> for ConvertedSentences {
    fn name(&self) -> &str {
        self.from.format_name()
    }

    fn open<'a>(&'a self, source: &'a str, options: ReadOptions) -> Result<Samples<'a, SentenceSamples>, FormatError> {
        let detokenizer = Detokenizer::new(&self.dictionary);
        Ok(match self.from {
            ConvertFrom::Names => {
                Box::new(ToSentenceSamples::new(NameReader::new(source, options.whitespace)?, detokenizer, self.chunk))
            }
            ConvertFrom::Pos => Box::new(ToSentenceSamples::new(
                WordTagReader::new(source, options.whitespace)?,
                detokenizer,
                self.chunk,
            )),
        })
    }
}

/// Part-of-speech samples from CoNLL-U (`ConlluPOSSampleStreamFactory`,
/// whose tag set defaults to [`Tagset::U`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConlluPos {
    pub tagset: Tagset,
}

impl Format<PosSamples> for ConlluPos {
    fn name(&self) -> &str {
        CONLLU_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, PosSamples>, FormatError> {
        Ok(Box::new(ConlluPosReader::new(source, self.tagset)?))
    }
}

/// Lemma samples from CoNLL-U (`ConlluLemmaSampleStreamFactory`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConlluLemmas {
    pub tagset: Tagset,
}

impl Format<LemmaSamples> for ConlluLemmas {
    fn name(&self) -> &str {
        CONLLU_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, LemmaSamples>, FormatError> {
        Ok(Box::new(ConlluLemmaReader::new(source, self.tagset)?))
    }
}

/// Sentence samples from CoNLL-U (`ConlluSentenceSampleStreamFactory`,
/// whose `sentencesPerSample` has no default).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConlluSentences {
    pub per_sample: usize,
}

impl Format<SentenceSamples> for ConlluSentences {
    fn name(&self) -> &str {
        CONLLU_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, SentenceSamples>, FormatError> {
        Ok(Box::new(ConlluSentenceReader::new(source, self.per_sample)?))
    }
}

/// Token samples from CoNLL-U (`ConlluTokenSampleStreamFactory`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConlluTokens;

impl Format<TokenSamples> for ConlluTokens {
    fn name(&self) -> &str {
        CONLLU_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, options: ReadOptions) -> Result<Samples<'a, TokenSamples>, FormatError> {
        Ok(Box::new(ConlluTokenReader::new(source, options.whitespace)?))
    }
}

/// Dependency samples from CoNLL-U (`ConlluDependencySampleStreamFactory`,
/// whose tag set defaults to [`Tagset::U`]), registered as [`CONLLU_FORMAT`].
/// Blank lines and relations are tested in [`ReadOptions::whitespace`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConlluDependencies {
    pub tagset: Tagset,
}

impl Format<DependencySamples> for ConlluDependencies {
    fn name(&self) -> &str {
        CONLLU_FORMAT
    }

    fn open<'a>(
        &'a self,
        source: &'a str,
        options: ReadOptions,
    ) -> Result<Samples<'a, DependencySamples>, FormatError> {
        Ok(Box::new(ConlluDependencyReader::new(source, self.tagset, options.whitespace)?))
    }
}

/// The same format as [`ConlluDependencies`], registered as
/// [`OPENNLP_FORMAT`], the default format of dependency samples.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NativeDependencies {
    pub tagset: Tagset,
}

impl Format<DependencySamples> for NativeDependencies {
    fn name(&self) -> &str {
        OPENNLP_FORMAT
    }

    fn open<'a>(
        &'a self,
        source: &'a str,
        options: ReadOptions,
    ) -> Result<Samples<'a, DependencySamples>, FormatError> {
        Ok(Box::new(ConlluDependencyReader::new(source, self.tagset, options.whitespace)?))
    }
}

impl NativeFormat for DependencySamples {
    fn native() -> Box<dyn Format<Self>> {
        Box::new(NativeDependencies::default())
    }
}

/// Name samples from CoNLL-2002 (`Conll02NameSampleStreamFactory`, whose
/// `lang` and `types` parameters have no default).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Conll02Names {
    pub language: Conll02Language,
    pub types: EntityTypes,
}

impl Format<NameSamples> for Conll02Names {
    fn name(&self) -> &str {
        CONLL02_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, NameSamples>, FormatError> {
        Ok(Box::new(Conll02Reader::new(source, self.language, self.types)?))
    }
}

/// Name samples from CoNLL-2003 (`Conll03NameSampleStreamFactory`, whose
/// `lang` and `types` parameters have no default).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Conll03Names {
    pub language: Conll03Language,
    pub types: EntityTypes,
}

impl Format<NameSamples> for Conll03Names {
    fn name(&self) -> &str {
        CONLL03_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, NameSamples>, FormatError> {
        Ok(Box::new(Conll03Reader::new(source, self.language, self.types)?))
    }
}

/// Part-of-speech samples from CoNLL-X (`ConllXPOSSampleStreamFactory`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConllXPos;

impl Format<PosSamples> for ConllXPos {
    fn name(&self) -> &str {
        CONLLX_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, PosSamples>, FormatError> {
        Ok(Box::new(ConllXPosReader::new(source)?))
    }
}

/// Token samples converted from CoNLL-X part-of-speech samples with a
/// detokenizer dictionary (`ConllXTokenSampleStreamFactory`).
#[derive(Clone, Debug)]
pub struct ConllXTokens {
    pub dictionary: Arc<DetokenizerDictionary>,
}

impl Format<TokenSamples> for ConllXTokens {
    fn name(&self) -> &str {
        CONLLX_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, TokenSamples>, FormatError> {
        Ok(Box::new(ToTokenSamples::new(ConllXPosReader::new(source)?, Detokenizer::new(&self.dictionary))))
    }
}

/// Sentence samples converted from CoNLL-X part-of-speech samples with a
/// detokenizer dictionary (`ConllXSentenceSampleStreamFactory`, which
/// uses a chunk size of 30, [`ChunkSize::COMMAND_LINE`]).
#[derive(Clone, Debug)]
pub struct ConllXSentences {
    pub dictionary: Arc<DetokenizerDictionary>,
    pub chunk: ChunkSize,
}

impl Format<SentenceSamples> for ConllXSentences {
    fn name(&self) -> &str {
        CONLLX_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, SentenceSamples>, FormatError> {
        let detokenizer = Detokenizer::new(&self.dictionary);
        Ok(Box::new(ToSentenceSamples::new(ConllXPosReader::new(source)?, detokenizer, self.chunk)))
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::super::{NativeTokens, OpenError, Registry};
    use super::*;

    #[test]
    fn native_formats_open_by_name() {
        let r = Registry::<PosSamples>::default();
        assert_eq!(r.names().collect::<Vec<_>>(), [OPENNLP_FORMAT]);
        let samples: Vec<_> = r.open("opennlp", "a_X b_Y\nc_Z", ReadOptions::default()).unwrap().collect();
        assert_eq!(samples.len(), 2);
        let Err(err) = r.open("conllu", "", ReadOptions::default()) else { panic!("expected unknown format") };
        assert!(err.source().is_none());
        let wrapped = OpenError::from(FormatError::NoTokens);
        assert!(wrapped.source().is_some());
    }

    #[test]
    fn conllu_formats_register_by_name() {
        let mut r = Registry::<LemmaSamples>::with_native();
        r.register(ConlluLemmas { tagset: Tagset::X }).unwrap();
        let src = "1\ta\tla\tX\tx\t_\t_\t_\t_\t_\n";
        let samples: Vec<_> = r.open(CONLLU_FORMAT, src, ReadOptions::default()).unwrap().map(Result::unwrap).collect();
        assert_eq!(samples[0].tags()[0].as_str(), "x");
    }

    #[test]
    fn a_duplicate_keeps_the_first() {
        let mut r = Registry::<TokenSamples>::with_native();
        let dictionary = Arc::new(DetokenizerDictionary::new());
        r.register(ConvertedTokens { from: ConvertFrom::Names, dictionary: dictionary.clone() }).unwrap();
        r.register(ConvertedTokens { from: ConvertFrom::Pos, dictionary }).unwrap();
        let e = r.register(NativeTokens).unwrap_err();
        assert_eq!(e.name, "opennlp");
        assert_eq!(r.names().collect::<Vec<_>>(), ["opennlp", "namefinder", "pos"]);
        let samples: Vec<_> =
            r.open("namefinder", "<START:x> a <END> b", ReadOptions::default()).unwrap().map(Result::unwrap).collect();
        assert_eq!(samples[0].text(), "a b");
    }

    #[test]
    fn dependency_formats_register_as_java_registers_them() {
        let mut r = Registry::<DependencySamples>::with_native();
        r.register(ConlluDependencies { tagset: Tagset::from_parameter("x").unwrap() }).unwrap();
        assert_eq!(r.names().collect::<Vec<_>>(), [OPENNLP_FORMAT, CONLLU_FORMAT]);
        let src = "1\ta\t_\tU\tX\t_\t0\troot\t_\t_\n";
        let read = |name: &str| -> Vec<String> {
            r.open(name, src, ReadOptions::default()).unwrap().map(|s| s.unwrap().to_string()).collect()
        };
        assert_eq!(read(OPENNLP_FORMAT), ["1\ta\tU\t0\troot\n"]);
        assert_eq!(read(CONLLU_FORMAT), ["1\ta\tX\t0\troot\n"]);
        assert_eq!(Tagset::from_parameter("U"), None);
    }

    #[test]
    fn conll_formats_register_by_name() {
        let mut names = Registry::<NameSamples>::with_native();
        names.register(Conll02Names { language: Conll02Language::Dutch, types: EntityTypes::ALL }).unwrap();
        names.register(Conll03Names { language: Conll03Language::English, types: EntityTypes::ALL }).unwrap();
        assert_eq!(names.names().collect::<Vec<_>>(), [OPENNLP_FORMAT, CONLL02_FORMAT, CONLL03_FORMAT]);
        let read = |format: &str, src: &str| -> Vec<String> {
            names.open(format, src, ReadOptions::default()).unwrap().map(|s| s.unwrap().to_string()).collect()
        };
        assert_eq!(read(CONLL02_FORMAT, "Gent N B-LOC\nis V O\n"), ["<START:location> Gent <END> is"]);
        assert_eq!(read(CONLL03_FORMAT, "Gent NNP I-NP I-LOC\nis VBZ I-VP O\n"), ["<START:location> Gent <END> is"]);

        let mut tokens = Registry::<TokenSamples>::with_native();
        tokens.register(ConllXTokens { dictionary: Arc::new(DetokenizerDictionary::new()) }).unwrap();
        let src = "1\tHet\thet\tArt\tDT\n2\tregent\tregenen\tV\tVBZ\n";
        let samples: Vec<_> =
            tokens.open(CONLLX_FORMAT, src, ReadOptions::default()).unwrap().map(Result::unwrap).collect();
        assert_eq!(samples[0].text(), "Het regent");
    }
}
