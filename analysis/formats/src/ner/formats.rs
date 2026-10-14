//! The registry formats of the named entity corpora.

use super::{
    BIONLP_FORMAT, BioNlpReader, BioNlpTypes, EVALITA_FORMAT, EvalitaLanguage, EvalitaReader, EvalitaTypes,
    GERMEVAL_FORMAT, GermEvalLayer, GermEvalReader, GermEvalTypes, MUC6_FORMAT, MucReader, RuleTokenizer, TokenSpans,
};
use crate::error::FormatError;
use crate::registry::{Format, NameSamples, ReadOptions, Samples};

/// Name samples from GermEval 2014 (`GermEval2014NameSampleStreamFactory`),
/// named [`GERMEVAL_FORMAT`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GermEvalNames {
    pub types: GermEvalTypes,
    pub layer: GermEvalLayer,
}

impl Format<NameSamples> for GermEvalNames {
    fn name(&self) -> &str {
        GERMEVAL_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, NameSamples>, FormatError> {
        Ok(Box::new(GermEvalReader::new(source, self.types, self.layer)?))
    }
}

/// Name samples from BioNLP 2004 (`BioNLP2004NameSampleStreamFactory`),
/// named [`BIONLP_FORMAT`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct BioNlpNames {
    pub types: BioNlpTypes,
}

impl Format<NameSamples> for BioNlpNames {
    fn name(&self) -> &str {
        BIONLP_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, NameSamples>, FormatError> {
        Ok(Box::new(BioNlpReader::new(source, self.types)?))
    }
}

/// Name samples from Evalita (`EvalitaNameSampleStreamFactory`), named
/// [`EVALITA_FORMAT`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EvalitaNames {
    pub language: EvalitaLanguage,
    pub types: EvalitaTypes,
}

impl Format<NameSamples> for EvalitaNames {
    fn name(&self) -> &str {
        EVALITA_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, NameSamples>, FormatError> {
        Ok(Box::new(EvalitaReader::new(source, self.language, self.types)?))
    }
}

/// Name samples from a MUC-6 document file (`Muc6NameSampleStreamFactory`),
/// named [`MUC6_FORMAT`]. The factory reads every `.sgm` file of a
/// directory, each a text of this format, and splits them with a maxent
/// tokenizer model, which `tokenizer` stands in for.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Muc6Names<T = RuleTokenizer> {
    pub tokenizer: T,
}

impl<T: TokenSpans + Send + Sync> Format<NameSamples> for Muc6Names<T> {
    fn name(&self) -> &str {
        MUC6_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, options: ReadOptions) -> Result<Samples<'a, NameSamples>, FormatError> {
        Ok(Box::new(MucReader::new(source, &self.tokenizer, options.whitespace)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Registry;

    #[test]
    fn formats_register_by_name() {
        let mut r = Registry::<NameSamples>::with_native();
        r.register(GermEvalNames::default()).unwrap();
        r.register(BioNlpNames::default()).unwrap();
        r.register(EvalitaNames::default()).unwrap();
        r.register(Muc6Names::<RuleTokenizer>::default()).unwrap();
        assert_eq!(r.names().collect::<Vec<_>>(), ["opennlp", "germeval2014", "bionlp2004", "evalita", "muc6"]);
        let read = |name: &str, text: &'static str| -> String {
            r.open(name, text, ReadOptions::default()).unwrap().next().unwrap().unwrap().to_string()
        };
        assert_eq!(read("germeval2014", "1\ta\tB-LOC\tO\n"), "<START:location> a  <END>");
        assert_eq!(read("bionlp2004", "a\tB-RNA\n"), "<START:RNA> a  <END>");
        assert_eq!(read("evalita", "a S B-NP B-GPE\n"), "\n<START:gpe> a  <END>");
        assert_eq!(read("muc6", "<DOC><s><NUMEX TYPE=\"MONEY\">$5</NUMEX></s>"), "\n<START:money> $5  <END>");
    }
}
