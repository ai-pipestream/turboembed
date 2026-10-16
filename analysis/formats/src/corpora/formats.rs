//! The registry formats of the sentence and token corpora.

use std::sync::Arc;

use turbo_tokenize::detok::{Detokenizer, DetokenizerDictionary};

use super::{
    IRISH_SENTENCE_BANK_FORMAT, IrishSentenceBank, LETSMT_FORMAT, LetsmtDocument, LetsmtReader, MOSES_FORMAT,
    MosesReader, NKJP_FORMAT, NkjpReader, NkjpSegmentation, NkjpText,
};
use crate::error::FormatError;
use crate::registry::{Format, ReadOptions, Samples, SentenceSamples, TokenSamples};

/// Sentence samples from a Moses file (`MosesSentenceSampleStreamFactory`),
/// named [`MOSES_FORMAT`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MosesSentences;

impl Format<SentenceSamples> for MosesSentences {
    fn name(&self) -> &str {
        MOSES_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, SentenceSamples>, FormatError> {
        Ok(Box::new(MosesReader::new(source)?))
    }
}

/// Sentence samples from a LetsMT document (`LetsmtSentenceStreamFactory`),
/// named [`LETSMT_FORMAT`], detokenized when the factory's optional
/// `detokenizer` dictionary is given. The document is parsed when the
/// format opens it, as the factory parses it.
#[derive(Clone, Debug, Default)]
pub struct LetsmtSentences {
    pub detokenizer: Option<Arc<DetokenizerDictionary>>,
}

impl Format<SentenceSamples> for LetsmtSentences {
    fn name(&self) -> &str {
        LETSMT_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, options: ReadOptions) -> Result<Samples<'a, SentenceSamples>, FormatError> {
        let document = LetsmtDocument::parse(source)?;
        let detokenizer = self.detokenizer.as_deref().map(|d| (Detokenizer::new(d), options.whitespace));
        Ok(Box::new(LetsmtReader::new(document, detokenizer)))
    }
}

/// Sentence samples from an NKJP segmentation file
/// (`NKJPSentenceSampleStreamFactory`), named [`NKJP_FORMAT`]: the source
/// is the segmentation file, and `text` is the text file the factory's
/// `text` parameter names.
#[derive(Clone, Debug)]
pub struct NkjpSentences {
    pub text: Arc<NkjpText>,
}

impl Format<SentenceSamples> for NkjpSentences {
    fn name(&self) -> &str {
        NKJP_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, SentenceSamples>, FormatError> {
        let segmentation = NkjpSegmentation::parse(source)?;
        Ok(Box::new(NkjpReader::new(segmentation, &self.text)))
    }
}

/// The sentence sample of an Irish Sentence Bank document
/// (`IrishSentenceBankSentenceStreamFactory`), named
/// [`IRISH_SENTENCE_BANK_FORMAT`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct IrishSentences;

impl Format<SentenceSamples> for IrishSentences {
    fn name(&self) -> &str {
        IRISH_SENTENCE_BANK_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, SentenceSamples>, FormatError> {
        let bank = IrishSentenceBank::parse(source)?;
        Ok(Box::new(bank.sentence_sample().into_iter()))
    }
}

/// The token samples of an Irish Sentence Bank document
/// (`IrishSentenceBankTokenSampleStreamFactory`), named
/// [`IRISH_SENTENCE_BANK_FORMAT`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct IrishTokens;

impl Format<TokenSamples> for IrishTokens {
    fn name(&self) -> &str {
        IRISH_SENTENCE_BANK_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, TokenSamples>, FormatError> {
        Ok(Box::new(IrishSentenceBank::parse(source)?.into_token_samples()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Registry;

    #[test]
    fn formats_register_by_name() {
        let text = Arc::new(
            NkjpText::parse(
                r#"<teiCorpus><TEI><text xml:id="t"><body><div xml:id="d"><p xml:id="p">Tak.</p></div></body></text></TEI></teiCorpus>"#,
            )
            .unwrap(),
        );
        let mut r = Registry::<SentenceSamples>::with_native();
        r.register(MosesSentences).unwrap();
        r.register(LetsmtSentences::default()).unwrap();
        r.register(NkjpSentences { text }).unwrap();
        r.register(IrishSentences).unwrap();
        assert_eq!(r.names().collect::<Vec<_>>(), ["opennlp", "moses", "letsmt", "nkjp", "irishsentencebank"]);
        let read = |name: &str, text: &'static str| -> String {
            r.open(name, text, ReadOptions::default()).unwrap().next().unwrap().unwrap().document().to_owned()
        };
        assert_eq!(read("moses", " a \nb\n"), "a b ");
        assert_eq!(read("letsmt", "<d><s><w>a</w><w>b</w></s></d>"), "a b ");
        assert_eq!(
            read(
                "nkjp",
                r#"<teiCorpus><TEI><text><body><p><s><seg xml:id="x" corresp="t#r(p,0,3)"/></s></p></body></text></TEI></teiCorpus>"#
            ),
            "Tak "
        );
        let irish = r#"<sentences><sentence source="s"><original><token slot="1">Dia</token> duit</original></sentence></sentences>"#;
        assert_eq!(read("irishsentencebank", irish), "Dia duit ");

        let mut tokens = Registry::<TokenSamples>::with_native();
        tokens.register(IrishTokens).unwrap();
        let sample = tokens.open("irishsentencebank", irish, ReadOptions::default()).unwrap().next().unwrap().unwrap();
        assert_eq!(sample.to_string(), "Dia duit");
    }
}
