//! Sample conversion streams: token and sentence samples made from name
//! and part-of-speech samples, as `NameToTokenSampleStream`,
//! `POSToTokenSampleStream`, `NameToSentenceSampleStream` and
//! `POSToSentenceSampleStream` make them, with a detokenizer that
//! rebuilds the text the tokens came from.
//!
//! A token sample's text is the tokens joined by `TokenSample`'s rule:
//! a space between two tokens unless the first attaches to the right or
//! the second to the left. A sentence sample's document is each sample's
//! detokenized text, one after the other with nothing between them, so
//! the last token of a sentence touches the first of the next. Each
//! token and sentence keeps the span of the source it came from.
//!
//! `AbstractToSentenceSampleStream` reads one sample past a full chunk
//! before it sees the chunk is full, and drops that sample: with the
//! command line's chunk size of 30, every 31st sample is lost. This is
//! reproduced ([`ToSentenceSamples`]).

use std::borrow::Cow;
use std::marker::PhantomData;

use turbo_analysis::Span;
use turbo_tokenize::detok::{Detokenizer, Operation};

use crate::error::FormatError;
use crate::sample::{SentenceSample, TokenSample, TokenizedSample};

fn merges_left(op: Operation) -> bool {
    matches!(op, Operation::MergeToLeft | Operation::MergeBoth)
}

fn merges_right(op: Operation) -> bool {
    matches!(op, Operation::MergeToRight | Operation::MergeBoth)
}

/// The token sample of `sample`'s tokens: `new TokenSample(detokenizer,
/// tokens)`.
pub fn to_token_sample<'a, S: TokenizedSample<'a>>(
    detokenizer: &Detokenizer<'_>,
    sample: &S,
) -> Result<TokenSample<'a>, FormatError> {
    let fields = sample.tokens();
    let texts: Vec<&str> = fields.iter().map(|f| f.as_str()).collect();
    let mut ops = Vec::with_capacity(texts.len());
    detokenizer.operations(&texts, &mut ops);
    let mut text = String::new();
    let mut tokens = Vec::with_capacity(texts.len());
    for (i, token) in texts.iter().enumerate() {
        if i > 0 && !merges_right(ops[i - 1]) && !merges_left(ops[i]) {
            text.push(' ');
        }
        let start = offset(&text)?;
        text.push_str(token);
        tokens.push(Span::new(start, offset(&text)?));
    }
    TokenSample::new(Cow::Owned(text), tokens, fields.iter().map(|f| f.span).collect(), sample.source())
}

fn offset(text: &str) -> Result<u32, FormatError> {
    u32::try_from(text.len()).map_err(|_| FormatError::TooLong { len: text.len() })
}

/// Token samples from a stream of name or part-of-speech samples, one per
/// sample: `NameToTokenSampleStream` and `POSToTokenSampleStream`.
pub struct ToTokenSamples<'a, 'd, I> {
    inner: I,
    detokenizer: Detokenizer<'d>,
    source: PhantomData<fn() -> &'a str>,
}

impl<'a, 'd, I> ToTokenSamples<'a, 'd, I> {
    pub fn new(inner: I, detokenizer: Detokenizer<'d>) -> ToTokenSamples<'a, 'd, I> {
        ToTokenSamples { inner, detokenizer, source: PhantomData }
    }
}

impl<'a, S, I> Iterator for ToTokenSamples<'a, '_, I>
where
    S: TokenizedSample<'a>,
    I: Iterator<Item = Result<S, FormatError>>,
{
    type Item = Result<TokenSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.inner.next()?.and_then(|s| to_token_sample(&self.detokenizer, &s)))
    }
}

/// The number of samples a sentence sample is made of.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ChunkSize {
    /// At most this many (a chunk size above zero).
    Samples(std::num::NonZeroUsize),
    /// Every sample of the stream (a chunk size of zero).
    All,
}

impl ChunkSize {
    /// The chunk size the command line's conversion factories use.
    pub const COMMAND_LINE: ChunkSize = ChunkSize::Samples(std::num::NonZeroUsize::new(30).unwrap());

    fn limit(self) -> usize {
        match self {
            ChunkSize::Samples(n) => n.get(),
            ChunkSize::All => usize::MAX,
        }
    }
}

/// Sentence samples from a stream of name or part-of-speech samples,
/// each made of up to a chunk of samples: `NameToSentenceSampleStream`
/// and `POSToSentenceSampleStream`.
///
/// After a full chunk the next sample is read and dropped, as OpenNLP
/// drops it. An error from the stream is passed on, and the samples read
/// for the chunk so far are dropped with it, as Java's exception drops
/// them.
pub struct ToSentenceSamples<'a, 'd, I> {
    inner: I,
    detokenizer: Detokenizer<'d>,
    chunk: ChunkSize,
    source: PhantomData<fn() -> &'a str>,
}

impl<'a, 'd, I> ToSentenceSamples<'a, 'd, I> {
    pub fn new(inner: I, detokenizer: Detokenizer<'d>, chunk: ChunkSize) -> ToSentenceSamples<'a, 'd, I> {
        ToSentenceSamples { inner, detokenizer, chunk, source: PhantomData }
    }
}

impl<'a, S, I> Iterator for ToSentenceSamples<'a, '_, I>
where
    S: TokenizedSample<'a>,
    I: Iterator<Item = Result<S, FormatError>>,
{
    type Item = Result<SentenceSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let limit = self.chunk.limit();
        let mut document = String::new();
        let mut sentences = Vec::new();
        let mut sources = Vec::new();
        loop {
            let sample = match self.inner.next() {
                None => break,
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(sample)) => sample,
            };
            if sentences.len() >= limit {
                // Read past the full chunk and dropped.
                break;
            }
            let texts: Vec<&str> = sample.tokens().iter().map(|f| f.as_str()).collect();
            let start = match offset(&document) {
                Ok(o) => o,
                Err(e) => return Some(Err(e)),
            };
            self.detokenizer.detokenize(&texts, None, &mut document);
            let end = match offset(&document) {
                Ok(o) => o,
                Err(e) => return Some(Err(e)),
            };
            sentences.push(Span::new(start, end));
            sources.push(sample.source());
        }
        let first = sources.first()?;
        let source = Span::new(first.start, sources.last().expect("not empty").end);
        Some(SentenceSample::new(Cow::Owned(document), sentences, sources, source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::WordTagReader;
    use turbo_tokenize::detok::{DetokenizerDictionary, DictionaryOperation};
    use turbo_unicode::WhitespaceMode;

    fn dictionary() -> DetokenizerDictionary {
        let mut d = DetokenizerDictionary::new();
        d.insert(".", DictionaryOperation::MoveLeft);
        d.insert("(", DictionaryOperation::MoveRight);
        d.insert("-", DictionaryOperation::MoveBoth);
        d
    }

    #[test]
    fn token_samples_follow_the_operations() {
        let d = dictionary();
        let src = "a_X (_X b_X -_X c_X ._X";
        let samples: Vec<TokenSample<'_>> =
            ToTokenSamples::new(WordTagReader::new(src, WhitespaceMode::Unicode).unwrap(), Detokenizer::new(&d))
                .map(Result::unwrap)
                .collect();
        assert_eq!(samples[0].text(), "a (b-c.");
        assert_eq!(samples[0].to_string(), "a (<SPLIT>b<SPLIT>-<SPLIT>c<SPLIT>.");
        let sources: Vec<&str> = samples[0].sources().iter().map(|s| &src[s.range()]).collect();
        assert_eq!(sources, ["a", "(", "b", "-", "c", "."]);
    }

    #[test]
    fn sentence_chunks_drop_the_sample_after_a_full_chunk() {
        let d = dictionary();
        let src = "a_X ._X\nb_X\nc_X\nd_X\ne_X\n";
        let reader = WordTagReader::new(src, WhitespaceMode::Unicode).unwrap();
        let two = ChunkSize::Samples(std::num::NonZeroUsize::new(2).unwrap());
        let samples: Vec<SentenceSample<'_>> =
            ToSentenceSamples::new(reader, Detokenizer::new(&d), two).map(Result::unwrap).collect();
        let docs: Vec<&str> = samples.iter().map(|s| s.document()).collect();
        // c is read after the first chunk is full and dropped.
        assert_eq!(docs, ["a.b", "de"]);
        assert_eq!(samples[0].to_string(), "a.\nb\n");
        let reader = WordTagReader::new(src, WhitespaceMode::Unicode).unwrap();
        let all: Vec<_> = ToSentenceSamples::new(reader, Detokenizer::new(&d), ChunkSize::All).collect();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].as_ref().unwrap().document(), "a.bcde");
    }
}
