use turbo_formats::masc::{ChunkSkip, MascChunkRead, MascChunkReader, MascCorpus, MascError, MascWarning};
use turbo_formats::sample::ChunkSample;
use turbo_formats::sample::chunk::phrase_spans;
use turbo_formats::source::Field;
use turbo_zoo::Sha256;

use super::{MAX_SKIPPED_SENTENCES, REFERENCE_TAGS};
use crate::card::ChunksCard;
use crate::error::{LeftOut, Result};
use crate::spans::SpanTypes;

/// The sentences [`read_samples`] skipped, by why.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Skipped {
    pub no_tokens: usize,
    pub null_tag: usize,
    pub partial_chunk: usize,
    pub overlapping_chunks: usize,
    pub chunk_with_gap: usize,
}

impl Skipped {
    pub(super) fn add(&mut self, reason: ChunkSkip) {
        let count = match reason {
            ChunkSkip::NoTokens => &mut self.no_tokens,
            ChunkSkip::NullTag => &mut self.null_tag,
            ChunkSkip::PartialChunk => &mut self.partial_chunk,
            ChunkSkip::OverlappingChunks => &mut self.overlapping_chunks,
            ChunkSkip::ChunkWithGap => &mut self.chunk_with_gap,
        };
        *count += 1;
    }

    /// Every sentence skipped.
    pub fn total(&self) -> usize {
        self.no_tokens + self.null_tag + self.partial_chunk + self.overlapping_chunks + self.chunk_with_gap
    }
}

/// Reads the chunk samples of `corpus` ([`MascChunkReader`]), giving each
/// to `each`, and returns the sentences it skipped.
///
/// # Errors
///
/// A corpus read without its chunk files, or a token the text cannot
/// give.
pub fn read_samples<'c>(corpus: &'c MascCorpus, mut each: impl FnMut(ChunkSample<'c>)) -> Result<Skipped> {
    let mut skipped = Skipped::default();
    for read in MascChunkReader::new(corpus)? {
        match read? {
            MascChunkRead::Sample(s) => each(s.sample),
            MascChunkRead::Skipped { reason, .. } => skipped.add(reason),
        }
    }
    Ok(skipped)
}

/// The chunks of `corpus` in no sentence the reader reads, by why, as the
/// documents warned of them when their chunks were placed
/// (`MascDocument::add_chunks`), and every chunk of the documents' chunk
/// files: those placed in a sentence and those left out. A document with
/// chunk files and no Penn tokens (MASC 3.0.0 has none) has no chunks
/// placed, and its chunks are not counted.
pub fn chunk_counts(corpus: &MascCorpus) -> ChunksCard {
    let mut counts = ChunksCard::default();
    let mut placed = 0;
    for doc in corpus.documents() {
        placed += doc.sentences().iter().map(|s| s.chunks().len() as u64).sum::<u64>();
        for warning in doc.warnings() {
            let count = match warning {
                MascWarning::ChunkOutsideSentences { .. } => &mut counts.outside_sentences,
                MascWarning::ChunkCrossesSentences { .. } => &mut counts.across_sentences,
                MascWarning::ChunkWithoutTokens { .. } => &mut counts.without_tokens,
                _ => continue,
            };
            *count += 1;
        }
    }
    counts.chunks = placed + counts.outside_sentences + counts.across_sentences + counts.without_tokens;
    counts
}

/// Checks the sentences the reader skipped against
/// [`MAX_SKIPPED_SENTENCES`]: `skipped` of `skipped + read`.
pub(super) fn check_skipped(what: LeftOut, skipped: usize, read: usize) -> Result<()> {
    MAX_SKIPPED_SENTENCES.check(what, skipped as u64, (skipped + read) as u64)
}

/// The tag scored for the chunk tag `tag` of a type `types` does not
/// score: `O`; any other tag as it is. A tag's type follows its `B-` or
/// `I-` prefix, as `ChunkSample` reads it.
pub fn scored_tag<'t>(tag: &'t str, types: &SpanTypes) -> &'t str {
    match tag.strip_prefix("B-").or_else(|| tag.strip_prefix("I-")).filter(|k| !k.is_empty()) {
        Some(kind) if !types.predicted(kind) => "O",
        _ => tag,
    }
}

/// The held-out chunk samples, their chunk tags of the types scored
/// ([`super::SPAN_TYPES`]: renamed where it renames them, `O` where it
/// leaves them out), borrowed from the held-out corpus.
#[derive(Debug, Clone)]
pub struct HeldOutChunks<'c> {
    samples: Vec<ChunkSample<'c>>,
    skipped: Skipped,
    /// The held-out documents' chunks in no sentence the reader reads.
    chunks: ChunksCard,
    /// The held-out phrases of the types scored.
    spans: u64,
    /// The held-out phrases of other types, made `O`.
    left_out: u64,
}

impl<'c> HeldOutChunks<'c> {
    /// The held-out samples of `samples`, their phrases of other types
    /// than `types` scores made `O` and the renamed ones renamed; the
    /// reader skipped `skipped` sentences and placed `chunks` in no
    /// sentence ([`chunk_counts`]).
    ///
    /// # Errors
    ///
    /// A chunk tag `ChunkSample` cannot read a phrase from
    /// ([`MascError::Sample`]).
    pub fn new(
        samples: Vec<ChunkSample<'c>>,
        skipped: Skipped,
        chunks: ChunksCard,
        types: &SpanTypes,
    ) -> Result<HeldOutChunks<'c>> {
        let mut out =
            HeldOutChunks { samples: Vec::with_capacity(samples.len()), skipped, chunks, spans: 0, left_out: 0 };
        for sample in samples {
            let tags: Vec<&str> = sample.chunks().iter().map(Field::as_str).collect();
            let phrases = phrase_spans(&tags).map_err(MascError::from)?;
            let mut chunks: Vec<Field<'c>> = sample.chunks().to_vec();
            for &(start, end, cut) in &phrases {
                let kind = &tags[start as usize][cut..];
                match types.held_out(kind) {
                    Some(scored) => {
                        out.spans += 1;
                        if scored != kind || !tags[start as usize].starts_with("B-") {
                            for (i, field) in chunks[start as usize..end as usize].iter_mut().enumerate() {
                                let prefix = if i == 0 { "B-" } else { "I-" };
                                *field = Field::made(format!("{prefix}{scored}"), field.span);
                            }
                        }
                    }
                    None => {
                        out.left_out += 1;
                        for field in &mut chunks[start as usize..end as usize] {
                            *field = Field::made("O".to_owned(), field.span);
                        }
                    }
                }
            }
            let source = sample.source();
            let sample = ChunkSample::new(sample.tokens().to_vec(), sample.tags().to_vec(), chunks, source);
            out.samples.push(sample.map_err(MascError::from)?);
        }
        Ok(out)
    }

    /// The samples scored.
    pub fn samples(&self) -> &[ChunkSample<'c>] {
        &self.samples
    }

    /// The held-out sentences [`read_samples`] skipped.
    pub fn skipped(&self) -> Skipped {
        self.skipped
    }

    /// The held-out documents' chunks in no sentence the reader reads
    /// ([`chunk_counts`]).
    pub fn chunks(&self) -> &ChunksCard {
        &self.chunks
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The held-out phrases scored.
    pub fn spans(&self) -> u64 {
        self.spans
    }

    /// The held-out phrases of other types, left out of both models'
    /// scores.
    pub fn left_out(&self) -> u64 {
        self.left_out
    }

    /// The tokens of the samples.
    pub fn tokens(&self) -> u64 {
        self.samples.iter().map(|s| s.tokens().len() as u64).sum()
    }

    /// The SHA-256 of the samples as scored: for each sample, its number
    /// of tokens as a little endian `u64`, then for each token its text,
    /// its tag and its chunk tag, each as its length in bytes as a little
    /// endian `u64` and its bytes; after the samples, the types scored,
    /// the renamed types (each held-out type and the type it is scored
    /// as) and [`REFERENCE_TAGS`] (each held-out tag and the reference's),
    /// each list as its length and its strings.
    pub fn sha256(&self, types: &SpanTypes) -> String {
        let mut bytes = Vec::new();
        let put = |bytes: &mut Vec<u8>, s: &str| {
            bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
            bytes.extend_from_slice(s.as_bytes());
        };
        for sample in &self.samples {
            bytes.extend_from_slice(&(sample.tokens().len() as u64).to_le_bytes());
            for ((token, tag), chunk) in sample.tokens().iter().zip(sample.tags()).zip(sample.chunks()) {
                put(&mut bytes, token.as_str());
                put(&mut bytes, tag.as_str());
                put(&mut bytes, chunk.as_str());
            }
        }
        bytes.extend_from_slice(&(types.types.len() as u64).to_le_bytes());
        for t in types.types {
            put(&mut bytes, t);
        }
        bytes.extend_from_slice(&(types.renamed.len() as u64).to_le_bytes());
        for (held_out, scored) in types.renamed {
            put(&mut bytes, held_out);
            put(&mut bytes, scored);
        }
        bytes.extend_from_slice(&(REFERENCE_TAGS.len() as u64).to_le_bytes());
        for m in REFERENCE_TAGS {
            put(&mut bytes, m.held_out);
            put(&mut bytes, m.reference);
        }
        Sha256::of(&bytes).to_string()
    }
}
