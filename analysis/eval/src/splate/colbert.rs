//! The upstream ColBERT token protocol for the trained study (#913).
//!
//! Upstream ColBERT does not encode bare text. A query is
//! `[CLS] [Q] pieces [SEP]` padded with live `[MASK]` tokens to
//! `QUERY_MAXLEN` rows; a document is `[CLS] [D] pieces [SEP]`
//! truncated to `DOC_MAXLEN` rows, and punctuation piece rows are
//! dropped from the document's vectors before MaxSim. `[Q]` and
//! `[D]` are the BERT vocabulary's `[unused0]` (id 1) and
//! `[unused1]` (id 2). Every kept row is projected and
//! L2-normalized. The generic bundle template (`[CLS] $TEXT [SEP]`,
//! no markers, no augmentation) is a different encoding, which is
//! what the archived trained run measured; this module builds the
//! protocol sequences explicitly and runs them through the encoder
//! one token range at a time.

use turbo::bundle::Bundle;
use turbo::head::ProjectionHead;
use turbo::turbo_token_range;
use turbo_analysis_embed::encoder::{Encoder, EncoderOptions, Rows};

use crate::Result;

/// Total rows in an encoded query, markers and MASK padding included.
pub const QUERY_MAXLEN: usize = 32;
/// Most rows in an encoded document, markers included.
pub const DOC_MAXLEN: usize = 180;

const CLS: i32 = 101;
const SEP: i32 = 102;
const MASK: i32 = 103;
const QUERY_MARKER: i32 = 1; // [unused0], upstream's [Q]
const DOC_MARKER: i32 = 2; // [unused1], upstream's [D]

/// Upstream's punctuation set (Python `string.punctuation`), matched
/// against a piece's source text.
const PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// The protocol rows for a query over its piece ids: exactly
/// `QUERY_MAXLEN` ids, MASK-padded when short.
pub fn query_ids(pieces: &[i32]) -> Vec<i32> {
    let keep = pieces.len().min(QUERY_MAXLEN - 3);
    let mut ids = Vec::with_capacity(QUERY_MAXLEN);
    ids.extend([CLS, QUERY_MARKER]);
    ids.extend_from_slice(&pieces[..keep]);
    ids.push(SEP);
    ids.resize(QUERY_MAXLEN, MASK);
    ids
}

/// The protocol rows for a document over its piece ids: at most
/// `DOC_MAXLEN` ids, SEP last.
pub fn doc_ids(pieces: &[i32]) -> Vec<i32> {
    let keep = pieces.len().min(DOC_MAXLEN - 3);
    let mut ids = Vec::with_capacity(keep + 3);
    ids.extend([CLS, DOC_MARKER]);
    ids.extend_from_slice(&pieces[..keep]);
    ids.push(SEP);
    ids
}

/// Whether a piece's source text is punctuation upstream drops.
pub fn is_punctuation_piece(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| PUNCTUATION.contains(c))
}

/// One text's protocol encoding: the rows retrieval keeps and the
/// piece rows (before the document punctuation drop) the adapter fit
/// aligns against the sparse teacher.
pub struct ProtocolEncoded {
    /// Ids parallel to `token_vecs`: the full sequence for a query;
    /// for a document, the sequence minus its punctuation pieces.
    pub ids: Vec<i32>,
    /// L2-normalized projected rows parallel to `ids`, flattened.
    pub token_vecs: Vec<f32>,
    /// Projected rows of the content pieces only, in piece order
    /// (documents: before punctuation filtering; queries: unused).
    pub piece_vecs: Vec<f32>,
}

/// The ColBERT bundle run under the upstream protocol.
pub struct ColbertProtocol {
    encoder: Encoder,
    projection: ProjectionHead,
    pub dim: usize,
}

impl ColbertProtocol {
    pub fn open(dir: &std::path::Path, options: &EncoderOptions) -> Result<Self> {
        let bundle = Bundle::open(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let projection = ProjectionHead::load(&bundle).map_err(|e| format!("{}: {e}", dir.display()))?;
        let dim = projection.dim;
        let mut opts = options.clone();
        let batch = if opts.max_batch == 0 { 32 } else { opts.max_batch };
        opts.max_ranges = opts.max_ranges.max(batch * DOC_MAXLEN as u32);
        let encoder = Encoder::open(dir, &opts).map_err(|e| e.to_string())?;
        Ok(Self { encoder, projection, dim })
    }

    /// Encodes protocol sequences built by `build` from each text's
    /// piece ids (and the pieces' source texts, for the document
    /// punctuation drop). `pieces[i]` is `(piece ids, piece texts)`.
    pub fn encode(&mut self, pieces: &[(Vec<i32>, Vec<String>)], query: bool) -> Result<Vec<ProtocolEncoded>> {
        let stride = if query { QUERY_MAXLEN } else { DOC_MAXLEN } as u32;
        let info = self.encoder.info();
        let batch = (info.max_ranges as usize / stride as usize).max(1).min(info.max_batch.max(1) as usize);
        let mut out: Vec<ProtocolEncoded> = Vec::with_capacity(pieces.len());
        for chunk in pieces.chunks(batch) {
            let mut rows = Rows::new(stride, 0);
            let mut ranges: Vec<turbo_token_range> = Vec::new();
            let mut seqs: Vec<Vec<i32>> = Vec::with_capacity(chunk.len());
            for (ids, _) in chunk {
                let seq = if query { query_ids(ids) } else { doc_ids(ids) };
                let row = rows.push(&[seq.as_slice()]);
                for pos in 0..seq.len() as u32 {
                    ranges.push(turbo_token_range { row, start: pos, end: pos + 1 });
                }
                seqs.push(seq);
            }
            let mut states = Vec::new();
            self.encoder.run_ranges(&rows, &ranges, &mut states).map_err(|e| e.to_string())?;
            let projected = self.projection.project(&states, true).map_err(|e| e.to_string())?;
            let mut at = 0usize;
            for (seq, (piece_ids, piece_texts)) in seqs.iter().zip(chunk) {
                let rows_here = &projected[at * self.dim..(at + seq.len()) * self.dim];
                at += seq.len();
                let keep = piece_ids.len().min(seq.len().saturating_sub(3));
                let piece_vecs = rows_here[2 * self.dim..(2 + keep) * self.dim].to_vec();
                let (ids, token_vecs) = if query {
                    (seq.clone(), rows_here.to_vec())
                } else {
                    let mut ids = vec![seq[0], seq[1]];
                    let mut vecs = rows_here[..2 * self.dim].to_vec();
                    for (k, text) in piece_texts.iter().take(keep).enumerate() {
                        if !is_punctuation_piece(text) {
                            ids.push(seq[2 + k]);
                            vecs.extend_from_slice(&rows_here[(2 + k) * self.dim..(3 + k) * self.dim]);
                        }
                    }
                    ids.push(SEP);
                    vecs.extend_from_slice(&rows_here[(seq.len() - 1) * self.dim..]);
                    (ids, vecs)
                };
                out.push(ProtocolEncoded { ids, token_vecs, piece_vecs });
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_sequences_are_marked_and_mask_padded_to_32() {
        let ids = query_ids(&[11, 12, 13]);
        assert_eq!(ids.len(), 32);
        assert_eq!(&ids[..6], &[101, 1, 11, 12, 13, 102]);
        assert!(ids[6..].iter().all(|&i| i == MASK));
        // An over-long query keeps 29 pieces and no MASK row.
        let long: Vec<i32> = (0..40).collect();
        let ids = query_ids(&long);
        assert_eq!(ids.len(), 32);
        assert_eq!(ids[31], SEP);
        assert!(!ids.contains(&MASK));
        assert_eq!(query_ids(&[])[..3], [101, 1, 102]);
    }

    #[test]
    fn doc_sequences_are_marked_and_capped_at_180() {
        let ids = doc_ids(&[11, 12]);
        assert_eq!(ids, [101, 2, 11, 12, 102]);
        let long: Vec<i32> = (0..400).collect();
        let ids = doc_ids(&long);
        assert_eq!(ids.len(), 180);
        assert_eq!(ids[179], SEP);
        assert_eq!(ids[178], 176);
    }

    #[test]
    fn punctuation_pieces_are_the_upstream_set() {
        for text in [".", ",", "!", "(", "\"", "...", ",."] {
            assert!(is_punctuation_piece(text), "{text}");
        }
        for text in ["", "a", "##ing", ".a", "3.14", "don't"] {
            assert!(!is_punctuation_piece(text), "{text}");
        }
    }
}
