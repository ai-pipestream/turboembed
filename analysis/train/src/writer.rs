//! The bytes of a trained GIS model: `GenericModelWriter` over a
//! `GISModel`, which is `BinaryGISModelWriter` and `GISModelWriter.persist`
//! writing through `DataOutputStream` and `ModelParameterChunker`.
//!
//! The layout, read back by `turbo-maxent` and by OpenNLP's
//! `GISModelReader`:
//!
//! 1. `writeUTF("GIS")`, `writeInt(1)` and `writeDouble(1)`: the model type
//!    and the correction constant and parameter, which nothing reads.
//! 2. The outcome labels: their count, then each with `writeUTF`.
//! 3. The outcome patterns: the predicates are sorted by their pattern
//!    (the outcome ids a predicate has parameters for, compared
//!    lexicographically, then by length; a stable sort, so predicates with
//!    one pattern keep their index order) and consecutive equal patterns
//!    are grouped. The count of groups, then each group as the text
//!    `"<size> <o1> <o2> ..."`.
//! 4. The predicate labels in the sorted order, their count first.
//! 5. The parameters: for every predicate in the sorted order, one
//!    `writeDouble` per entry of its pattern.
//!
//! Every string goes through `ModelParameterChunker.writeUTF`: a string of
//! at most 65535 bytes of modified UTF-8 is written as `writeUTF` writes
//! it; a longer one is written as `"CHUNKED-MODEL-PARAMS:<n>"` and then
//! `n` chunks, each a `writeUTF` of a run of the string that
//! `CharsetEncoder` for UTF-8 fit into 65535 bytes. Two consequences of
//! measuring the chunks in standard UTF-8 are kept, as they are what the
//! JVM writes: a chunk full of characters outside the BMP can exceed
//! 65535 bytes of modified UTF-8, which `writeUTF` then refuses
//! ([`Error::StringTooLong`] here, `UTFDataFormatException` there), and
//! the encoder stops at a lone surrogate, so a chunked string is cut
//! there and the rest of it is lost. U+0000 counts one byte in a chunk
//! and takes two when written, so a chunk of them can overflow as well.

use crate::error::{Error, Result};
use crate::gis::GisModel;
use crate::java_string::JavaString;

/// The type string `GISModelWriter.persist` starts with.
pub const GIS_TYPE: &str = "GIS";
/// `ModelParameterChunker.SIGNATURE_CHUNKED_PARAMS`.
pub const CHUNKED_SIGNATURE: &str = "CHUNKED-MODEL-PARAMS:";
/// `writeUTF`'s limit on a string's modified UTF-8 bytes.
pub const MAX_UTF_BYTES: usize = 65535;

/// `GenericModelWriter(model, DataOutputStream).persist()` for a GIS
/// model: the bytes of the `.model` entry.
pub fn write_gis(model: &GisModel) -> Result<Vec<u8>> {
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_utf(&JavaString::new(GIS_TYPE))?;
    out.write_int(1);
    out.write_double(1.0);
    out.write_int(to_java_int(model.outcome_labels().len()));
    for label in model.outcome_labels() {
        out.write_utf(&JavaString::new(label))?;
    }
    let sorted = sort_by_pattern(model);
    let groups = pattern_groups(model, &sorted);
    out.write_int(to_java_int(groups.len()));
    for (size, first) in &groups {
        let mut text = size.to_string();
        for oid in model.pattern(*first) {
            text.push(' ');
            text.push_str(&oid.to_string());
        }
        out.write_utf(&JavaString::new(&text))?;
    }
    out.write_int(to_java_int(model.num_predicates()));
    for &p in &sorted {
        out.write_utf(&model.pred_labels()[p])?;
    }
    for &p in &sorted {
        for &param in model.parameters(p) {
            out.write_double(param);
        }
    }
    Ok(out.bytes)
}

/// `GISModelWriter.sortValues`: the predicate indices sorted by
/// `ComparablePredicate.compareTo`, a stable sort as `Arrays.sort` on
/// objects is.
fn sort_by_pattern(model: &GisModel) -> Vec<usize> {
    let mut sorted: Vec<usize> = (0..model.num_predicates()).collect();
    sorted.sort_by(|&a, &b| model.pattern(a).cmp(model.pattern(b)));
    sorted
}

/// `compressOutcomes`: runs of predicates with equal patterns, as the
/// run's size and its first predicate.
fn pattern_groups(model: &GisModel, sorted: &[usize]) -> Vec<(usize, usize)> {
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for &p in sorted {
        match groups.last_mut() {
            Some((size, first)) if model.pattern(*first) == model.pattern(p) => *size += 1,
            _ => groups.push((1, p)),
        }
    }
    groups
}

fn to_java_int(n: usize) -> i32 {
    i32::try_from(n).expect("a model has fewer than 2^31 labels")
}

/// A `DataOutputStream` into memory.
struct ModelOutput {
    bytes: Vec<u8>,
}

impl ModelOutput {
    fn write_int(&mut self, i: i32) {
        self.bytes.extend_from_slice(&i.to_be_bytes());
    }

    fn write_double(&mut self, d: f64) {
        self.bytes.extend_from_slice(&d.to_bits().to_be_bytes());
    }

    /// `ModelParameterChunker.writeUTF`.
    fn write_utf(&mut self, s: &JavaString) -> Result<()> {
        if s.len() <= MAX_UTF_BYTES {
            self.write_utf_plain(s.as_bytes());
            return Ok(());
        }
        let chunks = split_by_utf8_length(s);
        self.write_utf_plain(format!("{CHUNKED_SIGNATURE}{}", chunks.len()).as_bytes());
        for chunk in chunks {
            if chunk.len() > MAX_UTF_BYTES {
                return Err(Error::StringTooLong { what: "chunk of a model parameter", bytes: chunk.len() });
            }
            self.write_utf_plain(chunk);
        }
        Ok(())
    }

    /// `DataOutputStream.writeUTF` of bytes known to fit.
    fn write_utf_plain(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
        self.bytes.extend_from_slice(bytes);
    }
}

/// `ModelParameterChunker.splitByByteLength`: the string cut into runs
/// whose standard UTF-8 encoding fits 65535 bytes, each as the run's
/// modified UTF-8 bytes. The UTF-8 encoder stops at a lone surrogate, so
/// the run before it is the last chunk and the rest is dropped.
fn split_by_utf8_length(s: &JavaString) -> Vec<&[u8]> {
    let bytes = s.as_bytes();
    let mut chunks = Vec::new();
    let mut chunk_start = 0;
    let mut chunk_utf8 = 0;
    let mut pos = 0;
    while pos < bytes.len() {
        let (modified_len, utf8_len) = match next_char(bytes, pos) {
            Some(lens) => lens,
            None => {
                // A lone surrogate: malformed input ends the encoding.
                chunks.push(&bytes[chunk_start..pos]);
                return chunks;
            }
        };
        if chunk_utf8 + utf8_len > MAX_UTF_BYTES {
            chunks.push(&bytes[chunk_start..pos]);
            chunk_start = pos;
            chunk_utf8 = 0;
        }
        chunk_utf8 += utf8_len;
        pos += modified_len;
    }
    chunks.push(&bytes[chunk_start..]);
    chunks
}

/// The character at `pos` of canonical modified UTF-8: its length there
/// and in standard UTF-8, or `None` for a lone surrogate.
fn next_char(bytes: &[u8], pos: usize) -> Option<(usize, usize)> {
    match bytes[pos] {
        0x00..=0x7F => return Some((1, 1)),
        // U+0000, which canonical modified UTF-8 writes as C0 80 and
        // standard UTF-8 as one zero byte.
        0xC0 => return Some((2, 1)),
        0xC1..=0xDF => return Some((2, 2)),
        _ => {}
    }
    let unit = (u32::from(bytes[pos] & 0x0F) << 12)
        | (u32::from(bytes[pos + 1] & 0x3F) << 6)
        | u32::from(bytes[pos + 2] & 0x3F);
    match unit {
        0xD800..=0xDBFF => {
            let next = pos + 3;
            if next + 3 <= bytes.len() && bytes[next] == 0xED && (0xB0..=0xBF).contains(&bytes[next + 1]) {
                Some((6, 4))
            } else {
                None
            }
        }
        0xDC00..=0xDFFF => None,
        _ => Some((3, 3)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Event;
    use crate::index::{DataIndex, IndexParams, IndexerKind};
    use crate::monitor::TrainingConfiguration;
    use crate::params::{Key, TrainingParameters};
    use turbo_maxent::MaxentModel;

    fn trained() -> GisModel {
        let events = [
            Event::from_strs("yes", &["a", "b"]),
            Event::from_strs("no", &["c", "b"]),
            Event::from_strs("maybe", &["d"]),
            Event::from_strs("yes", &["a", "d"]),
        ];
        let index = DataIndex::index(&events, &IndexParams::new(IndexerKind::TwoPass, 0, true)).unwrap();
        let mut params = TrainingParameters::new();
        params.put(Key::ITERATIONS, 10);
        crate::gis::train(&index, &params, &mut TrainingConfiguration::default()).unwrap()
    }

    #[test]
    fn the_written_model_reads_back_with_the_same_parameters() {
        let model = trained();
        let bytes = write_gis(&model).unwrap();
        let read = MaxentModel::read(&bytes).unwrap();
        assert_eq!(read.outcomes().collect::<Vec<_>>(), model.outcome_labels());
        assert_eq!(read.num_predicates(), model.num_predicates());
        for p in ["a", "b", "c", "d"] {
            assert!(read.has_predicate(p), "{p}");
        }
        // The written parameters evaluate as the trained ones.
        let mut out = vec![0.0; 3];
        read.eval(&["a", "b"], &mut out);
        let yes = model.outcome_labels().iter().position(|o| o == "yes").unwrap();
        assert!(out[yes] > 0.5, "{out:?}");
    }

    #[test]
    fn the_layout_is_persist_byte_for_byte() {
        let model = trained();
        let bytes = write_gis(&model).unwrap();
        let mut expect = Vec::new();
        expect.extend_from_slice(&[0, 3, b'G', b'I', b'S']);
        expect.extend_from_slice(&1i32.to_be_bytes());
        expect.extend_from_slice(&1.0f64.to_bits().to_be_bytes());
        expect.extend_from_slice(&3i32.to_be_bytes());
        for o in ["yes", "no", "maybe"] {
            expect.extend_from_slice(&(o.len() as u16).to_be_bytes());
            expect.extend_from_slice(o.as_bytes());
        }
        // Patterns in sorted order: a [0], b [0 1], d [0 2], c [1].
        expect.extend_from_slice(&4i32.to_be_bytes());
        for g in ["1 0", "1 0 1", "1 0 2", "1 1"] {
            expect.extend_from_slice(&(g.len() as u16).to_be_bytes());
            expect.extend_from_slice(g.as_bytes());
        }
        expect.extend_from_slice(&4i32.to_be_bytes());
        for p in ["a", "b", "d", "c"] {
            expect.extend_from_slice(&[0, 1, p.as_bytes()[0]]);
        }
        assert_eq!(&bytes[..expect.len()], &expect[..]);
        assert_eq!(bytes.len(), expect.len() + 6 * 8);
    }

    fn utf8_len(s: &JavaString) -> usize {
        s.to_str().unwrap().len()
    }

    #[test]
    fn long_strings_are_chunked_by_utf8_bytes() {
        // 70000 ASCII bytes: two chunks of 65535 and 4465 bytes.
        let long = JavaString::new(&"x".repeat(70000));
        let chunks = split_by_utf8_length(&long);
        assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 4465]);
        // Three byte characters do not straddle a chunk end.
        let long = JavaString::new(&"\u{20AC}".repeat(30000));
        let chunks = split_by_utf8_length(&long);
        assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 24465]);
        assert_eq!(utf8_len(&long), 90000);
        let mut out = ModelOutput { bytes: Vec::new() };
        out.write_utf(&long).unwrap();
        let signature = b"CHUNKED-MODEL-PARAMS:2";
        assert_eq!(&out.bytes[..2 + signature.len()], [&[0, signature.len() as u8][..], signature].concat());
        assert_eq!(out.bytes.len(), 2 + signature.len() + 2 + 65535 + 2 + 24465);
    }

    #[test]
    fn a_chunk_of_supplementary_characters_overflows_write_utf() {
        // 16383 emoji fit a 65532 byte UTF-8 chunk but are 98298 bytes of
        // modified UTF-8, which writeUTF refuses, as the JVM's does.
        let long = JavaString::new(&"\u{1F600}".repeat(17000));
        let chunks = split_by_utf8_length(&long);
        assert_eq!(chunks[0].len(), 16383 * 6);
        let mut out = ModelOutput { bytes: Vec::new() };
        assert_eq!(
            out.write_utf(&long),
            Err(Error::StringTooLong { what: "chunk of a model parameter", bytes: 16383 * 6 })
        );
    }

    #[test]
    fn a_nul_is_one_utf8_byte_but_two_written() {
        assert_eq!(next_char(b"\xC0\x80a", 0), Some((2, 1)));
        assert_eq!(next_char(b"\xC3\xA9", 0), Some((2, 2)));
        // 40000 NULs: 80000 bytes of modified UTF-8, so the string is
        // chunked, but 40000 bytes of UTF-8, so it is one chunk, whose
        // writeUTF throws in the JVM.
        let nuls = JavaString::new(&"\0".repeat(40000));
        assert_eq!(nuls.len(), 80000);
        let chunks = split_by_utf8_length(&nuls);
        assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [80000]);
        let mut out = ModelOutput { bytes: Vec::new() };
        assert_eq!(
            out.write_utf(&nuls),
            Err(Error::StringTooLong { what: "chunk of a model parameter", bytes: 80000 })
        );
        // NULs after the first chunk make a second chunk of twice their count.
        let late = JavaString::new(&format!("{}{}", "x".repeat(65535), "\0".repeat(1000)));
        let chunks = split_by_utf8_length(&late);
        assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 2000]);
        let mut out = ModelOutput { bytes: Vec::new() };
        out.write_utf(&late).unwrap();
        let signature = b"CHUNKED-MODEL-PARAMS:2";
        assert_eq!(out.bytes.len(), 2 + signature.len() + 2 + 65535 + 2 + 2000);
    }

    #[test]
    fn a_lone_surrogate_cuts_a_chunked_string() {
        let mut units: Vec<u16> = vec![b'y' as u16; 70000];
        units[66000] = 0xD83D;
        let long = JavaString::from_utf16(&units);
        let chunks = split_by_utf8_length(&long);
        assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 465]);
    }
}
