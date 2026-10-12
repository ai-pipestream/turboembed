//! The fields of SentencePiece's `ModelProto` (sentencepiece_model.proto,
//! proto2) that encoding and decoding read, from the protobuf wire format.
//!
//! The reader follows what protobuf's C++ parser does with these
//! messages: a scalar field seen twice keeps its last value, a message
//! field seen twice is merged, a field with another wire type than the
//! schema's and a field the schema does not have are skipped, and an
//! enum value the schema does not list leaves the field at its default.

use crate::status::{Result, invalid};

/// A piece's type, `ModelProto.SentencePiece.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl PieceType {
    fn from_wire(v: u64) -> Option<PieceType> {
        Some(match v {
            1 => PieceType::Normal,
            2 => PieceType::Unknown,
            3 => PieceType::Control,
            4 => PieceType::UserDefined,
            5 => PieceType::Unused,
            6 => PieceType::Byte,
            _ => return None,
        })
    }
}

/// `TrainerSpec.ModelType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Unigram,
    Bpe,
    Word,
    Char,
}

pub struct Piece {
    /// The piece's bytes as the file holds them.
    pub piece: Vec<u8>,
    pub score: f32,
    pub kind: PieceType,
}

/// `TrainerSpec`: what encoding and decoding read of it.
pub struct TrainerSpec {
    pub model_type: ModelType,
    pub byte_fallback: bool,
    pub treat_whitespace_as_suffix: bool,
    pub unk_piece: Vec<u8>,
    pub bos_piece: Vec<u8>,
    pub eos_piece: Vec<u8>,
    pub pad_piece: Vec<u8>,
    pub unk_surface: Vec<u8>,
}

impl Default for TrainerSpec {
    fn default() -> Self {
        TrainerSpec {
            model_type: ModelType::Unigram,
            byte_fallback: false,
            treat_whitespace_as_suffix: false,
            unk_piece: b"<unk>".to_vec(),
            bos_piece: b"<s>".to_vec(),
            eos_piece: b"</s>".to_vec(),
            pad_piece: b"<pad>".to_vec(),
            unk_surface: " \u{2047} ".as_bytes().to_vec(),
        }
    }
}

/// `NormalizerSpec`.
pub struct NormalizerSpec {
    pub name: Vec<u8>,
    pub precompiled_charsmap: Vec<u8>,
    pub add_dummy_prefix: bool,
    pub remove_extra_whitespaces: bool,
    pub escape_whitespaces: bool,
}

impl Default for NormalizerSpec {
    fn default() -> Self {
        NormalizerSpec {
            name: Vec::new(),
            precompiled_charsmap: Vec::new(),
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
            escape_whitespaces: true,
        }
    }
}

/// One `SelfTestData.Sample`.
pub struct Sample {
    pub input: Vec<u8>,
    pub expected: Vec<u8>,
}

#[derive(Default)]
pub struct ModelProto {
    pub pieces: Vec<Piece>,
    pub trainer: TrainerSpec,
    pub normalizer: NormalizerSpec,
    pub denormalizer: Option<NormalizerSpec>,
    pub self_test: Vec<Sample>,
}

impl ModelProto {
    pub fn parse(bytes: &[u8]) -> Result<ModelProto> {
        let mut m = ModelProto::default();
        let mut r = Reader::new(bytes);
        while let Some((field, value)) = r.next()? {
            match (field, value) {
                (1, Value::Bytes(b)) => m.pieces.push(piece(b)?),
                (2, Value::Bytes(b)) => trainer(b, &mut m.trainer)?,
                (3, Value::Bytes(b)) => normalizer(b, &mut m.normalizer)?,
                (4, Value::Bytes(b)) => self_test(b, &mut m.self_test)?,
                (5, Value::Bytes(b)) => normalizer(b, m.denormalizer.get_or_insert_with(NormalizerSpec::default))?,
                _ => {}
            }
        }
        Ok(m)
    }
}

fn piece(bytes: &[u8]) -> Result<Piece> {
    let mut p = Piece { piece: Vec::new(), score: 0.0, kind: PieceType::Normal };
    let mut r = Reader::new(bytes);
    while let Some((field, value)) = r.next()? {
        match (field, value) {
            (1, Value::Bytes(b)) => p.piece = b.to_vec(),
            (2, Value::Fixed32(v)) => p.score = f32::from_bits(v),
            (3, Value::Varint(v)) => {
                if let Some(k) = PieceType::from_wire(v) {
                    p.kind = k;
                }
            }
            _ => {}
        }
    }
    Ok(p)
}

fn trainer(bytes: &[u8], t: &mut TrainerSpec) -> Result<()> {
    let mut r = Reader::new(bytes);
    while let Some((field, value)) = r.next()? {
        match (field, value) {
            (3, Value::Varint(v)) => {
                t.model_type = match v {
                    1 => ModelType::Unigram,
                    2 => ModelType::Bpe,
                    3 => ModelType::Word,
                    4 => ModelType::Char,
                    _ => t.model_type,
                }
            }
            (24, Value::Varint(v)) => t.treat_whitespace_as_suffix = v != 0,
            (35, Value::Varint(v)) => t.byte_fallback = v != 0,
            (44, Value::Bytes(b)) => t.unk_surface = b.to_vec(),
            (45, Value::Bytes(b)) => t.unk_piece = b.to_vec(),
            (46, Value::Bytes(b)) => t.bos_piece = b.to_vec(),
            (47, Value::Bytes(b)) => t.eos_piece = b.to_vec(),
            (48, Value::Bytes(b)) => t.pad_piece = b.to_vec(),
            _ => {}
        }
    }
    Ok(())
}

fn normalizer(bytes: &[u8], n: &mut NormalizerSpec) -> Result<()> {
    let mut r = Reader::new(bytes);
    while let Some((field, value)) = r.next()? {
        match (field, value) {
            (1, Value::Bytes(b)) => n.name = b.to_vec(),
            (2, Value::Bytes(b)) => n.precompiled_charsmap = b.to_vec(),
            (3, Value::Varint(v)) => n.add_dummy_prefix = v != 0,
            (4, Value::Varint(v)) => n.remove_extra_whitespaces = v != 0,
            (5, Value::Varint(v)) => n.escape_whitespaces = v != 0,
            _ => {}
        }
    }
    Ok(())
}

fn self_test(bytes: &[u8], out: &mut Vec<Sample>) -> Result<()> {
    let mut r = Reader::new(bytes);
    while let Some((field, value)) = r.next()? {
        if let (1, Value::Bytes(b)) = (field, value) {
            let mut s = Sample { input: Vec::new(), expected: Vec::new() };
            let mut r = Reader::new(b);
            while let Some((field, value)) = r.next()? {
                match (field, value) {
                    (1, Value::Bytes(b)) => s.input = b.to_vec(),
                    (2, Value::Bytes(b)) => s.expected = b.to_vec(),
                    _ => {}
                }
            }
            out.push(s);
        }
    }
    Ok(())
}

/// A field's value by its wire type. Groups, which these messages never
/// hold, are refused.
enum Value<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Reader<'a> {
        Reader { buf, at: 0 }
    }

    fn truncated(&self) -> crate::status::Error {
        invalid(format!("SentencePiece model: the protobuf message ends inside a field at byte {}", self.at))
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in 0..10 {
            let Some(&b) = self.buf.get(self.at) else { return Err(self.truncated()) };
            self.at += 1;
            v |= ((b & 0x7f) as u64) << (7 * shift);
            if b < 0x80 {
                return Ok(v);
            }
        }
        Err(invalid(format!("SentencePiece model: a varint longer than ten bytes at byte {}", self.at)))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.buf.len()).ok_or_else(|| self.truncated())?;
        let b = &self.buf[self.at..end];
        self.at = end;
        Ok(b)
    }

    /// The next field's number and value, or None at the end.
    fn next(&mut self) -> Result<Option<(u32, Value<'a>)>> {
        if self.at == self.buf.len() {
            return Ok(None);
        }
        // A key is 32 bits, as protobuf reads it: a longer one is not a
        // field whose number is cut short.
        let key = self.varint()?;
        let key = u32::try_from(key)
            .map_err(|_| invalid(format!("SentencePiece model: a field key past 32 bits at byte {}", self.at)))?;
        let field = key >> 3;
        if field == 0 {
            return Err(invalid(format!("SentencePiece model: field number 0 at byte {}", self.at)));
        }
        let value = match key & 7 {
            0 => Value::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Value::Fixed64
            }
            2 => {
                let n = self.varint()?;
                let n = usize::try_from(n).map_err(|_| self.truncated())?;
                Value::Bytes(self.take(n)?)
            }
            5 => {
                let b = self.take(4)?;
                Value::Fixed32(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            w => return Err(invalid(format!("SentencePiece model: wire type {w} at byte {}", self.at))),
        };
        Ok(Some((field, value)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_hold_when_a_field_is_absent() {
        let m = ModelProto::parse(&[]).unwrap();
        assert!(m.pieces.is_empty());
        assert_eq!(m.trainer.model_type, ModelType::Unigram);
        assert!(m.normalizer.add_dummy_prefix && m.normalizer.remove_extra_whitespaces);
        assert!(m.normalizer.escape_whitespaces && m.denormalizer.is_none());
        assert_eq!(m.trainer.unk_surface, " \u{2047} ".as_bytes());
    }

    #[test]
    fn a_piece_with_an_unknown_type_is_normal_and_a_later_value_wins() {
        // pieces { piece: "ab" score: 1.5 type: 9 type: ... }
        let score = 1.5f32.to_bits().to_le_bytes();
        let mut inner = vec![0x0a, 2, b'a', b'b', 0x15];
        inner.extend_from_slice(&score);
        inner.extend_from_slice(&[0x18, 9]);
        let mut bytes = vec![0x0a, inner.len() as u8];
        bytes.extend_from_slice(&inner);
        // trainer_spec { model_type: BPE } then trainer_spec { byte_fallback: true }:
        // the two are merged.
        bytes.extend_from_slice(&[0x12, 2, 0x18, 2, 0x12, 3, 0x98, 0x02, 1]);
        let m = ModelProto::parse(&bytes).unwrap();
        assert_eq!(m.pieces[0].piece, b"ab");
        assert_eq!(m.pieces[0].score, 1.5);
        assert_eq!(m.pieces[0].kind, PieceType::Normal);
        assert_eq!(m.trainer.model_type, ModelType::Bpe);
        assert!(m.trainer.byte_fallback);
    }

    #[test]
    fn a_truncated_message_is_refused() {
        assert!(ModelProto::parse(&[0x0a, 5, 1]).is_err());
        assert!(ModelProto::parse(&[0x0a]).is_err());
        assert!(ModelProto::parse(&[0x0b]).is_err());
    }

    #[test]
    fn a_varint_that_does_not_end_is_refused() {
        // trainer_spec's varint field cut short, at the end and inside.
        assert!(ModelProto::parse(&[0x12, 2, 0x18, 0x80]).is_err());
        assert!(ModelProto::parse(&[0x08, 0x80]).is_err());
        // Eleven bytes of continuation.
        let mut bytes = vec![0x08];
        bytes.extend_from_slice(&[0xff; 10]);
        bytes.push(0);
        assert!(ModelProto::parse(&bytes).is_err());
        // A key of ten bytes.
        assert!(ModelProto::parse(&[0xff; 10]).is_err());
    }

    #[test]
    fn a_length_past_the_message_is_refused_without_allocating_it() {
        // pieces of 2^32 - 1 bytes, of 2^64 - 1 bytes, and one byte past.
        assert!(ModelProto::parse(&[0x0a, 0xff, 0xff, 0xff, 0xff, 0x0f, 0]).is_err());
        let mut bytes = vec![0x0a];
        bytes.extend_from_slice(&[0xff; 9]);
        bytes.push(0x01);
        assert!(ModelProto::parse(&bytes).is_err());
        assert!(ModelProto::parse(&[0x0a, 3, 0x0a, 1, b'a']).is_ok());
        assert!(ModelProto::parse(&[0x0a, 3, 0x0a, 2, b'a']).is_err());
        // Inside a nested message.
        assert!(ModelProto::parse(&[0x1a, 2, 0x12, 9]).is_err());
    }

    #[test]
    fn a_key_past_32_bits_is_refused() {
        // Field 2^29 + 1 with wire type 2 would be field 1 cut to 32 bits.
        let key = ((1u64 << 32) | 1) << 3 | 2;
        let mut bytes = Vec::new();
        let mut v = key;
        while v >= 0x80 {
            bytes.push(v as u8 | 0x80);
            v >>= 7;
        }
        bytes.push(v as u8);
        bytes.extend_from_slice(&[2, 0x0a, 0]);
        assert!(ModelProto::parse(&bytes).is_err());
    }

    #[test]
    fn a_field_of_another_wire_type_is_skipped_and_groups_are_refused() {
        // pieces { piece: varint 7, score: varint 3, type: fixed32 } and
        // trainer_spec { model_type: fixed64 }: each skipped, defaults kept.
        let m =
            ModelProto::parse(&[0x0a, 9, 0x08, 7, 0x10, 3, 0x1d, 4, 0, 0, 0, 0x12, 9, 0x19, 2, 0, 0, 0, 0, 0, 0, 0])
                .unwrap();
        assert!(m.pieces[0].piece.is_empty());
        assert_eq!(m.pieces[0].score, 0.0);
        assert_eq!(m.pieces[0].kind, PieceType::Normal);
        assert_eq!(m.trainer.model_type, ModelType::Unigram);
        // Start group, end group and the two wire types protobuf lacks.
        for wire in [3, 4, 6, 7] {
            assert!(ModelProto::parse(&[0x08 | wire, 0]).is_err(), "wire type {wire}");
        }
        // Field number 0.
        assert!(ModelProto::parse(&[0x02, 0]).is_err());
    }
}
