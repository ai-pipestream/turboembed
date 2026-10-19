//! Writes the protobuf wire format of the `ModelProto` fields a trained
//! SentencePiece BPE model needs (sentencepiece_model.proto, proto2).

/// A message being written, field by field.
#[derive(Default)]
pub(crate) struct Message(Vec<u8>);

impl Message {
    fn key(&mut self, field: u32, wire: u32) {
        self.varint(((field << 3) | wire) as u64);
    }

    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }

    /// A string, bytes or embedded message field.
    pub fn bytes(&mut self, field: u32, b: &[u8]) {
        self.key(field, 2);
        self.varint(b.len() as u64);
        self.0.extend_from_slice(b);
    }

    pub fn message(&mut self, field: u32, m: Message) {
        self.bytes(field, &m.0);
    }

    /// An int32, enum or bool field. A negative int32 is ten bytes long,
    /// as protobuf writes it.
    pub fn int(&mut self, field: u32, v: i64) {
        self.key(field, 0);
        self.varint(v as u64);
    }

    pub fn float(&mut self, field: u32, v: f32) {
        self.key(field, 5);
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_are_written_as_protobuf_writes_them() {
        let mut inner = Message::default();
        inner.bytes(1, b"ab");
        inner.float(2, -1.0);
        inner.int(3, 2);
        let mut m = Message::default();
        m.message(1, inner);
        m.int(43, -1);
        m.int(4, 300);
        let mut want = vec![0x0a, 11, 0x0a, 2, b'a', b'b', 0x15, 0, 0, 0x80, 0xbf, 0x18, 2];
        // Field 43, varint: key 344 in two bytes, then -1 in ten.
        want.extend_from_slice(&[0xd8, 0x02, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]);
        want.extend_from_slice(&[0x20, 0xac, 0x02]);
        assert_eq!(m.into_bytes(), want);
    }
}
