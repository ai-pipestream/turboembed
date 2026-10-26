//! An upload's body cut into the request messages of its RPC as it
//! arrives: frames in the Connect envelope layout, `[flags: u8 = 0]
//! [length: u32 BE][payload]`, each a message, and for ImportDictionary
//! one header frame and then the file's raw bytes, cut into `chunk`
//! messages of `FRAME_BYTES`. A header or document is wrapped in its
//! request's oneof field as bytes, never decoded. At most one frame, and
//! the body chunk it arrived in, is held at a time.

use bytes::{Buf, BufMut, Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use tonic::Code;

use crate::limits::{FRAME_BYTES, MAX_FRAME_BYTES, MAX_STREAM_BYTES};

/// How a method's body becomes its messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Each frame is a request message (IndexDocuments, StreamDocuments).
    Messages,
    /// The first frame is the header (field 1), each after a document
    /// (field 2): LearnVocabulary.
    HeaderThenDocuments,
    /// The first frame is the header (field 1), then raw bytes, cut into
    /// chunks (field 2): ImportDictionary.
    HeaderThenRaw,
}

impl Kind {
    pub(crate) fn of(path: &str) -> Kind {
        match path.rsplit('/').next() {
            Some("ImportDictionary") => Kind::HeaderThenRaw,
            Some("LearnVocabulary") => Kind::HeaderThenDocuments,
            _ => Kind::Messages,
        }
    }
}

/// Why an upload is refused: the gRPC code, and 413 in place of its HTTP
/// status when the body is over the streamed ceiling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Refusal {
    pub code: Code,
    pub message: String,
    pub too_large: bool,
}

impl Refusal {
    pub(crate) fn new(code: Code, message: impl Into<String>) -> Refusal {
        Refusal { code, message: message.into(), too_large: false }
    }

    pub(crate) fn too_large() -> Refusal {
        let message = format!("the upload is over {MAX_STREAM_BYTES} bytes");
        Refusal { code: Code::ResourceExhausted, message, too_large: true }
    }
}

/// The messages of one upload, read from `body` as they are asked for.
pub(crate) struct Frames<B> {
    body: B,
    kind: Kind,
    buf: BytesMut,
    frames: u64,
    raw: bool,
    forwarded: u64,
    ended: bool,
}

impl<B, E> Frames<B>
where
    B: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    pub(crate) fn new(body: B, kind: Kind) -> Frames<B> {
        Frames { body, kind, buf: BytesMut::new(), frames: 0, raw: false, forwarded: 0, ended: false }
    }

    /// The next request message; `None` at the body's end.
    pub(crate) async fn next(&mut self) -> Result<Option<Bytes>, Refusal> {
        loop {
            if let Some(m) = self.cut()? {
                self.forwarded += 5 + m.len() as u64;
                if self.forwarded > MAX_STREAM_BYTES {
                    return Err(Refusal::too_large());
                }
                return Ok(Some(m));
            }
            if self.ended {
                return match (self.raw, self.buf.is_empty()) {
                    (_, true) => Ok(None),
                    (true, false) => Ok(Some(self.chunk(self.buf.len()))),
                    (false, false) => Err(Refusal::new(Code::InvalidArgument, "the body ends inside a frame")),
                };
            }
            match self.body.next().await {
                Some(Ok(data)) => self.buf.extend_from_slice(&data),
                Some(Err(e)) => return Err(Refusal::new(Code::InvalidArgument, format!("the request body: {e}"))),
                None => self.ended = true,
            }
        }
    }

    /// A whole message from what is held, if there is one yet.
    fn cut(&mut self) -> Result<Option<Bytes>, Refusal> {
        if self.raw {
            return Ok((self.buf.len() >= FRAME_BYTES).then(|| self.chunk(FRAME_BYTES)));
        }
        if self.buf.len() < 5 {
            return Ok(None);
        }
        if self.buf[0] != 0 {
            let msg =
                format!("frame {} has flags {}: compressed or other frames are refused", self.frames, self.buf[0]);
            return Err(Refusal::new(Code::InvalidArgument, msg));
        }
        let len = u32::from_be_bytes([self.buf[1], self.buf[2], self.buf[3], self.buf[4]]) as usize;
        if len > MAX_FRAME_BYTES {
            let msg = format!("frame {} of {len} bytes is over {MAX_FRAME_BYTES}", self.frames);
            return Err(Refusal::new(Code::ResourceExhausted, msg));
        }
        if self.buf.len() < 5 + len {
            self.buf.reserve(5 + len - self.buf.len());
            return Ok(None);
        }
        self.buf.advance(5);
        let payload = self.buf.split_to(len).freeze();
        let first = self.frames == 0;
        self.frames += 1;
        Ok(Some(match self.kind {
            Kind::Messages => payload,
            Kind::HeaderThenDocuments => wrap(if first { 1 } else { 2 }, &payload),
            Kind::HeaderThenRaw => {
                self.raw = true;
                wrap(1, &payload)
            }
        }))
    }

    fn chunk(&mut self, n: usize) -> Bytes {
        let raw = self.buf.split_to(n);
        wrap(2, &raw)
    }
}

/// `payload` as length-delimited field `field` of a message.
fn wrap(field: u8, payload: &[u8]) -> Bytes {
    let mut out = BytesMut::with_capacity(payload.len() + 11);
    out.put_u8((field << 3) | 2);
    let mut n = payload.len() as u64;
    while n >= 0x80 {
        out.put_u8((n as u8) | 0x80);
        n >>= 7;
    }
    out.put_u8(n as u8);
    out.put_slice(payload);
    out.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    fn frame(flags: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![flags];
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// Every message of `body`, delivered in pieces of `piece` bytes.
    async fn all(body: &[u8], piece: usize, kind: Kind) -> Result<Vec<Vec<u8>>, Refusal> {
        let pieces: Vec<Result<Bytes, Infallible>> =
            body.chunks(piece).map(|c| Ok(Bytes::copy_from_slice(c))).collect();
        let mut f = Frames::new(futures_util::stream::iter(pieces), kind);
        let mut out = Vec::new();
        while let Some(m) = f.next().await? {
            out.push(m.to_vec());
        }
        Ok(out)
    }

    #[tokio::test]
    async fn frames_become_messages_however_the_body_arrives() {
        let body = [frame(0, b"one"), frame(0, b""), frame(0, b"three")].concat();
        for piece in [1, 2, 7, 1000] {
            assert_eq!(all(&body, piece, Kind::Messages).await.unwrap(), [b"one".to_vec(), vec![], b"three".to_vec()]);
            let learn = all(&body, piece, Kind::HeaderThenDocuments).await.unwrap();
            assert_eq!(learn, [b"\x0a\x03one".to_vec(), b"\x12\x00".to_vec(), b"\x12\x05three".to_vec()]);
        }
        let body = [frame(0, b"head"), b"raw bytes".to_vec()].concat();
        let import = all(&body, 3, Kind::HeaderThenRaw).await.unwrap();
        assert_eq!(import, [b"\x0a\x04head".to_vec(), b"\x12\x09raw bytes".to_vec()]);
        assert_eq!(all(b"", 1, Kind::Messages).await.unwrap(), Vec::<Vec<u8>>::new());
    }

    #[tokio::test]
    async fn raw_bytes_are_cut_into_frame_sized_chunks() {
        let raw = vec![7u8; FRAME_BYTES * 2 + 3];
        let body = [frame(0, b"h"), raw].concat();
        let got = all(&body, 65536, Kind::HeaderThenRaw).await.unwrap();
        let sizes: Vec<usize> = got.iter().map(Vec::len).collect();
        assert_eq!(sizes, [3, FRAME_BYTES + 5, FRAME_BYTES + 5, 5]);
    }

    #[tokio::test]
    async fn flagged_truncated_and_oversized_frames_are_refused() {
        let e = all(&frame(1, b"x"), 10, Kind::Messages).await.unwrap_err();
        assert_eq!(e.code, Code::InvalidArgument);
        let e = all(&frame(0, b"xyz")[..6], 10, Kind::Messages).await.unwrap_err();
        assert_eq!((e.code, e.message.as_str()), (Code::InvalidArgument, "the body ends inside a frame"));
        let mut over = vec![0u8];
        over.extend_from_slice(&(MAX_FRAME_BYTES as u32 + 1).to_be_bytes());
        let e = all(&over, 10, Kind::Messages).await.unwrap_err();
        assert!(e.code == Code::ResourceExhausted && !e.too_large);
        assert_eq!(wrap(2, &vec![0u8; 300])[..3], [0x12, 0xac, 0x02]);
    }
}
