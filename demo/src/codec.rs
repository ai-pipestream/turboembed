//! A gRPC codec that leaves messages as the bytes they are. The proxy
//! hands the page's protobuf to the server and the server's protobuf to
//! the page without decoding either: gRPC's own framing, message size
//! limits and compression handling still apply, so only the message type
//! is unknown to the demo.

use bytes::{Buf, BufMut, Bytes};
use tonic::Status;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};

#[derive(Debug, Clone, Copy, Default)]
pub struct Raw;

impl Codec for Raw {
    type Encode = Bytes;
    type Decode = Bytes;
    type Encoder = Raw;
    type Decoder = Raw;

    fn encoder(&mut self) -> Raw {
        Raw
    }

    fn decoder(&mut self) -> Raw {
        Raw
    }
}

impl Encoder for Raw {
    type Item = Bytes;
    type Error = Status;

    fn encode(&mut self, item: Bytes, dst: &mut EncodeBuf<'_>) -> Result<(), Status> {
        dst.put(item);
        Ok(())
    }
}

impl Decoder for Raw {
    type Item = Bytes;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<Bytes>, Status> {
        Ok(Some(src.copy_to_bytes(src.remaining())))
    }
}
