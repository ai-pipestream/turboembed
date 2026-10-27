//! The demo's message limits, each matched to the server's default
//! (docs/grpc.md). A Connect request carries one message, so its body is
//! held whole up to the request limit and refused at once, before it is
//! read, when it says it is larger. The streamed upload ceiling (2 GiB,
//! the server's `--max-stream-bytes`) is for uploads in frames, which the
//! demo does not forward yet.

/// The largest request message forwarded: the server's default
/// `--max-message-bytes`, 64 MiB.
pub const MAX_REQUEST_BYTES: usize = 64 << 20;

/// The largest reply message forwarded: the server's default reply limit,
/// 256 MiB.
pub const MAX_REPLY_BYTES: usize = 256 << 20;

/// What the body of a request may be: the message, and for the streaming
/// protocol its five-byte envelope.
pub(crate) fn body_limit(enveloped: bool) -> usize {
    MAX_REQUEST_BYTES + if enveloped { 5 } else { 0 }
}

/// Why a body declaring `length` bytes is refused before it is read.
pub(crate) fn declared(length: Option<u64>, limit: usize) -> Result<(), String> {
    match length {
        Some(n) if n > limit as u64 => Err(format!("the request is over {limit} bytes")),
        _ => Ok(()),
    }
}

/// Why an envelope declaring a message of `length` bytes is refused
/// before the message is read.
pub(crate) fn enveloped(length: u32) -> Result<(), String> {
    match length as usize > MAX_REQUEST_BYTES {
        true => Err(format!("the request message is over {MAX_REQUEST_BYTES} bytes")),
        false => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declared_size_over_the_limit_is_refused() {
        assert!(declared(None, 10).is_ok());
        assert!(declared(Some(10), 10).is_ok());
        assert!(declared(Some(11), 10).is_err());
        assert!(enveloped(MAX_REQUEST_BYTES as u32).is_ok());
        assert!(enveloped(MAX_REQUEST_BYTES as u32 + 1).is_err());
        assert_eq!(body_limit(true), body_limit(false) + 5);
    }
}
