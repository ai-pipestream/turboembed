//! Policy inputs contain only opaque identifiers. Implementations are trusted
//! application capabilities, supplied after authenticating the caller.

use super::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrincipalId(pub [u8; 16]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodingId(pub [u8; 16]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Begin,
    Publish,
    Read,
    List,
    Delete,
    Abort,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub principal: PrincipalId,
    /// Random opaque 128-bit artifact identifier; absent only for listing.
    pub artifact: Option<[u8; 16]>,
    pub operation: Operation,
}

/// Checks every operation, including publication after a long-running write.
/// A caller cannot choose its authenticated principal through a request method.
/// Implementations must deny absent grants and must not include inputs in errors.
pub trait Authorization: Send + Sync {
    fn authorize(&self, request: &Request) -> Result<(), Error>;
}

/// The application's authoritative retention ledger, outside mutable provider
/// metadata. It enforces expiry, legal holds and minimum retention before delete.
/// Failure to consult the ledger must return an error, never permission.
pub trait Retention: Send + Sync {
    fn allow_delete(&self, request: &Request) -> Result<(), Error>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Succeeded,
    Failed,
    /// Publication occurred but durability or staging cleanup was unsuccessful.
    PublishedWithWarning,
}

/// Persist an intent before access; failure prevents the operation. Completion
/// callbacks cannot veto an already committed operation. A deployment must use
/// a reliable sink or reconcile incomplete intents, never log document content.
pub trait Audit: Send + Sync {
    fn intent(&self, request: &Request) -> Result<(), Error>;
    fn outcome(&self, request: &Request, outcome: Outcome);
}
