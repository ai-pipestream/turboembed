//! The one rule of an identity that withholds its source (#2403, #2408).
//!
//! An unkeyed SHA-256 names its input to anyone who can guess it: for a
//! short source (an address, a phone number, an id) the digest of the
//! source text, of the raw bytes, or of the graph bytes that hold that
//! text, gives the value back by hashing candidates. So a digest leaves
//! only with the text it is a digest of. Every answer that names its
//! input and its result asks this module, on both sides of the wire:
//!
//! - what is written: [`identities`] for a projection, [`input`] and
//!   [`result`] for a receipt made without one, [`view`] for each listed
//!   view;
//! - what is read back: [`check_input`] and [`check_result`], which hold
//!   an identity to all or nothing and a shown result to its digest and
//!   the served versions, and [`check_source_view`], which holds the
//!   source view's listed digest to the input's.
//!
//! The typed projection (`InspectDocumentGraph`, the findings of a
//! withheld graph, sink items), the search projection
//! (`ProjectSearchDocument`) and the processing receipt of either and of
//! the progressive stream all go through here, so the rule cannot differ
//! between them. The one thing that does differ is a parameter,
//! [`WithheldLengths`]: whether the size of a withheld text is itself
//! part of the answer.

use turbo_analysis::export::FORMAT_VERSION;
use turbo_analysis::{Document, View};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::{InputBinding, InputIdentity, ResultIdentity};
use crate::results::disclosure::Disclosure;
use crate::results::typed::identity::{ENVELOPE_SCHEMA_VERSION, input_identity, result_identity};

/// What an answer says of the size of a text it withholds. A digest of
/// that text never leaves; its length does or does not by the answer's
/// own contract, which each surface names here and nowhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WithheldLengths {
    /// The findings contract of `InspectDocumentGraph`, and of every
    /// answer and receipt made by the same projection: the finding spans,
    /// the edit chain's offsets and the assessed ranges are counted in
    /// the withheld text, so its length is part of the answer and stays
    /// (`raw_bytes_len`, `source_text_len`, a view's `text_len`).
    Kept,
    /// The contract of `ProjectSearchDocument`: the answer carries no
    /// span of the source, so the lengths go with the digests
    /// (`raw_bytes_len`, `source_text_len` and a withheld view's
    /// `utf8_bytes` are 0). A rewrite that does not keep length (a fixed
    /// replacement) would otherwise tell two values apart by their size.
    Zeroed,
}

/// The input and result identity of an answer about `doc`, whose TGRF
/// bytes are `graph`, under `disclosure`: whole when the graph of `doc`
/// may leave, else as [`input`] and [`result`] withhold them. `binding`
/// is the caller's, copied as given either way.
pub fn identities(
    doc: &Document,
    binding: Option<&InputBinding>,
    graph: &[u8],
    disclosure: &Disclosure,
    lengths: WithheldLengths,
) -> (InputIdentity, ResultIdentity) {
    let shown = disclosure.graph(doc);
    (input(doc, binding, shown, lengths), result(graph, shown))
}

/// The input identity of `doc`: whole when its source is `shown`, else
/// the binding with no digest, and its two lengths as `lengths` says.
pub fn input(doc: &Document, binding: Option<&InputBinding>, shown: bool, lengths: WithheldLengths) -> InputIdentity {
    let mut input = input_identity(doc, binding);
    if !shown {
        input.raw_bytes_digest.clear();
        input.source_text_digest.clear();
        if lengths == WithheldLengths::Zeroed {
            input.raw_bytes_len = 0;
            input.source_text_len = 0;
        }
    }
    input
}

/// The identity of the result whose TGRF bytes are `graph`: whole when
/// the graph is `shown`, else the format and schema versions alone. The
/// graph holds the source text, so its digest, the id that is that
/// digest's hexadecimal, and its length stay with it under either
/// [`WithheldLengths`]. A withheld graph is not hashed.
pub fn result(graph: &[u8], shown: bool) -> ResultIdentity {
    if shown {
        return result_identity(graph);
    }
    ResultIdentity {
        graph_format_version: u32::from(FORMAT_VERSION),
        envelope_schema_version: ENVELOPE_SCHEMA_VERSION,
        ..ResultIdentity::default()
    }
}

/// The UTF-8 length and the text digest `view` is listed with: both when
/// its text is `shown`, else no digest, and its length as `lengths` says.
pub fn view(view: &View, shown: bool, lengths: WithheldLengths) -> (u32, Vec<u8>) {
    let len = view.text().len() as u32;
    match (shown, lengths) {
        (true, _) => (len, view.text_digest().as_bytes().to_vec()),
        (false, WithheldLengths::Kept) => (len, Vec::new()),
        (false, WithheldLengths::Zeroed) => (0, Vec::new()),
    }
}

/// Whether `input` withholds its source: it carries no digest of it.
/// Read on an identity [`check_input`] accepted, or one made here.
pub fn is_withheld(input: &InputIdentity) -> bool {
    input.source_text_digest.is_empty()
}

fn bad(message: String) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

/// Refuses half an input identity: one digest without the other, or,
/// where `lengths` says a withheld text gives no length, a length beside
/// no digest. `at` is the path of the message `input` is on, such as
/// `provenance.`, for the refusal. Returns whether the source is
/// withheld. The 32 bytes of a digest that is there are the caller's to
/// check.
pub fn check_input(at: &str, input: &InputIdentity, lengths: WithheldLengths) -> Result<bool, Failure> {
    if input.raw_bytes_digest.is_empty() != input.source_text_digest.is_empty() {
        return Err(bad(format!(
            "{at}input carries one of the two input digests (raw_bytes_digest, source_text_digest) without the other"
        )));
    }
    let withheld = is_withheld(input);
    if withheld && lengths == WithheldLengths::Zeroed && (input.raw_bytes_len != 0 || input.source_text_len != 0) {
        return Err(bad(format!("{at}input gives the length of a text and carries no digest of it")));
    }
    Ok(withheld)
}

/// Refuses a result identity that does not go with its input's, or is
/// not one the server makes: `withheld` is what [`check_input`]
/// returned. A result with no digest names no graph by id or length
/// either, and the result's digest is there exactly when the input's
/// are. A shown result's digest is 32 bytes and its id is that digest's
/// lowercase hexadecimal. Either way it carries the served graph format
/// and envelope schema versions. That the digest is the one of a
/// particular graph is [`crate::results::typed::identity::verify_result`]'s.
pub fn check_result(at: &str, result: &ResultIdentity, withheld: bool) -> Result<(), Failure> {
    if result.content_digest.is_empty() && (!result.result_id.is_empty() || result.graph_bytes_len != 0) {
        return Err(bad(format!("{at}result names a graph by id or length and carries no content_digest")));
    }
    if result.content_digest.is_empty() != withheld {
        return Err(bad(format!("{at}input and result digests are given one without the other")));
    }
    if !withheld {
        let digest = &result.content_digest;
        if digest.len() != 32 {
            return Err(bad(format!("{at}result.content_digest is {} bytes, not the 32 of a SHA-256", digest.len())));
        }
        if !is_hex_of(&result.result_id, digest) {
            return Err(bad(format!("{at}result.result_id is not the hexadecimal of {at}result.content_digest")));
        }
    }
    if result.envelope_schema_version != ENVELOPE_SCHEMA_VERSION {
        return Err(bad(format!(
            "{at}result envelope schema version {} is not the served {ENVELOPE_SCHEMA_VERSION}",
            result.envelope_schema_version
        )));
    }
    if result.graph_format_version != u32::from(FORMAT_VERSION) {
        return Err(bad(format!(
            "{at}result graph format version {} is not the served {FORMAT_VERSION}",
            result.graph_format_version
        )));
    }
    Ok(())
}

/// Whether `id` is the lowercase hexadecimal of `digest`, two digits a
/// byte.
fn is_hex_of(id: &str, digest: &[u8]) -> bool {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    id.len() == 2 * digest.len()
        && id
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .zip(digest)
            .all(|(&[hi, lo], &b)| hi == DIGITS[usize::from(b >> 4)] && lo == DIGITS[usize::from(b & 15)])
}

/// Refuses a source view digest that does not go with `input`:
/// `digest` is the text digest the answer lists for its source view
/// (empty when it lists none) and `withheld` is what [`check_input`]
/// returned. A withheld source is listed with no digest; a shown one
/// with the input's own `source_text_digest`. `what` names the field
/// for the refusal.
pub fn check_source_view(what: &str, digest: &[u8], input: &InputIdentity, withheld: bool) -> Result<(), Failure> {
    if withheld && !digest.is_empty() {
        return Err(bad(format!("{what} is listed and the input digests are withheld")));
    }
    if !withheld && digest != input.source_text_digest.as_slice() {
        return Err(bad(format!("{what} is not the input's source_text_digest")));
    }
    Ok(())
}
