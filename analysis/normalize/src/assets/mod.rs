//! Embedded assets: binary payloads that survived text extraction, found
//! by [`AssetDetector`] and folded into short captions by [`AssetFold`].
//!
//! A port of OpenNLP's `embedded-assets` add-on (`CursorAssetDetector`,
//! `KnownMagics`, `SvgHeader`, `JsonObjectParser`,
//! `Base64PayloadInputStream`, `AssetAnnotator`, `AssetFolder`). The
//! detector finds:
//!
//! - bare base64 runs of at least 32 characters, standard or URL-safe,
//!   possibly wrapped at 64 or 76 columns, whose decoded header starts
//!   with a known file signature;
//! - `data:` URIs with a `;base64,` payload of any length, identified by
//!   their header or by the media type they declare;
//! - RFC 7468 PEM envelopes (certificates, keys, CRLs, CMS and others);
//! - compact JSON Web Tokens whose header and claims decode to JSON
//!   objects;
//! - text formats whose payload is base64 too: an SVG root element,
//!   an iCalendar opening line, a WebVTT header.
//!
//! Three additions are off by default, so the defaults give OpenNLP's
//! output: hex dumps ([`AssetDetector::hex_dumps`]), base64 runs with no
//! known signature ([`AssetDetector::report_unknown_base64`]) and
//! quoted-printable soft line breaks inside a payload
//! ([`AssetDetector::soft_line_breaks`]).
//!
//! Nothing here allocates a whole decoded payload: a header is at most 45
//! bytes, and the SVG and calendar checks read the payload through a
//! streaming decoder.

use turbo_analysis::{Annotator, Error, Refs, Span, View, ViewId};
use turbo_normalize_layers::{AssetEncoding, AssetFormat, Assets, subtype};

mod detect;
mod fold;
mod hex;
mod json;
mod magic;
mod payload;
mod sniff;
mod svg;
mod unknown;

pub use fold::{AssetFold, CaptionFn, write_caption};
pub use magic::{FIRST_BYTES, MAGICS, Magic, known as magic_format};
pub use payload::DecodeError;

/// One embedded asset. Offsets are UTF-8 byte offsets of the text it was
/// found in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Asset {
    /// The whole asset: a `data:` URI from `data:` on, a PEM envelope from
    /// its begin line to the end of its end line, a token from its first
    /// segment to its last, a bare run or a hex dump alone.
    pub span: Span,
    /// The encoded bytes inside `span`, with any line breaks between them.
    /// For a token, its claims segment.
    pub payload: Span,
    pub format: AssetFormat,
    /// The media type a `data:` URI declared, when [`subtype`] reads a type
    /// and subtype from it; OpenNLP then reports it instead of the format's
    /// own.
    pub declared: Option<Span>,
    pub encoding: AssetEncoding,
    /// The number of bytes the payload decodes to.
    pub decoded_len: u64,
    /// The pixel size a PNG or GIF header states, when both are positive.
    pub width: Option<u32>,
    pub height: Option<u32>,
}

impl Asset {
    /// The media type: the declared one, or the format's.
    pub fn media_type<'t>(&self, text: &'t str) -> &'t str {
        match self.declared {
            Some(declared) => &text[declared.range()],
            None => self.format.media_type(),
        }
    }

    /// The format name OpenNLP reports: the format's name, or for a
    /// `data:` URI no signature identified, the subtype of its declared
    /// media type.
    pub fn format_name<'t>(&self, text: &'t str) -> &'t str {
        match self.format {
            AssetFormat::Declared => subtype(self.media_type(text)).unwrap_or_default(),
            format => format.name(),
        }
    }

    /// The decoded payload, read from the text the asset was found in
    /// (OpenNLP's `EmbeddedAsset.decode`).
    pub fn decode(&self, text: &str) -> Result<Vec<u8>, DecodeError> {
        let (start, end) = (self.payload.start as usize, self.payload.end as usize);
        match self.encoding {
            AssetEncoding::Hex => payload::decode_hex(text, start, end),
            AssetEncoding::Base64 | AssetEncoding::Base64Url => payload::decode_base64(text, start, end),
        }
    }
}

/// Finds the embedded assets of a text, OpenNLP's `CursorAssetDetector`.
///
/// The assets are in text order and do not overlap: the scan resumes
/// after each asset it reports. The default value has every addition off
/// and reports what OpenNLP reports; each field turns one on.
///
/// As an `Annotator` it reads a `View` and writes an [`Assets`] layer
/// over it, with one row per asset.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AssetDetector {
    /// Also report hex dumps of 32 or more bytes whose decoded header is
    /// recognized, with [`AssetEncoding::Hex`].
    pub hex_dumps: bool,
    /// Also report base64 runs of 32 or more characters that no signature
    /// identifies, as [`AssetFormat::Unknown`], when their characters look
    /// like encoded data rather than words (see `unknown.rs`).
    pub report_unknown_base64: bool,
    /// Read a quoted-printable soft line break (`=` before CRLF or LF)
    /// inside a bare run or a `data:` URI payload as part of the payload.
    pub soft_line_breaks: bool,
}

impl AssetDetector {
    /// The detector with OpenNLP's behaviour.
    pub fn new() -> AssetDetector {
        AssetDetector::default()
    }

    /// The assets of `text`, in text order.
    pub fn detect(&self, text: &str) -> Vec<Asset> {
        let mut found = Vec::new();
        self.scan(text, &mut |asset| found.push(asset));
        found
    }

    /// Reports each asset of `text` to `sink`, in text order.
    pub fn scan(&self, text: &str, sink: &mut dyn FnMut(Asset)) {
        detect::scan(self, text, sink);
    }

    /// The assets of `text` as a layer over `view`.
    pub fn layer(&self, text: &str, view: ViewId) -> Assets {
        let mut layer = Assets::new(view);
        self.scan(text, &mut |asset| {
            layer.spans.push(asset.span);
            layer.payload.push(asset.payload);
            layer.format.push(asset.format);
            layer.declared_type.push(asset.declared.map_or("", |d| &text[d.range()]));
            layer.encoding.push(asset.encoding);
            layer.decoded_len.push(asset.decoded_len);
            layer.width.push(asset.width);
            layer.height.push(asset.height);
        });
        layer
    }
}

impl Annotator for AssetDetector {
    const ID: &'static str = "turbo:assets";
    type Reads = (View,);
    type Writes = (Assets,);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(Assets,), Error> {
        Ok((self.layer(src.text(), src.id()),))
    }

    fn config_hash(&self) -> u64 {
        crate::hash_config(self)
    }
}
