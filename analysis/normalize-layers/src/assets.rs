//! Binary payload rows (`turbo:assets`).

use turbo_analysis::{Layer, Span, Spans, TextSpans, ViewId};
use turbo_analysis_layers::Strs;

use crate::{AssetEncoding, AssetFormat, subtype};

/// Binary payloads embedded in a text: base64 runs, `data:` URIs, PEM
/// envelopes, JSON Web Tokens and, when the detector is configured for
/// them, hex dumps. Each row is one asset:
///
/// - `spans`: the whole asset (a `data:` URI from its scheme, a PEM
///   envelope from its begin line to its end line, a token from its first
///   segment to its last);
/// - `payload`: the encoded bytes inside it, in the same view (the base64
///   body with its line breaks, a token's claims segment);
/// - `format` and `declared_type`: what the header identifies, and the
///   media type a `data:` URI declared (empty when the asset declared
///   none); [`Assets::format_name`] and [`Assets::media_type`] give the
///   strings OpenNLP reports;
/// - `encoding`: how the payload is written;
/// - `decoded_len`: the number of bytes the payload decodes to;
/// - `width` and `height`: the pixel size, for a PNG or GIF whose header
///   states a positive one. OpenNLP writes -1 for an unknown dimension;
///   here it is `None`, so a column cannot hold a negative or zero size.
///
/// The spans are in text order and do not overlap.
pub struct Assets {
    pub spans: Spans<Assets>,
    pub payload: Vec<Span>,
    pub format: Vec<AssetFormat>,
    pub declared_type: Strs,
    pub encoding: Vec<AssetEncoding>,
    pub decoded_len: Vec<u64>,
    pub width: Vec<Option<u32>>,
    pub height: Vec<Option<u32>>,
}

impl Assets {
    pub fn new(view: ViewId) -> Assets {
        Assets {
            spans: Spans::new(view),
            payload: Vec::new(),
            format: Vec::new(),
            declared_type: Strs::default(),
            encoding: Vec::new(),
            decoded_len: Vec::new(),
            width: Vec::new(),
            height: Vec::new(),
        }
    }

    /// The media type of asset `i`: the one its `data:` URI declared, or
    /// else the one its format implies.
    pub fn media_type(&self, i: usize) -> &str {
        match self.declared_type.get(i) {
            "" => self.format[i].media_type(),
            declared => declared,
        }
    }

    /// The format name of asset `i` as OpenNLP reports it: the format's
    /// name, or for [`AssetFormat::Declared`] the subtype of the declared
    /// media type.
    pub fn format_name(&self, i: usize) -> &str {
        match self.format[i] {
            AssetFormat::Declared => subtype(self.declared_type.get(i)).unwrap_or_default(),
            format => format.name(),
        }
    }
}

impl Layer for Assets {
    const ID: &'static str = "turbo:assets";

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let columns = [
            self.payload.len(),
            self.format.len(),
            self.declared_type.len(),
            self.encoding.len(),
            self.decoded_len.len(),
            self.width.len(),
            self.height.len(),
        ];
        if columns.iter().any(|&len| len != n) {
            return Err("a payload, format, declared type, encoding, length and size for every asset".into());
        }
        if !self.spans.is_ordered() {
            return Err("assets in text order, not overlapping".into());
        }
        self.spans.check(Self::OVERLAP)?;
        for (ix, span) in self.spans.iter() {
            let i = ix.get();
            let payload = self.payload[i];
            if span.start >= span.end || payload.start >= payload.end || !span.contains(payload) {
                return Err(format!("asset {i}: a non-empty payload {payload:?} inside its span {span:?}"));
            }
            let declared = self.declared_type.get(i);
            let declares = !declared.is_empty() || self.format[i] == AssetFormat::Declared;
            if declares && subtype(declared).is_none() {
                return Err(format!("asset {i}: a declared media type {declared:?} of a type and subtype"));
            }
            if self.format[i] == AssetFormat::Unknown && !declared.is_empty() {
                return Err(format!("asset {i}: an unknown run declares no media type, got {declared:?}"));
            }
            let positive = |d: Option<u32>| d.is_none_or(|d| d > 0 && d <= i32::MAX as u32);
            let (w, h) = (self.width[i], self.height[i]);
            if w.is_some() != h.is_some() || !positive(w) || !positive(h) {
                return Err(format!("asset {i}: both dimensions or neither, positive, got {w:?} x {h:?}"));
            }
        }
        Ok(())
    }
}
