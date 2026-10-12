//! The layers the normalizers write besides their views, so another
//! add-on can read what a normalizer found without depending on the
//! normalizer itself.

use turbo_analysis::{Layer, Spans, ViewId};
use turbo_analysis_layers::Strs;

mod asset_format;
pub use asset_format::{AssetEncoding, AssetFormat, subtype};

/// What a placeholder stands for.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PlaceholderKind {
    Url,
    Email,
    Hashtag,
    Mention,
    RetweetMarker,
    Emoticon,
    Number,
    Emoji,
    /// A binary payload folded away: base64, a data URL, a hex dump.
    Asset,
}

/// The stretches of a text a normalizer replaced with placeholders: a URL
/// written as a space, an e-mail address, a hashtag. The spans are in the
/// view the normalizer read, so the original text of each is at hand, and
/// the alignment of the view it wrote says where the placeholder landed.
pub struct Placeholders {
    pub spans: Spans<Placeholders>,
    pub kind: Vec<PlaceholderKind>,
}

impl Placeholders {
    pub fn new(view: ViewId) -> Placeholders {
        Placeholders { spans: Spans::new(view), kind: Vec::new() }
    }
}

impl Layer for Placeholders {
    const ID: &'static str = "turbo:placeholders";

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<(ViewId, &[u32], &[u32])> {
        let (start, end) = self.spans.columns();
        Some((self.spans.view(), start, end))
    }

    fn check(&self) -> Result<(), String> {
        if self.kind.len() != self.spans.len() {
            return Err("a kind for every placeholder".into());
        }
        if !self.spans.is_ordered() {
            return Err("placeholders in text order, not overlapping".into());
        }
        Ok(())
    }
}

/// How an emoji is put together, following UTS #51.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EmojiKind {
    /// One emoji character, with or without U+FE0F.
    Basic,
    /// A digit, `#` or `*` with U+FE0F and U+20E3.
    Keycap,
    /// Two regional indicators: a country flag.
    Flag,
    /// A base followed by a skin tone modifier.
    Modifier,
    /// U+1F3F4 and tag characters: a subdivision flag.
    Tag,
    /// Several emoji joined by U+200D.
    Zwj,
}

/// The emoji in a text, each as one whole sequence: a ZWJ family, a flag,
/// a keycap, a skin-toned hand. `name` is the CLDR short name of a
/// sequence Unicode lists as RGI, empty otherwise; `region` is the ISO
/// 3166 code a flag or subdivision flag spells, empty otherwise.
pub struct Emoji {
    pub spans: Spans<Emoji>,
    pub kind: Vec<EmojiKind>,
    pub name: Strs,
    pub region: Strs,
}

impl Emoji {
    pub fn new(view: ViewId) -> Emoji {
        Emoji { spans: Spans::new(view), kind: Vec::new(), name: Strs::default(), region: Strs::default() }
    }
}

impl Layer for Emoji {
    const ID: &'static str = "turbo:emoji";

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<(ViewId, &[u32], &[u32])> {
        let (start, end) = self.spans.columns();
        Some((self.spans.view(), start, end))
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.kind.len() != n || self.name.len() != n || self.region.len() != n {
            return Err("a kind, a name and a region for every emoji".into());
        }
        if !self.spans.is_ordered() {
            return Err("emoji in text order, not overlapping".into());
        }
        Ok(())
    }
}

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
    pub payload: Vec<turbo_analysis::Span>,
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
            AssetFormat::Declared => subtype(self.declared_type.get(i)),
            format => format.name(),
        }
    }
}

impl Layer for Assets {
    const ID: &'static str = "turbo:assets";

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<(ViewId, &[u32], &[u32])> {
        let (start, end) = self.spans.columns();
        Some((self.spans.view(), start, end))
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
        for (ix, span) in self.spans.iter() {
            let i = ix.get();
            let payload = self.payload[i];
            if span.is_empty() || payload.is_empty() || !span.contains(payload) {
                return Err(format!("asset {i}: a non-empty payload {payload:?} inside its span {span:?}"));
            }
            let declared = self.declared_type.get(i);
            if (self.format[i] == AssetFormat::Declared && declared.is_empty())
                || !(declared.is_empty() || declared.find('/').is_some_and(|slash| slash > 0))
            {
                return Err(format!("asset {i}: a declared media type {declared:?} with a subtype"));
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
