//! The layers the normalizers write besides their views, so another
//! add-on can read what a normalizer found without depending on the
//! normalizer itself.

use turbo_analysis::{Layer, Spans, ViewId};
use turbo_analysis_layers::Strs;

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
