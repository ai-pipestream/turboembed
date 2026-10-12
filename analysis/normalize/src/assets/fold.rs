//! `AssetFold`: OpenNLP's `AssetFolder` as a normalizer, replacing each
//! embedded asset with a caption.

use std::fmt::Write as _;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Refs, View, ViewBuilder};
use turbo_normalize_layers::{PlaceholderKind, Placeholders};

use super::{Asset, AssetDetector};
use crate::placeholder::{self, Sink};
use crate::{Normalizer, hash_config};

/// The function that writes the text standing in for an asset: it gets
/// the asset and the text the asset was found in (to read the declared
/// media type or decode the payload) and appends the caption to the
/// string it is given.
pub type CaptionFn = dyn Fn(&Asset, &str, &mut String) + Send + Sync;

/// Replaces each embedded asset [`AssetDetector`] finds with a caption, by
/// default OpenNLP's `AssetFolder.caption()`: the format name, the pixel
/// size when known, and the decoded length, as in
/// `[png 640x480, 12345 bytes]` or `[pdf, 51200 bytes]`.
///
/// Each asset is one `replace` on the view builder, so its caption maps
/// back to exactly the asset's span, also when two assets touch (two
/// `data:` URIs back to back): each keeps its own edit.
///
/// As an `Annotator` it reads a `View` and writes the folded `View` and a
/// `Placeholders` layer with one `Asset` row per asset, spanning the
/// asset in the view it read.
#[derive(Clone)]
pub struct AssetFold {
    detector: AssetDetector,
    caption: Caption,
}

#[derive(Clone)]
enum Caption {
    /// `AssetFolder.caption()`.
    Default,
    /// A caller's function, named by `id` in the configuration hash.
    Custom { id: &'static str, write: Arc<CaptionFn> },
}

impl Default for AssetFold {
    fn default() -> AssetFold {
        AssetFold::new()
    }
}

impl std::fmt::Debug for AssetFold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let caption = match &self.caption {
            Caption::Default => "default",
            Caption::Custom { id, .. } => id,
        };
        f.debug_struct("AssetFold").field("detector", &self.detector).field("caption", &caption).finish()
    }
}

impl AssetFold {
    /// Folds what OpenNLP's detector finds into OpenNLP's captions.
    pub fn new() -> AssetFold {
        AssetFold::with_detector(AssetDetector::new())
    }

    /// Folds what `detector` finds.
    pub fn with_detector(detector: AssetDetector) -> AssetFold {
        AssetFold { detector, caption: Caption::Default }
    }

    /// Writes captions with `write` instead, as OpenNLP's
    /// `AssetFolder.fold(document, replacement)` does. `id` names the
    /// function in the configuration hash, so views folded with different
    /// captions are told apart.
    pub fn with_caption(
        self,
        id: &'static str,
        write: impl Fn(&Asset, &str, &mut String) + Send + Sync + 'static,
    ) -> AssetFold {
        AssetFold { caption: Caption::Custom { id, write: Arc::new(write) }, ..self }
    }

    pub fn detector(&self) -> &AssetDetector {
        &self.detector
    }

    /// Writes `text` with every asset replaced by its caption and reports
    /// each asset's span to `sink`.
    pub fn scan(&self, text: &str, out: &mut ViewBuilder<'_>, sink: &mut Sink<'_>) {
        let mut caption = String::new();
        self.detector.scan(text, &mut |asset| {
            caption.clear();
            match &self.caption {
                Caption::Default => write_caption(&asset, text, &mut caption),
                Caption::Custom { write, .. } => write(&asset, text, &mut caption),
            }
            out.keep(asset.span.start as usize);
            out.replace(asset.span.end as usize, &caption);
            sink(asset.span, PlaceholderKind::Asset);
        });
    }
}

/// OpenNLP's default caption, `AssetFolder.caption()`: `[`, the format
/// name, a space and `WxH` when the size is known, then `, `, the decoded
/// length and ` bytes]`.
pub fn write_caption(asset: &Asset, text: &str, out: &mut String) {
    out.push('[');
    out.push_str(asset.format_name(text));
    if let (Some(width), Some(height)) = (asset.width, asset.height) {
        let _ = write!(out, " {width}x{height}");
    }
    let _ = write!(out, ", {} bytes]", asset.decoded_len);
}

impl Normalizer for AssetFold {
    const ID: &'static str = "turbo:normalize:asset-fold";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        self.scan(text, out, &mut placeholder::discard);
    }

    fn config_hash(&self) -> u64 {
        let caption = match &self.caption {
            Caption::Default => None,
            Caption::Custom { id, .. } => Some(*id),
        };
        hash_config(&(self.detector, caption))
    }
}

impl Annotator for AssetFold {
    const ID: &'static str = <AssetFold as Normalizer>::ID;
    type Reads = (View,);
    type Writes = (View, Placeholders);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View, Placeholders), Error> {
        Ok(placeholder::annotate(src, |text, out, sink| self.scan(text, out, sink)))
    }

    fn config_hash(&self) -> u64 {
        <AssetFold as Normalizer>::config_hash(self)
    }
}
