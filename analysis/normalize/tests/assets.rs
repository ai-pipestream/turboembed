//! The embedded-asset detector and fold. The expectations of OpenNLP's 29
//! embedded-assets test classes come first, class by class, with OpenNLP's
//! UTF-16 offsets written as UTF-8 byte offsets (equal for the ASCII texts
//! they use). Then false positives with every addition on, UTF-8 offsets,
//! the three additions, the fold's alignment, and property loops over
//! generated text. Every detection goes through `Layer::check` and a
//! document (`assets_support::detect_with`), and every fold through the
//! alignment check of `assets_support::fold_checked`.
//!
//! The tests are in `assets/`, one file for each group of classes or
//! additions. The helpers here are the ones more than one group uses.

#[path = "assets/additions.rs"]
mod additions;
mod assets_support;
#[path = "assets/detector.rs"]
mod detector;
#[path = "assets/dwf_jpeg2000.rs"]
mod dwf_jpeg2000;
#[path = "assets/fold_properties.rs"]
mod fold_properties;
#[path = "assets/forms.rs"]
mod forms;
#[path = "assets/headers.rs"]
mod headers;
#[path = "assets/known_magics.rs"]
mod known_magics;
#[path = "assets/layer.rs"]
mod layer;
#[path = "assets/media_types.rs"]
mod media_types;
#[path = "assets/payloads.rs"]
mod payloads;
#[path = "assets/svg.rs"]
mod svg;

use assets_support::*;
use turbo_analysis::{Analyzer, Annotator, Bound, Document, Key, Layer, Span, View};
use turbo_normalize::assets::{DecodeError, MAGICS, magic_format};
use turbo_normalize::{Asset, AssetDetector, AssetFold, Chain, NormalizeStep, Normalizer, hash_config, normalize};
use turbo_normalize_layers::{AssetEncoding, AssetFormat, Assets, PlaceholderKind, Placeholders};

/// `CursorAssetDetectorTest.embed`.
fn embed(run: &str) -> String {
    format!("x {run} y")
}

/// A RIFF header of form `kind` with 20 zero bytes after it.
fn riff(kind: &[u8; 4]) -> Vec<u8> {
    let mut b = vec![b'R', b'I', b'F', b'F', 100, 0, 0, 0];
    b.extend_from_slice(kind);
    b.resize(32, 0);
    b
}

/// CRLF MIME wrapping at `width`, `CursorAssetDetectorTest.wrapped`.
fn wrapped(bytes: &[u8], width: usize) -> String {
    mime(bytes, width, "\r\n")
}

/// The default detector finds no asset in the standard base64 of `bytes`,
/// alone or in an untyped data URI.
fn assert_unrecognized_default(bytes: &[u8]) {
    assert_unrecognized(&AssetDetector::new(), bytes);
}
