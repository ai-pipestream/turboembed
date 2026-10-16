//! Comparing a layer on a document with its golden.
//!
//! Each shared layer type with a golden shape implements `GoldenCompare`:
//! it names the columns its golden has and lays its own columns and the
//! golden's side by side as `Items`, which one loop compares item by item
//! (span first, then token range, UTF-16 span and cut, label and value),
//! stopping at the first difference. The layer's id and anchor come from
//! its `Layer` impl, so a golden is matched to a type by the same id the
//! document uses.

mod items;
mod layers;
mod mismatch;
mod view;

use std::fmt;

use turbo_analysis::{Anchor, Document, Key, KeyId, Layer, Status};

pub use items::{CutCol, Items, Labels, Names, SpanCols, TokenCols, UnitCols, compare_items};
pub use mismatch::{Item, Mismatch};

use crate::format::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer};
use crate::sha::sha256;

/// How close a value must be to the golden's.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Tolerance {
    /// The same bits: -0.0 and 0.0 differ, and a NaN equals a NaN with the
    /// same bits.
    Exact,
    /// Within this distance, both finite; or both the same infinity, or
    /// both NaN.
    Abs(f64),
}

impl Tolerance {
    pub const fn exact() -> Tolerance {
        Tolerance::Exact
    }

    /// Within `d`, which must be finite and not negative.
    pub fn abs(d: f64) -> Tolerance {
        assert!(d.is_finite() && d >= 0.0, "a tolerance is finite and not negative, found {d}");
        Tolerance::Abs(d)
    }

    /// Whether `rust` is close enough to `golden`.
    pub fn accepts(self, rust: f64, golden: f64) -> bool {
        match self {
            Tolerance::Exact => rust.to_bits() == golden.to_bits(),
            Tolerance::Abs(d) => {
                if rust.is_finite() && golden.is_finite() {
                    (rust - golden).abs() <= d
                } else {
                    (rust.is_nan() && golden.is_nan()) || rust == golden
                }
            }
        }
    }
}

impl fmt::Display for Tolerance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tolerance::Exact => f.write_str("exact bits"),
            Tolerance::Abs(d) => write!(f, "{d:e}"),
        }
    }
}

/// A layer type that has a golden shape: the columns its golden layer has,
/// and how its own columns line up with them.
pub trait GoldenCompare: Layer {
    /// The golden layer's columns, in order.
    const COLUMNS: &'static [ColumnSpec];

    /// The golden layer's anchor: by default the layer's own `Layer::anchor`.
    fn golden_anchor() -> GoldenAnchor {
        match Self::anchor() {
            Anchor::Text => GoldenAnchor::Text,
            Anchor::Aligned { layer, .. } => GoldenAnchor::Aligned(layer.to_string()),
            Anchor::Document => GoldenAnchor::Document,
        }
    }

    /// Compares this layer with its golden layer, whose anchor and columns
    /// are already checked against the type.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch>;
}

/// What a comparison works in: the document and the instance of the layer
/// being compared, and the golden file it comes from.
pub struct Cx<'a> {
    pub doc: &'a Document,
    pub key: KeyId,
    pub golden: &'a Golden,
    /// The layer as the document names it, for reports.
    pub layer: String,
}

impl<'a> Cx<'a> {
    /// The document's source text: the golden's input, as checked.
    pub fn text(&self) -> &'a str {
        self.doc.source()
    }

    /// The spans of the instance of `A` the compared layer was made on (an
    /// aligned layer's anchor), when it is on the text: what a tag's report
    /// shows as the token it is for.
    pub fn rust_anchor<A: Layer>(&self) -> Option<SpanCols<'a>> {
        let doc: &'a Document = self.doc;
        let target = Key::<A>::DEFAULT.id();
        let name = doc.made_on(self.key).iter().find(|k| k.ty == target.ty).map_or("", |k| k.name);
        let ts = doc.get_key(&Key::<A>::named(name))?.text_spans()?;
        Some(SpanCols { view: ts.view, start: ts.start, end: ts.end })
    }

    /// The spans of a golden layer of this file, when it has them.
    pub fn golden_spans(&self, id: &str) -> Option<SpanCols<'a>> {
        let l = self.golden.layer(id)?;
        Some(SpanCols { view: turbo_analysis::ViewId::SOURCE, start: l.u32s("start")?, end: l.u32s("end")? })
    }
}

/// What a layer type's golden layer looks like: its id, its anchor and
/// its columns, all from the type.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub id: String,
    pub anchor: GoldenAnchor,
    pub columns: &'static [ColumnSpec],
}

/// The golden shape of `L`. The id is the layer's own name as
/// `turbo_analysis::LayerName` prints it (`turbo:confidence<opennlp:tokens>`
/// for a generic layer), and so is an aligned anchor's.
pub fn shape_of<L: GoldenCompare>() -> Shape {
    Shape { id: L::NAME.to_string(), anchor: L::golden_anchor(), columns: L::COLUMNS }
}

/// The shapes of the layer types of turbo-analysis and
/// turbo-analysis-layers that have one: what `Golden::read` checks each
/// layer block against. A layer type from another crate is not here; that
/// crate implements `GoldenCompare` for it and reads its goldens with
/// `Golden::read_with`, passing this set with `shape_of::<ItsLayer>()`
/// added.
pub fn known_shapes() -> Vec<Shape> {
    use turbo_analysis::View;
    use turbo_analysis_layers::{
        Aspects, Categories, ChunkTags, Confidence, DocumentDate, Entities, Incivility, Language, Lemmas, Money,
        Numbers, Opinions, PhraseChunks, Pii, PosTags, Pragmatics, Quantities, Sentences, Subwords, Temporals, Tokens,
        Utf16Spans,
    };
    vec![
        shape_of::<Sentences>(),
        shape_of::<Confidence<Sentences>>(),
        shape_of::<Utf16Spans<Sentences>>(),
        shape_of::<Tokens>(),
        shape_of::<Confidence<Tokens>>(),
        shape_of::<Utf16Spans<Tokens>>(),
        shape_of::<PosTags>(),
        shape_of::<Lemmas>(),
        shape_of::<Confidence<Lemmas>>(),
        shape_of::<Entities>(),
        shape_of::<ChunkTags>(),
        shape_of::<PhraseChunks>(),
        shape_of::<Utf16Spans<PhraseChunks>>(),
        shape_of::<Language>(),
        shape_of::<Categories>(),
        shape_of::<Numbers>(),
        shape_of::<Money>(),
        shape_of::<Quantities>(),
        shape_of::<Temporals>(),
        shape_of::<DocumentDate>(),
        shape_of::<Pii>(),
        shape_of::<Opinions>(),
        shape_of::<Aspects>(),
        shape_of::<Incivility>(),
        shape_of::<Pragmatics>(),
        shape_of::<Subwords>(),
        shape_of::<View>(),
    ]
}

/// `known_shapes()` and the four layers the golden generator writes whose
/// types live in other crates or are the generator's own: a UAX #29
/// tokenizer's `turbo:word-types` (a word type name per token), the
/// detokenizer's `turbo:detokenization` (an operation name per token) and
/// `turbo:detokenized` (the text it returned, without and with a split
/// marker), and the lattice tokenizer's `turbo:morphemes` (per token,
/// whether unknown word handling made it and its entry's features joined
/// with commas). The shapes are written out here, so a test reads those goldens
/// with `Golden::read_with` without the crates that own the types
/// depending on this one.
pub fn segmentation_shapes() -> Vec<Shape> {
    use turbo_analysis_layers::Tokens;
    const KIND: &[ColumnSpec] = &[ColumnSpec::new("kind", ColumnType::Str)];
    const OPERATION: &[ColumnSpec] = &[ColumnSpec::new("operation", ColumnType::Str)];
    const DETOKENIZED: &[ColumnSpec] =
        &[ColumnSpec::new("text", ColumnType::Str), ColumnSpec::new("marked", ColumnType::Str)];
    const MORPHEMES: &[ColumnSpec] =
        &[ColumnSpec::new("unknown", ColumnType::U32), ColumnSpec::new("features", ColumnType::Str)];
    let tokens = GoldenAnchor::Aligned(<Tokens as Layer>::ID.to_string());
    let mut shapes = known_shapes();
    shapes.push(Shape { id: "turbo:word-types".into(), anchor: tokens.clone(), columns: KIND });
    shapes.push(Shape { id: "turbo:morphemes".into(), anchor: tokens.clone(), columns: MORPHEMES });
    shapes.push(Shape { id: "turbo:detokenization".into(), anchor: tokens, columns: OPERATION });
    shapes.push(Shape { id: "turbo:detokenized".into(), anchor: GoldenAnchor::Document, columns: DETOKENIZED });
    shapes
}

/// Compares the default instance of layer `L` on `doc` with its layer in
/// `golden`: the first difference, or `Ok`. A golden for a run that threw
/// on the JVM has no layers and gives `Mismatch::JvmError`.
pub fn matches_golden<L: GoldenCompare>(doc: &Document, golden: &Golden, tol: Tolerance) -> Result<(), Mismatch> {
    matches_golden_key(doc, &Key::<L>::DEFAULT, golden, tol)
}

/// As `matches_golden`, for a named instance of `L`.
pub fn matches_golden_key<L: GoldenCompare>(
    doc: &Document,
    key: &Key<L>,
    golden: &Golden,
    tol: Tolerance,
) -> Result<(), Mismatch> {
    let shape = shape_of::<L>();
    let layer = key.id().to_string();
    if let Err(found) = check_input(golden, doc.source()) {
        let input = &golden.header.input;
        return Err(Mismatch::Input {
            layer,
            expected_len: input.utf8_len,
            found_len: found.len,
            expected_sha256: input.sha256,
            found_sha256: found.sha256,
        });
    }
    if let Some(e) = &golden.header.error {
        return Err(Mismatch::JvmError { layer, class: e.class.clone(), message: e.message.clone() });
    }
    let Some(g) = golden.layer(&shape.id) else {
        return Err(Mismatch::MissingGolden { layer, available: golden.layers.iter().map(|l| l.id.clone()).collect() });
    };
    check_shape(&layer, &shape, g)?;
    let Some(rust) = doc.get_key(key) else {
        let status = match doc.status(key.id()) {
            Some(Status::Skipped(why)) => Some(format!("skipped: {why}")),
            Some(Status::Failed(e)) => Some(format!("failed: {e}")),
            _ => None,
        };
        return Err(Mismatch::MissingLayer { layer, status });
    };
    let cx = Cx { doc, key: key.id(), golden, layer };
    rust.compare(&cx, g, tol)
}

/// A text's length in UTF-8 bytes and its SHA-256: what `check_input`
/// found when the text is not the golden's input.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InputFound {
    pub len: usize,
    pub sha256: [u8; 32],
}

/// Checks that `text` is the golden's input, by its length in UTF-8 bytes
/// and its SHA-256, and returns what the text has when it is not.
pub fn check_input(golden: &Golden, text: &str) -> Result<(), InputFound> {
    let input = &golden.header.input;
    let digest = sha256(text.as_bytes());
    if text.len() != input.utf8_len as usize || digest != input.sha256 {
        return Err(InputFound { len: text.len(), sha256: digest });
    }
    Ok(())
}

/// Checks a golden layer's anchor and columns against the type's.
pub fn check_shape(layer: &str, shape: &Shape, g: &GoldenLayer) -> Result<(), Mismatch> {
    if g.anchor != shape.anchor {
        return Err(Mismatch::Anchor { layer: layer.into(), expected: shape.anchor.clone(), found: g.anchor.clone() });
    }
    let same = g.columns.len() == shape.columns.len()
        && g.columns.iter().zip(shape.columns).all(|(c, s)| c.name == s.name && c.data.ty() == s.ty);
    let rows = g.len();
    if !same || g.columns.iter().any(|c| c.data.len() != rows) {
        return Err(Mismatch::Columns {
            layer: layer.into(),
            expected: shape.columns.iter().map(|c| c.to_string()).collect(),
            found: g.column_names(),
        });
    }
    Ok(())
}

/// As `matches_golden`, panicking with the report on a difference.
#[track_caller]
pub fn assert_matches_golden<L: GoldenCompare>(doc: &Document, golden: &Golden, tol: Tolerance) {
    if let Err(m) = matches_golden::<L>(doc, golden, tol) {
        panic!("layer does not match its golden:\n{m}");
    }
}

/// As `assert_matches_golden`, for a named instance of `L`.
#[track_caller]
pub fn assert_matches_golden_key<L: GoldenCompare>(doc: &Document, key: &Key<L>, golden: &Golden, tol: Tolerance) {
    if let Err(m) = matches_golden_key(doc, key, golden, tol) {
        panic!("layer does not match its golden:\n{m}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_compares_bits() {
        let t = Tolerance::exact();
        assert!(t.accepts(0.5, 0.5));
        assert!(!t.accepts(0.0, -0.0));
        assert!(t.accepts(f64::NAN, f64::NAN));
        assert!(!t.accepts(f64::NAN, -f64::NAN));
        assert!(!t.accepts(0.5, f64::from_bits(0.5f64.to_bits() + 1)));
    }

    #[test]
    fn abs_needs_finite_values_within_the_distance() {
        let t = Tolerance::abs(1e-9);
        assert!(t.accepts(0.5, 0.5 + 5e-10));
        assert!(!t.accepts(0.5, 0.5 + 1.5e-9));
        assert!(t.accepts(0.0, -0.0));
        assert!(t.accepts(f64::INFINITY, f64::INFINITY));
        assert!(!t.accepts(f64::INFINITY, f64::NEG_INFINITY));
        assert!(!t.accepts(f64::INFINITY, 1.0));
        assert!(t.accepts(f64::NAN, f64::NAN));
        assert!(!t.accepts(f64::NAN, 0.5));
        assert!(!t.accepts(f64::MAX, f64::INFINITY));
    }

    #[test]
    #[should_panic(expected = "finite and not negative")]
    fn a_negative_tolerance_is_refused() {
        Tolerance::abs(-1.0);
    }

    #[test]
    fn shapes_use_the_layers_own_names() {
        use turbo_analysis_layers::{Confidence, PosTags, Tokens};
        let c = shape_of::<Confidence<Tokens>>();
        assert_eq!(c.id, "turbo:confidence<opennlp:tokens>");
        assert_eq!(c.anchor, GoldenAnchor::Aligned("opennlp:tokens".into()));
        assert_eq!(shape_of::<PosTags>().anchor, GoldenAnchor::Aligned("opennlp:tokens".into()));
        let ids: Vec<String> = known_shapes().into_iter().map(|s| s.id).collect();
        assert_eq!(
            ids,
            [
                "opennlp:sentences",
                "turbo:confidence<opennlp:sentences>",
                "turbo:utf16<opennlp:sentences>",
                "opennlp:tokens",
                "turbo:confidence<opennlp:tokens>",
                "turbo:utf16<opennlp:tokens>",
                "opennlp:pos",
                "opennlp:lemmas",
                "turbo:confidence<opennlp:lemmas>",
                "opennlp:entities",
                "opennlp:chunk-tags",
                "opennlp:chunks",
                "turbo:utf16<opennlp:chunks>",
                "opennlp:language",
                "turbo:categories",
                "turbo:numbers",
                "turbo:money",
                "turbo:quantities",
                "turbo:temporal",
                "turbo:document-date",
                "turbo:pii",
                "turbo:opinions",
                "turbo:aspects",
                "turbo:incivility",
                "turbo:pragmatics",
                "opennlp:subwords",
                "turbo:view"
            ]
        );
    }
}
