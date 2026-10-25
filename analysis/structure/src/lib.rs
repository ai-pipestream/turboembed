//! Document structure from text: `turbo:structure` for a view of plain
//! text or Markdown, for the chunker's sections and the blocks it must
//! not cut.
//!
//! [`StructureAnnotator`] reads a `View` and writes `Structure`: sections,
//! headings with their levels, paragraphs, lists and their items, tables
//! with their rows and cells, code blocks and block quotes, as a tree in
//! text order. [`detect`] does the same outside a pipeline, and
//! [`StructureProvider`] offers it to a registry as "structure".
//! [`StructureProjector`] maps a `Structure` layer from a parent view into a
//! normalized child view (`docs/structure-projection.md`).
//!
//! The rules are written here, with no Markdown library: one pass over
//! the lines, every offset a UTF-8 byte offset into the view.
//! docs/analysis.md says which parts of CommonMark and GFM are followed
//! and where the detector differs from them.

mod columns;
mod html;
mod kv;
mod line;
mod marker;
mod parse;
mod project;
mod table;
mod table_grid;
mod tree;

pub use kv::KeyValueAnnotator;
pub use project::StructureProjector;
pub use table_grid::{TableGridAnnotator, detect_tables};

use turbo_analysis::annotate::Step;
use turbo_analysis::registry::{Provider, Spec};
use turbo_analysis::{Annotator, Bound, Error, Refs, View, ViewId};
use turbo_analysis_layers::Structure;

/// Which conventions a text is read with.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Syntax {
    /// CommonMark blocks and GFM pipe tables.
    #[default]
    Markdown,
    /// Plain text: paragraphs between blank lines, list items (ordered
    /// ones numbered with at most three digits), quotes marked with `>`,
    /// and headings: a line underlined with three or more `=` or `-`, or
    /// a short line in capitals at the top level, either on its own or
    /// over a line with lower case letters, from which it is split off
    /// ("INTRODUCTION", then "This report covers the year."). No `#`
    /// headings or code. Repeated aligned columns form tables; otherwise an indented line is text, as in a
    /// letter or a contract.
    Plain,
}

impl Syntax {
    /// The name a configuration uses: `markdown` or `plain`.
    pub fn name(self) -> &'static str {
        match self {
            Syntax::Markdown => "markdown",
            Syntax::Plain => "plain",
        }
    }

    pub fn from_name(name: &str) -> Option<Syntax> {
        match name {
            "markdown" => Some(Syntax::Markdown),
            "plain" => Some(Syntax::Plain),
            _ => None,
        }
    }
}

/// The structure of `text`, a view of `view`'s, read with `syntax`.
///
/// # Panics
///
/// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`.
pub fn detect(text: &str, syntax: Syntax, view: ViewId) -> Structure {
    assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
    parse::Parser::new(text, syntax).run().into_structure(view)
}

/// Writes the structure of a view.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct StructureAnnotator {
    pub syntax: Syntax,
}

impl StructureAnnotator {
    pub fn new(syntax: Syntax) -> StructureAnnotator {
        StructureAnnotator { syntax }
    }
}

impl Annotator for StructureAnnotator {
    const VERSION: &'static str = "1";
    const ID: &'static str = "turbo-structure:blocks";
    type Reads = (View,);
    type Writes = (Structure,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Structure,), Error> {
        let text = view.text();
        if u32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the u32 offset range", text.len()),
            });
        }
        Ok((detect(text, self.syntax, view.id()),))
    }

    /// FNV-1a over a number per syntax, fixed so stored provenance stays
    /// comparable across builds.
    fn config_hash(&self) -> u64 {
        let x: u64 = match self.syntax {
            Syntax::Markdown => 1,
            Syntax::Plain => 2,
        };
        (0xcbf2_9ce4_8422_2325 ^ x).wrapping_mul(0x0100_0000_01b3)
    }
}

/// Offers `StructureAnnotator` to pipelines built from configuration, as
/// "structure". It takes no model location and one option, `syntax`
/// (`markdown`, the default, or `plain`).
pub struct StructureProvider;

impl Provider<dyn Step> for StructureProvider {
    fn name(&self) -> &'static str {
        "structure"
    }

    fn supports(&self, spec: &Spec) -> bool {
        spec.location.is_none() && spec.options.iter().all(|(k, v)| k == "syntax" && Syntax::from_name(v).is_some())
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let name = spec.option("syntax", "markdown");
        let syntax =
            Syntax::from_name(name).ok_or_else(|| Error::Pipeline(format!("structure: no syntax named {name}")))?;
        Ok(Box::new(Bound::new(StructureAnnotator::new(syntax))))
    }
}
