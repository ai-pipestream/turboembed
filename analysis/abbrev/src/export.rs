//! Registration of the document graph codecs for what this crate
//! writes, so abbreviation definitions and their links cross the
//! export boundary with the rest of the graph.
//!
//! The layer types themselves ([`Abbreviations`] and the
//! [`Edges`] between its items) live in `turbo-analysis-layers` and
//! core, and so do their codec implementations: the spans and forms
//! in the layers crate, the relation endpoints in core, where an
//! empty endpoint instance the importing record's provenance would
//! rebind to a named instance is refused instead of being checked
//! against the default instance and silently attached to the named
//! one. This module is the registration entry point for a consumer
//! of this crate: it registers exactly the two writes of
//! [`AbbreviationAnnotator`](crate::AbbreviationAnnotator), and
//! composes with the shared layers' registrations for the layers
//! the annotator only reads.

use turbo_analysis::Edges;
use turbo_analysis::export::Codecs;
use turbo_analysis_layers::Abbreviations;

/// Registers the codecs for this crate's writes on `codecs`: the
/// abbreviations layer and the links between its items.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<Abbreviations>().register::<Edges<Abbreviations, Abbreviations>>();
}

/// A registry holding this crate's codecs and nothing else.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
