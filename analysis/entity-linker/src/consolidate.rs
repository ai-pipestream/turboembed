//! [`EntityConsolidator`]: the document-level entity graph
//! ([`DocumentEntities`]) from whichever of `Entities`, coreference chains
//! (`Groups<Mentions>`, `Groups<Entities>`), `LinkedEntities`, `Locations`,
//! `Dependencies`, `Aspects` and relation edges (`Edges<Entities,
//! Entities>`) a document has.
//!
//! Mentions are grouped first by coreference chain, then by knowledge-base
//! id, then, for names no chain or link joins, by the same folded surface
//! of the same kind, by overlapping spans of the same kind, and a one-word
//! name by the longer name of its kind it ends (`Lincoln` with `Abraham
//! Lincoln`) when that name is the only one it could be. Each group with at
//! least one name is an entity. Its canonical name is its knowledge-base
//! name, else its longest name; its roles are its mentions' head relations;
//! its sentiment the mean polarity of the aspects on its mentions. Each
//! relation edge is lifted to the entities its arguments belong to, so a
//! relation found on `He` or `Lincoln` is a relation of `Abraham Lincoln`.

use turbo_analysis::{Annotator, Document, Edges, Error, Groups, Key, Layer, Refs, View};
use turbo_analysis_layers::{Aspects, Dependencies, DocumentEntities, Entities, Tokens};
use turbo_coref::Mentions;
use turbo_geo::Locations;

use crate::LinkedEntities;

mod build;
mod groups;

/// What the consolidator merges beyond coreference and knowledge-base
/// links. Both are on by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsolidatorConfig {
    /// Names of one kind with the same folded surface, or overlapping
    /// spans, are one entity.
    pub surface_merge: bool,
    /// A one-word name is the entity of the only longer name of its kind
    /// that ends with that word.
    pub alias_merge: bool,
}

impl Default for ConsolidatorConfig {
    fn default() -> Self {
        ConsolidatorConfig { surface_merge: true, alias_merge: true }
    }
}

/// Builds [`DocumentEntities`] for the default view.
#[derive(Clone, Debug, Default)]
pub struct EntityConsolidator {
    pub config: ConsolidatorConfig,
}

impl EntityConsolidator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(mut self, config: ConsolidatorConfig) -> Self {
        self.config = config;
        self
    }

    /// The entity graph of the default layers of `doc`, without adding it.
    pub fn consolidate(&self, doc: &Document) -> Result<DocumentEntities, Error> {
        let (entities,) = self.annotate((
            doc.require(&Key::<View>::DEFAULT)?,
            doc.require(&Key::<Entities>::DEFAULT)?,
            doc.get_key(&Key::<Tokens>::DEFAULT),
            doc.get_key(&Key::<Mentions>::DEFAULT),
            doc.get_key(&Key::<Groups<Mentions>>::DEFAULT),
            doc.get_key(&Key::<Groups<Entities>>::DEFAULT),
            doc.get_key(&Key::<LinkedEntities>::DEFAULT),
            doc.get_key(&Key::<Dependencies>::DEFAULT),
            doc.get_key(&Key::<Aspects>::DEFAULT),
            doc.get_key(&Key::<Locations>::DEFAULT),
            doc.get_key(&Key::<Edges<Entities, Entities>>::DEFAULT),
        ))?;
        Ok(entities)
    }
}

/// The optional inputs, as the annotator reads them.
pub(crate) struct Inputs<'a> {
    pub view: &'a View,
    pub entities: &'a Entities,
    pub mentions: Option<&'a Mentions>,
    pub mention_groups: Option<&'a Groups<Mentions>>,
    pub entity_groups: Option<&'a Groups<Entities>>,
    pub linked: Option<&'a LinkedEntities>,
    pub dependencies: Option<&'a Dependencies>,
    pub aspects: Option<&'a Aspects>,
    pub locations: Option<&'a Locations>,
    pub relations: Option<&'a Edges<Entities, Entities>>,
}

impl Annotator for EntityConsolidator {
    const ID: &'static str = "turbo:entity_consolidator";
    const VERSION: &'static str = "1";

    type Reads = (
        View,
        Entities,
        Option<Tokens>,
        Option<Mentions>,
        Option<Groups<Mentions>>,
        Option<Groups<Entities>>,
        Option<LinkedEntities>,
        Option<Dependencies>,
        Option<Aspects>,
        Option<Locations>,
        Option<Edges<Entities, Entities>>,
    );
    type Writes = (DocumentEntities,);

    /// FNV-1a over the version and each switch.
    fn config_hash(&self) -> u64 {
        turbo_analysis::Fnv1a::new()
            .bytes(Self::VERSION.as_bytes())
            .bool(self.config.surface_merge)
            .bool(self.config.alias_merge)
            .finish()
    }

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let (
            view,
            entities,
            tokens,
            mentions,
            mention_groups,
            entity_groups,
            linked,
            dependencies,
            aspects,
            locations,
            relations,
        ) = input;
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if entities.spans.view() != view.id() {
            return Err(fail("the entities are not of the view read".into()));
        }
        if let Some(l) = linked
            && l.len() != entities.spans.len()
        {
            return Err(fail(format!("{} linked rows for {} entities", l.len(), entities.spans.len())));
        }
        let inputs = Inputs {
            view,
            entities,
            mentions: mentions.filter(|m| m.spans.view() == view.id()),
            mention_groups,
            entity_groups,
            linked,
            dependencies: dependencies
                .filter(|d| tokens.is_some_and(|t| t.spans.view() == view.id() && t.spans.len() == d.head.len())),
            aspects,
            locations: locations.filter(|l| l.spans.view() == view.id()),
            relations,
        };
        Ok((build::build(&inputs, self.config).map_err(fail)?,))
    }
}
