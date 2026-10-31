//! Pipeline [`Step`] and [`Annotator`] implementations for deterministic
//! rule-to-[`Entities`] projection.

use std::any::Any;

use turbo_analysis::annotate::{AnchorUse, Annotator, Refs, Step};
use turbo_analysis::{Document, Error, Fnv1a, Key, KeyId, Provenance, Status, View, ViewId};
use turbo_analysis_layers::{Entities, Money, Quantities, Temporals, Times, Tokens};

use super::{
    TimeEntityFilter, money_to_entities, quantities_to_entities, rule_entities_in_text, temporals_to_entities,
    times_to_entities_in_text, times_to_entities_with,
};

const TEMPORAL_STEP_ID: &str = "turbo-namefind:temporal-entities";
const TIMES_STEP_ID: &str = "turbo-namefind:time-entities";
const MONEY_STEP_ID: &str = "turbo-namefind:money-entities";
const QUANTITIES_STEP_ID: &str = "turbo-namefind:percent-entities";

const DEFAULT_TEMPORAL_PRODUCER: &str = "turbo-numeric:temporal";
const DEFAULT_TIMES_PRODUCER: &str = "turbo-grammar:times";
const DEFAULT_MONEY_PRODUCER: &str = "turbo-numeric:money";
const DEFAULT_QUANTITIES_PRODUCER: &str = "turbo-numeric:quantities";

/// Source layer kind read by a [`RuleEntityProjector`].
#[derive(Copy, Clone, Debug)]
pub enum RuleEntitySource {
    /// Reads [`Temporals`] (`turbo:temporal`) and emits `"date"` entities.
    Temporals(Key<Temporals>),
    /// Reads [`Times`] (`turbo:times`) and emits `"date"` and/or `"time"`
    /// entities according to [`TimeEntityFilter`].
    Times(Key<Times>, TimeEntityFilter),
    /// Reads [`Money`] (`turbo:money`) and emits `"money"` entities.
    Money(Key<Money>),
    /// Reads [`Quantities`] (`turbo:quantities`) and emits `"percent"` entities
    /// for percentage mentions (`unit == "%"`).
    Quantities(Key<Quantities>),
}

impl RuleEntitySource {
    /// Returns the [`KeyId`] of the underlying structured rule layer.
    pub fn key_id(&self) -> KeyId {
        match self {
            RuleEntitySource::Temporals(k) => k.id(),
            RuleEntitySource::Times(k, _) => k.id(),
            RuleEntitySource::Money(k) => k.id(),
            RuleEntitySource::Quantities(k) => k.id(),
        }
    }

    /// Static step identifier for this projection source.
    pub const fn step_id(&self) -> &'static str {
        match self {
            RuleEntitySource::Temporals(_) => TEMPORAL_STEP_ID,
            RuleEntitySource::Times(_, _) => TIMES_STEP_ID,
            RuleEntitySource::Money(_) => MONEY_STEP_ID,
            RuleEntitySource::Quantities(_) => QUANTITIES_STEP_ID,
        }
    }

    /// Fallback producer annotator name when the source layer on the document
    /// was inserted with blank provenance (`Document::put`).
    pub const fn default_producer(&self) -> &'static str {
        match self {
            RuleEntitySource::Temporals(_) => DEFAULT_TEMPORAL_PRODUCER,
            RuleEntitySource::Times(_, _) => DEFAULT_TIMES_PRODUCER,
            RuleEntitySource::Money(_) => DEFAULT_MONEY_PRODUCER,
            RuleEntitySource::Quantities(_) => DEFAULT_QUANTITIES_PRODUCER,
        }
    }

    fn config_code(&self) -> u64 {
        match self {
            RuleEntitySource::Temporals(_) => 1,
            RuleEntitySource::Times(_, filter) => (filter.code() << 8) | 2,
            RuleEntitySource::Money(_) => 3,
            RuleEntitySource::Quantities(_) => 4,
        }
    }
}

/// Projects a single deterministic rule layer ([`Temporals`], [`Times`],
/// [`Money`], or [`Quantities`]) into an [`Entities`] (`opennlp:entities`)
/// layer aligned to [`Tokens`], preserving the source layer's provenance.
///
/// Because the stored [`Provenance`] retains the upstream rule layer's
/// `annotator`, `version`, `model`, and `execution` fields while recording both
/// the source layer and `tokens` in `reads`, [`ConsensusEntityMerger`](crate::ConsensusEntityMerger)
/// attributes every winning rule entity to its original deterministic producer
/// (for example `"turbo-numeric:temporal"` or `"turbo-numeric:money"`) in
/// [`Evidence<Entities>`](turbo_analysis::Evidence).
#[derive(Copy, Clone, Debug)]
pub struct RuleEntityProjector {
    pub source: RuleEntitySource,
    pub output: Key<Entities>,
    pub tokens: Key<Tokens>,
}

impl RuleEntityProjector {
    pub const VERSION: &'static str = "1";

    /// Projects [`Temporals`] (`source`) into `output` (`"date"` entities).
    pub fn temporals(source: Key<Temporals>, output: Key<Entities>) -> Self {
        Self { source: RuleEntitySource::Temporals(source), output, tokens: Key::DEFAULT }
    }

    /// Projects [`Times`] (`source`) into `output` (`"date"` and `"time"`
    /// entities under [`TimeEntityFilter::DatesAndTimes`]).
    pub fn times(source: Key<Times>, output: Key<Entities>) -> Self {
        Self::times_with_filter(source, output, TimeEntityFilter::DatesAndTimes)
    }

    /// Projects [`Times`] (`source`) into `output` using `filter`.
    pub fn times_with_filter(source: Key<Times>, output: Key<Entities>, filter: TimeEntityFilter) -> Self {
        Self { source: RuleEntitySource::Times(source, filter), output, tokens: Key::DEFAULT }
    }

    /// Projects [`Money`] (`source`) into `output` (`"money"` entities).
    pub fn money(source: Key<Money>, output: Key<Entities>) -> Self {
        Self { source: RuleEntitySource::Money(source), output, tokens: Key::DEFAULT }
    }

    /// Projects percentage mentions from [`Quantities`] (`source`) into
    /// `output` (`"percent"` entities).
    pub fn quantities(source: Key<Quantities>, output: Key<Entities>) -> Self {
        Self { source: RuleEntitySource::Quantities(source), output, tokens: Key::DEFAULT }
    }

    /// Selects a named [`Tokens`] instance instead of [`Key::DEFAULT`].
    pub fn with_tokens(mut self, tokens: Key<Tokens>) -> Self {
        self.tokens = tokens;
        self
    }

    fn build(&self, doc: &Document) -> Result<Entities, Error> {
        let tokens = doc.require(&self.tokens)?;
        let projected = match &self.source {
            RuleEntitySource::Temporals(k) => temporals_to_entities(doc.require(k)?, tokens),
            RuleEntitySource::Times(k, filter) => {
                let times = doc.require(k)?;
                if tokens.spans.view() == ViewId::SOURCE {
                    times_to_entities_in_text(doc.source(), times, tokens, *filter)
                } else {
                    times_to_entities_with(times, tokens, *filter)
                }
            }
            RuleEntitySource::Money(k) => money_to_entities(doc.require(k)?, tokens),
            RuleEntitySource::Quantities(k) => quantities_to_entities(doc.require(k)?, tokens),
        };
        projected.map_err(|detail| Error::Annotator { annotator: self.source.step_id(), detail })
    }

    fn provenance(&self, doc: &Document) -> Provenance {
        let mut reads = Vec::with_capacity(2);
        let src_id = self.source.key_id();
        if doc.has(src_id) {
            reads.push(src_id);
        }
        if doc.has(self.tokens.id()) {
            reads.push(self.tokens.id());
        }
        if let Some(src) = doc.provenance(src_id) {
            let annotator = if src.annotator.is_empty() { self.source.default_producer() } else { src.annotator };
            let version = if src.version.is_empty() { Self::VERSION } else { src.version };
            let mut h = Fnv1a::new();
            h.u64(src.config);
            h.u64(Step::config_hash(self));
            Provenance {
                annotator,
                version,
                config: h.finish(),
                model: src.model.clone(),
                execution: src.execution.clone(),
                reads,
                on: Vec::new(),
            }
        } else {
            Provenance {
                reads,
                version: Self::VERSION,
                config: Step::config_hash(self),
                ..Provenance::by(self.source.default_producer())
            }
        }
    }
}

type Out = Vec<Result<Entities, Error>>;

impl Step for RuleEntityProjector {
    fn id(&self) -> &'static str {
        self.source.step_id()
    }

    fn reads(&self) -> Vec<KeyId> {
        vec![self.source.key_id(), self.tokens.id()]
    }

    fn required(&self) -> Vec<bool> {
        vec![true, true]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![self.output.id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        vec![AnchorUse::None]
    }

    fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.u64(self.source.config_code());
        h.bytes(self.source.key_id().name.as_bytes());
        h.bytes(self.tokens.name.as_bytes());
        h.finish()
    }

    fn expected_provenance(&self, doc: &Document) -> Provenance {
        self.provenance(doc)
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let out: Out = docs
            .iter()
            .map(|d| {
                for k in self.reads() {
                    match d.status(k) {
                        Some(Status::Complete) => {}
                        Some(Status::Skipped(why)) => return Err(Error::Skipped(format!("{k} was skipped: {why}"))),
                        Some(Status::Failed(e)) => return Err(Error::Skipped(format!("{k} failed: {e}"))),
                        None => return Err(Error::Missing(k)),
                    }
                }
                self.build(d)
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let out = *out.downcast::<Out>().expect("this step's own output");
        for (i, r) in out.into_iter().enumerate() {
            let p = self.provenance(&docs[i]);
            let failure = match r {
                Ok(entities) => docs[i].add(&self.output, entities, p.clone()),
                Err(Error::Skipped(why)) => docs[i].skip(self.output.id(), why, p.clone()),
                Err(e) => Err(e),
            };
            let Err(e) = failure else { continue };
            if !docs[i].has(self.output.id()) {
                let _ = docs[i].fail(self.output.id(), e.clone(), p);
            }
            if status[i].is_ok() {
                status[i] = Err(e);
            }
        }
    }
}

/// Combined [`Annotator`] that projects any present [`Temporals`], [`Times`],
/// [`Money`], and [`Quantities`] layers on a [`Document`] into a single
/// [`Entities`] (`opennlp:entities`) layer over `"date"`, `"time"`, `"money"`,
/// and `"percent"`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RuleEntitiesAnnotator {
    time_filter: TimeEntityFilter,
}

impl RuleEntitiesAnnotator {
    /// Creates an annotator using [`TimeEntityFilter::DatesAndTimes`].
    pub const fn new() -> Self {
        Self { time_filter: TimeEntityFilter::DatesAndTimes }
    }

    /// Creates an annotator with an explicit [`TimeEntityFilter`].
    pub const fn with_time_filter(time_filter: TimeEntityFilter) -> Self {
        Self { time_filter }
    }

    /// Returns the configured [`TimeEntityFilter`].
    pub const fn time_filter(&self) -> TimeEntityFilter {
        self.time_filter
    }
}

impl Annotator for RuleEntitiesAnnotator {
    const ID: &'static str = "turbo-namefind:rule-entities";
    type Reads = (View, Tokens, Option<Temporals>, Option<Times>, Option<Money>, Option<Quantities>);
    type Writes = (Entities,);

    fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.u64(self.time_filter.code());
        h.finish()
    }

    fn annotate(
        &self,
        (view, tokens, temporals, times, money, quantities): Refs<'_, Self>,
    ) -> Result<(Entities,), Error> {
        let entities =
            rule_entities_in_text(view.text(), tokens, temporals, times, money, quantities, self.time_filter)
                .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((entities,))
    }
}
