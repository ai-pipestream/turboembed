//! `GeneratorFactory`: builds the feature generators a model describes in
//! its `generator.featuregen` entry.
//!
//! ```xml
//! <featureGenerators cache="true" name="nameFinder">
//!   <generator class="opennlp.tools.util.featuregen.WindowFeatureGeneratorFactory">
//!     <int name="prevLength">2</int>
//!     <int name="nextLength">2</int>
//!     <generator class="opennlp.tools.util.featuregen.TokenClassFeatureGeneratorFactory"/>
//!   </generator>
//!   <generator class="opennlp.tools.util.featuregen.BrownClusterTokenFeatureGeneratorFactory">
//!     <str name="dict">brownCluster.txt</str>
//!   </generator>
//! </featureGenerators>
//! ```
//!
//! The root must be `featureGenerators` (OpenNLP dropped the older format
//! in OPENNLP-1174); its `generator` children are aggregated, and wrapped
//! in a [`Cached`] when `cache` is `true` in any case. A `generator`
//! element names a factory class and holds its parameters (`int`, `long`,
//! `float`, `double`, `str`, `bool`, each with a `name`) and the
//! generators it wraps. The factories and their parameters are those of
//! `opennlp.tools.util.featuregen`; [`build`] lists them. Errors carry the
//! messages OpenNLP's `InvalidFormatException`s carry.
//!
//! Two behaviours of OpenNLP are reproduced because the features depend
//! on them:
//!
//! - A `generator` element with several `generator` children passes their
//!   aggregate to its factory as the first child, and the second and later
//!   children again on their own. The window and cache factories read only
//!   the first, so a window over several generators is a window over all
//!   of them; but `AggregatedFeatureGeneratorFactory` reads them all and so
//!   writes the features of every child but the first twice.
//! - `TokenClassFeatureGeneratorFactory` defaults `wordAndClass` to true,
//!   unlike the class's own constructor.
//!
//! Resources (Brown clusters, word clusters, dictionaries) are entries of
//! the model archive, named by the `dict` parameter. Each name is read as
//! the kind the last factory naming it, in document order, expects, as
//! OpenNLP's model loader does, and a factory that finds another kind
//! under its name fails.

mod args;
mod prepared;
mod tree;
pub use prepared::Prepared;

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use turbo_maxent::{Dictionary, ModelFile};

use self::args::Args;
use self::tree::Element;
use super::{
    Aggregated, BigramName, BoxedGenerator, BrownBigram, BrownCluster, BrownToken, BrownTokenClass, Cached,
    CharacterNgram, DocumentBegin, InDictionary, OutcomePrior, Prefix, PreviousMap, PreviousTags, SentencePosition,
    Suffix, TokenClass, TokenGenerator, TokenPattern, TrigramName, Window, WordCluster, WordClusterDictionary,
};
use crate::error::{Error, Result};
use crate::pattern::PatternMode;

/// The model entry that holds the descriptor
/// (`GENERATOR_DESCRIPTOR_ENTRY_NAME`).
pub const ENTRY_NAME: &str = "generator.featuregen";

/// The package every factory class is in.
const PACKAGE: &str = "opennlp.tools.util.featuregen.";

/// Where the descriptor's resources are read from: the entries of a model
/// archive, by name.
pub trait Resources {
    /// The bytes of the named resource, or `None` when there is none.
    fn resource(&self, name: &str) -> Result<Option<Cow<'_, [u8]>>>;
}

impl Resources for ModelFile {
    fn resource(&self, name: &str) -> Result<Option<Cow<'_, [u8]>>> {
        match self.entry(name) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(turbo_maxent::Error::MissingEntry(_)) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// Builds the generators the model's descriptor describes, or `None` when
/// the model has no descriptor. Resources are read from the model.
pub fn from_model(model: &ModelFile, mode: PatternMode) -> Result<Option<BoxedGenerator>> {
    match model.resource(ENTRY_NAME)? {
        Some(xml) => build(&xml, Some(model), mode).map(Some),
        None => Ok(None),
    }
}

/// Builds the generators a descriptor describes.
///
/// Without `resources`, the factories that need one (the Brown cluster
/// and word cluster factories) make nothing, as OpenNLP's do when they are
/// given no resource manager, and the dictionary factory fails (OpenNLP
/// would look on its class path). `mode` is the [`PatternMode`] of every
/// generator that classifies tokens.
///
/// The factories, all in `opennlp.tools.util.featuregen`, and what they
/// make:
///
/// | Factory | Parameters | Generator |
/// |---|---|---|
/// | `AggregatedFeatureGeneratorFactory` | generators | [`Aggregated`] |
/// | `BigramNameFeatureGeneratorFactory` | | [`BigramName`] |
/// | `BrownClusterBigramFeatureGeneratorFactory` | `dict` | [`BrownBigram`] |
/// | `BrownClusterTokenClassFeatureGeneratorFactory` | `dict` | [`BrownTokenClass`] |
/// | `BrownClusterTokenFeatureGeneratorFactory` | `dict` | [`BrownToken`] |
/// | `CachedFeatureGeneratorFactory` | a generator | [`Cached`] |
/// | `CharacterNgramFeatureGeneratorFactory` | `min`, `max` | [`CharacterNgram`] |
/// | `DefinitionFeatureGeneratorFactory` | | [`OutcomePrior`] |
/// | `DictionaryFeatureGeneratorFactory` | `dict` | [`InDictionary`] |
/// | `DocumentBeginFeatureGeneratorFactory` | | [`DocumentBegin`] |
/// | `PosTaggerFeatureGeneratorFactory` | | [`PreviousTags`] |
/// | `PrefixFeatureGeneratorFactory` | `length` (4) | [`Prefix`] |
/// | `PreviousMapFeatureGeneratorFactory` | | [`PreviousMap`] |
/// | `SentenceFeatureGeneratorFactory` | `begin` (true), `end` (true) | [`SentencePosition`] |
/// | `SuffixFeatureGeneratorFactory` | `length` (4) | [`Suffix`] |
/// | `TokenClassFeatureGeneratorFactory` | `wordAndClass` (true) | [`TokenClass`] |
/// | `TokenFeatureGeneratorFactory` | `lowercase` (true) | [`TokenGenerator`] |
/// | `TokenPatternFeatureGeneratorFactory` | | [`TokenPattern`] |
/// | `TrigramNameFeatureGeneratorFactory` | | [`TrigramName`] |
/// | `WindowFeatureGeneratorFactory` | `prevLength`, `nextLength`, a generator | [`Window`] |
/// | `WordClusterFeatureGeneratorFactory` | `dict`, `lowerCase` | [`WordCluster`] |
///
/// `POSTaggerNameFeatureGeneratorFactory` and
/// `EmojiAnnotationFeatureGeneratorFactory` are [`Error::Unsupported`].
pub fn build(xml: &[u8], resources: Option<&dyn Resources>, mode: PatternMode) -> Result<BoxedGenerator> {
    let root = tree::parse(xml)?;
    if root.name != "featureGenerators" {
        return Err(Error::Descriptor("[OPENNLP-1174] - Classic configuration format is no longer supported!".into()));
    }
    let mut kinds = HashMap::new();
    resource_kinds(&root, &mut kinds);
    let mut builder = Builder { resources, mode, kinds: &kinds, loaded: HashMap::new(), shared: None };
    build_root(&root, &mut builder)
}

fn build_root(root: &Element, builder: &mut Builder<'_>) -> Result<BoxedGenerator> {
    let mut generators = Vec::new();
    let mut single_is_cached = false;
    for child in root.elements() {
        if child.name != "generator" {
            // Java names the parent in this message, not the child.
            return Err(Error::Descriptor("Unexpected element: featureGenerators".into()));
        }
        match builder.build(child)? {
            Some(g) => generators.push(g),
            None => {
                return Err(Error::Descriptor(format!(
                    "{} made no generator, which needs the model's resources",
                    child.attribute("class")
                )));
            }
        }
        single_is_cached = class_name(child) == Some("CachedFeatureGeneratorFactory");
    }
    let generator: BoxedGenerator = match generators.len() {
        0 => return Err(Error::Descriptor("featureGenerators must have one or more generators".into())),
        1 => {
            if single_is_cached {
                return Err(Error::Descriptor(
                    "CachedFeatureGeneratorFactory cannot be specified manually.\
                     Use cache=\"true\" attribute in featureGenerators element instead."
                        .into(),
                ));
            }
            generators.pop().expect("one generator")
        }
        _ => Box::new(Aggregated::new(generators)),
    };
    if root.attribute("cache").eq_ignore_ascii_case("true") {
        Ok(Box::new(Cached::new(generator)))
    } else {
        Ok(generator)
    }
}

/// The factory class's name within [`PACKAGE`].
fn class_name(element: &Element) -> Option<&str> {
    element.attribute("class").strip_prefix(PACKAGE)
}

/// What a resource is read as.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Kind {
    Brown,
    WordCluster,
    Dictionary,
    PosModel,
}

/// A resource read from the model.
#[derive(Clone)]
enum Resource {
    Brown(Arc<BrownCluster>),
    WordCluster(Arc<WordClusterDictionary>),
    Dictionary(Arc<Dictionary>),
}

/// The kind each resource name is read as: that of the last factory in
/// document order (an element before the elements inside it) that names
/// it in its `dict` (or, for the part-of-speech factory, `model`)
/// parameter, as `GeneratorFactory.extractArtifactSerializerMappings`
/// collects them.
fn resource_kinds(element: &Element, kinds: &mut HashMap<String, Kind>) {
    for child in element.elements().filter(|e| e.name == "generator") {
        let kind = match class_name(child) {
            Some(
                "BrownClusterBigramFeatureGeneratorFactory"
                | "BrownClusterTokenClassFeatureGeneratorFactory"
                | "BrownClusterTokenFeatureGeneratorFactory",
            ) => Some((Kind::Brown, "dict")),
            Some("WordClusterFeatureGeneratorFactory") => Some((Kind::WordCluster, "dict")),
            Some("DictionaryFeatureGeneratorFactory") => Some((Kind::Dictionary, "dict")),
            Some("POSTaggerNameFeatureGeneratorFactory") => Some((Kind::PosModel, "model")),
            _ => None,
        };
        if let Some((kind, parameter)) = kind
            && let Ok(args) = Args::read(child)
            && let Ok(name) = args.str(parameter)
        {
            kinds.insert(name.to_owned(), kind);
        }
        resource_kinds(child, kinds);
    }
}

struct Builder<'r> {
    resources: Option<&'r dyn Resources>,
    mode: PatternMode,
    kinds: &'r HashMap<String, Kind>,
    loaded: HashMap<String, Resource>,
    shared: Option<&'r HashMap<String, Resource>>,
}

impl Builder<'_> {
    /// Builds the generators of the `generator` children of `element`, in
    /// order, leaving out those whose factory made nothing.
    fn children(&mut self, element: &Element) -> Result<Vec<BoxedGenerator>> {
        let mut out = Vec::new();
        for child in element.elements().filter(|e| e.name == "generator") {
            if let Some(g) = self.build(child)? {
                out.push(g);
            }
        }
        Ok(out)
    }

    /// The factory's `generator#0`: the one child, or the aggregate of all
    /// of them.
    fn first_child(&mut self, element: &Element) -> Result<Option<BoxedGenerator>> {
        let mut children = self.children(element)?;
        Ok(match children.len() {
            0 => None,
            1 => children.pop(),
            _ => Some(Box::new(Aggregated::new(children))),
        })
    }

    /// `buildGenerator`: the generator one `generator` element describes,
    /// or `None` when its factory makes none.
    fn build(&mut self, element: &Element) -> Result<Option<BoxedGenerator>> {
        let class = element.attribute("class");
        if class.trim().is_empty() {
            return Err(Error::Descriptor("generator must have class attribute".into()));
        }
        // `init` reads every parameter before the factory runs, so a bad
        // value fails whichever factory it is given to.
        let args = Args::read(element)?;
        let Some(name) = class.strip_prefix(PACKAGE) else {
            return Err(Error::UnknownGenerator(class.into()));
        };
        let mode = self.mode;
        let generator: BoxedGenerator = match name {
            "AggregatedFeatureGeneratorFactory" => {
                // generator#0 is the aggregate of every child when there
                // are several, and generator#1 onwards are the children
                // after the first again.
                let mut all = self.children(element)?;
                let mut generators: Vec<BoxedGenerator> = Vec::new();
                if all.len() > 1 {
                    let again = self.children(element)?;
                    generators.push(Box::new(Aggregated::new(std::mem::take(&mut all))));
                    generators.extend(again.into_iter().skip(1));
                } else {
                    generators.append(&mut all);
                }
                Box::new(Aggregated::new(generators))
            }
            "BigramNameFeatureGeneratorFactory" => Box::new(BigramName::new(mode)),
            "BrownClusterBigramFeatureGeneratorFactory" => match self.brown(&args)? {
                Some(c) => Box::new(BrownBigram::new(c)),
                None => return Ok(None),
            },
            "BrownClusterTokenClassFeatureGeneratorFactory" => match self.brown(&args)? {
                Some(c) => Box::new(BrownTokenClass::new(c, mode)),
                None => return Ok(None),
            },
            "BrownClusterTokenFeatureGeneratorFactory" => match self.brown(&args)? {
                Some(c) => Box::new(BrownToken::new(c)),
                None => return Ok(None),
            },
            "CachedFeatureGeneratorFactory" => match self.first_child(element)? {
                Some(g) => Box::new(Cached::new(g)),
                None => return Err(Error::Descriptor("Could not find containing generator element!".into())),
            },
            "CharacterNgramFeatureGeneratorFactory" => {
                let min = non_negative(args.int("min")?, "min")?;
                let max = non_negative(args.int("max")?, "max")?;
                Box::new(CharacterNgram::new(min, max))
            }
            "DefinitionFeatureGeneratorFactory" => Box::new(OutcomePrior),
            "DictionaryFeatureGeneratorFactory" => {
                let key = args.str("dict")?;
                if !self.has_resources() {
                    return Err(Error::Descriptor(format!("No dictionary resource at: '{key}")));
                }
                match self.resource(key)? {
                    Some(Resource::Dictionary(d)) => Box::new(InDictionary::new("", d)),
                    _ => return Err(Error::Descriptor(format!("No dictionary resource for key: {key}"))),
                }
            }
            "DocumentBeginFeatureGeneratorFactory" => Box::new(DocumentBegin::new()),
            "EmojiAnnotationFeatureGeneratorFactory" => return Err(Error::Unsupported(class.into())),
            "POSTaggerNameFeatureGeneratorFactory" => {
                if !self.has_resources() {
                    return Ok(None);
                }
                return Err(Error::Unsupported(class.into()));
            }
            "PosTaggerFeatureGeneratorFactory" => Box::new(PreviousTags),
            "PrefixFeatureGeneratorFactory" => {
                Box::new(Prefix::new(non_negative(args.int_or("length", Prefix::DEFAULT_LENGTH as i32)?, "length")?))
            }
            "PreviousMapFeatureGeneratorFactory" => Box::new(PreviousMap::new()),
            "SentenceFeatureGeneratorFactory" => {
                Box::new(SentencePosition::new(args.bool_or("begin", true)?, args.bool_or("end", true)?))
            }
            "SuffixFeatureGeneratorFactory" => {
                Box::new(Suffix::new(non_negative(args.int_or("length", Suffix::DEFAULT_LENGTH as i32)?, "length")?))
            }
            "TokenClassFeatureGeneratorFactory" => Box::new(TokenClass::new(args.bool_or("wordAndClass", true)?, mode)),
            "TokenFeatureGeneratorFactory" => Box::new(TokenGenerator::new(args.bool_or("lowercase", true)?)),
            "TokenPatternFeatureGeneratorFactory" => Box::new(TokenPattern::new(mode)),
            "TrigramNameFeatureGeneratorFactory" => Box::new(TrigramName::new(mode)),
            "WindowFeatureGeneratorFactory" => {
                let Some(inner) = self.first_child(element)? else {
                    return Err(Error::Descriptor(
                        "window feature generator must contain an aggregator element".into(),
                    ));
                };
                // A negative window is an empty one: Java's loops run from
                // 1 while below the length plus one.
                let prev = args.int("prevLength")?.max(0) as usize;
                let next = args.int("nextLength")?.max(0) as usize;
                Box::new(Window::new(inner, prev, next))
            }
            "WordClusterFeatureGeneratorFactory" => {
                if !self.has_resources() {
                    return Ok(None);
                }
                let key = args.str("dict")?;
                let lowercase = args.bool("lowerCase")?;
                match self.resource(key)? {
                    Some(Resource::WordCluster(d)) => Box::new(WordCluster::new(d, key, lowercase)),
                    _ => {
                        return Err(Error::Descriptor(format!("Not a WordClusterDictionary resource for key: {key}")));
                    }
                }
            }
            _ => return Err(Error::UnknownGenerator(class.into())),
        };
        Ok(Some(generator))
    }

    /// The Brown cluster a Brown factory names, `None` without resources.
    fn brown(&mut self, args: &Args) -> Result<Option<Arc<BrownCluster>>> {
        if !self.has_resources() {
            return Ok(None);
        }
        let key = args.str("dict")?;
        match self.resource(key)? {
            Some(Resource::Brown(c)) => Ok(Some(c)),
            _ => Err(Error::Descriptor(format!("Not a BrownLexicon resource for key: {key}"))),
        }
    }

    /// The named resource, read as the kind [`resource_kinds`] gave it, or
    /// `None` when the model does not have it or holds it as a kind this
    /// crate does not read.
    fn resource(&mut self, name: &str) -> Result<Option<Resource>> {
        if let Some(r) = self.shared.and_then(|shared| shared.get(name)) {
            return Ok(Some(r.clone()));
        }
        if let Some(r) = self.loaded.get(name) {
            return Ok(Some(r.clone()));
        }
        let Some(resources) = self.resources else {
            return Ok(None);
        };
        let Some(kind) = self.kinds.get(name).copied() else {
            return Ok(None);
        };
        let Some(bytes) = resources.resource(name)? else {
            return Ok(None);
        };
        let resource = match kind {
            Kind::Brown => Resource::Brown(Arc::new(BrownCluster::read(name, &bytes)?)),
            Kind::WordCluster => Resource::WordCluster(Arc::new(WordClusterDictionary::read(&bytes))),
            Kind::Dictionary => Resource::Dictionary(Arc::new(
                Dictionary::read(&bytes).map_err(|e| Error::Resource { name: name.into(), message: e.to_string() })?,
            )),
            Kind::PosModel => return Ok(None),
        };
        self.loaded.insert(name.to_owned(), resource.clone());
        Ok(Some(resource))
    }

    fn has_resources(&self) -> bool {
        self.resources.is_some() || self.shared.is_some()
    }
}

/// A length parameter as a `usize`. Java would fail later, when the
/// generator runs (a negative array size); here the descriptor is refused.
fn non_negative(value: i32, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::Descriptor(format!("parameter {name} must not be negative")))
}
