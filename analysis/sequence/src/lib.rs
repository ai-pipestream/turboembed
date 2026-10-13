//! The sequence machinery of OpenNLP 3.x's taggers, ported for the
//! part-of-speech tagger, the chunker and the name finder: the beam
//! search, the BIO and BILOU sequence codecs, and the feature generation
//! framework with the XML descriptors models store.
//!
//! - [`beam`]: `BeamSearch` over a [`turbo_maxent::MaxentModel`], with
//!   the [`ContextGenerator`] and [`SequenceValidator`] traits a tagger
//!   plugs into it.
//! - [`codec`]: `BioCodec` and `BilouCodec` and their validators.
//! - [`featuregen`]: the feature generators and `GeneratorFactory`
//!   ([`featuregen::descriptor`]).
//! - [`pattern`]: `StringPattern` and `FeatureGeneratorUtil.tokenFeature`,
//!   the token classes several generators write, in both the UTF-16 unit
//!   version every published model was trained with and the code point
//!   version of the `ready/string-pattern-unicode` branch.
//! - [`postag`], [`chunker`], [`namefind`]: the context generators and
//!   validators of the three taggers, and the choices their factories make
//!   for a model.
//! - [`postag::dictionary`]: the part-of-speech tag dictionary a model
//!   stores as `tags.tagdict`, which only the sequence validator reads.
//! - [`postag::format`]: `POSTagFormatMapper`, the Penn Treebank and
//!   Universal Dependencies tag sets as enums, and the conversion
//!   `POSTaggerME` applies to the tags it returns.
//!
//! Everything here reproduces the Java of OpenNLP 3.x at commit
//! `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` (of the OpenNLP fork), down
//! to the order features are written in, so a model sums the same
//! parameters in the same order and the beam search returns the same
//! sequences with bit-identical probabilities and scores. Features are
//! built as Java modified UTF-8 ([`Features`]), so a feature cut inside a
//! surrogate pair, as Java's `substring` can cut it, is the same bytes the
//! model was trained with. The tests compare against files a JVM running
//! OpenNLP wrote (`golden/`, see the crate README).
//!
//! # Where OpenNLP's behaviour is surprising
//!
//! These are reproduced, because the models were trained with them, and
//! documented where they are implemented:
//!
//! - [`featuregen::Cached`] keys its cache by token index only, so in the
//!   default part-of-speech descriptor (`cache="true"` around the
//!   previous-tag features) every path of the beam sees the previous tags
//!   of the first path that reached a token.
//! - `AggregatedFeatureGeneratorFactory` with several children writes the
//!   features of every child but the first twice
//!   ([`featuregen::descriptor`]).
//! - `TokenClassFeatureGeneratorFactory` defaults `wordAndClass` to true,
//!   the class's constructor to false.
//! - OpenNLP 1.5 name finder models carry no descriptor, and 3.x runs
//!   them with its default one ([`namefind::DEFAULT_DESCRIPTOR`]). The
//!   generator hard-coded in `TokenNameFinderFactory.createContextGenerator`
//!   for a model without a descriptor is never used, since the default
//!   descriptor is always found. The two write the same features, but the
//!   descriptor writes the token class window before the token window.
//! - The UTF-16 version of `StringPattern` classes the empty token as all
//!   Hiragana (`jah`), and a letter outside the Basic Multilingual Plane
//!   as two non-letters.
//! - A `featureGenerators` element with a child other than `generator` is
//!   refused with the message "Unexpected element: featureGenerators",
//!   naming the parent instead of the child.
//! - `getStr` reports a parameter of the wrong type as "must be double!".
//! - `DefaultChunkerSequenceValidator` throws on a previous tag shorter
//!   than two characters; here it rejects the outcome.
//! - A case-insensitive tag dictionary is lowercased by walking a Java
//!   `HashMap`, so when several words lowercase to the same one, Java's
//!   hash order picks whose tags are kept ([`postag::TagDictionary::read`]).
//! - `POSTaggerME` returns UD tags unless asked for another format, so it
//!   converts the tags of a Penn model, such as the OpenNLP 1.5 English
//!   ones, to UD. The Penn table has no opening quote, so `` `` `` becomes
//!   `?`; so does every joined outcome of the UD English model (such as
//!   `AUX+PART`) when Penn tags are asked for ([`postag::format`] lists
//!   the rest).
//!
//! One is not reproduced yet: `QNModel.eval(context, probs)` adds its sums
//! to what `probs` already holds, and `BeamSearch` passes the same array to
//! every evaluation (one per thread, kept across sentences), so inside a
//! beam search each evaluation of a QN model starts from the probabilities
//! of the one before. GIS and perceptron models overwrite the array. The
//! search here evaluates every context from zero, which matches OpenNLP
//! for GIS and perceptron models but not for QN models; matching QN needs
//! an evaluation in `turbo-maxent` that starts from the caller's sums
//! (`tests/beam.rs` holds the ignored test and its golden file).

pub mod beam;
pub mod chunker;
pub mod codec;
mod error;
pub mod featuregen;
pub mod features;
mod java;
pub mod namefind;
pub mod pattern;
pub mod postag;

pub use beam::{AcceptAll, BeamSearch, ContextGenerator, Sequence, SequenceValidator, manifest_beam_size};
pub use codec::{BilouCodec, BioCodec, SequenceCodec, Span};
pub use error::{Error, Result};
pub use featuregen::{BoxedGenerator, FeatureGenerator};
pub use features::Features;
pub use java::compare_doubles;
pub use pattern::{PatternMode, StringPattern, token_feature};
