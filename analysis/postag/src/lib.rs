//! Part-of-speech tagging, ported from OpenNLP 3.x's `POSTaggerME` onto
//! the analysis API.
//!
//! [`PosModel`] opens a `POSModel` file and makes the choices
//! `POSTaggerFactory` and `POSTaggerME` make for it: the context
//! generator (the model's descriptor, or OpenNLP's default one, behind
//! the beam-sized cache), the tag dictionary as the sequence validator,
//! the beam size from the manifest and the tag format mapper.
//! [`PosTagger`] runs the beam search over the tokens of each sentence
//! and is the annotator that reads `opennlp:tokens` and writes
//! `opennlp:pos`: a tag and its probability per token, with the tag set
//! the tags are in. The beam search, the feature generators, the tag
//! dictionary and the Penn Treebank and Universal Dependencies mapping are
//! `turbo-sequence`'s, whose documentation lists the OpenNLP behaviour
//! they reproduce that is surprising.
//!
//! The tags are the JVM's and the probabilities are the JVM's doubles,
//! bit for bit; the tests compare both with files the JVM wrote
//! (`analysis/goldens/data/postag`, see `README.md` here).
//!
//! # Where OpenNLP's behaviour is surprising
//!
//! - `POSTaggerME` returns Universal Dependencies tags unless asked for
//!   another format, so the tags of a model with Penn Treebank outcomes
//!   come out converted, and a Penn outcome the table lacks (such as
//!   `` `` ``) comes out as `?`. Asked for Penn tags, a Universal
//!   Dependencies model's joined outcomes (`AUX+PART`, which OpenNLP's
//!   CoNLL-U reader makes of a multiword token) come out as `?` too.
//! - Asking for the `UNKNOWN` format converts like asking for the other
//!   format: a Universal Dependencies model then returns Penn tags.
//! - The context generator's cache is keyed by the token index and the
//!   two previous tags, and the default descriptor caches its generators
//!   by the token index alone, so every path of the beam sees the
//!   previous tags of the first path that reached a token
//!   (`turbo_sequence::featuregen::Cached`). The models were trained
//!   this way, and the search reproduces it.
//! - `tag` on a sentence where no sequence scores above the search's
//!   minimum returns an array of nulls, and `probs` then throws. The
//!   annotator fails the document with a message instead.
//! - A `BeamSize` of zero in a manifest makes a tagger that tags nothing;
//!   it is refused when the model is opened.

pub mod error;
pub mod model;
pub mod tagger;

pub use error::ModelError;
pub use model::{COMPONENT_NAME, FACTORY, PosModel};
pub use tagger::PosTagger;
pub use turbo_sequence::postag::{PosTagFormat, TagDictionary};
