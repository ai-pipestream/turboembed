//! The learned components of an in-process analyzer: the heads of the
//! bundle `turbo_analyzer_desc.bundle_path` names (`manifest.heads`,
//! docs/bundle.md "Heads"), run over one encoder pass a sentence as the
//! analysis annotators run them (`turbo-tagger`), through the same
//! decoders (`head::run`). The part-of-speech, entity, dependency and
//! lemma layers each come from the one head of the bundle that writes
//! that layer; a head of a kind the analyzer does not run leaves its
//! layer unsupported, named in the refusal.

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo_analysis::TagSet;
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, PosTags, Tokens};

use crate::bundle::Bundle;
use crate::head::run::{Decoded, LemmaTags, Parse, decode_tagger, lemmatize_sequences, parse_sequences, same_layer};
use crate::head::{BiaffineParser, EditTreeLemmatizer, TokenTagger};
use crate::manifest::{HeadKind, HeadSpec, Scheme, WordPooling};
use crate::status::{BUNDLE_INVALID, Error, INVALID_ARGUMENT, Result, UNSUPPORTED_TASK};
use turbo_analysis::Layer as _;

mod words;
use words::Words;

/// An entity a sequence's labels mark: words `start..end` of the
/// sequence, its kind in the head's kinds, and the mean of its words'
/// probabilities.
pub(crate) struct Entity {
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) kind: u32,
    pub(crate) prob: f64,
}

/// What a task asks of the heads: each layer wanted, and for the lemmas
/// the tags given by the caller, when any.
#[derive(Default)]
pub(crate) struct Want<'a> {
    pub(crate) pos: bool,
    pub(crate) entities: bool,
    pub(crate) dependencies: bool,
    pub(crate) lemmas: bool,
    pub(crate) tags: Option<&'a [Vec<u32>]>,
}

/// The layers the heads wrote, each for every sequence given.
#[derive(Default)]
pub(crate) struct Written {
    pub(crate) pos: Option<Vec<Decoded>>,
    pub(crate) entities: Option<Vec<Vec<Entity>>>,
    pub(crate) dependencies: Option<Vec<Parse>>,
    pub(crate) lemmas: Option<Vec<Vec<String>>>,
}

/// Why a layer is not written: no bundle, no head for it, or a head the
/// analyzer does not run.
#[derive(Clone, Debug)]
pub(crate) enum Missing {
    NoHead,
    Kind { head: String, kind: HeadKind },
}

/// A layer's head, or why the bundle gives none.
type Loaded<T> = std::result::Result<T, Missing>;

/// The heads of one bundle and the encoder that feeds them.
pub(crate) struct BundleHeads {
    words: Mutex<Words>,
    pos: Loaded<(TokenTagger, Arc<TagSet>)>,
    entities: Loaded<(TokenTagger, Arc<TagSet>)>,
    dependencies: Loaded<(BiaffineParser, Arc<TagSet>)>,
    lemmas: Loaded<(EditTreeLemmatizer, Option<TokenTagger>)>,
    pub(crate) model_id: String,
    pub(crate) model_revision: String,
    pub(crate) manifest_sha256: String,
    pub(crate) backend: String,
    pub(crate) device: String,
}

/// The task's refusal when the bundle gives no head for it.
pub(crate) fn unsupported(task: &str, layer: &str, why: Option<&Missing>) -> Error {
    let reason = match why {
        None => "requires a bundle with a head for it (turbo_analyzer_desc.bundle_path)".to_owned(),
        Some(Missing::NoHead) => format!("the bundle has no head writing {layer}"),
        Some(Missing::Kind { head, kind }) => {
            format!("head {head:?} writes {layer} as a {kind:?} head, which turbo_analyzer does not run")
        }
    };
    Error::new(UNSUPPORTED_TASK, format!("{task}: {reason}; no fallback task is substituted"))
}

/// The one head of `bundle` that writes `layer`, if any, or the refusal
/// when several do.
fn head_for<'a>(bundle: &'a Bundle, layer: &str) -> Result<Option<&'a HeadSpec>> {
    let mut found = bundle.manifest.heads.iter().filter(|h| same_layer(&h.layer, layer));
    match (found.next(), found.next()) {
        (Some(h), None) => Ok(Some(h)),
        (None, _) => Ok(None),
        _ => Err(Error::new(BUNDLE_INVALID, format!("several heads write {layer}; turbo_analyzer runs one a layer"))),
    }
}

fn kind_of(spec: &HeadSpec, want: HeadKind) -> Loaded<()> {
    if spec.kind == want { Ok(()) } else { Err(Missing::Kind { head: spec.name.clone(), kind: spec.kind }) }
}

fn tagset(name: &str, labels: &[String]) -> Arc<TagSet> {
    let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
    TagSet::new(name, &labels)
}

impl BundleHeads {
    /// The heads of the bundle at `path`. A head the analyzer cannot run
    /// is remembered for its layer's refusal; a head whose width is not
    /// the encoder's is refused here.
    pub(crate) fn open(path: &str) -> Result<BundleHeads> {
        let dir = Path::new(path);
        let bundle = Bundle::open(dir)?;
        let words = Words::open(dir, &bundle)?;
        let hidden = words.hidden();
        let width = |name: &str, h: usize| -> Result<()> {
            if h != hidden {
                return Err(Error::new(
                    INVALID_ARGUMENT,
                    format!("head {name:?} reads {h} values a word, the encoder gives {hidden}"),
                ));
            }
            Ok(())
        };
        let pos = match head_for(&bundle, PosTags::ID)? {
            None => Err(Missing::NoHead),
            Some(spec) => match kind_of(spec, HeadKind::TokenTagger) {
                Err(m) => Err(m),
                Ok(()) => {
                    if spec.scheme != Scheme::Tags {
                        return Err(Error::new(
                            BUNDLE_INVALID,
                            format!("head {:?}: a part-of-speech head reads SCHEME_TAGS labels", spec.name),
                        ));
                    }
                    let t = TokenTagger::load(&bundle, &spec.name)?;
                    width(&spec.name, t.hidden())?;
                    let set = tagset(&spec.name, &spec.labels);
                    Ok((t, set))
                }
            },
        };
        let entities = match head_for(&bundle, Entities::ID)? {
            None => Err(Missing::NoHead),
            Some(spec) => match kind_of(spec, HeadKind::TokenTagger) {
                Err(m) => Err(m),
                Ok(()) => {
                    if spec.scheme == Scheme::Tags {
                        return Err(Error::new(
                            BUNDLE_INVALID,
                            format!("head {:?}: an entity head reads SCHEME_BIO or SCHEME_BILOU labels", spec.name),
                        ));
                    }
                    let t = TokenTagger::load(&bundle, &spec.name)?;
                    width(&spec.name, t.hidden())?;
                    let set = tagset(&spec.name, t.types());
                    Ok((t, set))
                }
            },
        };
        let dependencies = match head_for(&bundle, Dependencies::<Tokens>::ID)? {
            None => Err(Missing::NoHead),
            Some(spec) => match kind_of(spec, HeadKind::BiaffineParser) {
                Err(m) => Err(m),
                Ok(()) => {
                    let p = BiaffineParser::load(&bundle, &spec.name)?;
                    width(&spec.name, p.hidden())?;
                    let set = tagset(&spec.name, p.labels());
                    Ok((p, set))
                }
            },
        };
        let lemmas = match head_for(&bundle, Lemmas::ID)? {
            None => Err(Missing::NoHead),
            Some(spec) => match kind_of(spec, HeadKind::EditTreeLemmatizer) {
                Err(m) => Err(m),
                Ok(()) => {
                    let l = EditTreeLemmatizer::load(&bundle, &spec.name)?;
                    width(&spec.name, l.hidden())?;
                    let pos = match &spec.pos_input {
                        Some(p) => {
                            let t = TokenTagger::load(&bundle, &p.head)?;
                            width(&p.head, t.hidden())?;
                            Some(t)
                        }
                        None => None,
                    };
                    Ok((l, pos))
                }
            },
        };
        let m = &bundle.manifest.model;
        Ok(BundleHeads {
            backend: words.backend.clone(),
            device: words.device.clone(),
            words: Mutex::new(words),
            pos,
            entities,
            dependencies,
            lemmas,
            model_id: m.id.clone(),
            model_revision: m.revision.clone(),
            manifest_sha256: bundle.manifest_sha256.clone(),
        })
    }

    /// The part-of-speech head's tag set, or why there is none.
    pub(crate) fn pos_tagset(&self) -> Result<&Arc<TagSet>> {
        self.pos.as_ref().map(|(_, set)| set).map_err(|m| unsupported("POS tagging", "pos", Some(m)))
    }

    pub(crate) fn entity_kinds(&self) -> Result<&Arc<TagSet>> {
        self.entities
            .as_ref()
            .map(|(_, set)| set)
            .map_err(|m| unsupported("named entity recognition", "entities", Some(m)))
    }

    pub(crate) fn relations(&self) -> Result<&Arc<TagSet>> {
        self.dependencies
            .as_ref()
            .map(|(_, set)| set)
            .map_err(|m| unsupported("dependency parsing", "dependencies", Some(m)))
    }

    /// Whether the bundle gives each layer.
    pub(crate) fn has(&self) -> Want<'static> {
        Want {
            pos: self.pos.is_ok(),
            entities: self.entities.is_ok(),
            dependencies: self.dependencies.is_ok(),
            lemmas: self.lemmas.is_ok(),
            tags: None,
        }
    }

    /// The labels of the tagger the lemma head reads tags from: the
    /// caller's tags for `turbo_analyzer_lemmatize` must be among them.
    /// None when the bundle lemmatizes without tags.
    pub(crate) fn lemma_tag_labels(&self) -> Result<Option<&[String]>> {
        match &self.lemmas {
            Ok((_, pos)) => Ok(pos.as_ref().map(TokenTagger::labels)),
            Err(m) => Err(unsupported("lemmatization", "lemmas", Some(m))),
        }
    }

    /// Every layer `want` asks for over `sequences` of words, from one
    /// read of each window whatever the heads: the poolings each head
    /// reads are taken from the same rows.
    pub(crate) fn run(&self, sequences: &[Vec<&str>], want: &Want<'_>) -> Result<Written> {
        let mut out = Written::default();
        if sequences.iter().all(Vec::is_empty) {
            if want.pos {
                self.pos_tagset()?;
                out.pos = Some(sequences.iter().map(|_| Decoded { tags: Vec::new(), prob: Vec::new() }).collect());
            }
            if want.entities {
                self.entity_kinds()?;
                out.entities = Some(sequences.iter().map(|_| Vec::new()).collect());
            }
            if want.dependencies {
                self.relations()?;
                out.dependencies =
                    Some(sequences.iter().map(|_| Parse { heads: Vec::new(), relations: Vec::new() }).collect());
            }
            if want.lemmas {
                self.lemma_tag_labels()?;
                out.lemmas = Some(sequences.iter().map(|_| Vec::new()).collect());
            }
            return Ok(out);
        }
        let mut poolings: Vec<WordPooling> = Vec::new();
        let mut pool = |p: WordPooling| {
            if !poolings.contains(&p) {
                poolings.push(p);
            }
        };
        let pos =
            want.pos.then(|| self.pos.as_ref().map_err(|m| unsupported("POS tagging", "pos", Some(m)))).transpose()?;
        let ents = want
            .entities
            .then(|| self.entities.as_ref().map_err(|m| unsupported("named entity recognition", "entities", Some(m))))
            .transpose()?;
        let deps = want
            .dependencies
            .then(|| self.dependencies.as_ref().map_err(|m| unsupported("dependency parsing", "dependencies", Some(m))))
            .transpose()?;
        let lem = want
            .lemmas
            .then(|| self.lemmas.as_ref().map_err(|m| unsupported("lemmatization", "lemmas", Some(m))))
            .transpose()?;
        if let Some((t, _)) = pos {
            pool(t.spec().word_pooling);
        }
        if let Some((t, _)) = ents {
            pool(t.spec().word_pooling);
        }
        if let Some((p, _)) = deps {
            pool(p.spec().word_pooling);
        }
        if let Some((l, t)) = lem {
            pool(l.spec().word_pooling);
            if let (Some(t), None) = (t, want.tags) {
                pool(t.spec().word_pooling);
            }
        }
        let features = self.words.lock().unwrap_or_else(PoisonError::into_inner).features(sequences, &poolings)?;
        let of = |p: WordPooling| -> &[Vec<f32>] { &features[poolings.iter().position(|&q| q == p).expect("pooled")] };
        if let Some((t, _)) = pos {
            out.pos = Some(decode_tagger(t, of(t.spec().word_pooling), sequences)?);
        }
        if let Some((t, _)) = ents {
            let decoded = decode_tagger(t, of(t.spec().word_pooling), sequences)?;
            let mut all = Vec::with_capacity(decoded.len());
            for d in &decoded {
                let mut found = Vec::new();
                for w in t.spans(&d.tags)? {
                    let p = &d.prob[w.start as usize..w.end as usize];
                    let prob = p.iter().sum::<f64>() / p.len() as f64;
                    found.push(Entity { start: w.start, end: w.end, kind: w.kind, prob });
                }
                all.push(found);
            }
            out.entities = Some(all);
        }
        if let Some((p, _)) = deps {
            out.dependencies = Some(parse_sequences(p, of(p.spec().word_pooling))?);
        }
        if let Some((l, pos_tagger)) = lem {
            let own = l.spec().word_pooling;
            let pos_features = match (pos_tagger, want.tags) {
                (Some(t), None) if t.spec().word_pooling != own => Some(of(t.spec().word_pooling)),
                _ => None,
            };
            let tags = LemmaTags { pos_tagger: pos_tagger.as_ref(), pos: pos_features, given: want.tags };
            out.lemmas = Some(lemmatize_sequences(l, tags, None, of(own), sequences)?);
        }
        Ok(out)
    }
}
