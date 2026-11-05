//! Head opening and decoding for the layers MultiTask writes.

use turbo::bundle::Bundle;
use turbo::head::{BiaffineParser, EditTreeLemmatizer, SentenceBoundary, TokenSplitter, TokenTagger};
use turbo::manifest::{HeadKind, HeadSpec, Scheme};
use turbo::status::Error as CoreError;
use turbo_analysis::annotate::Writes;
use turbo_analysis::{Ix, Layer, Span, Spans, TagSet};
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, MorphFeatures, PosTags, Sentences, Strs, Tokens, Words};

use crate::annotate::{decode_with, entity_layer, find_head, pos_layer, refuse, tail_taps};
use crate::lemma::{lemma_layer, lemmas_with};
use crate::parse::{dep_layer, parse_with};
use crate::segment::{sentence_ranges, split_with, word_layer};

use super::{Head, Input, language_of, pooling_of, spec_taps, wrong_head};

/// A layer a head writes: its head kind, how it is opened, and how it decodes shared features.
pub trait HeadLayer: Layer {
    const KIND: HeadKind;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError>;
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String>;
}

fn open_tags(bundle: &Bundle, spec: &HeadSpec, role: &str) -> Result<Head, CoreError> {
    if spec.scheme != Scheme::Tags {
        return Err(refuse(format!("head {:?}: a {role} head reads SCHEME_TAGS labels", spec.name)));
    }
    let tagger = TokenTagger::load(bundle, &spec.name)?;
    let labels: Vec<&str> = spec.labels.iter().map(String::as_str).collect();
    Ok(Head::Tagger { set: TagSet::new(&spec.name, &labels), tagger })
}

impl HeadLayer for PosTags {
    const KIND: HeadKind = HeadKind::TokenTagger;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        open_tags(bundle, spec, "part-of-speech")
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Tagger { tagger, set } = head else { return wrong_head() };
        let f = input.tapped(tagger.spec().word_pooling, tagger.taps())?;
        let decoded =
            decode_with(tagger, &f, &input.forms, language_of(tagger.spec(), input)).map_err(|e| e.to_string())?;
        pos_layer(input.tokens, &input.seqs, &decoded, set)
    }
}

impl HeadLayer for Entities {
    const KIND: HeadKind = HeadKind::TokenTagger;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        if spec.scheme == Scheme::Tags {
            return Err(refuse(format!(
                "head {:?}: an entity head reads SCHEME_BIO or SCHEME_BILOU labels",
                spec.name
            )));
        }
        let tagger = TokenTagger::load(bundle, &spec.name)?;
        let types: Vec<&str> = tagger.types().iter().map(String::as_str).collect();
        Ok(Head::Tagger { set: TagSet::new(&spec.name, &types), tagger })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Tagger { tagger, set } = head else { return wrong_head() };
        let f = input.tapped(tagger.spec().word_pooling, tagger.taps())?;
        let decoded =
            decode_with(tagger, &f, &input.forms, language_of(tagger.spec(), input)).map_err(|e| e.to_string())?;
        entity_layer(tagger, input.view, input.tokens, &input.seqs, &decoded, set)
    }
}

impl HeadLayer for MorphFeatures<Tokens> {
    const KIND: HeadKind = HeadKind::TokenTagger;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        open_tags(bundle, spec, "morphology")
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Tagger { tagger, set } = head else { return wrong_head() };
        let f = input.tapped(tagger.spec().word_pooling, tagger.taps())?;
        let decoded =
            decode_with(tagger, &f, &input.forms, language_of(tagger.spec(), input)).map_err(|e| e.to_string())?;
        let tags = pos_layer(input.tokens, &input.seqs, &decoded, set)?;
        let mut feats = Strs::default();
        for &t in &tags.tags {
            feats.push(set.label(t));
        }
        Ok(MorphFeatures::new(feats))
    }
}

impl HeadLayer for Dependencies<Tokens> {
    const KIND: HeadKind = HeadKind::BiaffineParser;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        let parser = BiaffineParser::load(bundle, &spec.name)?;
        let labels: Vec<&str> = parser.labels().iter().map(String::as_str).collect();
        Ok(Head::Parser { relations: TagSet::new(&spec.name, &labels), parser })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Parser { parser, relations } = head else { return wrong_head() };
        let f = input.tapped(parser.spec().word_pooling, parser.taps())?;
        let parses = parse_with(parser, &f, language_of(parser.spec(), input)).map_err(|e| e.to_string())?;
        dep_layer(input.tokens, &input.seqs, &parses, relations)
    }
}

impl HeadLayer for Lemmas {
    const KIND: HeadKind = HeadKind::EditTreeLemmatizer;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        let lemmatizer = EditTreeLemmatizer::load(bundle, &spec.name)?;
        let pos = spec.pos_input.as_ref().map(|p| TokenTagger::load(bundle, &p.head)).transpose()?;
        Ok(Head::Lemma { lemmatizer, pos })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Lemma { lemmatizer, pos } = head else { return wrong_head() };
        let own = lemmatizer.spec().word_pooling;
        let f = input.tapped(own, lemmatizer.taps())?;
        let pf = match pos {
            Some(t) if t.spec().word_pooling != own || t.taps() != lemmatizer.taps() => {
                Some(input.tapped(t.spec().word_pooling, t.taps())?)
            }
            _ => None,
        };
        let lemmas = lemmas_with(
            lemmatizer,
            pos.as_ref(),
            None,
            &f,
            pf.as_deref(),
            &input.forms,
            language_of(lemmatizer.spec(), input),
        )
        .map_err(|e| e.to_string())?;
        lemma_layer(input.tokens, &input.seqs, lemmas)
    }
}

impl HeadLayer for Sentences {
    const KIND: HeadKind = HeadKind::SentenceBoundary;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        Ok(Head::Sentence { boundary: SentenceBoundary::load(bundle, &spec.name)? })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Sentence { boundary } = head else { return wrong_head() };
        let mut out = Sentences { spans: Spans::new(input.view.id()) };
        let push = |out: &mut Sentences, s: &[usize]| {
            let (a, b) = (s[0] as u32, s[s.len() - 1] as u32);
            out.spans.push(Span::new(input.tokens.spans.get(Ix::new(a)).start, input.tokens.spans.get(Ix::new(b)).end));
        };
        if let Some(whole) = &input.whole {
            let p = pooling_of(&input.poolings, boundary.spec().word_pooling)?;
            let wf = tail_taps(&whole.features[p], input.taps, spec_taps(boundary.spec()));
            for r in sentence_ranges(boundary, &wf, &whole.forms).map_err(|e| e.to_string())? {
                push(&mut out, &whole.seqs[0][r]);
            }
            return Ok(out);
        }
        for s in &input.seqs {
            push(&mut out, s);
        }
        Ok(out)
    }
}

impl HeadLayer for Words {
    const KIND: HeadKind = HeadKind::TokenSplit;
    fn open(bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        Ok(Head::Split { splitter: TokenSplitter::load(bundle, &spec.name)? })
    }
    fn write(head: &Head, input: &Input<'_>) -> Result<Self, String> {
        let Head::Split { splitter } = head else { return wrong_head() };
        let f = input.tapped(splitter.spec().word_pooling, spec_taps(splitter.spec()))?;
        let split =
            split_with(splitter, &f, &input.forms, language_of(splitter.spec(), input)).map_err(|e| e.to_string())?;
        word_layer(input.view, input.tokens, &input.seqs, split)
    }
}

/// The layers a [`MultiTask`](super::MultiTask) writes: one head each.
pub trait Outputs: Writes + Sized {
    fn open(bundle: &Bundle, names: &[Option<&str>]) -> Result<Vec<Head>, CoreError>;
    fn write(heads: &[Head], input: &Input<'_>) -> Result<Self, String>;
}

macro_rules! outputs {
    ($($L:ident $i:tt),+) => {
        impl<$($L: HeadLayer),+> Outputs for ($($L,)+) {
            fn open(bundle: &Bundle, names: &[Option<&str>]) -> Result<Vec<Head>, CoreError> {
                Ok(vec![$($L::open(bundle, find_head(bundle, names[$i], $L::ID, $L::KIND)?)?),+])
            }
            fn write(heads: &[Head], input: &Input<'_>) -> Result<Self, String> {
                std::thread::scope(|s| {
                    let running = ($(s.spawn(|| $L::write(&heads[$i], input)),)+);
                    Ok(($(running.$i.join().map_err(|_| format!("head {:?} panicked", heads[$i].spec().name))??,)+))
                })
            }
        }
    };
}

outputs!(A 0);
outputs!(A 0, B 1);
outputs!(A 0, B 1, C 2);
outputs!(A 0, B 1, C 2, D 3);
outputs!(A 0, B 1, C 2, D 3, E 4);
outputs!(A 0, B 1, C 2, D 3, E 4, F 5);
outputs!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
outputs!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
