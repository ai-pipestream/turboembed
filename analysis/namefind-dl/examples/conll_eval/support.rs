use std::borrow::Cow;
use std::collections::BTreeSet;

use turbo_formats::conll::{Conll03Language, Conll03Reader, EntityTypes};
use turbo_namefind_dl::decode::Candidate;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Person,
    Organization,
    Location,
    Misc,
}

impl Kind {
    pub const ALL: [Self; 4] = [Self::Person, Self::Organization, Self::Location, Self::Misc];

    pub fn parse(label: &str) -> Result<Self, String> {
        match label {
            "PER" | "person" => Ok(Self::Person),
            "ORG" | "organization" => Ok(Self::Organization),
            "LOC" | "location" => Ok(Self::Location),
            "MISC" | "misc" => Ok(Self::Misc),
            _ => Err("entity type is outside the four CoNLL-2003 kinds".into()),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Person => "person",
            Self::Organization => "organization",
            Self::Location => "location",
            Self::Misc => "misc",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Entity {
    pub start: usize,
    pub end: usize,
    pub kind: Kind,
}

#[derive(Debug)]
pub struct Sentence<'a> {
    pub words: Vec<Cow<'a, str>>,
    pub entities: BTreeSet<Entity>,
}

pub fn read(source: &str) -> Result<Vec<Sentence<'_>>, String> {
    // Reader errors can contain Reuters text. Report positions, not source excerpts.
    let reader = Conll03Reader::new(source, Conll03Language::English, EntityTypes::ALL)
        .map_err(|_| "CoNLL input is too large".to_string())?;
    let mut out = Vec::new();
    for (i, sample) in reader.enumerate() {
        let sample = sample.map_err(|_| format!("invalid CoNLL record at sample {}", i + 1))?;
        if sample.tokens().iter().any(|t| t.text.is_empty() || t.text.chars().any(char::is_whitespace)) {
            return Err(format!("empty or whitespace token at sample {}", i + 1));
        }
        let entities = sample
            .names()
            .iter()
            .map(|n| {
                Ok(Entity { start: n.start as usize, end: n.end as usize, kind: Kind::parse(n.kind.unwrap_or(""))? })
            })
            .collect::<Result<_, String>>()?;
        out.push(Sentence { words: sample.tokens().iter().map(|t| t.text.clone()).collect(), entities });
    }
    if out.is_empty() {
        return Err("CoNLL input has no samples".into());
    }
    Ok(out)
}

pub fn aligned(gold: &[Sentence<'_>], peer: &[Sentence<'_>]) -> Result<(), String> {
    if gold.len() != peer.len() {
        return Err(format!("sample count differs: gold {}, upstream {}", gold.len(), peer.len()));
    }
    for (i, (a, b)) in gold.iter().zip(peer).enumerate() {
        if a.words != b.words {
            return Err(format!("token alignment differs at sample {}", i + 1));
        }
    }
    Ok(())
}

pub fn joined(words: &[impl AsRef<str>]) -> (String, Vec<(usize, usize)>) {
    let mut text = String::new();
    let spans = words
        .iter()
        .map(|word| {
            let word = word.as_ref();
            if !text.is_empty() {
                text.push(' ');
            }
            let start = text.len();
            text.push_str(word);
            (start, text.len())
        })
        .collect();
    (text, spans)
}

pub fn project(
    text: &str,
    spans: &[(usize, usize)],
    candidates: &[Candidate],
    kinds: &[Kind],
) -> Result<BTreeSet<Entity>, String> {
    let mut out = BTreeSet::new();
    for c in candidates {
        if c.start >= c.end || text.get(c.start..c.end).is_none() {
            return Err("native entity has invalid UTF-8 byte bounds".into());
        }
        if !c.probability.is_finite() || !(0.0..=1.0).contains(&c.probability) {
            return Err("native entity probability is invalid".into());
        }
        let kind = *kinds.get(c.kind).ok_or("native entity kind is out of bounds")?;
        let start = spans.partition_point(|&(_, end)| end <= c.start);
        let end = spans.partition_point(|&(start, _)| start < c.end);
        if start >= end {
            return Err("native entity overlaps no input token".into());
        }
        // As in the existing optional corpus test, repeated subword names
        // that project to one token/type count once.
        out.insert(Entity { start, end, kind });
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub gold: usize,
    pub predicted: usize,
    pub correct: usize,
}

impl Counts {
    pub fn update(&mut self, gold: &BTreeSet<Entity>, predicted: &BTreeSet<Entity>, kind: Option<Kind>) {
        let keep = |e: &&Entity| kind.is_none_or(|k| e.kind == k);
        self.gold += gold.iter().filter(keep).count();
        self.predicted += predicted.iter().filter(keep).count();
        self.correct += predicted.intersection(gold).filter(keep).count();
    }

    pub fn precision(self) -> f64 {
        if self.predicted == 0 { 0.0 } else { self.correct as f64 / self.predicted as f64 }
    }

    pub fn recall(self) -> f64 {
        if self.gold == 0 { 0.0 } else { self.correct as f64 / self.gold as f64 }
    }

    pub fn f1(self) -> f64 {
        if self.gold + self.predicted == 0 {
            0.0
        } else {
            2.0 * self.correct as f64 / (self.gold + self.predicted) as f64
        }
    }
}
