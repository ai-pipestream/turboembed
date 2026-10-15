//! The span ruler: a token pattern matcher and phrase matchers run
//! together, their labels merged, overlaps settled by one policy, written
//! as a `turbo:matches` layer (spaCy's `SpanRuler`).

use std::sync::Arc;

use turbo_analysis::{IxRange, Tag, TagSet, ViewId};
use turbo_matcher_layers::Matches;

use crate::matcher::{Greedy, Match, MatchError, Matcher, keep_greedy};
use crate::pattern::PatternError;
use crate::phrase::PhraseMatcher;
use crate::seq::Sequence;
use crate::vocab::Fnv;

/// Which matches a ruler keeps where they overlap, across all labels.
/// Each rule's own `Greedy` applies first.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Overlap {
    /// Every match, overlapping and nested ones included: a span group.
    All,
    /// The longest, then the longest of the rest that overlap none kept,
    /// the leftmost first among equally long ones, as spaCy's
    /// `filter_spans`. Where one run of tokens has several labels, the
    /// first label added wins.
    Longest,
    /// The leftmost, the longest first among those that start together,
    /// then the leftmost of the rest that overlap none kept.
    First,
}

/// A token pattern matcher and any number of phrase matchers, run as one.
pub struct Ruler {
    matcher: Option<Matcher>,
    phrases: Vec<PhraseMatcher>,
    overlap: Overlap,
    labels: Arc<TagSet>,
    /// Per source (the token matcher first, when there is one), its label
    /// numbers in `labels`.
    maps: Vec<Vec<u16>>,
    config: u64,
}

impl Ruler {
    /// The labels are merged by name: a label in two sources is one label,
    /// numbered where it first appears, token matcher first.
    pub fn new(matcher: Option<Matcher>, phrases: Vec<PhraseMatcher>, overlap: Overlap) -> Result<Ruler, PatternError> {
        let mut names: Vec<String> = Vec::new();
        let mut maps = Vec::new();
        let sources = matcher.iter().map(Matcher::labels).chain(phrases.iter().map(PhraseMatcher::labels));
        for set in sources {
            let mut map = Vec::with_capacity(set.len());
            for i in 0..set.len() {
                let name = set.label(Tag(i as u16));
                let at = match names.iter().position(|n| n == name) {
                    Some(at) => at,
                    None => {
                        names.push(name.into());
                        names.len() - 1
                    }
                };
                if at > usize::from(u16::MAX) {
                    return Err(PatternError::Labels);
                }
                map.push(at as u16);
            }
            maps.push(map);
        }
        let mut config = Fnv::default();
        config.u64(overlap as u64);
        config.u64(matcher.as_ref().map_or(0, Matcher::config_hash));
        for p in &phrases {
            config.u64(p.config_hash());
        }
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        Ok(Ruler { matcher, phrases, overlap, labels: TagSet::new("ruler", &refs), maps, config: config.0 })
    }

    pub fn labels(&self) -> &Arc<TagSet> {
        &self.labels
    }

    pub fn matcher(&self) -> Option<&Matcher> {
        self.matcher.as_ref()
    }

    pub fn phrases(&self) -> &[PhraseMatcher] {
        &self.phrases
    }

    pub fn config_hash(&self) -> u64 {
        self.config
    }

    /// Every source's matches under the merged labels, with the overlap
    /// policy applied, in order of start, end and label.
    pub fn find(&self, seq: &Sequence<'_>) -> Result<Vec<Match>, MatchError> {
        let mut all = Vec::new();
        let mut maps = self.maps.iter();
        if let Some(m) = &self.matcher {
            let map = maps.next().expect("a map per source");
            all.extend(m.find(seq)?.into_iter().map(|x| Match { label: Tag(map[x.label.0 as usize]), ..x }));
        }
        for (p, map) in self.phrases.iter().zip(maps) {
            all.extend(p.find(seq)?.into_iter().map(|x| Match { label: Tag(map[x.label.0 as usize]), ..x }));
        }
        all.sort_unstable();
        all.dedup();
        let mut kept = match self.overlap {
            Overlap::All => all,
            Overlap::Longest => keep_greedy(&all, Greedy::Longest),
            Overlap::First => keep_greedy(&all, Greedy::First),
        };
        kept.sort_unstable();
        Ok(kept)
    }

    /// The matches as a layer over the tokens' view.
    pub fn layer(&self, seq: &Sequence<'_>, view: ViewId) -> Result<Matches, MatchError> {
        let found = self.find(seq)?;
        let mut out = Matches::new(view, self.labels.clone());
        for m in found {
            out.spans.push(seq.span(m.start as usize, m.end as usize));
            out.tokens.push(IxRange::new(m.start, m.end));
            out.label.push(m.label);
        }
        Ok(out)
    }
}
