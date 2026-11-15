//! The precision-ranked sieves of Lee et al. (Computational Linguistics
//! 2013), "Deterministic Coreference Resolution Based on Entity-Centric,
//! Precision-Ranked Rules" (`SieveResolver`).
//!
//! Each sieve visits unresolved mentions in text order and tests their
//! candidate antecedents in salience order; the first that passes joins
//! the anaphor's cluster. The speaker sieve handles first and second
//! person pronouns before the string sieves, from the speakers layer when
//! there is one and from quoted-speech attribution otherwise.

use crate::clusters::Clusters;
use crate::input::Labeled;
use crate::layer::Person;
use crate::lexicon;
use crate::mention::Mention;

mod matches;
mod speaker;

/// How many sentences back a pronoun may find its antecedent.
pub(crate) const PRONOUN_WINDOW: usize = 3;
/// No limit on how many sentences back an antecedent may lie.
const UNLIMITED: usize = usize::MAX;

/// The step that linked an anaphor to its antecedent, the tag of each
/// edge in `Edges<Mentions, Mentions>`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Sieve {
    Speaker,
    ExactMatch,
    RelaxedStringMatch,
    Acronym,
    PersonName,
    StrictHead,
    HeadNoModifiers,
    HeadNoInclusion,
    ProperHead,
    RelaxedHead,
    PronounMatch,
}

impl Sieve {
    pub const ALL: [Sieve; 11] = [
        Sieve::Speaker,
        Sieve::ExactMatch,
        Sieve::RelaxedStringMatch,
        Sieve::Acronym,
        Sieve::PersonName,
        Sieve::StrictHead,
        Sieve::HeadNoModifiers,
        Sieve::HeadNoInclusion,
        Sieve::ProperHead,
        Sieve::RelaxedHead,
        Sieve::PronounMatch,
    ];

    /// The edge tag label.
    pub fn label(self) -> &'static str {
        match self {
            Sieve::Speaker => "speaker",
            Sieve::ExactMatch => "exact",
            Sieve::RelaxedStringMatch => "relaxed",
            Sieve::Acronym => "acronym",
            Sieve::PersonName => "personName",
            Sieve::StrictHead => "strictHead",
            Sieve::HeadNoModifiers => "headNoModifiers",
            Sieve::HeadNoInclusion => "headNoInclusion",
            Sieve::ProperHead => "properHead",
            Sieve::RelaxedHead => "relaxedHead",
            Sieve::PronounMatch => "pronounMatch",
        }
    }
}

/// One resolved link: the anaphor joined the antecedent's cluster.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Link {
    pub anaphor: usize,
    pub antecedent: usize,
    pub sieve: Sieve,
    pub score: f64,
}

/// The resolver over one document's mentions.
pub(crate) struct Resolver<'a> {
    pub mentions: Vec<Mention>,
    pub clusters: Clusters,
    pub forms: &'a [&'a str],
    pub sentence_of_token: &'a [usize],
    pub speakers: Option<&'a [Labeled]>,
    pub person_types: &'a [String],
    pub neutral_types: &'a [String],
    pub by_sentence: Vec<Vec<usize>>,
    pub speaker_of: Option<Vec<String>>,
    pub links: Vec<Link>,
}

impl<'a> Resolver<'a> {
    pub fn new(
        mentions: Vec<Mention>,
        forms: &'a [&'a str],
        sentence_of_token: &'a [usize],
        speakers: Option<&'a [Labeled]>,
        person_types: &'a [String],
        neutral_types: &'a [String],
    ) -> Resolver<'a> {
        let mut by_sentence: Vec<Vec<usize>> = Vec::new();
        for (i, m) in mentions.iter().enumerate() {
            if by_sentence.len() <= m.sentence {
                by_sentence.resize(m.sentence + 1, Vec::new());
            }
            by_sentence[m.sentence].push(i);
        }
        Resolver {
            clusters: Clusters::new(&mentions),
            mentions,
            forms,
            sentence_of_token,
            speakers,
            person_types,
            neutral_types,
            by_sentence,
            speaker_of: None,
            links: Vec::new(),
        }
    }

    /// Joins the anaphor to the antecedent's cluster and records the link
    /// when the clusters were apart and their types allow the merge.
    pub fn link(&mut self, antecedent: usize, anaphor: usize, sieve: Sieve, score: f64) -> bool {
        let apart = self.clusters.find(antecedent) != self.clusters.find(anaphor);
        let joined = self.clusters.union(antecedent, anaphor);
        if apart && joined {
            self.links.push(Link { anaphor, antecedent, sieve, score });
        }
        joined
    }

    /// Runs every sieve in precision order: speaker, the string sieves,
    /// the head sieves, then pronouns.
    pub fn resolve(&mut self) {
        self.speaker_sieve();
        let nominal = |r: &Resolver, j: usize| r.nominal(j);
        self.pass(nominal, UNLIMITED, |r, i, j| r.exact_match(i, j).then_some(Sieve::ExactMatch));
        self.pass(nominal, UNLIMITED, |r, i, j| r.relaxed_string_match(i, j).then_some(Sieve::RelaxedStringMatch));
        self.pass(nominal, UNLIMITED, |r, i, j| {
            if r.acronym(i, j) { Some(Sieve::Acronym) } else { r.person_name(i, j).then_some(Sieve::PersonName) }
        });
        self.pass(nominal, UNLIMITED, |r, i, j| r.strict_head_match(i, j, true, true).then_some(Sieve::StrictHead));
        self.pass(nominal, UNLIMITED, |r, i, j| {
            r.strict_head_match(i, j, true, false).then_some(Sieve::HeadNoModifiers)
        });
        self.pass(nominal, UNLIMITED, |r, i, j| {
            r.strict_head_match(i, j, false, true).then_some(Sieve::HeadNoInclusion)
        });
        self.pass(nominal, UNLIMITED, |r, i, j| r.proper_head_match(i, j).then_some(Sieve::ProperHead));
        self.pass(nominal, UNLIMITED, |r, i, j| r.relaxed_head_match(i, j).then_some(Sieve::RelaxedHead));
        // Demonstratives mostly point at clauses and events; the rules
        // leave them unresolved.
        let pronoun = |r: &Resolver, j: usize| {
            let m = &r.mentions[j];
            m.pronoun() && m.person == Person::Third && lexicon::demonstrative(&m.head).is_none()
        };
        self.pass(pronoun, PRONOUN_WINDOW, |r, i, j| r.pronoun_match(i, j).then_some(Sieve::PronounMatch));
    }

    fn nominal(&self, j: usize) -> bool {
        let m = &self.mentions[j];
        !m.pronoun() && !m.indefinite
    }

    fn admissible(&mut self, i: usize, j: usize) -> bool {
        self.clusters.find(i) != self.clusters.find(j)
            && !self.mentions[j].nests(&self.mentions[i])
            && self.clusters.types_compatible(i, j)
    }

    /// One sieve: for each anaphor that heads its own cluster, links the
    /// first admissible candidate the test accepts.
    fn pass(
        &mut self,
        anaphor: impl Fn(&Resolver, usize) -> bool,
        window: usize,
        test: impl Fn(&mut Resolver, usize, usize) -> Option<Sieve>,
    ) {
        for j in 0..self.mentions.len() {
            if self.clusters.find(j) != j || !anaphor(self, j) {
                continue;
            }
            let limit = if self.reflexive(j) { 0 } else { window };
            for i in self.candidates(j, limit) {
                if !self.admissible(i, j) {
                    continue;
                }
                if let Some(sieve) = test(self, i, j)
                    && self.link(i, j, sieve, 1.0)
                {
                    break;
                }
            }
        }
    }

    fn reflexive(&self, j: usize) -> bool {
        let m = &self.mentions[j];
        m.pronoun() && (m.head.ends_with("self") || m.head.ends_with("selves"))
    }

    /// The candidate antecedents of an anaphor: the same sentence left to
    /// right, then earlier sentences nearest first, each right to left
    /// for a pronoun and left to right otherwise.
    fn candidates(&self, j: usize, window: usize) -> Vec<usize> {
        let m = &self.mentions[j];
        let sentence = m.sentence;
        let mut ordered: Vec<usize> = self.by_sentence[sentence].iter().copied().filter(|&i| i < j).collect();
        let farthest = if window == UNLIMITED { 0 } else { sentence.saturating_sub(window) };
        for s in (farthest..sentence).rev() {
            let earlier = &self.by_sentence[s];
            if m.pronoun() {
                ordered.extend(earlier.iter().rev());
            } else {
                ordered.extend(earlier.iter());
            }
        }
        ordered
    }
}
