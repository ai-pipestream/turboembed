//! Mention attributes: heads, number, gender, animacy and definiteness
//! (`MentionDetector.mention` and its helpers).

use super::{Candidate, Detector, NOUN_TAG_PREFIX, OF, UNTYPED};
use crate::layer::{Animacy, Gender, Number, Origin, Person};
use crate::lexicon;
use crate::mention::Mention;

/// The Penn Treebank tags of proper nouns.
const PROPER_TAGS: [&str; 2] = ["NNP", "NNPS"];
/// The Penn Treebank tags of plural nouns.
const PLURAL_TAGS: [&str; 2] = ["NNS", "NNPS"];
/// Tags of the words that open a definite noun phrase.
const DEFINITE_OPENER_TAGS: [&str; 4] = ["DT", "PRP$", "WDT", "CD"];
/// The comma, which separates a head segment from an appositive.
const COMMA: &str = ",";
/// Entity type labels of organizations, whose number is left open.
const ORGANIZATION_TYPES: [&str; 2] = ["organization", "org"];

impl Detector<'_> {
    /// Computes a candidate's attributes; `None` for an entity whose span
    /// meets no token.
    pub(super) fn mention(&self, candidate: &Candidate) -> Option<Mention> {
        if candidate.first < 0 || candidate.last < candidate.first {
            return None;
        }
        let (first, last) = (candidate.first as usize, candidate.last as usize);
        let words: Vec<String> = (first..=last).map(|t| self.p.lower[t].clone()).collect();
        let head_token = if candidate.head >= 0 { candidate.head as usize } else { self.head_token(first, last) };
        let head = self.p.lower[head_token].clone();
        let head_prefix = join(&words, head_token - first + 1);
        let end = if self.is_possessive(last) || self.lower(last) == "'" { words.len() - 1 } else { words.len() };
        let normalized = join(&words, end.max(1));
        let pronoun = candidate.origin == Origin::Pronoun;
        let entity = candidate.entity.is_some();
        let tag = self.tag(head_token);
        let proper = entity || PROPER_TAGS.contains(&tag);
        // An unknown entity type supplies no type evidence, but its proper
        // name can still supply gender and animacy evidence.
        let ty = candidate.ty.clone().filter(|t| t != UNTYPED);
        let person_type = ty.as_ref().is_some_and(|t| self.person_types.contains(t));

        let mut number = Number::Unknown;
        let mut gender = Gender::Unknown;
        let mut animacy = Animacy::Unknown;
        let mut person = Person::Third;
        if pronoun {
            let form = lexicon::pronoun(&head)
                .or_else(|| lexicon::demonstrative(&head))
                .expect("a pronoun candidate is a lexicon pronoun or demonstrative");
            (number, gender, animacy, person) = (form.number, form.gender, form.animacy, form.person);
        } else {
            if self.coordinated(first, last) {
                number = Number::Plural;
            } else if ty.as_deref().is_some_and(|t| ORGANIZATION_TYPES.contains(&t)) {
                number = Number::Unknown;
            } else if PLURAL_TAGS.contains(&tag) {
                number = Number::Plural;
            } else if tag.starts_with(NOUN_TAG_PREFIX) {
                number = Number::Singular;
            }
            if person_type {
                gender = self.name_gender(first, last);
                animacy = Animacy::Animate;
            } else if ty.is_some() {
                gender = Gender::Neutral;
                animacy = Animacy::Inanimate;
            } else if proper {
                gender = self.name_gender(first, last);
                animacy = Animacy::Animate;
            } else if !entity {
                gender = lexicon::noun_gender(&head);
                if lexicon::animate_noun(&head) {
                    animacy = Animacy::Animate;
                }
            }
        }
        let indefinite = !pronoun && !proper && self.indefinite(first, head_token);
        Some(Mention {
            span: candidate.span,
            fragments: candidate.fragments.clone(),
            origin: candidate.origin,
            entity: candidate.entity,
            ty,
            sentence: self.p.sentence_of_token[first],
            first,
            last,
            head_token,
            words,
            head,
            head_prefix,
            normalized,
            proper,
            indefinite,
            number,
            gender,
            animacy,
            person,
        })
    }

    /// The head token of a token range: the last noun before the first
    /// `of` or comma, skipping a possessive marker; else the last token
    /// of that segment.
    pub(crate) fn head_token(&self, first: usize, last: usize) -> usize {
        let mut stop = last;
        for t in first + 1..=last {
            if self.lower(t) == OF || self.lower(t) == COMMA {
                stop = t - 1;
                break;
            }
        }
        if stop > first && self.is_possessive(stop) {
            stop -= 1;
        }
        (first..=stop).rev().find(|&t| self.tag(t).starts_with(NOUN_TAG_PREFIX)).unwrap_or(stop)
    }

    /// The gender a person name implies from its title or first name.
    fn name_gender(&self, first: usize, last: usize) -> Gender {
        let mut t = first;
        if lexicon::title(self.lower(t)) {
            let titled = lexicon::noun_gender(self.lower(t));
            if titled != Gender::Unknown || t == last {
                return titled;
            }
            t += 1;
        }
        lexicon::first_name_gender(self.lower(t))
    }

    /// Whether a nominal mention is indefinite or generic: opened by an
    /// indefinite word, or a bare common noun phrase.
    fn indefinite(&self, first: usize, head_token: usize) -> bool {
        if lexicon::indefinite_word(self.lower(first)) {
            return true;
        }
        let tag = self.tag(head_token);
        tag.starts_with(NOUN_TAG_PREFIX)
            && !PROPER_TAGS.contains(&tag)
            && !DEFINITE_OPENER_TAGS.contains(&self.tag(first))
            && self.lower(first) != "the"
    }
}

/// Joins the first `to` words with spaces.
fn join(words: &[String], to: usize) -> String {
    words[..to.min(words.len())].join(" ")
}
