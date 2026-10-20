//! The pair tests of the string, head and pronoun sieves.

use std::collections::HashSet;

use super::Resolver;
use crate::layer::Gender;
use crate::lexicon;
use crate::mention::Mention;

impl Resolver<'_> {
    /// Whether two mentions have identical normalized text.
    pub fn exact_match(&self, i: usize, j: usize) -> bool {
        let a = &self.mentions[i];
        !a.pronoun() && !a.normalized.is_empty() && a.normalized == self.mentions[j].normalized
    }

    /// Whether two mentions have identical text through their heads.
    pub fn relaxed_string_match(&self, i: usize, j: usize) -> bool {
        let a = &self.mentions[i];
        !a.pronoun() && !a.head_prefix.is_empty() && a.head_prefix == self.mentions[j].head_prefix
    }

    /// Whether one proper name is the acronym of the other.
    pub fn acronym(&self, i: usize, j: usize) -> bool {
        let (a, b) = (&self.mentions[i], &self.mentions[j]);
        a.proper && b.proper && (self.acronym_of(b, a) || self.acronym_of(a, b))
    }

    /// Whether a one-token, all-uppercase mention of at least two code
    /// points spells the uppercase initials of a multi-token name.
    fn acronym_of(&self, candidate: &Mention, name: &Mention) -> bool {
        if candidate.first != candidate.last || name.last == name.first {
            return false;
        }
        let acronym = self.forms[candidate.first];
        if acronym.chars().count() < 2 || !acronym.chars().all(char::is_uppercase) {
            return false;
        }
        let initials: String = (name.first..=name.last)
            .filter_map(|t| self.forms[t].chars().next())
            .filter(|c| c.is_uppercase())
            .collect();
        initials.chars().count() >= 2 && initials == acronym
    }

    /// Links a person's one-word name to a longer name of the same person
    /// that starts or ends with it, titles aside.
    pub fn person_name(&mut self, i: usize, j: usize) -> bool {
        let (a, b) = (&self.mentions[i], &self.mentions[j]);
        if !a.proper || !b.proper || !self.person_type(a.ty.as_deref()) || !self.person_type(b.ty.as_deref()) {
            return false;
        }
        let (short, long) = if a.words.len() <= b.words.len() { (a, b) } else { (b, a) };
        let (shorter, longer) = (name_words(short), name_words(long));
        let matched = shorter.len() == 1
            && longer.len() >= 2
            && (longer[0] == shorter[0] || longer[longer.len() - 1] == shorter[0]);
        matched && self.clusters.numbers_agree(i, j)
    }

    fn person_type(&self, ty: Option<&str>) -> bool {
        ty.is_some_and(|t| self.person_types.iter().any(|p| p == t))
    }

    /// Links an anaphor whose head is a head of the candidate's cluster,
    /// optionally requiring cluster word inclusion and that the anaphor's
    /// modifiers appear in the candidate.
    pub fn strict_head_match(&mut self, i: usize, j: usize, word_inclusion: bool, compatible_modifiers: bool) -> bool {
        if self.mentions[i].pronoun() || !self.clusters.heads(i).contains(&self.mentions[j].head) {
            return false;
        }
        if word_inclusion {
            let anaphor_words = self.clusters.words(j).clone();
            if !anaphor_words.is_subset(self.clusters.words(i)) {
                return false;
            }
        }
        !compatible_modifiers || modifiers_appear_in(&self.mentions[j], &self.mentions[i])
    }

    /// Links two proper names with the same head, agreeing number, no
    /// differing numerals and, for a compound name head, shared modifiers.
    pub fn proper_head_match(&mut self, i: usize, j: usize) -> bool {
        {
            let (a, b) = (&self.mentions[i], &self.mentions[j]);
            if !a.proper || !b.proper || a.head != b.head {
                return false;
            }
        }
        if !self.clusters.numbers_agree(i, j) {
            return false;
        }
        let (a, b) = (&self.mentions[i], &self.mentions[j]);
        if numbers_differ(a, b) {
            return false;
        }
        !lexicon::compound_name_head(&b.head) || modifiers_appear_in(b, a) || modifiers_appear_in(a, b)
    }

    /// Links two entity mentions of one type when the anaphor's head and
    /// every content word of its cluster appear in the candidate cluster.
    pub fn relaxed_head_match(&mut self, i: usize, j: usize) -> bool {
        let (a, b) = (&self.mentions[i], &self.mentions[j]);
        if a.pronoun() || a.ty.is_none() || a.ty != b.ty {
            return false;
        }
        let head = b.head.clone();
        if !self.clusters.words(i).contains(&head) {
            return false;
        }
        let anaphor_words = self.clusters.words(j).clone();
        if !anaphor_words.is_subset(self.clusters.words(i)) {
            return false;
        }
        let (a, b) = (&self.mentions[i], &self.mentions[j]);
        !compound_differs(a, b) && !compound_differs(b, a)
    }

    /// Links a pronoun to a candidate whose cluster agrees in number,
    /// gender, animacy and person, and whose known type suits the
    /// pronoun's gender.
    pub fn pronoun_match(&mut self, i: usize, j: usize) -> bool {
        if !self.clusters.attributes_agree(i, j) {
            return false;
        }
        let Some(ty) = self.clusters.ty(i) else { return true };
        let person = self.person_types.contains(&ty);
        let neutral = self.neutral_types.contains(&ty);
        match self.mentions[j].gender {
            Gender::Neutral => neutral,
            Gender::Male | Gender::Female => person,
            Gender::Unknown => person || neutral,
        }
    }
}

/// A name's words after any leading titles, keeping at least one.
fn name_words(m: &Mention) -> &[String] {
    let mut start = 0;
    while start + 1 < m.words.len() && lexicon::title(&m.words[start]) {
        start += 1;
    }
    &m.words[start..]
}

/// Whether every content word before the anaphor's head occurs in the
/// antecedent.
fn modifiers_appear_in(anaphor: &Mention, antecedent: &Mention) -> bool {
    let words: HashSet<&String> = antecedent.words.iter().collect();
    for word in &anaphor.words {
        if *word == anaphor.head {
            break;
        }
        if !lexicon::stop_word(word) && !words.contains(word) {
            return false;
        }
    }
    true
}

/// Whether both mentions hold numerals and the numeral sets differ.
fn numbers_differ(a: &Mention, b: &Mention) -> bool {
    let (x, y) = (numerals(a), numerals(b));
    !x.is_empty() && !y.is_empty() && x != y
}

/// The words that begin with a Unicode digit (`Character.isDigit`).
fn numerals(m: &Mention) -> HashSet<&str> {
    m.words.iter().filter(|w| w.chars().next().is_some_and(turbo_unicode::java::is_digit)).map(String::as_str).collect()
}

/// Whether the first name has a compound head absent from the other.
fn compound_differs(compound: &Mention, other: &Mention) -> bool {
    lexicon::compound_name_head(&compound.head) && !other.words.contains(&compound.head)
}
