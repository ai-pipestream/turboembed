//! The speaker sieve and speaker assignment: the speakers layer, or
//! quoted-speech attribution without one.

use std::collections::HashMap;

use super::{Resolver, Sieve};
use crate::layer::{Number, Person};
use crate::quotes::{QuoteRegions, speech_verb};

/// The speaker of text outside every quotation when no speakers layer
/// says otherwise.
const NARRATOR: &str = "";
/// How many tokens a verb of speech may lie from the person it
/// attributes a quote to.
const ATTRIBUTION_DISTANCE: usize = 2;

impl Resolver<'_> {
    /// The speaker of mention `j`, assigned once.
    pub fn speaker_of(&mut self, j: usize) -> &str {
        if self.speaker_of.is_none() {
            let speakers = self.speaker_of_mentions();
            self.speaker_of = Some(speakers);
        }
        &self.speaker_of.as_ref().expect("assigned above")[j]
    }

    /// Chains the first and second person mentions of each speaker,
    /// keeping a speaker's `I` apart from `we`.
    pub(super) fn speaker_sieve(&mut self) {
        if self.speaker_of.is_none() {
            let speakers = self.speaker_of_mentions();
            self.speaker_of = Some(speakers);
        }
        let mut first_of_group: HashMap<String, usize> = HashMap::new();
        for j in 0..self.mentions.len() {
            let m = &self.mentions[j];
            if !m.pronoun() || m.person == Person::Third {
                continue;
            }
            let we = m.person == Person::First && m.number == Number::Plural;
            let group = format!(
                "{}\u{0}{}\u{0}{}",
                self.speaker_of.as_ref().expect("assigned above")[j],
                m.person.java_name(),
                we
            );
            match first_of_group.get(&group) {
                Some(&first) => {
                    self.link(first, j, Sieve::Speaker, 1.0);
                }
                None => {
                    first_of_group.insert(group, j);
                }
            }
        }
    }

    /// Every mention's speaker: the covering speakers layer label, or
    /// without the layer, `mention:<i>` inside a quote attributed to
    /// mention `i`, `quote:<q>` inside an unattributed quote, and the
    /// narrator outside quotes. A singular first person mention inside an
    /// attributed quote joins the person it is attributed to.
    fn speaker_of_mentions(&mut self) -> Vec<String> {
        if let Some(speakers) = self.speakers {
            return self
                .mentions
                .iter()
                .map(|m| {
                    speakers
                        .iter()
                        .find(|s| s.span.start <= m.span.start && m.span.start < s.span.end)
                        .map_or(NARRATOR.to_owned(), |s| s.label.clone())
                })
                .collect();
        }
        let QuoteRegions { quote_of, starts, ends } = QuoteRegions::scan(self.forms);
        let mut attributed_to: HashMap<usize, usize> = HashMap::new();
        for q in 0..starts.len() {
            if let Some(attributed) = self.attribute(starts[q], ends[q], &quote_of) {
                attributed_to.insert(q + 1, attributed);
            }
        }
        let mut speaker_of = Vec::with_capacity(self.mentions.len());
        for j in 0..self.mentions.len() {
            let quote = quote_of[self.mentions[j].first];
            if quote == 0 {
                speaker_of.push(NARRATOR.to_owned());
            } else if let Some(&speaker) = attributed_to.get(&quote) {
                speaker_of.push(format!("mention:{speaker}"));
                let m = &self.mentions[j];
                if m.person == Person::First && m.number == Number::Singular {
                    self.link(speaker, j, Sieve::Speaker, 1.0);
                }
            } else {
                speaker_of.push(format!("quote:{quote}"));
            }
        }
        speaker_of
    }

    /// The person a quotation is attributed to: a person mention outside
    /// every quotation, in the sentence the quote opens or closes in, next
    /// to a verb of speech.
    fn attribute(&self, quote_start: usize, quote_end: usize, quote_of: &[usize]) -> Option<usize> {
        let sentences = (self.sentence_of_token[quote_start], self.sentence_of_token[quote_end]);
        self.mentions.iter().position(|m| {
            quote_of[m.first] == 0
                && !m.pronoun()
                && m.ty.as_ref().is_some_and(|t| self.person_types.contains(t))
                && (m.sentence == sentences.0 || m.sentence == sentences.1)
                && ((m.first > 0 && self.speech_verb_near(m.first - 1, false, quote_of))
                    || self.speech_verb_near(m.last + 1, true, quote_of))
        })
    }

    /// Whether a verb of speech outside quotations lies within
    /// `ATTRIBUTION_DISTANCE` tokens from `from` in one direction.
    fn speech_verb_near(&self, from: usize, forward: bool, quote_of: &[usize]) -> bool {
        let mut t = from;
        for _ in 0..ATTRIBUTION_DISTANCE {
            if t >= self.forms.len() || quote_of[t] != 0 {
                return false;
            }
            if speech_verb(self.forms[t]) {
                return true;
            }
            if forward {
                t += 1;
            } else if t == 0 {
                return false;
            } else {
                t -= 1;
            }
        }
        false
    }
}
