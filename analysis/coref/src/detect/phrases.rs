//! Noun phrase candidates from chunks and parse phrases, possessors and
//! pronouns (`MentionDetector` phrase collection).

use turbo_analysis::Span;

use super::{AND, Candidate, Detector, NOUN_TAG_PREFIX, OF};
use crate::layer::Origin;
use crate::lexicon;

/// The chunk and phrase label of noun phrases.
const NOUN_PHRASE: &str = "NP";
/// The tag of a coordinating conjunction.
const CONJUNCTION_TAG: &str = "CC";
/// The tag of a determiner, which a standalone demonstrative carries.
const DETERMINER_TAG: &str = "DT";
/// POS tags that mark pronoun tokens: Penn Treebank PRP and PRP$, Universal PRON.
const PRONOUN_TAGS: [&str; 3] = ["PRP", "PRP$", "PRON"];
/// How many tokens after a pleonastic verb its complement may follow.
const PLEONASTIC_WINDOW: usize = 6;

/// One noun phrase with its token range and head. A merged phrase joins
/// two chunks over a lone `of` or `and`.
#[derive(Clone, Debug)]
pub(crate) struct NounPhrase {
    pub span: Span,
    pub first: usize,
    pub last: usize,
    pub head: usize,
    pub merged: bool,
}

impl Detector<'_> {
    /// Reads the noun phrase chunks, joining a chunk that opens with a
    /// possessive marker to the chunk before it.
    pub(super) fn chunk_noun_phrases(&self) -> Vec<NounPhrase> {
        let mut phrases = Vec::new();
        let mut pending: Option<(usize, usize)> = None;
        for chunk in self.p.chunks.as_deref().unwrap_or(&[]) {
            let range = if chunk.label == NOUN_PHRASE {
                let (first, last) = (self.first_token(chunk.span), self.last_token(chunk.span));
                (first >= 0 && last >= first).then_some((first as usize, last as usize))
            } else {
                None
            };
            if let (Some((first, last)), Some((p_first, p_last))) = (range, pending)
                && first == p_last + 1
                && self.is_possessive(first)
            {
                pending = Some((p_first, last));
                continue;
            }
            if let Some((first, last)) = pending {
                phrases.push(self.noun_phrase(first, last));
            }
            pending = range;
        }
        if let Some((first, last)) = pending {
            phrases.push(self.noun_phrase(first, last));
        }
        self.with_merged_phrases(phrases)
    }

    /// Adds, for two chunks a lone `of` or `and` separates, the phrase
    /// spanning both.
    fn with_merged_phrases(&self, phrases: Vec<NounPhrase>) -> Vec<NounPhrase> {
        let mut all = Vec::with_capacity(phrases.len() * 2);
        for (p, phrase) in phrases.iter().enumerate() {
            all.push(phrase.clone());
            let Some(next) = phrases.get(p + 1) else { continue };
            let between = phrase.last + 1;
            if next.first != between + 1
                || self.p.sentence_of_token[phrase.first] != self.p.sentence_of_token[next.last]
            {
                continue;
            }
            let of = self.lower(between) == OF;
            let and = self.lower(between) == AND && self.tag(between) == CONJUNCTION_TAG;
            if of || and {
                all.push(NounPhrase {
                    span: self.token_span(phrase.first, next.last),
                    first: phrase.first,
                    last: next.last,
                    head: phrase.head,
                    merged: true,
                });
            }
        }
        all
    }

    fn noun_phrase(&self, first: usize, last: usize) -> NounPhrase {
        NounPhrase {
            span: self.token_span(first, last),
            first,
            last,
            head: self.head_token(first, last),
            merged: false,
        }
    }

    /// Reads the noun phrases of the parse. When phrases share a head
    /// token, only the widest is kept.
    pub(super) fn maximal_noun_phrases(&self) -> Vec<NounPhrase> {
        let mut noun_phrases = Vec::new();
        for phrase in self.p.phrases.as_deref().unwrap_or(&[]) {
            if phrase.label != NOUN_PHRASE {
                continue;
            }
            let (first, last) = (self.first_token(phrase.span), self.last_token(phrase.span));
            let mut head = self.first_token(phrase.head);
            if first < 0 || last < first || head < first || head > last {
                continue;
            }
            let (first, last) = (first as usize, last as usize);
            // Head rules make the possessive marker the head of a possessive
            // phrase; the mention's head is the noun the marker follows.
            if head as usize > first && self.is_possessive(head as usize) {
                head = self.head_token(first, head as usize - 1) as isize;
            }
            noun_phrases.push(NounPhrase { span: phrase.span, first, last, head: head as usize, merged: false });
        }
        let widest = |i: usize| {
            let phrase = &noun_phrases[i];
            !noun_phrases.iter().enumerate().any(|(j, other)| {
                j != i && other.head == phrase.head && other.span.contains(phrase.span) && other.span != phrase.span
            })
        };
        (0..noun_phrases.len()).filter(|&i| widest(i)).map(|i| noun_phrases[i].clone()).collect()
    }

    /// Reconciles one noun phrase with the entity candidates it meets and
    /// adds a nominal candidate when no entity owns it.
    pub(super) fn add_noun_phrase(&self, phrase: &NounPhrase, candidates: &mut Vec<Candidate>, entity_count: usize) {
        let (first, last) = (phrase.first as isize, phrase.last as isize);
        if candidates[entity_count..].iter().any(|c| c.span == phrase.span) {
            return;
        }
        let mut headed_by = None;
        let mut touching = 0;
        for (e, entity) in candidates[..entity_count].iter().enumerate() {
            if entity.last < first || entity.first > last || entity.first < 0 {
                continue;
            }
            touching += 1;
            if entity.first == first && entity.last == last {
                return;
            }
            if entity.first < first || entity.last > last {
                return;
            }
            if entity.last == phrase.head as isize {
                headed_by = Some(e);
            }
        }
        // A phrase headed by an entity is that entity's full mention, unless
        // the phrase coordinates it with others: the coordination is then a
        // mention of its own.
        if let Some(e) = headed_by
            && !(touching > 1 && self.coordinated(phrase.first, phrase.last))
        {
            if !phrase.merged {
                let entity = &mut candidates[e];
                entity.span = phrase.span;
                entity.fragments = None;
                entity.first = first;
                entity.last = last;
                entity.head = phrase.head as isize;
            }
            return;
        }
        let word = self.lower(phrase.first);
        if first == last && (lexicon::pronoun(word).is_some() || lexicon::demonstrative(word).is_some()) {
            return;
        }
        if !phrase.merged {
            self.add_possessors(phrase.first, phrase.last, candidates, entity_count);
        }
        if !self.tag(phrase.head).starts_with(NOUN_TAG_PREFIX) {
            return;
        }
        candidates.push(Candidate {
            span: phrase.span,
            fragments: None,
            origin: Origin::NounPhrase,
            entity: None,
            ty: None,
            first,
            last,
            head: phrase.head as isize,
        });
    }

    /// Adds the possessor inside a noun phrase as a mention of its own,
    /// `Norton 's` in `Norton 's company`, or widens the entity it ends.
    fn add_possessors(&self, first: usize, last: usize, candidates: &mut Vec<Candidate>, entity_count: usize) {
        for t in first + 1..last {
            if !self.is_possessive(t) {
                continue;
            }
            let head = self.head_token(first, t - 1);
            let mut widened = false;
            let ti = t as isize;
            for entity in &mut candidates[..entity_count] {
                if entity.last == ti - 1 && entity.first >= first as isize {
                    entity.last = ti;
                    entity.span = Span::new(entity.span.start, self.p.tokens[t].end);
                    entity.fragments = None;
                    widened = true;
                } else if entity.first <= ti && entity.last >= ti {
                    widened = true;
                }
            }
            if widened || !self.tag(head).starts_with(NOUN_TAG_PREFIX) || lexicon::pronoun(self.lower(head)).is_some() {
                continue;
            }
            candidates.push(Candidate {
                span: self.token_span(first, t),
                fragments: None,
                origin: Origin::NounPhrase,
                entity: None,
                ty: None,
                first: first as isize,
                last: ti,
                head: head as isize,
            });
        }
    }

    /// Adds every personal pronoun token no entity covers as a pronoun
    /// candidate, and every standalone demonstrative.
    pub(super) fn add_pronouns(&self, candidates: &mut Vec<Candidate>, entity_count: usize) {
        for t in 0..self.p.tokens.len() {
            if PRONOUN_TAGS.contains(&self.tag(t)) {
                if lexicon::pronoun(self.lower(t)).is_none() {
                    continue;
                }
            } else if !self.standalone_demonstrative(t) {
                continue;
            }
            let span = self.p.tokens[t];
            if candidates[..entity_count].iter().any(|c| c.span.contains(span)) || self.pleonastic(t) {
                continue;
            }
            candidates.push(Candidate {
                span,
                fragments: None,
                origin: Origin::Pronoun,
                entity: None,
                ty: None,
                first: t as isize,
                last: t as isize,
                head: -1,
            });
        }
    }

    /// Whether a demonstrative determiner has no noun phrase after it.
    fn standalone_demonstrative(&self, t: usize) -> bool {
        if self.tag(t) != DETERMINER_TAG || lexicon::demonstrative(self.lower(t)).is_none() {
            return false;
        }
        if t == self.sentence_end[t] {
            return true;
        }
        let next = self.tag(t + 1);
        !(next.starts_with(NOUN_TAG_PREFIX)
            || next.starts_with("JJ")
            || next == "CD"
            || next == DETERMINER_TAG
            || next == "PRP$")
    }

    /// Recognizes a pleonastic `it`: one followed, possibly after an
    /// adverb, by a pleonastic verb and then a complement cue.
    fn pleonastic(&self, t: usize) -> bool {
        if self.lower(t) != "it" {
            return false;
        }
        let end = self.sentence_end[t];
        let mut verb = t + 1;
        if verb <= end && self.tag(verb).starts_with("RB") {
            verb += 1;
        }
        if verb > end || !lexicon::pleonastic_verb(self.lower(verb)) {
            return false;
        }
        (verb + 1..=end.min(verb + PLEONASTIC_WINDOW)).any(|k| lexicon::pleonastic_complement(self.lower(k)))
    }
}
