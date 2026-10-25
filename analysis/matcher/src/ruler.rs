//! The span ruler, spaCy's `SpanRuler`: token patterns and phrases under
//! labels, run as an analysis step that writes their matches as a
//! `turbo:matches` layer.

use std::hash::Hasher;
use std::sync::Arc;

use turbo_analysis::{Annotator, Bound, Error, FilterRule, IxRange, Refs, Spans, TagSet, View, ViewId, filter_spans};
use turbo_analysis_layers::{Entities, Lemmas, Matches, MorphFeatures, Numbers, PosTags, Tokens};
use turbo_stem_layers::Stems;
use turbo_stopword_layers::Stopwords;

use crate::attr::Field;
use crate::error::{MatchError, PatternError};
use crate::hash::FoldHasher;
use crate::input::Input;
use crate::matcher::{Greedy, Matcher, keep_greedy};
use crate::pattern::Tok;
use crate::phrase::PhraseMatcher;

/// What the ruler does with matches that overlap.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Writes them all: the layer's items may overlap and nest.
    Keep,
    /// Keeps a subset that does not overlap, by `filter_spans` with this
    /// rule. [`FilterRule::LongestFirst`] is spaCy's `filter_spans` (its
    /// `first_longest_spans_filter`), except that lengths are compared in
    /// bytes, not tokens.
    Filter(FilterRule),
    /// Keeps a subset that does not overlap as spaCy's `filter_spans`
    /// does, lengths counted in tokens: the longest first, the earliest
    /// first among equally long ones, each kept when none of its tokens
    /// is in one kept before. Of matches over the same tokens under
    /// several labels, the label added first is kept.
    LongestTokens,
}

/// Token patterns and phrases under labels. Phrases go to one phrase
/// matcher per [`Field`]: [`SpanRuler::add_phrase`] to the field the
/// ruler was made with, [`SpanRuler::add_phrase_on`] to any, so one ruler
/// can hold phrases by lowercase and phrases by lemma. As a step it reads
/// a text view and its tokens, and whichever of `opennlp:pos`, `opennlp:lemmas`,
/// `opennlp:entities`, `turbo:stopwords`, `turbo:numbers` and
/// `turbo:stems` the document has, and writes [`Matches`]; bind it with
/// [`SpanRuler::bound`] to write a named instance. A match's rule is the
/// number [`SpanRuler::add_pattern`] or [`SpanRuler::add_phrase`]
/// returned. Matches of one label over the same tokens are written once,
/// with the first rule added.
pub struct SpanRuler {
    matcher: Matcher,
    /// The phrase matchers, one per field, the ruler's own field first.
    phrases: Vec<PhraseMatcher>,
    /// Per rule, its label's name, and whether the label is still in use.
    rule_label: Vec<Box<str>>,
    rule_live: Vec<bool>,
    rule_of_pattern: Vec<u32>,
    /// Per phrase matcher, the rule of each of its phrases.
    rule_of_phrase: Vec<Vec<u32>>,
    /// The labels in use, in the order they were first added.
    labels: Arc<TagSet>,
    policy: Policy,
    config: u64,
}

impl SpanRuler {
    /// A ruler whose phrases match over `phrase_field`, spaCy's
    /// `phrase_matcher_attr`.
    pub fn new(phrase_field: Field, policy: Policy) -> SpanRuler {
        SpanRuler {
            matcher: Matcher::new(),
            phrases: vec![PhraseMatcher::new(phrase_field)],
            rule_label: Vec::new(),
            rule_live: Vec::new(),
            rule_of_pattern: Vec::new(),
            rule_of_phrase: vec![Vec::new()],
            labels: TagSet::new("rules", &[]),
            policy,
            config: 0,
        }
    }

    /// The step writing instance `name` of [`Matches`] (`Key::named`).
    pub fn bound(self, name: &'static str) -> Bound<SpanRuler> {
        Bound::new(self).write(0, name)
    }

    fn note(&mut self, kind: &str, label: &str, rule: &str) {
        let mut h = FoldHasher::default();
        h.write_u64(self.config);
        h.write(kind.as_bytes());
        h.write(label.as_bytes());
        h.write(rule.as_bytes());
        self.config = h.finish();
    }

    fn new_rule(&mut self, label: &str) -> u32 {
        if self.labels.tag(label).is_none() {
            let mut names: Vec<&str> =
                (0..self.labels.len()).map(|t| self.labels.label(turbo_analysis::Tag(t as u16))).collect();
            names.push(label);
            self.labels = TagSet::new("rules", &names);
        }
        self.rule_label.push(label.into());
        self.rule_live.push(true);
        (self.rule_label.len() - 1) as u32
    }

    /// Adds a token pattern under `label` and returns its rule number.
    pub fn add_pattern(&mut self, label: &str, pattern: &[Tok]) -> Result<u32, PatternError> {
        let id = self.matcher.add(label, pattern)?;
        debug_assert_eq!(id.0 as usize, self.rule_of_pattern.len());
        let rule = self.new_rule(label);
        self.rule_of_pattern.push(rule);
        self.note("pattern", label, &format!("{pattern:?}"));
        Ok(rule)
    }

    /// Adds a phrase under `label`, matched over the ruler's own field,
    /// and returns its rule number; a phrase already under the label keeps
    /// its number.
    pub fn add_phrase(&mut self, label: &str, tokens: &[&str]) -> Result<u32, PatternError> {
        self.add_phrase_at(0, label, tokens)
    }

    /// Adds a phrase under `label`, matched over `field`, and returns its
    /// rule number; a phrase already under the label and field keeps its
    /// number. The first phrase over a field makes its phrase matcher.
    pub fn add_phrase_on(&mut self, field: Field, label: &str, tokens: &[&str]) -> Result<u32, PatternError> {
        let at = match self.phrases.iter().position(|p| p.field() == field) {
            Some(at) => at,
            None => {
                self.phrases.push(PhraseMatcher::new(field));
                self.rule_of_phrase.push(Vec::new());
                self.phrases.len() - 1
            }
        };
        self.add_phrase_at(at, label, tokens)
    }

    fn add_phrase_at(&mut self, at: usize, label: &str, tokens: &[&str]) -> Result<u32, PatternError> {
        let id = self.phrases[at].add(label, tokens)?;
        if let Some(&rule) = self.rule_of_phrase[at].get(id.0 as usize) {
            return Ok(rule);
        }
        let rule = self.new_rule(label);
        self.rule_of_phrase[at].push(rule);
        let field = format!("{:?}", self.phrases[at].field());
        self.note("phrase", label, &format!("{field}\u{1f}{}", tokens.join("\u{1f}")));
        Ok(rule)
    }

    /// Removes a label and all its patterns and phrases.
    pub fn remove(&mut self, label: &str) -> Result<(), MatchError> {
        let in_patterns = self.matcher.remove(label).is_ok();
        let mut in_phrases = false;
        for p in &mut self.phrases {
            in_phrases |= p.remove(label).is_ok();
        }
        if !in_patterns && !in_phrases {
            return Err(MatchError::NoLabel(label.into()));
        }
        let names: Vec<&str> = (0..self.labels.len())
            .map(|t| self.labels.label(turbo_analysis::Tag(t as u16)))
            .filter(|&l| l != label)
            .collect();
        self.labels = TagSet::new("rules", &names);
        for (live, l) in self.rule_live.iter_mut().zip(&self.rule_label) {
            *live &= &**l != label;
        }
        self.note("remove", label, "");
        Ok(())
    }

    /// The labels in use, in the order they were first added.
    pub fn labels(&self) -> &Arc<TagSet> {
        &self.labels
    }

    /// The label of a rule.
    pub fn rule_label(&self, rule: u32) -> &str {
        &self.rule_label[rule as usize]
    }

    /// How many rules were added, removed ones included.
    pub fn rules(&self) -> usize {
        self.rule_label.len()
    }

    /// How many rules are in use: those whose label was not removed, as
    /// spaCy's `len(ruler.patterns)`.
    pub fn len(&self) -> usize {
        self.rule_live.iter().filter(|&&l| l).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn matcher(&self) -> &Matcher {
        &self.matcher
    }

    /// The phrase matchers, one per field, the ruler's own field first.
    pub fn phrases(&self) -> &[PhraseMatcher] {
        &self.phrases
    }

    /// The matches over `input`'s tokens, as a layer of view `view`, the
    /// view the tokens are in.
    pub fn matches(&self, input: &Input<'_>, view: ViewId) -> Result<Matches, MatchError> {
        if input.view() != view {
            return Err(MatchError::ViewMismatch { layer: "opennlp:tokens", expected: view, actual: input.view() });
        }
        if let Some(numbers) = input.numbers()
            && numbers.spans.view() != view
        {
            return Err(MatchError::ViewMismatch {
                layer: "turbo:numbers",
                expected: view,
                actual: numbers.spans.view(),
            });
        }
        let mut found: Vec<(u32, u32, u16, u32)> = Vec::new();
        let tag = |name: &str| self.labels.tag(name).expect("a label in use").0;
        if !self.matcher.is_empty() {
            for m in self.matcher.find(input)? {
                let label = tag(self.matcher.label_name(m.label));
                found.push((
                    m.tokens.start.get() as u32,
                    m.tokens.end.get() as u32,
                    label,
                    self.rule_of_pattern[m.pattern.0 as usize],
                ));
            }
        }
        for (phrases, rules) in self.phrases.iter().zip(&self.rule_of_phrase) {
            if phrases.is_empty() {
                continue;
            }
            for m in phrases.find(input)? {
                let label = tag(phrases.label_name(m.label));
                found.push((m.tokens.start.get() as u32, m.tokens.end.get() as u32, label, rules[m.phrase.0 as usize]));
            }
        }
        found.sort_unstable();
        found.dedup_by_key(|m| (m.0, m.1, m.2));
        let mut out = Matches::new(view, self.labels.clone());
        let keep: Vec<usize> = match self.policy {
            Policy::Keep => (0..found.len()).collect(),
            Policy::Filter(rule) => {
                let mut spans: Spans<()> = Spans::with_capacity(view, found.len());
                for m in &found {
                    spans.push(input.span(IxRange::new(m.0, m.1)));
                }
                let mut kept: Vec<usize> = filter_spans(&spans, None, rule).into_iter().map(|i| i.get()).collect();
                kept.sort_unstable();
                kept
            }
            Policy::LongestTokens => {
                let order: Vec<usize> = (0..found.len()).collect();
                let mut kept = Vec::new();
                keep_greedy(&order, |&i| (found[i].0, found[i].1), Greedy::Longest, 0, &mut Vec::new(), &mut kept);
                kept.sort_unstable();
                kept
            }
        };
        for i in keep {
            let (s, e, label, rule) = found[i];
            let tokens = IxRange::new(s, e);
            out.push(input.span(tokens), tokens, turbo_analysis::Tag(label), rule);
        }
        Ok(out)
    }
}

impl Annotator for SpanRuler {
    const ID: &'static str = "turbo-matcher:span-ruler";
    type Reads = (
        View,
        Tokens,
        Option<PosTags>,
        Option<Lemmas>,
        Option<Entities>,
        Option<Stopwords>,
        Option<Numbers>,
        Option<Stems>,
        Option<MorphFeatures<Tokens>>,
    );
    type Writes = (Matches,);

    fn annotate(&self, reads: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let (view, tokens, pos, lemmas, entities, stop, numbers, stems, morph) = reads;
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if tokens.spans.view() != view.id() {
            return Err(fail("the tokens are in another view".into()));
        }
        if entities.is_some_and(|e| e.spans.view() != view.id()) {
            return Err(fail("the entities are in another view".into()));
        }
        if numbers.is_some_and(|n| n.spans.view() != view.id()) {
            return Err(fail("the numbers are in another view".into()));
        }
        let input = (|| {
            let mut input = Input::new(view.text(), tokens)?;
            if let Some(p) = pos {
                input = input.with_pos(p)?;
            }
            if let Some(l) = lemmas {
                input = input.with_lemmas(&l.lemma)?;
            }
            if let Some(e) = entities {
                input = input.with_entities(e)?;
            }
            if let Some(s) = stop {
                input = input.with_stopwords(&s.stop)?;
            }
            if let Some(n) = numbers {
                input = input.with_numbers(n);
            }
            if let Some(s) = stems {
                input = input.with_stems(&s.stem)?;
            }
            if let Some(m) = morph {
                input = input.with_morph(m)?;
            }
            Ok(input)
        })()
        .map_err(|e: MatchError| fail(e.to_string()))?;
        let matches = self.matches(&input, view.id()).map_err(|e| fail(e.to_string()))?;
        Ok((matches,))
    }

    fn config_hash(&self) -> u64 {
        let mut h = FoldHasher::default();
        h.write_u64(self.config);
        h.write(format!("{:?}", self.policy).as_bytes());
        for p in &self.phrases {
            h.write(format!("{:?}", p.field()).as_bytes());
        }
        h.finish()
    }
}
