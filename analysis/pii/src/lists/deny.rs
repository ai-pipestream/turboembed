//! Deny lists: caller terms matched with `turbo_glossary::TextMatcher`
//! (Aho-Corasick) and optional bounded Levenshtein fuzzy matching over word
//! tokens, always reported as a caller-named [`PiiKind::Custom`].

use std::collections::HashSet;
use std::path::Path;

use turbo_analysis::Fnv1a;
use turbo_analysis_layers::{PiiEvidence, PiiKind};
use turbo_glossary::{Candidate, Case, EntryIx, Glossary, GlossaryMatch, TextMatcher, select};
use turbo_unicode::java::to_lower_char;

use crate::hits::Hits;
use crate::text::{letter_or_digit, on_end, on_word_start};

use super::fuzzy::{FuzzyIndex, build_fuzzy_term, is_capitalised};
use super::{ListError, case_tag, read_file_lines};

/// Terms always reported as a caller-named [`PiiKind::Custom`], matched with
/// [`turbo_glossary::TextMatcher`] (Aho-Corasick) and optional bounded
/// Levenshtein edit distance (`0`, `1`, or `2` edits) over word tokens.
#[derive(Clone, Debug)]
pub struct DenyList {
    kind: PiiKind,
    terms: Vec<String>,
    case: Case,
    max_edits: u8,
    capitalised_only: bool,
    matcher: Option<TextMatcher>,
    fuzzy: FuzzyIndex,
}

impl PartialEq for DenyList {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.terms == other.terms
            && self.case == other.case
            && self.max_edits == other.max_edits
            && self.capitalised_only == other.capitalised_only
    }
}

impl Eq for DenyList {}

impl DenyList {
    /// Creates an empty deny list reporting matches as `kind`, which must be
    /// a [`PiiKind::Custom`]. [`crate::PiiDetector::with_deny_list`] refuses
    /// it when `kind` is outside the detector's custom-kind table.
    pub fn new(kind: PiiKind) -> Result<DenyList, ListError> {
        if !kind.is_custom() {
            return Err(ListError::BuiltinKind(kind));
        }
        Ok(DenyList {
            kind,
            terms: Vec::new(),
            case: Case::Sensitive,
            max_edits: 0,
            capitalised_only: false,
            matcher: None,
            fuzzy: FuzzyIndex::default(),
        })
    }

    /// Creates a deny list for `kind` initialized from `terms`.
    pub fn from_terms<I, S>(kind: PiiKind, terms: I) -> Result<DenyList, ListError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::new(kind)?.with_terms(terms)
    }

    /// Creates a deny list for `kind` loaded from `path`, one term per line,
    /// ignoring blank lines and `#` comments.
    pub fn from_file(kind: PiiKind, path: impl AsRef<Path>) -> Result<DenyList, ListError> {
        Self::new(kind)?.with_file(path)
    }

    /// The [`PiiKind::Custom`] this deny list emits.
    pub const fn kind(&self) -> PiiKind {
        self.kind
    }

    /// The terms registered in this deny list.
    pub fn terms(&self) -> &[String] {
        &self.terms
    }

    /// Whether this deny list holds no terms.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// How letters are compared (`Case::Sensitive` or `Case::Insensitive`).
    pub const fn case(&self) -> Case {
        self.case
    }

    /// Maximum Levenshtein edit distance (`0`, `1`, or `2`).
    pub const fn max_edits(&self) -> u8 {
        self.max_edits
    }

    /// Whether only capitalised candidates in the input text are matched.
    pub const fn is_capitalised_only(&self) -> bool {
        self.capitalised_only
    }

    /// Alias for [`DenyList::is_capitalised_only`].
    pub const fn is_capitalized_only(&self) -> bool {
        self.capitalised_only
    }

    /// Sets how letters are compared (`Case::Sensitive` or `Case::Insensitive`).
    pub fn with_case(mut self, case: Case) -> DenyList {
        self.case = case;
        self.rebuild();
        self
    }

    /// Sets [`Case::Insensitive`] on this deny list.
    pub fn case_insensitive(self) -> DenyList {
        self.with_case(Case::Insensitive)
    }

    /// Sets [`Case::Sensitive`] on this deny list.
    pub fn case_sensitive(self) -> DenyList {
        self.with_case(Case::Sensitive)
    }

    /// Sets the maximum Levenshtein edit distance (`0`, `1`, or `2`) over
    /// word tokens. Words shorter than `4` characters only match at `0` edits;
    /// words of `4..=5` characters allow at most `1` edit; words of `6` or more
    /// characters allow up to `2` edits.
    pub fn with_max_edits(mut self, max_edits: u8) -> Result<DenyList, ListError> {
        if max_edits > 2 {
            return Err(ListError::InvalidEditDistance(max_edits));
        }
        self.max_edits = max_edits;
        Ok(self)
    }

    /// Requires candidate spans in the text to begin with an uppercase or
    /// titlecase letter when `enabled` is `true`.
    pub fn with_capitalised_only(mut self, enabled: bool) -> DenyList {
        self.capitalised_only = enabled;
        self
    }

    /// Alias for [`DenyList::with_capitalised_only`].
    pub fn with_capitalized_only(self, enabled: bool) -> DenyList {
        self.with_capitalised_only(enabled)
    }

    /// Enables the `capitalised_only` guard (`with_capitalised_only(true)`).
    pub fn capitalised_only(self) -> DenyList {
        self.with_capitalised_only(true)
    }

    /// Alias for [`DenyList::capitalised_only`].
    pub fn capitalized_only(self) -> DenyList {
        self.with_capitalised_only(true)
    }

    /// Adds a term to this deny list.
    pub fn add_term(&mut self, term: &str) -> Result<&mut Self, ListError> {
        let trimmed = term.trim();
        if trimmed.is_empty() {
            return Err(ListError::EmptyTerm);
        }
        self.insert_term(trimmed.to_owned());
        self.rebuild();
        Ok(self)
    }

    /// Builder method adding a term to this deny list.
    pub fn with_term(mut self, term: &str) -> Result<DenyList, ListError> {
        self.add_term(term)?;
        Ok(self)
    }

    /// Adds multiple terms to this deny list.
    pub fn add_terms<I, S>(&mut self, terms: I) -> Result<&mut Self, ListError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut batch = Vec::new();
        for term in terms {
            let trimmed = term.as_ref().trim();
            if trimmed.is_empty() {
                return Err(ListError::EmptyTerm);
            }
            batch.push(trimmed.to_owned());
        }
        for term in batch {
            self.insert_term(term);
        }
        self.rebuild();
        Ok(self)
    }

    /// Builder method adding multiple terms to this deny list.
    pub fn with_terms<I, S>(mut self, terms: I) -> Result<DenyList, ListError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.add_terms(terms)?;
        Ok(self)
    }

    /// Loads terms from `path`, one term per line, ignoring blank lines and
    /// `#` comments.
    pub fn add_file(&mut self, path: impl AsRef<Path>) -> Result<&mut Self, ListError> {
        for line in read_file_lines(path.as_ref())? {
            self.insert_term(line);
        }
        self.rebuild();
        Ok(self)
    }

    /// Builder method loading terms from `path`.
    pub fn with_file(mut self, path: impl AsRef<Path>) -> Result<DenyList, ListError> {
        self.add_file(path)?;
        Ok(self)
    }

    fn insert_term(&mut self, term: String) {
        let exists = match self.case {
            Case::Sensitive => self.terms.contains(&term),
            Case::Insensitive => {
                let key = fold_lower(&term);
                self.terms.iter().any(|t| fold_lower(t) == key)
            }
        };
        if !exists {
            self.terms.push(term);
        }
    }

    fn rebuild(&mut self) {
        let mut seen = HashSet::with_capacity(self.terms.len());
        let case = self.case;
        self.terms.retain(|t| {
            let key = match case {
                Case::Sensitive => t.clone(),
                Case::Insensitive => fold_lower(t),
            };
            seen.insert(key)
        });
        if self.terms.is_empty() {
            self.matcher = None;
            self.fuzzy = FuzzyIndex::default();
            return;
        }
        let mut glossary = Glossary::new();
        for term in &self.terms {
            glossary.push("deny", term).expect("validated non-blank kind and term");
        }
        self.matcher = Some(TextMatcher::new(&glossary, self.case).expect("non-empty glossary"));
        let terms = self
            .terms
            .iter()
            .enumerate()
            .filter_map(|(i, term)| build_fuzzy_term(EntryIx(i as u32), term, self.case))
            .collect();
        self.fuzzy = FuzzyIndex::new(terms);
    }

    /// Scans `text` and appends non-overlapping matches for this deny list
    /// to `hits`, each with a `DenyHit` fact naming `list` (this list's
    /// position in the detector's composition order) and the term.
    pub(crate) fn scan(&self, text: &str, list: u32, hits: &mut Hits) {
        let Some(matcher) = &self.matcher else {
            return;
        };
        let mut candidates: Vec<Candidate> = Vec::new();
        matcher.scan(text, |span, entry| {
            let (start, end) = (span.start as usize, span.end as usize);
            let covered = &text[start..end];
            if covered.chars().next().is_some_and(letter_or_digit) && !on_word_start(text, start) {
                return;
            }
            if covered.chars().next_back().is_some_and(letter_or_digit) && !on_end(text, end) {
                return;
            }
            if self.capitalised_only && !is_capitalised(covered) {
                return;
            }
            candidates.push(Candidate { span, tokens: 0, entry });
        });

        if self.max_edits > 0 && !self.fuzzy.is_empty() {
            self.fuzzy.scan(text, self.case, self.max_edits, self.capitalised_only, &mut candidates);
        }

        if candidates.is_empty() {
            return;
        }
        let mut selected: Vec<GlossaryMatch> = Vec::new();
        select(&mut candidates, &mut selected);
        for m in selected {
            let fact = PiiEvidence::DenyHit { list, term: m.entry.0 };
            hits.add_text(text, m.span.start as usize, m.span.end as usize, self.kind, fact);
        }
    }

    pub(crate) fn fold_hash(&self, h: &mut Fnv1a) {
        h.u32(self.kind.code())
            .u8(case_tag(self.case))
            .u8(self.max_edits)
            .bool(self.capitalised_only)
            .usize(self.terms.len());
        for term in &self.terms {
            h.str(term);
        }
    }
}

fn fold_lower(text: &str) -> String {
    text.chars().map(to_lower_char).collect()
}
