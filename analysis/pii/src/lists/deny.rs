//! Deny lists: caller terms matched with `turbo_glossary::TextMatcher`
//! (Aho-Corasick) and optional bounded Levenshtein fuzzy matching over word
//! tokens, always reported as a caller-named [`PiiKind::Custom`].

use std::collections::HashSet;
use std::path::Path;

use turbo_analysis::{Fnv1a, Span};
use turbo_analysis_layers::{PiiConfidence, PiiKind};
use turbo_glossary::{Candidate, Case, EntryIx, Glossary, GlossaryMatch, TextMatcher, select};
use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::java::to_lower_char;

use crate::hits::Hits;
use crate::text::{letter_or_digit, on_end, on_word_start};

use super::{ListError, case_tag, read_file_lines};

#[derive(Clone, Debug)]
struct FuzzyTerm {
    entry: EntryIx,
    words: Vec<Vec<char>>,
    separators: Vec<String>,
}

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
    fuzzy_terms: Vec<FuzzyTerm>,
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
    /// Creates an empty deny list reporting matches as `kind`, which must be a
    /// valid [`PiiKind::Custom`] (1 to [`PiiKind::MAX_CUSTOM_NAME_BYTES`] bytes
    /// and not a reserved built-in kind name).
    pub fn new(kind: PiiKind) -> Result<DenyList, ListError> {
        let name = kind.name();
        if name.is_empty() {
            return Err(ListError::EmptyKind);
        }
        if PiiKind::parse(name).is_some() {
            return Err(ListError::ReservedKind(name));
        }
        if name.len() > PiiKind::MAX_CUSTOM_NAME_BYTES {
            return Err(ListError::KindTooLong(name));
        }
        if !PiiKind::remember_custom(kind) {
            return Err(ListError::RegistryFull(name));
        }
        Ok(DenyList {
            kind,
            terms: Vec::new(),
            case: Case::Sensitive,
            max_edits: 0,
            capitalised_only: false,
            matcher: None,
            fuzzy_terms: Vec::new(),
        })
    }

    /// Creates an empty deny list reporting matches as `PiiKind::custom(name)`.
    pub fn custom(name: &'static str) -> Result<DenyList, ListError> {
        Self::new(PiiKind::custom(name))
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
            self.fuzzy_terms.clear();
            return;
        }
        let mut glossary = Glossary::new();
        let id = self.kind.name();
        for term in &self.terms {
            glossary.push(id, term).expect("validated non-blank kind and term");
        }
        self.matcher = Some(TextMatcher::new(&glossary, self.case).expect("non-empty glossary"));
        self.fuzzy_terms = self
            .terms
            .iter()
            .enumerate()
            .filter_map(|(i, term)| build_fuzzy_term(EntryIx(i as u32), term, self.case))
            .collect();
    }

    /// Scans `text` and appends non-overlapping matches for this deny list to `hits`.
    pub(crate) fn scan(&self, text: &str, hits: &mut Hits) {
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

        if self.max_edits > 0 && !self.fuzzy_terms.is_empty() {
            self.scan_fuzzy(text, &mut candidates);
        }

        if candidates.is_empty() {
            return;
        }
        let mut selected: Vec<GlossaryMatch> = Vec::new();
        select(&mut candidates, &mut selected);
        for m in selected {
            hits.add_text(text, m.span.start as usize, m.span.end as usize, self.kind, PiiConfidence::Format);
        }
    }

    fn scan_fuzzy(&self, text: &str, candidates: &mut Vec<Candidate>) {
        let words = word_spans(text);
        if words.is_empty() {
            return;
        }
        let folded_words: Vec<Vec<char>> = words.iter().map(|&(s, e)| fold_chars(&text[s..e], self.case)).collect();
        let mut prev = Vec::new();
        let mut curr = Vec::new();

        for ft in &self.fuzzy_terms {
            let k = ft.words.len();
            if words.len() < k {
                continue;
            }
            for i in 0..=words.len() - k {
                let start = words[i].0;
                let end = words[i + k - 1].1;
                let covered = &text[start..end];
                if self.capitalised_only && !is_capitalised(covered) {
                    continue;
                }
                if !separators_match(text, &words[i..i + k], &ft.separators) {
                    continue;
                }
                let mut total_edits = 0u8;
                let mut ok = true;
                for m in 0..k {
                    let rem = (self.max_edits - total_edits).min(max_word_edits(ft.words[m].len()));
                    match levenshtein_within(&folded_words[i + m], &ft.words[m], rem, &mut prev, &mut curr) {
                        Some(d) => total_edits += d,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    candidates.push(Candidate {
                        span: Span::new(start as u32, end as u32),
                        tokens: k as u32,
                        entry: ft.entry,
                    });
                }
            }
        }
    }

    pub(crate) fn fold_hash(&self, h: &mut Fnv1a) {
        h.str(self.kind.name())
            .u8(case_tag(self.case))
            .u8(self.max_edits)
            .bool(self.capitalised_only)
            .usize(self.terms.len());
        for term in &self.terms {
            h.str(term);
        }
    }
}

fn is_capitalised(text: &str) -> bool {
    text.chars()
        .next()
        .is_some_and(|c| c.is_uppercase() || matches!(general_category(c), GeneralCategory::Lu | GeneralCategory::Lt))
}

fn fold_lower(text: &str) -> String {
    text.chars().map(to_lower_char).collect()
}

fn fold_chars(text: &str, case: Case) -> Vec<char> {
    match case {
        Case::Sensitive => text.chars().collect(),
        Case::Insensitive => text.chars().map(to_lower_char).collect(),
    }
}

fn max_word_edits(word_len: usize) -> u8 {
    ((word_len.saturating_sub(2)) / 2).min(2) as u8
}

fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if letter_or_digit(c) {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            out.push((s, i));
        }
    }
    if let Some(s) = start {
        out.push((s, text.len()));
    }
    out
}

fn build_fuzzy_term(entry: EntryIx, term: &str, case: Case) -> Option<FuzzyTerm> {
    let spans = word_spans(term);
    if spans.is_empty() {
        return None;
    }
    let words: Vec<Vec<char>> = spans.iter().map(|&(s, e)| fold_chars(&term[s..e], case)).collect();
    if words.iter().all(|w| max_word_edits(w.len()) == 0) {
        return None;
    }
    let separators: Vec<String> = spans.windows(2).map(|w| term[w[0].1..w[1].0].to_owned()).collect();
    Some(FuzzyTerm { entry, words, separators })
}

fn separators_match(text: &str, window: &[(usize, usize)], term_seps: &[String]) -> bool {
    for (w, want) in window.windows(2).zip(term_seps) {
        let gap = &text[w[0].1..w[1].0];
        if want.chars().all(char::is_whitespace) {
            if gap.is_empty() || !gap.chars().all(char::is_whitespace) {
                return false;
            }
        } else if gap != want {
            return false;
        }
    }
    true
}

fn levenshtein_within(a: &[char], b: &[char], max: u8, prev: &mut Vec<usize>, curr: &mut Vec<usize>) -> Option<u8> {
    let max_usize = usize::from(max);
    if a.len().abs_diff(b.len()) > max_usize {
        return None;
    }
    if max == 0 {
        return (a == b).then_some(0);
    }
    let blen = b.len();
    if blen == 0 {
        return Some(a.len() as u8);
    }
    let inf = max_usize + 1;
    if prev.len() <= blen {
        prev.resize(blen + 1, inf);
        curr.resize(blen + 1, inf);
    }
    for (j, slot) in prev.iter_mut().enumerate().take((max_usize + 1).min(blen) + 1) {
        *slot = j;
    }
    for (i, &ca) in a.iter().enumerate() {
        let r = i + 1;
        let lo = r.saturating_sub(max_usize).max(1);
        let hi = (r + max_usize).min(blen);
        curr[lo - 1] = if lo == 1 { r } else { inf };
        if hi < blen {
            curr[hi + 1] = inf;
        }
        let mut row_min = curr[lo - 1];
        for j in lo..=hi {
            let cost = usize::from(ca != b[j - 1]);
            let v = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            curr[j] = v;
            row_min = row_min.min(v);
        }
        if row_min > max_usize {
            return None;
        }
        std::mem::swap(prev, curr);
    }
    (prev[blen] <= max_usize).then_some(prev[blen] as u8)
}
