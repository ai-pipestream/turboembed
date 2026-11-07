//! Allow lists (ignore rules) for suppressing false-positive or test PII mentions.

use std::collections::HashSet;
use std::path::Path;

use turbo_analysis::Fnv1a;
use turbo_analysis_layers::AllowRule;
use turbo_glossary::Case;
use turbo_unicode::java::to_lower_char;

use super::{ListError, Pattern, case_tag, read_file_lines};

fn fold_lower(text: &str) -> String {
    text.chars().map(to_lower_char).collect()
}

/// Rules that suppress a detected mention when its covered text or
/// normalized form matches an exact term or a [`Pattern`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllowList {
    case: Case,
    terms: Vec<(String, Case)>,
    exact_set: HashSet<String>,
    folded_set: HashSet<String>,
    patterns: Vec<Pattern>,
}

impl Default for AllowList {
    fn default() -> Self {
        Self::new()
    }
}

impl AllowList {
    /// An empty case-sensitive allow list.
    pub fn new() -> AllowList {
        AllowList {
            case: Case::Sensitive,
            terms: Vec::new(),
            exact_set: HashSet::new(),
            folded_set: HashSet::new(),
            patterns: Vec::new(),
        }
    }

    /// An allow list initialized from `terms`.
    pub fn from_terms<I, S>(terms: I) -> Result<AllowList, ListError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        AllowList::new().with_terms(terms)
    }

    /// Loads an allow list of exact terms from `path`, one term per line,
    /// ignoring blank lines and `#` comments.
    pub fn from_file(path: impl AsRef<Path>) -> Result<AllowList, ListError> {
        AllowList::new().with_file(path)
    }

    /// Sets the case mode for this allow list, updating all terms and patterns
    /// currently held by the list and using `case` for subsequent additions.
    pub fn with_case(mut self, case: Case) -> AllowList {
        self.case = case;
        for (_, term_case) in &mut self.terms {
            *term_case = case;
        }
        for pat in std::mem::take(&mut self.patterns) {
            self.add_pattern(pat.with_case(case));
        }
        self.rebuild_sets();
        self
    }

    /// Sets [`Case::Insensitive`] on this allow list.
    pub fn case_insensitive(self) -> AllowList {
        self.with_case(Case::Insensitive)
    }

    /// Sets [`Case::Sensitive`] on this allow list.
    pub fn case_sensitive(self) -> AllowList {
        self.with_case(Case::Sensitive)
    }

    /// The default [`Case`] of this allow list.
    pub const fn case(&self) -> Case {
        self.case
    }

    /// Whether this allow list holds no terms and no patterns.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty() && self.patterns.is_empty()
    }

    /// The exact terms and their case settings in registration order.
    pub fn terms(&self) -> &[(String, Case)] {
        &self.terms
    }

    /// The compiled patterns in registration order.
    pub fn patterns(&self) -> &[Pattern] {
        &self.patterns
    }

    /// Adds an exact term using this list's current [`Case`].
    pub fn add_term(&mut self, term: &str) -> Result<&mut Self, ListError> {
        self.add_term_with_case(term, self.case)
    }

    /// Adds an exact term with an explicit [`Case`].
    pub fn add_term_with_case(&mut self, term: &str, case: Case) -> Result<&mut Self, ListError> {
        let trimmed = term.trim();
        if trimmed.is_empty() {
            return Err(ListError::EmptyTerm);
        }
        self.insert_term(trimmed.to_owned(), case);
        Ok(self)
    }

    /// Builder method adding an exact term using this list's current [`Case`].
    pub fn with_term(mut self, term: &str) -> Result<AllowList, ListError> {
        self.add_term(term)?;
        Ok(self)
    }

    /// Adds multiple exact terms using this list's current [`Case`].
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
            self.insert_term(term, self.case);
        }
        Ok(self)
    }

    /// Builder method adding multiple exact terms using this list's current [`Case`].
    pub fn with_terms<I, S>(mut self, terms: I) -> Result<AllowList, ListError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.add_terms(terms)?;
        Ok(self)
    }

    /// Loads exact terms from `path`, one term per line, ignoring blank lines
    /// and `#` comments.
    pub fn add_file(&mut self, path: impl AsRef<Path>) -> Result<&mut Self, ListError> {
        for line in read_file_lines(path.as_ref())? {
            self.insert_term(line, self.case);
        }
        Ok(self)
    }

    /// Builder method loading exact terms from `path`.
    pub fn with_file(mut self, path: impl AsRef<Path>) -> Result<AllowList, ListError> {
        self.add_file(path)?;
        Ok(self)
    }

    /// Adds a compiled [`Pattern`].
    pub fn add_pattern(&mut self, pattern: Pattern) -> &mut Self {
        if !self.patterns.contains(&pattern) {
            self.patterns.push(pattern);
        }
        self
    }

    /// Builder method adding a compiled [`Pattern`].
    pub fn with_pattern(mut self, pattern: Pattern) -> AllowList {
        self.add_pattern(pattern);
        self
    }

    /// Compiles and adds a shape pattern (`d` digit, `a` letter, `w` alphanumeric,
    /// `?` any char, `*` any sequence, `\x` escaped literal) using this list's [`Case`].
    pub fn add_shape(&mut self, shape: &str) -> Result<&mut Self, ListError> {
        let pat = Pattern::shape(shape)?.with_case(self.case);
        self.add_pattern(pat);
        Ok(self)
    }

    /// Builder method adding a shape pattern using this list's [`Case`].
    pub fn with_shape(mut self, shape: &str) -> Result<AllowList, ListError> {
        self.add_shape(shape)?;
        Ok(self)
    }

    /// Compiles and adds a glob pattern (`*`, `?`, `\x`) using this list's [`Case`].
    pub fn add_glob(&mut self, glob: &str) -> Result<&mut Self, ListError> {
        let pat = Pattern::glob(glob)?.with_case(self.case);
        self.add_pattern(pat);
        Ok(self)
    }

    /// Builder method adding a glob pattern using this list's [`Case`].
    pub fn with_glob(mut self, glob: &str) -> Result<AllowList, ListError> {
        self.add_glob(glob)?;
        Ok(self)
    }

    /// Loads shape patterns from `path`, one shape pattern per line, ignoring
    /// blank lines and `#` comments.
    pub fn add_shapes_file(&mut self, path: impl AsRef<Path>) -> Result<&mut Self, ListError> {
        let mut batch = Vec::new();
        for line in read_file_lines(path.as_ref())? {
            batch.push(Pattern::shape(&line)?.with_case(self.case));
        }
        for pat in batch {
            self.add_pattern(pat);
        }
        Ok(self)
    }

    /// Builder method loading shape patterns from `path`.
    pub fn with_shapes_file(mut self, path: impl AsRef<Path>) -> Result<AllowList, ListError> {
        self.add_shapes_file(path)?;
        Ok(self)
    }

    /// Loads glob patterns from `path`, one glob pattern per line, ignoring
    /// blank lines and `#` comments.
    pub fn add_globs_file(&mut self, path: impl AsRef<Path>) -> Result<&mut Self, ListError> {
        let mut batch = Vec::new();
        for line in read_file_lines(path.as_ref())? {
            batch.push(Pattern::glob(&line)?.with_case(self.case));
        }
        for pat in batch {
            self.add_pattern(pat);
        }
        Ok(self)
    }

    /// Builder method loading glob patterns from `path`.
    pub fn with_globs_file(mut self, path: impl AsRef<Path>) -> Result<AllowList, ListError> {
        self.add_globs_file(path)?;
        Ok(self)
    }

    /// Merges all terms and patterns from `other` into `self`, preserving each
    /// entry's case mode.
    pub fn merge(&mut self, other: AllowList) {
        for (term, case) in other.terms {
            self.insert_term(term, case);
        }
        for pat in other.patterns {
            self.add_pattern(pat);
        }
    }

    fn insert_term(&mut self, term: String, case: Case) {
        let inserted = match case {
            Case::Sensitive => self.exact_set.insert(term.clone()),
            Case::Insensitive => self.folded_set.insert(fold_lower(&term)),
        };
        if inserted {
            self.terms.push((term, case));
        }
    }

    fn rebuild_sets(&mut self) {
        self.exact_set.clear();
        self.folded_set.clear();
        for (term, case) in std::mem::take(&mut self.terms) {
            self.insert_term(term, case);
        }
    }

    /// Whether `value` matches any exact term or pattern in this allow list.
    pub fn matches_value(&self, value: &str) -> bool {
        if self.exact_set.contains(value) {
            return true;
        }
        if !self.folded_set.is_empty() && self.folded_set.contains(&fold_lower(value)) {
            return true;
        }
        self.patterns.iter().any(|p| p.matches(value))
    }

    /// Whether a mention with `covered` surface text and `normalized` form
    /// matches any rule in this allow list.
    pub fn matches(&self, covered: &str, normalized: &str) -> bool {
        self.find_rule(covered, normalized).is_some()
    }

    /// The rule `value` matches: the first exact term in registration
    /// order, then the first pattern.
    pub fn find_rule_for_value(&self, value: &str) -> Option<AllowRule> {
        let exact = self.exact_set.contains(value);
        let folded = (!self.folded_set.is_empty()).then(|| fold_lower(value)).filter(|f| self.folded_set.contains(f));
        if exact || folded.is_some() {
            let at = self.terms.iter().position(|(term, case)| match case {
                Case::Sensitive => exact && term == value,
                Case::Insensitive => folded.as_deref().is_some_and(|f| fold_lower(term) == f),
            })?;
            return Some(AllowRule::Term(at as u32));
        }
        self.patterns.iter().position(|p| p.matches(value)).map(|at| AllowRule::Pattern(at as u32))
    }

    /// The rule a mention with `covered` surface text and `normalized` form
    /// matches: by the covered text first, then by the normalized form.
    pub fn find_rule(&self, covered: &str, normalized: &str) -> Option<AllowRule> {
        if self.is_empty() {
            return None;
        }
        self.find_rule_for_value(covered)
            .or_else(|| (normalized != covered).then(|| self.find_rule_for_value(normalized)).flatten())
    }

    pub(crate) fn fold_hash(&self, h: &mut Fnv1a) {
        h.u8(case_tag(self.case)).usize(self.terms.len());
        for (term, case) in &self.terms {
            h.str(term).u8(case_tag(*case));
        }
        h.usize(self.patterns.len());
        for pat in &self.patterns {
            pat.fold_hash(h);
        }
    }
}
