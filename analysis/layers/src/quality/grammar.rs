//! Syntactic and morphological grammar and agreement diagnostics (`turbo:grammar_diagnostics`)
//! and the clean-room Apache-2.0 [`GrammarChecker`] annotator.

use std::fmt;
use std::sync::{OnceLock, RwLock};

use turbo_analysis::{
    Annotator, Bound, Document, Error, FilterableLayer, Fnv1a, Ix, Key, Layer, Overlap, Refs, Span, Spans, TextSpans,
    View, ViewId,
};

use crate::{Dependencies, Lemmas, PosTags, Sentences, Tokens};

use super::agreement::check_sentence_rules;

static CUSTOM_RULE_IDS: OnceLock<RwLock<Vec<&'static str>>> = OnceLock::new();

fn custom_rule_registry() -> &'static RwLock<Vec<&'static str>> {
    CUSTOM_RULE_IDS.get_or_init(|| RwLock::new(Vec::new()))
}

/// Grammatical or morphological category of a [`GrammarDiagnostic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum GrammarCategory {
    SubjectVerbAgreement = 0,
    DeterminerNounAgreement = 1,
    ArticlePhonetics = 2,
    AuxiliaryVerbForm = 3,
    RepeatedToken = 4,
    CaseGovernment = 5,
    ClausePunctuation = 6,
}

impl GrammarCategory {
    /// All [`GrammarCategory`] variants in wire-tag order (`0..=6`).
    pub const ALL: [Self; 7] = [
        Self::SubjectVerbAgreement,
        Self::DeterminerNounAgreement,
        Self::ArticlePhonetics,
        Self::AuxiliaryVerbForm,
        Self::RepeatedToken,
        Self::CaseGovernment,
        Self::ClausePunctuation,
    ];

    /// Stable `u8` wire tag (`0..=6`).
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Decodes a [`GrammarCategory`] from its `u8` wire tag.
    pub fn from_byte(b: u8) -> Option<Self> {
        Self::ALL.get(b as usize).copied()
    }

    /// Canonical snake_case identifier used across wire projections.
    pub const fn name(self) -> &'static str {
        match self {
            Self::SubjectVerbAgreement => "subject_verb_agreement",
            Self::DeterminerNounAgreement => "determiner_noun_agreement",
            Self::ArticlePhonetics => "article_phonetics",
            Self::AuxiliaryVerbForm => "auxiliary_verb_form",
            Self::RepeatedToken => "repeated_token",
            Self::CaseGovernment => "case_government",
            Self::ClausePunctuation => "clause_punctuation",
        }
    }

    /// Alias for [`Self::name`].
    pub const fn as_str(self) -> &'static str {
        self.name()
    }

    /// Parses a [`GrammarCategory`] from its canonical snake_case or PascalCase name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim() {
            "subject_verb_agreement" | "SubjectVerbAgreement" => Some(Self::SubjectVerbAgreement),
            "determiner_noun_agreement" | "DeterminerNounAgreement" => Some(Self::DeterminerNounAgreement),
            "article_phonetics" | "ArticlePhonetics" => Some(Self::ArticlePhonetics),
            "auxiliary_verb_form" | "AuxiliaryVerbForm" => Some(Self::AuxiliaryVerbForm),
            "repeated_token" | "RepeatedToken" => Some(Self::RepeatedToken),
            "case_government" | "CaseGovernment" => Some(Self::CaseGovernment),
            "clause_punctuation" | "ClausePunctuation" => Some(Self::ClausePunctuation),
            _ => None,
        }
    }

    /// The canonical built-in rule identifier for this category.
    pub const fn canonical_rule_id(self) -> &'static str {
        match self {
            Self::SubjectVerbAgreement => "SVA-001",
            Self::DeterminerNounAgreement => "DET-001",
            Self::ArticlePhonetics => "ART-001",
            Self::AuxiliaryVerbForm => "AUX-001",
            Self::RepeatedToken => "REP-001",
            Self::CaseGovernment => "CASE-001",
            Self::ClausePunctuation => "PUNCT-001",
        }
    }
}

impl fmt::Display for GrammarCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Maximum byte length of a [`GrammarDiagnostic::rule_id`].
pub const MAX_RULE_ID_BYTES: usize = 64;

/// Maximum number of non-canonical custom rule IDs interned in the process registry.
pub const MAX_CUSTOM_RULE_IDS: usize = 256;

const BUILTIN_RULE_IDS: [&str; 7] = ["SVA-001", "DET-001", "ART-001", "AUX-001", "REP-001", "CASE-001", "PUNCT-001"];

/// Resolves a built-in rule identifier to its static slice without locking.
pub fn builtin_rule_id(id: &str) -> Option<&'static str> {
    BUILTIN_RULE_IDS.into_iter().find(|&known| known == id)
}

/// Registers a static rule ID so binary import can resolve it.
pub(crate) fn remember_rule_id(rule_id: &'static str) -> bool {
    let trimmed = rule_id.trim();
    if trimmed.is_empty() || trimmed.len() != rule_id.len() || rule_id.len() > MAX_RULE_ID_BYTES {
        return false;
    }
    if builtin_rule_id(rule_id).is_some()
        || custom_rule_registry().read().expect("grammar rule registry lock").contains(&rule_id)
    {
        return true;
    }
    let mut reg = custom_rule_registry().write().expect("grammar rule registry lock");
    if reg.contains(&rule_id) {
        return true;
    }
    if reg.len() >= MAX_CUSTOM_RULE_IDS {
        return false;
    }
    reg.push(rule_id);
    true
}

/// Resolves or interns `rule_id` during binary `DocumentGraph` import.
pub(crate) fn intern_rule_id(rule_id: &str) -> Option<&'static str> {
    let trimmed = rule_id.trim();
    if trimmed.is_empty() || trimmed.len() != rule_id.len() || rule_id.len() > MAX_RULE_ID_BYTES {
        return None;
    }
    if let Some(known) = builtin_rule_id(rule_id) {
        return Some(known);
    }
    if let Some(&existing) =
        custom_rule_registry().read().expect("grammar rule registry lock").iter().find(|&&r| r == rule_id)
    {
        return Some(existing);
    }
    let mut reg = custom_rule_registry().write().expect("grammar rule registry lock");
    if let Some(&existing) = reg.iter().find(|&&r| r == rule_id) {
        return Some(existing);
    }
    (reg.len() < MAX_CUSTOM_RULE_IDS).then(|| {
        let leaked: &'static str = Box::leak(rule_id.to_owned().into_boxed_str());
        reg.push(leaked);
        leaked
    })
}

/// One syntactic or morphological grammar diagnostic over a text view.
#[derive(Debug, Clone, PartialEq)]
pub struct GrammarDiagnostic {
    pub span: Span,
    pub rule_id: &'static str,
    pub category: GrammarCategory,
    pub message: Box<str>,
    pub replacements: Box<[Box<str>]>,
    pub confidence: f32,
}

impl GrammarDiagnostic {
    /// Creates a new [`GrammarDiagnostic`].
    pub fn new(
        span: Span,
        rule_id: &'static str,
        category: GrammarCategory,
        message: impl Into<Box<str>>,
        replacements: impl IntoIterator<Item = impl Into<Box<str>>>,
        confidence: f32,
    ) -> Self {
        let _ = remember_rule_id(rule_id);
        Self {
            span,
            rule_id,
            category,
            message: message.into(),
            replacements: replacements.into_iter().map(Into::into).collect::<Vec<_>>().into_boxed_slice(),
            confidence,
        }
    }

    /// Validates field invariants on this diagnostic.
    pub fn check(&self) -> Result<(), String> {
        if self.span.is_empty() {
            return Err("grammar diagnostic span must be nonempty".into());
        }
        if !remember_rule_id(self.rule_id) {
            return Err("grammar diagnostic rule_id must be nonempty, trimmed, and within registry bounds".into());
        }
        if self.message.trim().is_empty() {
            return Err("grammar diagnostic message must be nonempty".into());
        }
        if self.replacements.iter().any(|r| r.is_empty()) {
            return Err("grammar diagnostic replacements must be nonempty strings".into());
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err("grammar diagnostic confidence must be finite and in [0, 1]".into());
        }
        Ok(())
    }
}

/// View-anchored layer of syntactic and morphological grammar diagnostics (`turbo:grammar_diagnostics`).
#[derive(Debug, Clone)]
pub struct GrammarDiagnostics {
    pub spans: Spans<GrammarDiagnostics>,
    pub diagnostics: Vec<GrammarDiagnostic>,
}

impl PartialEq for GrammarDiagnostics {
    fn eq(&self, other: &Self) -> bool {
        self.spans.view() == other.spans.view()
            && self.diagnostics == other.diagnostics
            && self.spans.iter().all(|(ix, _)| self.spans.fragments(ix).eq(other.spans.fragments(ix)))
    }
}

impl GrammarDiagnostics {
    /// Creates an empty [`GrammarDiagnostics`] layer on `view`.
    pub fn new(view: ViewId) -> Self {
        Self { spans: Spans::new(view), diagnostics: Vec::new() }
    }

    /// Creates an empty [`GrammarDiagnostics`] layer on `view` with pre-allocated `capacity`.
    pub fn with_capacity(view: ViewId, capacity: usize) -> Self {
        Self { spans: Spans::with_capacity(view, capacity), diagnostics: Vec::with_capacity(capacity) }
    }

    /// The [`ViewId`] this layer's spans index.
    pub fn view(&self) -> ViewId {
        self.spans.view()
    }

    /// Appends a [`GrammarDiagnostic`] in start order.
    pub fn push(&mut self, diagnostic: GrammarDiagnostic) -> Ix<Self> {
        let span = diagnostic.span;
        self.diagnostics.push(diagnostic);
        self.spans.push(span)
    }

    /// Number of diagnostics in this layer.
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// Whether this layer contains no diagnostics.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Returns diagnostic `index` if in bounds.
    pub fn get(&self, index: usize) -> Option<&GrammarDiagnostic> {
        self.diagnostics.get(index)
    }

    /// Slice of all diagnostics in start order.
    pub fn as_slice(&self) -> &[GrammarDiagnostic] {
        &self.diagnostics
    }

    /// Iterates over all diagnostics in start order.
    pub fn iter(&self) -> std::slice::Iter<'_, GrammarDiagnostic> {
        self.diagnostics.iter()
    }
}

impl Layer for GrammarDiagnostics {
    const ID: &'static str = "turbo:grammar_diagnostics";
    const OVERLAP: Overlap = Overlap::Any;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        self.spans.check(Self::OVERLAP)?;
        if self.diagnostics.len() != self.spans.len() {
            return Err("one grammar diagnostic per span".into());
        }
        if !self.spans.is_start_ordered() {
            return Err("grammar diagnostics must be in start order".into());
        }
        for (i, span) in self.spans.iter() {
            if span.is_empty() || self.spans.is_discontinuous(i) {
                return Err("grammar diagnostic spans must be nonempty and continuous".into());
            }
            let d = &self.diagnostics[i.get()];
            if d.span != span {
                return Err("grammar diagnostic span must match layer span column".into());
            }
            d.check()?;
        }
        Ok(())
    }
}

impl FilterableLayer for GrammarDiagnostics {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Self {
            spans: self.spans.retain_indices(kept),
            diagnostics: kept.iter().map(|i| self.diagnostics[i.get()].clone()).collect(),
        }
    }
}

/// Clean-room syntactic and morphological agreement and grammar checker (`turbo:grammar_checker`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GrammarChecker {
    pub check_subject_verb_agreement: bool,
    pub check_determiner_noun_agreement: bool,
    pub check_article_phonetics: bool,
    pub check_auxiliary_verb_form: bool,
    pub check_repeated_tokens: bool,
    pub check_case_government: bool,
    pub check_clause_punctuation: bool,
    pub american_silent_h_herb: bool,
}

impl Default for GrammarChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl GrammarChecker {
    /// Creates a [`GrammarChecker`] with all 7 rule families enabled.
    pub const fn new() -> Self {
        Self {
            check_subject_verb_agreement: true,
            check_determiner_noun_agreement: true,
            check_article_phonetics: true,
            check_auxiliary_verb_form: true,
            check_repeated_tokens: true,
            check_case_government: true,
            check_clause_punctuation: true,
            american_silent_h_herb: true,
        }
    }

    pub const fn with_subject_verb_agreement(self, enabled: bool) -> Self {
        Self { check_subject_verb_agreement: enabled, ..self }
    }

    pub const fn with_determiner_noun_agreement(self, enabled: bool) -> Self {
        Self { check_determiner_noun_agreement: enabled, ..self }
    }

    pub const fn with_article_phonetics(self, enabled: bool) -> Self {
        Self { check_article_phonetics: enabled, ..self }
    }

    pub const fn with_auxiliary_verb_form(self, enabled: bool) -> Self {
        Self { check_auxiliary_verb_form: enabled, ..self }
    }

    pub const fn with_repeated_tokens(self, enabled: bool) -> Self {
        Self { check_repeated_tokens: enabled, ..self }
    }

    pub const fn with_case_government(self, enabled: bool) -> Self {
        Self { check_case_government: enabled, ..self }
    }

    pub const fn with_clause_punctuation(self, enabled: bool) -> Self {
        Self { check_clause_punctuation: enabled, ..self }
    }

    pub const fn with_american_silent_h_herb(self, enabled: bool) -> Self {
        Self { american_silent_h_herb: enabled, ..self }
    }

    /// Binds this checker to the default input and output layer keys.
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self)
    }

    /// Runs the configured grammar rules over `view` and its syntactic layers.
    pub fn check_view(
        &self,
        view: &View,
        sentences: &Sentences,
        tokens: &Tokens,
        pos: &PosTags,
        lemmas: Option<&Lemmas>,
        deps: Option<&Dependencies>,
    ) -> Result<GrammarDiagnostics, Error> {
        sentences.check_contiguous_in(view).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        tokens.check_contiguous_bounds_in(view).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        if pos.len() != tokens.len() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("pos tags length {} != tokens length {}", pos.len(), tokens.len()),
            });
        }
        if let Some(l) = lemmas
            && l.len() != tokens.len()
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("lemmas length {} != tokens length {}", l.len(), tokens.len()),
            });
        }
        if let Some(d) = deps {
            if d.len() != tokens.len() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("dependencies length {} != tokens length {}", d.len(), tokens.len()),
                });
            }
            d.check_forest().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        }

        let mut items = check_sentence_rules(self, view.text(), sentences, tokens, pos, lemmas, deps);
        items.sort_by(|a, b| {
            (a.span.start, a.span.end, a.category.as_u8()).cmp(&(b.span.start, b.span.end, b.category.as_u8()))
        });
        items.dedup_by(|a, b| a.span == b.span && a.rule_id == b.rule_id);

        let mut out = GrammarDiagnostics::with_capacity(view.id(), items.len());
        for item in items {
            out.push(item);
        }
        out.check().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok(out)
    }

    /// Convenience helper that fetches default layer instances from `doc` and runs [`Self::check_view`].
    pub fn check_document(&self, doc: &Document) -> Result<GrammarDiagnostics, Error> {
        let view = doc.require(&Key::<View>::DEFAULT)?;
        let sentences = doc.require(&Key::<Sentences>::DEFAULT)?;
        let tokens = doc.require(&Key::<Tokens>::DEFAULT)?;
        let pos = doc.require(&Key::<PosTags>::DEFAULT)?;
        let lemmas = doc.get::<Lemmas>();
        let deps = doc.get::<Dependencies>();
        self.check_view(view, sentences, tokens, pos, lemmas, deps)
    }
}

impl Annotator for GrammarChecker {
    const ID: &'static str = "turbo:grammar_checker";
    const VERSION: &'static str = "1";
    type Reads = (View, Sentences, Tokens, PosTags, Option<Lemmas>, Option<Dependencies>);
    type Writes = (GrammarDiagnostics,);

    fn annotate(&self, (view, sentences, tokens, pos, lemmas, deps): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let out = self.check_view(view, sentences, tokens, pos, lemmas, deps)?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        Fnv1a::new()
            .bytes(b"grammar-checker-v1")
            .u8(u8::from(self.check_subject_verb_agreement))
            .u8(u8::from(self.check_determiner_noun_agreement))
            .u8(u8::from(self.check_article_phonetics))
            .u8(u8::from(self.check_auxiliary_verb_form))
            .u8(u8::from(self.check_repeated_tokens))
            .u8(u8::from(self.check_case_government))
            .u8(u8::from(self.check_clause_punctuation))
            .u8(u8::from(self.american_silent_h_herb))
            .finish()
    }
}
