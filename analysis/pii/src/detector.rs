//! The packs and the detector that runs them, OpenNLP's `PiiPacks`,
//! `CompositePiiExtractor` and `PiiAnnotator`.

use std::path::Path;

use turbo_analysis::ViewId;
use turbo_analysis_layers::{AllowScope, CustomKinds, Pii, PiiCandidates, PiiKind, PiiKinds};

use crate::hits::{AllowDecision, Hits};
use crate::lists::{AllowList, DenyList, ListError, Pattern};
use crate::scan::{BUILTIN_VERSION, Extractor, Scanner};
use crate::window::WindowConfig;

/// The detector id deny list hits name.
pub const DENY_LIST_DETECTOR: &str = "turbo-pii:deny-list";

mod annotate;
mod audit;
mod custom;
mod labels;

/// The packs of `PiiPacks`, by kind of data.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pack {
    /// Cards, IBANs and ABA routing numbers.
    Payment,
    /// E-mail addresses and phone numbers.
    Contact,
    /// IPv4, IPv6 and MAC addresses.
    Network,
    /// AWS keys, GitHub tokens, JWTs and URL credentials.
    Secrets,
    /// Bitcoin and Ethereum addresses.
    Crypto,
    /// United States SSNs and ITINs.
    UsIdentity,
    /// European national and business identifiers (UK, DE, ES, FR, BIC).
    EuIdentity,
    /// Brazilian CPF and CNPJ identifiers.
    BrIdentity,
    /// Canadian SINs after a label.
    CaIdentity,
    /// IMEIs after a label.
    Device,
    /// Every detector: the default `CursorPiiExtractor` and every pack.
    AllStructured,
}

impl Pack {
    pub const ALL: [Pack; 11] = [
        Pack::Payment,
        Pack::Contact,
        Pack::Network,
        Pack::Secrets,
        Pack::Crypto,
        Pack::UsIdentity,
        Pack::EuIdentity,
        Pack::BrIdentity,
        Pack::CaIdentity,
        Pack::Device,
        Pack::AllStructured,
    ];

    /// The name of the `PiiPacks` factory method, in kebab case
    /// (`us-identity`, `all-structured`).
    pub const fn name(self) -> &'static str {
        match self {
            Pack::Payment => "payment",
            Pack::Contact => "contact",
            Pack::Network => "network",
            Pack::Secrets => "secrets",
            Pack::Crypto => "crypto",
            Pack::UsIdentity => "us-identity",
            Pack::EuIdentity => "eu-identity",
            Pack::BrIdentity => "br-identity",
            Pack::CaIdentity => "ca-identity",
            Pack::Device => "device",
            Pack::AllStructured => "all-structured",
        }
    }

    pub fn parse(name: &str) -> Option<Pack> {
        Pack::ALL.into_iter().find(|p| p.name() == name)
    }

    /// The scanners of the pack, in the order the Java composes them.
    pub fn scanners(self) -> Vec<Scanner> {
        use PiiKind::{Card, Email, Iban, Phone};
        let only = |e: Extractor, kinds: &[PiiKind]| {
            Scanner::with_kinds(e, PiiKinds::of(kinds)).expect("kinds of the extractor")
        };
        match self {
            Pack::Payment => vec![only(Extractor::Cursor, &[Iban, Card]), Scanner::all(Extractor::Banking)],
            Pack::Contact => vec![only(Extractor::Cursor, &[Email, Phone])],
            Pack::Network => vec![Scanner::all(Extractor::Network)],
            Pack::Secrets => vec![Scanner::all(Extractor::Secrets)],
            Pack::Crypto => vec![Scanner::all(Extractor::Crypto)],
            Pack::UsIdentity => vec![Scanner::all(Extractor::UsIdentity)],
            Pack::EuIdentity => vec![Scanner::all(Extractor::EuIdentity)],
            Pack::BrIdentity => vec![Scanner::all(Extractor::BrIdentity)],
            Pack::CaIdentity => vec![Scanner::all(Extractor::CaIdentity)],
            Pack::Device => vec![Scanner::all(Extractor::Device)],
            Pack::AllStructured => [
                Extractor::Cursor,
                Extractor::Secrets,
                Extractor::Crypto,
                Extractor::Network,
                Extractor::UsIdentity,
                Extractor::EuIdentity,
                Extractor::BrIdentity,
                Extractor::CaIdentity,
                Extractor::Device,
                Extractor::Banking,
            ]
            .into_iter()
            .map(Scanner::all)
            .collect(),
        }
    }
}

/// Runs scanners over a text and writes their mentions in OpenNLP's
/// order. A detector over several scanners is a `CompositePiiExtractor`:
/// candidates of all of them are ordered together, and ties keep the
/// scanner order. By default, a candidate strictly nested inside a
/// longer candidate is dropped; [`PiiDetector::keeping_nested`] keeps it
/// for OpenNLP parity.
#[derive(Clone, Debug, PartialEq)]
pub struct PiiDetector {
    scanners: Vec<Scanner>,
    custom_kinds: CustomKinds,
    keep_nested: bool,
    window: WindowConfig,
    global_allow: AllowList,
    kind_allow: Vec<(PiiKind, AllowList)>,
    deny_lists: Vec<DenyList>,
    min_score: Option<f32>,
}

impl PiiDetector {
    /// The scanners in the order given. An empty list finds nothing.
    pub fn new(scanners: Vec<Scanner>) -> PiiDetector {
        PiiDetector {
            scanners,
            custom_kinds: CustomKinds::new(),
            keep_nested: false,
            window: WindowConfig::default(),
            global_allow: AllowList::new(),
            kind_allow: Vec::new(),
            deny_lists: Vec::new(),
            min_score: None,
        }
    }

    /// The scanners of `pack`.
    pub fn pack(pack: Pack) -> PiiDetector {
        PiiDetector::new(pack.scanners())
    }

    /// The default `CursorPiiExtractor`: e-mail addresses, phones, IBANs
    /// and cards.
    pub fn cursor() -> PiiDetector {
        PiiDetector::new(vec![Scanner::all(Extractor::Cursor)])
    }

    /// This detector with `scanner` appended in composition order. A custom
    /// scanner's kinds are checked when it reports them; use
    /// [`PiiDetector::try_with_scanner`] to check them now.
    pub fn with_scanner(mut self, scanner: Scanner) -> PiiDetector {
        self.scanners.push(scanner);
        self
    }

    /// This detector, keeping mentions strictly nested inside a longer
    /// mention as OpenNLP does.
    pub fn keeping_nested(mut self) -> PiiDetector {
        self.keep_nested = true;
        self
    }

    /// Whether mentions strictly nested inside a longer mention are kept.
    pub const fn keeps_nested(&self) -> bool {
        self.keep_nested
    }

    /// This detector rejecting every scored candidate whose score is below
    /// `min` (`RejectReason::Threshold`). A candidate without a score is
    /// not subject to the threshold: absent is not zero.
    ///
    /// # Panics
    ///
    /// When `min` is not finite.
    pub fn with_score_threshold(mut self, min: f32) -> PiiDetector {
        assert!(min.is_finite(), "a finite score threshold");
        self.min_score = Some(min);
        self
    }

    /// The score threshold, when one is set.
    pub const fn score_threshold(&self) -> Option<f32> {
        self.min_score
    }

    /// Merges `allow` into this detector's global allow list, suppressing
    /// matching mentions across all [`PiiKind`]s.
    pub fn with_allow_list(mut self, allow: AllowList) -> PiiDetector {
        self.global_allow.merge(allow);
        self
    }

    /// Adds an exact term to the global allow list.
    pub fn allow_term(mut self, term: &str) -> Result<PiiDetector, ListError> {
        self.global_allow.add_term(term)?;
        Ok(self)
    }

    /// Adds multiple exact terms to the global allow list.
    pub fn allow_terms<I, S>(mut self, terms: I) -> Result<PiiDetector, ListError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.global_allow.add_terms(terms)?;
        Ok(self)
    }

    /// Loads exact terms from `path` into the global allow list.
    pub fn allow_file(mut self, path: impl AsRef<Path>) -> Result<PiiDetector, ListError> {
        self.global_allow.add_file(path)?;
        Ok(self)
    }

    /// Adds a compiled [`Pattern`] to the global allow list.
    pub fn allow_pattern(mut self, pattern: Pattern) -> PiiDetector {
        self.global_allow.add_pattern(pattern);
        self
    }

    /// Adds a shape pattern to the global allow list.
    pub fn allow_shape(mut self, shape: &str) -> Result<PiiDetector, ListError> {
        self.global_allow.add_shape(shape)?;
        Ok(self)
    }

    /// Adds a glob pattern to the global allow list.
    pub fn allow_glob(mut self, glob: &str) -> Result<PiiDetector, ListError> {
        self.global_allow.add_glob(glob)?;
        Ok(self)
    }

    /// Merges `allow` into the allow list scoped to `kind` only.
    pub fn with_kind_allow_list(mut self, kind: PiiKind, allow: AllowList) -> Result<PiiDetector, ListError> {
        self.check_kind(kind)?;
        if allow.is_empty() {
            return Ok(self);
        }
        self.kind_entry_mut(kind).merge(allow);
        Ok(self)
    }

    /// Adds an exact term to the allow list scoped to `kind` only.
    pub fn allow_kind_term(mut self, kind: PiiKind, term: &str) -> Result<PiiDetector, ListError> {
        self.check_kind(kind)?;
        let mut list = AllowList::new();
        list.add_term(term)?;
        self.kind_entry_mut(kind).merge(list);
        Ok(self)
    }

    /// Adds multiple exact terms to the allow list scoped to `kind` only.
    pub fn allow_kind_terms<I, S>(mut self, kind: PiiKind, terms: I) -> Result<PiiDetector, ListError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.check_kind(kind)?;
        let mut list = AllowList::new();
        list.add_terms(terms)?;
        if !list.is_empty() {
            self.kind_entry_mut(kind).merge(list);
        }
        Ok(self)
    }

    /// Loads exact terms from `path` into the allow list scoped to `kind` only.
    pub fn allow_kind_file(mut self, kind: PiiKind, path: impl AsRef<Path>) -> Result<PiiDetector, ListError> {
        self.check_kind(kind)?;
        let list = AllowList::from_file(path)?;
        if !list.is_empty() {
            self.kind_entry_mut(kind).merge(list);
        }
        Ok(self)
    }

    /// Adds a compiled [`Pattern`] to the allow list scoped to `kind` only.
    pub fn allow_kind_pattern(mut self, kind: PiiKind, pattern: Pattern) -> Result<PiiDetector, ListError> {
        self.check_kind(kind)?;
        self.kind_entry_mut(kind).add_pattern(pattern);
        Ok(self)
    }

    /// Adds a shape pattern to the allow list scoped to `kind` only.
    pub fn allow_kind_shape(mut self, kind: PiiKind, shape: &str) -> Result<PiiDetector, ListError> {
        self.check_kind(kind)?;
        let pat = Pattern::shape(shape)?;
        self.kind_entry_mut(kind).add_pattern(pat);
        Ok(self)
    }

    /// Adds a glob pattern to the allow list scoped to `kind` only.
    pub fn allow_kind_glob(mut self, kind: PiiKind, glob: &str) -> Result<PiiDetector, ListError> {
        self.check_kind(kind)?;
        let pat = Pattern::glob(glob)?;
        self.kind_entry_mut(kind).add_pattern(pat);
        Ok(self)
    }

    /// Appends a [`DenyList`] whose matched terms are reported under its
    /// [`PiiKind::Custom`], refused when that kind is outside
    /// [`PiiDetector::custom_kinds`].
    pub fn with_deny_list(mut self, deny: DenyList) -> Result<PiiDetector, ListError> {
        self.check_kind(deny.kind())?;
        self.deny_lists.push(deny);
        Ok(self)
    }

    fn kind_entry_mut(&mut self, kind: PiiKind) -> &mut AllowList {
        let key = kind.code();
        match self.kind_allow.binary_search_by_key(&key, |(k, _)| k.code()) {
            Ok(at) => &mut self.kind_allow[at].1,
            Err(at) => {
                self.kind_allow.insert(at, (kind, AllowList::new()));
                &mut self.kind_allow[at].1
            }
        }
    }

    /// The global allow list applied across all kinds.
    pub fn global_allow_list(&self) -> &AllowList {
        &self.global_allow
    }

    /// The allow list scoped to `kind`, when configured.
    pub fn kind_allow_list(&self, kind: PiiKind) -> Option<&AllowList> {
        let key = kind.code();
        self.kind_allow.binary_search_by_key(&key, |(k, _)| k.code()).ok().map(|at| &self.kind_allow[at].1)
    }

    /// All per-kind allow lists, ordered by kind code (built-in ids first,
    /// then custom kinds in table order).
    pub fn kind_allow_lists(&self) -> &[(PiiKind, AllowList)] {
        &self.kind_allow
    }

    /// All configured deny lists in composition order.
    pub fn deny_lists(&self) -> &[DenyList] {
        &self.deny_lists
    }

    pub fn scanners(&self) -> &[Scanner] {
        &self.scanners
    }

    /// The built-in kinds any scanner reports.
    pub fn kinds(&self) -> PiiKinds {
        self.scanners.iter().fold(PiiKinds::EMPTY, |k, s| k.union(s.kinds()))
    }

    /// The allow rule that suppresses a mention: the global list first,
    /// then the list scoped to `kind`.
    fn allow_rule(&self, kind: PiiKind, covered: &str, normalized: &str) -> AllowDecision {
        if let Some(rule) = self.global_allow.find_rule(covered, normalized) {
            return Some((AllowScope::Global, rule));
        }
        let rule = self.kind_allow_list(kind)?.find_rule(covered, normalized)?;
        Some((AllowScope::Kind, rule))
    }

    /// Appends every admitted mention in `text` to `out` and every
    /// candidate with its decision to `candidates`. Both layers must be
    /// empty and over the same view; they take this detector's custom-kind
    /// table and the run's detector table.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`,
    /// or when either layer already holds items.
    pub fn detect_with_candidates(&self, text: &str, out: &mut Pii, candidates: &mut PiiCandidates) {
        assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
        assert!(out.kind.is_empty() && candidates.kind.is_empty(), "empty layers for a detection run");
        out.custom_kinds.clone_from(&self.custom_kinds);
        candidates.custom_kinds.clone_from(&self.custom_kinds);
        let mut hits = Hits::default();
        for s in &self.scanners {
            hits.begin_detector(s.extractor().id(), s.version());
            s.scan(text, &self.custom_kinds, &mut hits);
        }
        if !self.deny_lists.is_empty() {
            hits.begin_detector(DENY_LIST_DETECTOR, BUILTIN_VERSION);
            for (i, d) in self.deny_lists.iter().enumerate() {
                d.scan(text, i as u32, &mut hits);
            }
        }
        hits.apply_label_policy(text, &self.window);
        if let Some(min) = self.min_score {
            hits.apply_threshold(min);
        }
        hits.resolve(
            self.keep_nested,
            |kind, start, end, normalized| self.allow_rule(kind, &text[start as usize..end as usize], normalized),
            out,
            candidates,
        );
    }

    /// Appends every mention in `text` to `out`, with byte offsets into
    /// `text`. `out` must be empty.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`,
    /// or when `out` already holds mentions.
    pub fn detect(&self, text: &str, out: &mut Pii) {
        let mut candidates = PiiCandidates::new(out.spans.view());
        self.detect_with_candidates(text, out, &mut candidates);
    }

    /// The mentions in `text` as a layer over `view`, together with every
    /// candidate and its decision.
    pub fn layer_with_candidates(&self, text: &str, view: ViewId) -> (Pii, PiiCandidates) {
        let mut out = Pii::with_custom_kinds(view, self.custom_kinds.clone());
        let mut candidates = PiiCandidates::with_tables(view, self.custom_kinds.clone(), Default::default());
        self.detect_with_candidates(text, &mut out, &mut candidates);
        (out, candidates)
    }

    /// The mentions in `text` as a layer over `view`.
    pub fn layer(&self, text: &str, view: ViewId) -> Pii {
        self.layer_with_candidates(text, view).0
    }
}
