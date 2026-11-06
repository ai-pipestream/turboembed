//! The packs and the detector that runs them, OpenNLP's `PiiPacks`,
//! `CompositePiiExtractor` and `PiiAnnotator`.

use std::path::Path;

use turbo_analysis::{Annotator, Error, Refs, View, ViewId};
use turbo_analysis_layers::{Pii, PiiKind, PiiKinds};

use crate::hits::{Hits, PiiSink};
use crate::lists::{AllowList, DenyList, ListError, Pattern};
use crate::rewrite::{HmacTokenizer, PiiAuditReport};
use crate::scan::{CustomExtractor, Extractor, Scanner};

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
    /// United Kingdom NHS numbers and German Steuer-IDs.
    EuIdentity,
    /// Canadian SINs after a label.
    CaIdentity,
    /// IMEIs after a label.
    Device,
    /// Every detector: the default `CursorPiiExtractor` and every pack.
    AllStructured,
}

impl Pack {
    pub const ALL: [Pack; 10] = [
        Pack::Payment,
        Pack::Contact,
        Pack::Network,
        Pack::Secrets,
        Pack::Crypto,
        Pack::UsIdentity,
        Pack::EuIdentity,
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
            Pack::CaIdentity => vec![Scanner::all(Extractor::CaIdentity)],
            Pack::Device => vec![Scanner::all(Extractor::Device)],
            Pack::AllStructured => [
                Extractor::Cursor,
                Extractor::Secrets,
                Extractor::Crypto,
                Extractor::Network,
                Extractor::UsIdentity,
                Extractor::EuIdentity,
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiiDetector {
    scanners: Vec<Scanner>,
    keep_nested: bool,
    global_allow: AllowList,
    kind_allow: Vec<(PiiKind, AllowList)>,
    deny_lists: Vec<DenyList>,
}

impl PiiDetector {
    /// The scanners in the order given. An empty list finds nothing.
    pub fn new(scanners: Vec<Scanner>) -> PiiDetector {
        PiiDetector {
            scanners,
            keep_nested: false,
            global_allow: AllowList::new(),
            kind_allow: Vec::new(),
            deny_lists: Vec::new(),
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

    /// This detector with `scanner` appended in composition order.
    pub fn with_scanner(mut self, scanner: Scanner) -> PiiDetector {
        self.scanners.push(scanner);
        self
    }

    /// This detector with a caller-defined [`CustomExtractor`] appended in
    /// composition order.
    pub fn with_custom<E: CustomExtractor + 'static>(mut self, extractor: E) -> PiiDetector {
        self.scanners.push(Scanner::custom(extractor));
        self
    }

    /// This detector with a stateless caller function for `kind` appended
    /// in composition order.
    pub fn with_custom_fn(
        mut self,
        name: &'static str,
        kind: PiiKind,
        scan: fn(&str, &mut PiiSink<'_>),
    ) -> PiiDetector {
        self.scanners.push(Scanner::custom_fn(name, kind, scan));
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
    pub fn with_kind_allow_list(mut self, kind: PiiKind, allow: AllowList) -> PiiDetector {
        let _ = PiiKind::remember_custom(kind);
        if allow.is_empty() {
            return self;
        }
        self.kind_entry_mut(kind).merge(allow);
        self
    }

    /// Adds an exact term to the allow list scoped to `kind` only.
    pub fn allow_kind_term(mut self, kind: PiiKind, term: &str) -> Result<PiiDetector, ListError> {
        let _ = PiiKind::remember_custom(kind);
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
        let _ = PiiKind::remember_custom(kind);
        let mut list = AllowList::new();
        list.add_terms(terms)?;
        if !list.is_empty() {
            self.kind_entry_mut(kind).merge(list);
        }
        Ok(self)
    }

    /// Loads exact terms from `path` into the allow list scoped to `kind` only.
    pub fn allow_kind_file(mut self, kind: PiiKind, path: impl AsRef<Path>) -> Result<PiiDetector, ListError> {
        let _ = PiiKind::remember_custom(kind);
        let list = AllowList::from_file(path)?;
        if !list.is_empty() {
            self.kind_entry_mut(kind).merge(list);
        }
        Ok(self)
    }

    /// Adds a compiled [`Pattern`] to the allow list scoped to `kind` only.
    pub fn allow_kind_pattern(mut self, kind: PiiKind, pattern: Pattern) -> PiiDetector {
        let _ = PiiKind::remember_custom(kind);
        self.kind_entry_mut(kind).add_pattern(pattern);
        self
    }

    /// Adds a shape pattern to the allow list scoped to `kind` only.
    pub fn allow_kind_shape(mut self, kind: PiiKind, shape: &str) -> Result<PiiDetector, ListError> {
        let _ = PiiKind::remember_custom(kind);
        let pat = Pattern::shape(shape)?;
        self.kind_entry_mut(kind).add_pattern(pat);
        Ok(self)
    }

    /// Adds a glob pattern to the allow list scoped to `kind` only.
    pub fn allow_kind_glob(mut self, kind: PiiKind, glob: &str) -> Result<PiiDetector, ListError> {
        let _ = PiiKind::remember_custom(kind);
        let pat = Pattern::glob(glob)?;
        self.kind_entry_mut(kind).add_pattern(pat);
        Ok(self)
    }

    /// Appends a [`DenyList`] whose matched terms are reported under its
    /// caller-named [`PiiKind::Custom`].
    pub fn with_deny_list(mut self, deny: DenyList) -> PiiDetector {
        self.deny_lists.push(deny);
        self
    }

    fn kind_entry_mut(&mut self, kind: PiiKind) -> &mut AllowList {
        let key = (kind.rank(), kind.name());
        match self.kind_allow.binary_search_by_key(&key, |(k, _)| (k.rank(), k.name())) {
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
        let key = (kind.rank(), kind.name());
        self.kind_allow.binary_search_by_key(&key, |(k, _)| (k.rank(), k.name())).ok().map(|at| &self.kind_allow[at].1)
    }

    /// All per-kind allow lists, ordered by kind rank and name.
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

    fn is_suppressed(&self, kind: PiiKind, covered: &str, normalized: &str) -> bool {
        if self.global_allow.matches(covered, normalized) {
            return true;
        }
        if let Some(allow) = self.kind_allow_list(kind) {
            return allow.matches(covered, normalized);
        }
        false
    }

    /// Appends every non-suppressed mention in `text` to `out`, returning the
    /// per-kind counts of mentions suppressed by allow rules.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`.
    pub fn detect_with_suppressed(&self, text: &str, out: &mut Pii) -> Vec<(PiiKind, usize)> {
        assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
        let mut hits = Hits::default();
        for s in &self.scanners {
            s.scan(text, &mut hits);
        }
        for d in &self.deny_lists {
            d.scan(text, &mut hits);
        }
        hits.resolve_with(
            self.keep_nested,
            |kind, start, end, normalized| self.is_suppressed(kind, &text[start as usize..end as usize], normalized),
            out,
        )
    }

    /// Appends every mention in `text` to `out`, with byte offsets into
    /// `text`.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`.
    pub fn detect(&self, text: &str, out: &mut Pii) {
        let _ = self.detect_with_suppressed(text, out);
    }

    /// The mentions in `text` as a layer over `view`, together with the
    /// per-kind counts of mentions suppressed by allow rules.
    pub fn layer_with_suppressed(&self, text: &str, view: ViewId) -> (Pii, Vec<(PiiKind, usize)>) {
        let mut out = Pii::new(view);
        let suppressed = self.detect_with_suppressed(text, &mut out);
        (out, suppressed)
    }

    /// The mentions in `text` as a layer over `view`.
    pub fn layer(&self, text: &str, view: ViewId) -> Pii {
        let mut out = Pii::new(view);
        self.detect(text, &mut out);
        out
    }

    /// Runs detection over `text` for `view` and builds a [`PiiAuditReport`]
    /// that includes both active mentions and per-kind suppressed counts.
    pub fn detect_and_audit(
        &self,
        text: &str,
        view: ViewId,
        tokenizer: &HmacTokenizer,
        samples: usize,
    ) -> (Pii, PiiAuditReport) {
        let (pii, suppressed) = self.layer_with_suppressed(text, view);
        let report = PiiAuditReport::with_suppressed(&pii, suppressed, tokenizer, samples);
        (pii, report)
    }

    /// Runs detection over `text` and returns the [`PiiAuditReport`],
    /// including per-kind counts of mentions suppressed by allow rules.
    pub fn audit(&self, text: &str, tokenizer: &HmacTokenizer, samples: usize) -> PiiAuditReport {
        self.detect_and_audit(text, ViewId::SOURCE, tokenizer, samples).1
    }
}

impl Annotator for PiiDetector {
    const ID: &'static str = "turbo-pii:pii";
    type Reads = (View,);
    type Writes = (Pii,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Pii,), Error> {
        let text = view.text();
        if u32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the u32 offset range", text.len()),
            });
        }
        Ok((self.layer(text, view.id()),))
    }

    fn requires(&self) -> &[turbo_analysis::KeyId] {
        const REQUIRES: &[turbo_analysis::KeyId] = &[turbo_analysis::Key::<View>::DEFAULT.id()];
        REQUIRES
    }

    fn produces(&self) -> &[turbo_analysis::KeyId] {
        const PRODUCES: &[turbo_analysis::KeyId] = &[turbo_analysis::Key::<Pii>::DEFAULT.id()];
        PRODUCES
    }

    /// FNV-1a over each scanner's extractor and kind bits, plus any
    /// configured allow lists and deny lists, fixed so that stored
    /// provenance stays comparable across builds.
    fn config_hash(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        for s in &self.scanners {
            let kinds = s.kinds().iter().fold(0u64, |b, k| b | 1 << k.rank());
            h.fold_u64(s.extractor().tag()).fold_u64(kinds);
            if let Extractor::Custom(name) = s.extractor() {
                h.bytes(name.as_bytes()).u8(0xff);
                for k in s.custom_kinds() {
                    h.bytes(k.name().as_bytes()).u8(0xff);
                }
                h.fold_u64(s.custom_config_hash());
            }
        }
        if self.keep_nested {
            h.fold_u64(1);
        }
        if !self.global_allow.is_empty() {
            h.u8(0xa1);
            self.global_allow.fold_hash(&mut h);
        }
        if !self.kind_allow.is_empty() {
            h.u8(0xa2).usize(self.kind_allow.len());
            for (kind, allow) in &self.kind_allow {
                h.str(kind.name());
                allow.fold_hash(&mut h);
            }
        }
        if !self.deny_lists.is_empty() {
            h.u8(0xd1).usize(self.deny_lists.len());
            for deny in &self.deny_lists {
                deny.fold_hash(&mut h);
            }
        }
        h.finish()
    }
}
