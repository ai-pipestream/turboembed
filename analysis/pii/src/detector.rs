//! The packs and the detector that runs them, OpenNLP's `PiiPacks`,
//! `CompositePiiExtractor` and `PiiAnnotator`.

use turbo_analysis::{Annotator, Error, Refs, View, ViewId};
use turbo_analysis_layers::{Pii, PiiKind, PiiKinds};

use crate::hits::{Hits, PiiSink};
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
}

impl PiiDetector {
    /// The scanners in the order given. An empty list finds nothing.
    pub fn new(scanners: Vec<Scanner>) -> PiiDetector {
        PiiDetector { scanners, keep_nested: false }
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

    pub fn scanners(&self) -> &[Scanner] {
        &self.scanners
    }

    /// The kinds any scanner reports.
    pub fn kinds(&self) -> PiiKinds {
        self.scanners.iter().fold(PiiKinds::EMPTY, |k, s| k.union(s.kinds()))
    }

    /// Appends every mention in `text` to `out`, with byte offsets into
    /// `text`.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`.
    pub fn detect(&self, text: &str, out: &mut Pii) {
        assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
        let mut hits = Hits::default();
        for s in &self.scanners {
            s.scan(text, &mut hits);
        }
        hits.resolve(self.keep_nested, out);
    }

    /// The mentions in `text` as a layer over `view`.
    pub fn layer(&self, text: &str, view: ViewId) -> Pii {
        let mut out = Pii::new(view);
        self.detect(text, &mut out);
        out
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

    /// FNV-1a over each scanner's extractor and kind bits, fixed so that
    /// stored provenance stays comparable across builds.
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
        h.finish()
    }
}
