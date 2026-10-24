//! The scanners, one per OpenNLP extractor class, each limited to a set
//! of the kinds it can find.

mod aba;
mod card;
mod crypto;
mod email;
mod iban;
mod jws;
mod jwt;
mod labelled;
mod national;
mod network;
mod phone;
mod secrets;

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use turbo_analysis_layers::{PiiKind, PiiKinds};

use crate::hits::{Hits, PiiSink};

/// One OpenNLP extractor class, or a caller-defined extractor named by
/// the caller.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Extractor {
    /// `CursorPiiExtractor`: e-mail addresses, IBANs, cards, phones.
    Cursor,
    /// `BankingPiiExtractor`: ABA routing numbers.
    Banking,
    /// `NetworkPiiExtractor`: IPv4, IPv6, MAC.
    Network,
    /// `SecretsPiiExtractor`: AWS keys, GitHub tokens, JWTs, URL
    /// credentials.
    Secrets,
    /// `CryptoPiiExtractor`: Bitcoin and Ethereum addresses.
    Crypto,
    /// `UsIdentityPiiExtractor`: SSNs and ITINs.
    UsIdentity,
    /// `EuIdentityPiiExtractor`: NHS numbers and Steuer-IDs.
    EuIdentity,
    /// `CaIdentityPiiExtractor`: labelled SINs.
    CaIdentity,
    /// `DevicePiiExtractor`: labelled IMEIs.
    Device,
    /// A caller-defined extractor (`PiiExtractor`).
    Custom(&'static str),
}

impl Extractor {
    /// The built-in kinds this extractor finds (`EMPTY` for
    /// [`Extractor::Custom`]).
    pub const fn kinds(self) -> PiiKinds {
        use PiiKind::*;
        match self {
            Extractor::Cursor => PiiKinds::of(&[Email, Phone, Iban, Card]),
            Extractor::Banking => PiiKinds::of(&[AbaRouting]),
            Extractor::Network => PiiKinds::of(&[Ipv4, Ipv6, Mac]),
            Extractor::Secrets => PiiKinds::of(&[AwsAccessKey, GithubToken, Jwt, UrlCredential]),
            Extractor::Crypto => PiiKinds::of(&[BtcAddress, EthAddress]),
            Extractor::UsIdentity => PiiKinds::of(&[UsSsn, UsItin]),
            Extractor::EuIdentity => PiiKinds::of(&[UkNhs, DeSteuerId]),
            Extractor::CaIdentity => PiiKinds::of(&[CaSin]),
            Extractor::Device => PiiKinds::of(&[Imei]),
            Extractor::Custom(_) => PiiKinds::EMPTY,
        }
    }

    /// The Java class's simple name, or the caller extractor's name.
    pub const fn java_name(self) -> &'static str {
        match self {
            Extractor::Cursor => "CursorPiiExtractor",
            Extractor::Banking => "BankingPiiExtractor",
            Extractor::Network => "NetworkPiiExtractor",
            Extractor::Secrets => "SecretsPiiExtractor",
            Extractor::Crypto => "CryptoPiiExtractor",
            Extractor::UsIdentity => "UsIdentityPiiExtractor",
            Extractor::EuIdentity => "EuIdentityPiiExtractor",
            Extractor::CaIdentity => "CaIdentityPiiExtractor",
            Extractor::Device => "DevicePiiExtractor",
            Extractor::Custom(name) => name,
        }
    }

    pub(crate) const fn tag(self) -> u64 {
        match self {
            Extractor::Cursor => 0,
            Extractor::Banking => 1,
            Extractor::Network => 2,
            Extractor::Secrets => 3,
            Extractor::Crypto => 4,
            Extractor::UsIdentity => 5,
            Extractor::EuIdentity => 6,
            Extractor::CaIdentity => 7,
            Extractor::Device => 8,
            Extractor::Custom(_) => 9,
        }
    }
}

/// A caller-defined PII extractor, OpenNLP's `PiiExtractor`.
pub trait CustomExtractor: Send + Sync {
    /// Identifier for this extractor in [`Extractor::Custom`] and
    /// [`crate::PiiDetector`]'s configuration hash.
    fn name(&self) -> &'static str;

    /// The kinds this extractor reports.
    fn kinds(&self) -> &[PiiKind];

    /// Scans `text` and appends candidate mentions to `sink`.
    fn scan(&self, text: &str, sink: &mut PiiSink<'_>);

    /// Fingerprint of this extractor's configuration, folded into
    /// [`crate::PiiDetector`]'s configuration hash and [`Scanner`] equality.
    fn config_hash(&self) -> u64 {
        0
    }
}

struct FnExtractor {
    name: &'static str,
    kinds: [PiiKind; 1],
    scan: fn(&str, &mut PiiSink<'_>),
}

impl CustomExtractor for FnExtractor {
    fn name(&self) -> &'static str {
        self.name
    }

    fn kinds(&self) -> &[PiiKind] {
        &self.kinds
    }

    fn scan(&self, text: &str, sink: &mut PiiSink<'_>) {
        (self.scan)(text, sink);
    }
}

/// A scanner configuration the extractor cannot honour.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScannerError {
    /// No kind selected.
    NoKinds(Extractor),
    /// A kind the extractor does not find.
    Foreign(Extractor, PiiKind),
}

impl fmt::Display for ScannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScannerError::NoKinds(e) => write!(f, "{} needs at least one kind", e.java_name()),
            ScannerError::Foreign(e, k) => write!(f, "{} does not find {k}", e.java_name()),
        }
    }
}

impl std::error::Error for ScannerError {}

#[derive(Clone)]
enum ScannerInner {
    Builtin { extractor: Extractor, kinds: PiiKinds },
    Custom(Arc<dyn CustomExtractor>),
}

/// An extractor with the kinds it reports, as the Java constructors that
/// take a type set build it, or a caller-defined [`CustomExtractor`].
#[derive(Clone)]
pub struct Scanner {
    inner: ScannerInner,
}

impl fmt::Debug for Scanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            ScannerInner::Builtin { extractor, kinds } => {
                f.debug_struct("Scanner").field("extractor", extractor).field("kinds", kinds).finish()
            }
            ScannerInner::Custom(c) => f
                .debug_struct("Scanner")
                .field("extractor", &Extractor::Custom(c.name()))
                .field("kinds", &c.kinds())
                .finish(),
        }
    }
}

impl PartialEq for Scanner {
    fn eq(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (ScannerInner::Builtin { extractor: a, kinds: ak }, ScannerInner::Builtin { extractor: b, kinds: bk }) => {
                a == b && ak == bk
            }
            (ScannerInner::Custom(a), ScannerInner::Custom(b)) => {
                Arc::ptr_eq(a, b)
                    || (a.name() == b.name() && a.kinds() == b.kinds() && a.config_hash() == b.config_hash())
            }
            _ => false,
        }
    }
}

impl Eq for Scanner {}

impl Hash for Scanner {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.extractor().hash(state);
        match &self.inner {
            ScannerInner::Builtin { kinds, .. } => kinds.hash(state),
            ScannerInner::Custom(c) => {
                c.kinds().hash(state);
                c.config_hash().hash(state);
            }
        }
    }
}

impl Scanner {
    /// The built-in extractor with every kind it finds.
    pub const fn all(extractor: Extractor) -> Scanner {
        Scanner { inner: ScannerInner::Builtin { extractor, kinds: extractor.kinds() } }
    }

    /// The built-in extractor limited to `kinds`, which must be a
    /// non-empty subset of [`Extractor::kinds`].
    pub fn with_kinds(extractor: Extractor, kinds: PiiKinds) -> Result<Scanner, ScannerError> {
        if kinds.is_empty() {
            return Err(ScannerError::NoKinds(extractor));
        }
        if let Some(k) = kinds.iter().find(|&k| !extractor.kinds().contains(k)) {
            return Err(ScannerError::Foreign(extractor, k));
        }
        Ok(Scanner { inner: ScannerInner::Builtin { extractor, kinds } })
    }

    /// A scanner running a caller-defined [`CustomExtractor`].
    pub fn custom<E: CustomExtractor + 'static>(extractor: E) -> Scanner {
        for &k in extractor.kinds() {
            PiiKind::remember_custom(k);
        }
        Scanner { inner: ScannerInner::Custom(Arc::new(extractor)) }
    }

    /// A scanner running a stateless function for a single [`PiiKind`].
    pub fn custom_fn(name: &'static str, kind: PiiKind, scan: fn(&str, &mut PiiSink<'_>)) -> Scanner {
        Scanner::custom(FnExtractor { name, kinds: [kind], scan })
    }

    pub fn extractor(&self) -> Extractor {
        match &self.inner {
            ScannerInner::Builtin { extractor, .. } => *extractor,
            ScannerInner::Custom(c) => Extractor::Custom(c.name()),
        }
    }

    /// The built-in kinds this scanner reports.
    pub fn kinds(&self) -> PiiKinds {
        match &self.inner {
            ScannerInner::Builtin { kinds, .. } => *kinds,
            ScannerInner::Custom(c) => PiiKinds::of(c.kinds()),
        }
    }

    /// All kinds declared by a custom scanner, or an empty slice for a
    /// built-in scanner.
    pub fn custom_kinds(&self) -> &[PiiKind] {
        match &self.inner {
            ScannerInner::Builtin { .. } => &[],
            ScannerInner::Custom(c) => c.kinds(),
        }
    }

    /// Configuration fingerprint for a custom scanner (`0` for a built-in
    /// scanner).
    pub fn custom_config_hash(&self) -> u64 {
        match &self.inner {
            ScannerInner::Builtin { .. } => 0,
            ScannerInner::Custom(c) => c.config_hash(),
        }
    }

    /// Appends this scanner's candidates in `text`, in the order the Java
    /// extractor's scans run.
    pub(crate) fn scan(&self, text: &str, hits: &mut Hits) {
        let ScannerInner::Builtin { extractor, kinds: k } = &self.inner else {
            if let ScannerInner::Custom(c) = &self.inner {
                let mut sink = PiiSink::new(text, hits);
                c.scan(text, &mut sink);
            }
            return;
        };
        let k = *k;
        match extractor {
            Extractor::Cursor => {
                if k.contains(PiiKind::Email) {
                    email::scan(text, hits);
                }
                if k.contains(PiiKind::Iban) {
                    iban::scan(text, hits);
                }
                if k.contains(PiiKind::Card) {
                    card::scan(text, hits);
                }
                if k.contains(PiiKind::Phone) {
                    phone::scan(text, hits);
                }
            }
            Extractor::Banking => aba::scan(text, hits),
            Extractor::Network => network::scan(text, k, hits),
            Extractor::Secrets => {
                if k.contains(PiiKind::UrlCredential) {
                    secrets::scan_url_credentials(text, hits);
                }
                if k.contains(PiiKind::Jwt) {
                    jwt::scan(text, hits);
                }
                if k.contains(PiiKind::AwsAccessKey) {
                    secrets::scan_aws(text, hits);
                }
                if k.contains(PiiKind::GithubToken) {
                    secrets::scan_github(text, hits);
                }
            }
            Extractor::Crypto => crypto::scan(text, k, hits),
            Extractor::UsIdentity => national::scan_us(text, k, hits),
            Extractor::EuIdentity => national::scan_eu(text, k, hits),
            Extractor::CaIdentity => labelled::scan_sin(text, hits),
            Extractor::Device => labelled::scan_imei(text, hits),
            Extractor::Custom(_) => {}
        }
    }
}
