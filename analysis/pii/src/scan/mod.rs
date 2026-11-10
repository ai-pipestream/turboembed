//! The scanners, one per OpenNLP extractor class, each limited to a set
//! of the kinds it can find.

mod aba;
mod address;
mod age;
mod br;
mod card;
mod crypto;
mod date;
mod email;
mod es_de;
mod extension;
mod fr;
mod iban;
mod jws;
mod jwt;
mod labelled;
mod logistics;
mod money;
mod national;
mod network;
mod phone;
mod secrets;
mod url;
mod us;
mod us_labelled;

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use turbo_analysis_layers::{CustomKinds, PiiKind, PiiKinds};

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
    /// `UsIdentityPiiExtractor`: SSNs, ITINs and EINs, and labelled
    /// passport, driver's licence and ZIP codes.
    UsIdentity,
    /// `EuIdentityPiiExtractor`: European national and business identifiers.
    EuIdentity,
    /// `BrIdentityPiiExtractor`: Brazilian CPF and CNPJ identifiers.
    BrIdentity,
    /// `CaIdentityPiiExtractor`: labelled SINs.
    CaIdentity,
    /// `DevicePiiExtractor`: labelled IMEIs.
    Device,
    /// `LogisticsPiiExtractor`: vehicle identification numbers and parcel
    /// tracking numbers.
    Logistics,
    /// `WebPiiExtractor`: web addresses.
    Web,
    /// `PhoneExtensionPiiExtractor`: extensions after a phone number.
    PhoneExtension,
    /// `UsAddressPiiExtractor`: United States street addresses and post
    /// office boxes.
    UsAddress,
    /// `AgePiiExtractor`: ages in years.
    Age,
    /// `TemporalPiiExtractor`: calendar dates naming a day, week or month.
    Temporal,
    /// `MoneyPiiExtractor`: amounts of money with their currency.
    Money,
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
            Extractor::UsIdentity => PiiKinds::of(&[UsSsn, UsItin, UsEin, UsPassport, UsDriversLicence, UsZip]),
            Extractor::EuIdentity => {
                PiiKinds::of(&[UkNhs, DeSteuerId, EsDni, EsNie, EsCif, DePersonalausweis, FrNir, FrSiren, FrSiret, Bic])
            }
            Extractor::BrIdentity => PiiKinds::of(&[BrCpf, BrCnpj]),
            Extractor::CaIdentity => PiiKinds::of(&[CaSin]),
            Extractor::Device => PiiKinds::of(&[Imei]),
            Extractor::Logistics => PiiKinds::of(&[Vin, TrackingNumber]),
            Extractor::Web => PiiKinds::of(&[Url]),
            Extractor::PhoneExtension => PiiKinds::of(&[PhoneExtension]),
            Extractor::UsAddress => PiiKinds::of(&[StreetAddress]),
            Extractor::Age => PiiKinds::of(&[Age]),
            Extractor::Temporal => PiiKinds::of(&[Date]),
            Extractor::Money => PiiKinds::of(&[Currency]),
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
            Extractor::BrIdentity => "BrIdentityPiiExtractor",
            Extractor::CaIdentity => "CaIdentityPiiExtractor",
            Extractor::Device => "DevicePiiExtractor",
            Extractor::Logistics => "LogisticsPiiExtractor",
            Extractor::Web => "WebPiiExtractor",
            Extractor::PhoneExtension => "PhoneExtensionPiiExtractor",
            Extractor::UsAddress => "UsAddressPiiExtractor",
            Extractor::Age => "AgePiiExtractor",
            Extractor::Temporal => "TemporalPiiExtractor",
            Extractor::Money => "MoneyPiiExtractor",
            Extractor::Custom(name) => name,
        }
    }

    /// The stable detector id every mention of this extractor names:
    /// `turbo-pii:` and the pack-style name (`turbo-pii:cursor`,
    /// `turbo-pii:us-identity`), or the caller extractor's own name.
    pub const fn id(self) -> &'static str {
        match self {
            Extractor::Cursor => "turbo-pii:cursor",
            Extractor::Banking => "turbo-pii:banking",
            Extractor::Network => "turbo-pii:network",
            Extractor::Secrets => "turbo-pii:secrets",
            Extractor::Crypto => "turbo-pii:crypto",
            Extractor::UsIdentity => "turbo-pii:us-identity",
            Extractor::EuIdentity => "turbo-pii:eu-identity",
            Extractor::BrIdentity => "turbo-pii:br-identity",
            Extractor::CaIdentity => "turbo-pii:ca-identity",
            Extractor::Device => "turbo-pii:device",
            Extractor::Logistics => "turbo-pii:logistics",
            Extractor::Web => "turbo-pii:web",
            Extractor::PhoneExtension => "turbo-pii:phone-extension",
            Extractor::UsAddress => "turbo-pii:us-address",
            Extractor::Age => "turbo-pii:age",
            Extractor::Temporal => "turbo-pii:temporal",
            Extractor::Money => "turbo-pii:money",
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
            Extractor::BrIdentity => 10,
            Extractor::Logistics => 11,
            Extractor::Web => 12,
            Extractor::PhoneExtension => 13,
            Extractor::UsAddress => 14,
            Extractor::Age => 15,
            Extractor::Temporal => 16,
            Extractor::Money => 17,
        }
    }
}

/// The version the built-in scanners and deny lists record on their
/// mentions: this crate's.
pub const BUILTIN_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A caller-defined PII extractor, OpenNLP's `PiiExtractor`.
pub trait CustomExtractor: Send + Sync {
    /// Identifier for this extractor in [`Extractor::Custom`] and
    /// [`crate::PiiDetector`]'s configuration hash.
    fn name(&self) -> &'static str;

    /// The kinds this extractor reports.
    fn kinds(&self) -> &[PiiKind];

    /// The version recorded with every mention this extractor finds, next
    /// to [`CustomExtractor::name`]. Unversioned extractors report `"0"`.
    fn version(&self) -> &str {
        "0"
    }

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
        Scanner { inner: ScannerInner::Custom(Arc::new(extractor)) }
    }

    /// A scanner running a stateless function for a single [`PiiKind`].
    pub fn custom_fn(name: &'static str, kind: PiiKind, scan: fn(&str, &mut PiiSink<'_>)) -> Scanner {
        Scanner::custom(FnExtractor { name, kinds: [kind], scan })
    }

    /// The detector version every mention of this scanner names: this
    /// crate's version for a built-in extractor, the caller's for a custom
    /// one.
    pub fn version(&self) -> &str {
        match &self.inner {
            ScannerInner::Builtin { .. } => BUILTIN_VERSION,
            ScannerInner::Custom(c) => c.version(),
        }
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
    pub(crate) fn scan(&self, text: &str, kinds: &CustomKinds, hits: &mut Hits) {
        let ScannerInner::Builtin { extractor, kinds: k } = &self.inner else {
            if let ScannerInner::Custom(c) = &self.inner {
                let mut sink = PiiSink::new(text, kinds, hits);
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
            Extractor::UsIdentity => {
                national::scan_us(text, k, hits);
                if k.contains(PiiKind::UsEin) {
                    us::scan_ein(text, hits);
                }
                us_labelled::scan(text, k, hits);
            }
            Extractor::EuIdentity => national::scan_eu(text, k, hits),
            Extractor::BrIdentity => br::scan_br(text, k, hits),
            Extractor::CaIdentity => labelled::scan_sin(text, hits),
            Extractor::Device => labelled::scan_imei(text, hits),
            Extractor::Logistics => logistics::scan(text, k, hits),
            Extractor::Web => url::scan(text, hits),
            Extractor::PhoneExtension => extension::scan(text, hits),
            Extractor::UsAddress => address::scan(text, hits),
            Extractor::Age => age::scan(text, hits),
            Extractor::Temporal => date::scan(text, hits),
            Extractor::Money => money::scan(text, hits),
            Extractor::Custom(_) => {}
        }
    }
}
