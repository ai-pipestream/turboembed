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

use turbo_analysis_layers::{PiiKind, PiiKinds};

use crate::hits::Hits;

/// One OpenNLP extractor class.
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
}

impl Extractor {
    /// The kinds this extractor finds.
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
        }
    }

    /// The Java class's simple name.
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
        }
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

/// An extractor with the kinds it reports, as the Java constructors that
/// take a type set build it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Scanner {
    extractor: Extractor,
    kinds: PiiKinds,
}

impl Scanner {
    /// The extractor with every kind it finds.
    pub const fn all(extractor: Extractor) -> Scanner {
        Scanner { extractor, kinds: extractor.kinds() }
    }

    /// The extractor limited to `kinds`, which must be a non-empty subset
    /// of [`Extractor::kinds`].
    pub fn with_kinds(extractor: Extractor, kinds: PiiKinds) -> Result<Scanner, ScannerError> {
        if kinds.is_empty() {
            return Err(ScannerError::NoKinds(extractor));
        }
        if let Some(k) = kinds.iter().find(|&k| !extractor.kinds().contains(k)) {
            return Err(ScannerError::Foreign(extractor, k));
        }
        Ok(Scanner { extractor, kinds })
    }

    pub const fn extractor(&self) -> Extractor {
        self.extractor
    }

    pub const fn kinds(&self) -> PiiKinds {
        self.kinds
    }

    /// Appends this scanner's candidates in `text`, in the order the Java
    /// extractor's scans run.
    pub(crate) fn scan(&self, text: &str, hits: &mut Hits) {
        let k = self.kinds;
        match self.extractor {
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
        }
    }
}
