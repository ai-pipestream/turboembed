//! Personal and sensitive identifiers written in a text view: e-mail
//! addresses, phone numbers, payment and bank account numbers, network
//! addresses, credentials, wallet addresses, national and device
//! identifiers.
//!
//! The layer holds what OpenNLP's PII extractors find (`PiiPacks` in the
//! `opennlp.tools.pii` package): one item per mention with its kind, the
//! normalized form OpenNLP writes, and how strongly the text supports the
//! kind. Mentions of different kinds may overlap, as in the Java, so the
//! layer allows any overlap.

use std::fmt;
use std::sync::{OnceLock, RwLock};

use turbo_analysis::{Ix, Layer, Overlap, Span, Spans, ViewId};

use crate::Strs;

static CUSTOM_KINDS: OnceLock<RwLock<Vec<&'static str>>> = OnceLock::new();

fn custom_registry() -> &'static RwLock<Vec<&'static str>> {
    CUSTOM_KINDS.get_or_init(|| RwLock::new(Vec::new()))
}

/// What a mention is. The built-in variants are in OpenNLP's
/// `PiiTypePriority` order, followed by [`PiiKind::Custom`] for caller
/// kinds, so the derived order is the rank that sorts mentions with the
/// same span.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PiiKind {
    /// A JSON Web Token in compact serialization.
    Jwt,
    /// An AWS access key identifier (`AKIA`, `ASIA`).
    AwsAccessKey,
    /// A GitHub access token (`ghp_`, `github_pat_`, `ghs_` and others).
    GithubToken,
    /// The `user:password` userinfo of a URL.
    UrlCredential,
    Email,
    /// An International Bank Account Number, mod-97 checked.
    Iban,
    /// A mobile device's IMEI after an `IMEI` label, Luhn checked.
    Imei,
    /// A payment card number, Luhn checked.
    Card,
    /// A Bitcoin address: base58check or bech32/bech32m.
    BtcAddress,
    /// An Ethereum address, EIP-55 checked when written in mixed case.
    EthAddress,
    /// A 48-bit IEEE 802 MAC address.
    Mac,
    Ipv6,
    Ipv4,
    /// A United States Social Security number.
    UsSsn,
    /// A United States Individual Taxpayer Identification Number.
    UsItin,
    /// A United Kingdom NHS number, mod-11 checked.
    UkNhs,
    /// A German tax identification number (Steuer-ID), ISO 7064 checked.
    DeSteuerId,
    /// A Canadian Social Insurance Number after a label, Luhn checked.
    CaSin,
    /// An ABA routing transit number, check digit verified.
    AbaRouting,
    Phone,
    /// A caller-defined identifier kind (`account-number`, `se-personnummer`),
    /// 1 to [`PiiKind::MAX_CUSTOM_NAME_BYTES`] UTF-8 bytes and not one of the
    /// reserved built-in names in [`PiiKind::ALL`].
    Custom(&'static str),
}

impl PiiKind {
    /// Maximum UTF-8 byte length of a [`PiiKind::Custom`] name.
    pub const MAX_CUSTOM_NAME_BYTES: usize = 64;

    /// Maximum number of distinct [`PiiKind::Custom`] names interned in the
    /// process-wide registry.
    pub const MAX_CUSTOM_KINDS: usize = 256;

    /// Every built-in kind, in rank order.
    pub const ALL: [PiiKind; 20] = [
        PiiKind::Jwt,
        PiiKind::AwsAccessKey,
        PiiKind::GithubToken,
        PiiKind::UrlCredential,
        PiiKind::Email,
        PiiKind::Iban,
        PiiKind::Imei,
        PiiKind::Card,
        PiiKind::BtcAddress,
        PiiKind::EthAddress,
        PiiKind::Mac,
        PiiKind::Ipv6,
        PiiKind::Ipv4,
        PiiKind::UsSsn,
        PiiKind::UsItin,
        PiiKind::UkNhs,
        PiiKind::DeSteuerId,
        PiiKind::CaSin,
        PiiKind::AbaRouting,
        PiiKind::Phone,
    ];

    /// A caller-defined kind named `name` (1 to
    /// [`PiiKind::MAX_CUSTOM_NAME_BYTES`] UTF-8 bytes), or the built-in
    /// [`PiiKind`] when `name` matches one of [`PiiKind::ALL`].
    pub const fn custom(name: &'static str) -> PiiKind {
        match name.as_bytes() {
            b"jwt" => PiiKind::Jwt,
            b"aws-access-key" => PiiKind::AwsAccessKey,
            b"github-token" => PiiKind::GithubToken,
            b"url-credential" => PiiKind::UrlCredential,
            b"email" => PiiKind::Email,
            b"iban" => PiiKind::Iban,
            b"imei" => PiiKind::Imei,
            b"card" => PiiKind::Card,
            b"btc-address" => PiiKind::BtcAddress,
            b"eth-address" => PiiKind::EthAddress,
            b"mac" => PiiKind::Mac,
            b"ipv6" => PiiKind::Ipv6,
            b"ipv4" => PiiKind::Ipv4,
            b"us-ssn" => PiiKind::UsSsn,
            b"us-itin" => PiiKind::UsItin,
            b"uk-nhs" => PiiKind::UkNhs,
            b"de-steuer-id" => PiiKind::DeSteuerId,
            b"ca-sin" => PiiKind::CaSin,
            b"aba-routing" => PiiKind::AbaRouting,
            b"phone" => PiiKind::Phone,
            _ => PiiKind::Custom(name),
        }
    }

    /// Whether this is a caller-defined kind outside [`PiiKind::ALL`].
    pub const fn is_custom(self) -> bool {
        matches!(self, PiiKind::Custom(_))
    }

    /// Records a valid custom kind's static name so binary layer import can
    /// resolve it without allocating, up to [`PiiKind::MAX_CUSTOM_KINDS`]
    /// entries of at most [`PiiKind::MAX_CUSTOM_NAME_BYTES`] bytes.
    pub fn remember_custom(kind: PiiKind) {
        if let PiiKind::Custom(name) = kind
            && !name.is_empty()
            && name.len() <= Self::MAX_CUSTOM_NAME_BYTES
            && PiiKind::parse(name).is_none()
        {
            if custom_registry().read().expect("custom pii kinds lock").contains(&name) {
                return;
            }
            let mut reg = custom_registry().write().expect("custom pii kinds lock");
            if !reg.contains(&name) && reg.len() < Self::MAX_CUSTOM_KINDS {
                reg.push(name);
            }
        }
    }

    /// Looks up a previously registered custom kind by `name`.
    pub fn lookup_custom(name: &str) -> Option<PiiKind> {
        let reg = custom_registry().read().expect("custom pii kinds lock");
        reg.iter().copied().find(|&k| k == name).map(PiiKind::Custom)
    }

    /// Looks up or interns a custom kind name during binary layer import,
    /// bounded to [`PiiKind::MAX_CUSTOM_KINDS`] custom kinds of at most
    /// [`PiiKind::MAX_CUSTOM_NAME_BYTES`] bytes each.
    pub fn intern_custom(name: &str) -> Option<PiiKind> {
        if name.is_empty() || PiiKind::parse(name).is_some() {
            return None;
        }
        if let Some(k) = PiiKind::lookup_custom(name) {
            return Some(k);
        }
        if name.len() > Self::MAX_CUSTOM_NAME_BYTES {
            return None;
        }
        let mut reg = custom_registry().write().expect("custom pii kinds lock");
        if let Some(&existing) = reg.iter().find(|&&k| k == name) {
            return Some(PiiKind::Custom(existing));
        }
        if reg.len() >= Self::MAX_CUSTOM_KINDS {
            return None;
        }
        let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
        reg.push(leaked);
        Some(PiiKind::Custom(leaked))
    }

    /// The name OpenNLP's `PiiMention` uses for the type (`email`,
    /// `aws-access-key`, `de-steuer-id`, or a caller's custom name).
    pub const fn name(self) -> &'static str {
        match self {
            PiiKind::Jwt => "jwt",
            PiiKind::AwsAccessKey => "aws-access-key",
            PiiKind::GithubToken => "github-token",
            PiiKind::UrlCredential => "url-credential",
            PiiKind::Email => "email",
            PiiKind::Iban => "iban",
            PiiKind::Imei => "imei",
            PiiKind::Card => "card",
            PiiKind::BtcAddress => "btc-address",
            PiiKind::EthAddress => "eth-address",
            PiiKind::Mac => "mac",
            PiiKind::Ipv6 => "ipv6",
            PiiKind::Ipv4 => "ipv4",
            PiiKind::UsSsn => "us-ssn",
            PiiKind::UsItin => "us-itin",
            PiiKind::UkNhs => "uk-nhs",
            PiiKind::DeSteuerId => "de-steuer-id",
            PiiKind::CaSin => "ca-sin",
            PiiKind::AbaRouting => "aba-routing",
            PiiKind::Phone => "phone",
            PiiKind::Custom(name) => name,
        }
    }

    /// The built-in kind OpenNLP names `name`.
    pub fn parse(name: &str) -> Option<PiiKind> {
        PiiKind::ALL.into_iter().find(|k| k.name() == name)
    }

    /// `PiiTypePriority.rank`: the position in [`PiiKind::ALL`], or `20`
    /// (`PiiKind::ALL.len()`) for [`PiiKind::Custom`].
    pub const fn rank(self) -> u8 {
        match self {
            PiiKind::Jwt => 0,
            PiiKind::AwsAccessKey => 1,
            PiiKind::GithubToken => 2,
            PiiKind::UrlCredential => 3,
            PiiKind::Email => 4,
            PiiKind::Iban => 5,
            PiiKind::Imei => 6,
            PiiKind::Card => 7,
            PiiKind::BtcAddress => 8,
            PiiKind::EthAddress => 9,
            PiiKind::Mac => 10,
            PiiKind::Ipv6 => 11,
            PiiKind::Ipv4 => 12,
            PiiKind::UsSsn => 13,
            PiiKind::UsItin => 14,
            PiiKind::UkNhs => 15,
            PiiKind::DeSteuerId => 16,
            PiiKind::CaSin => 17,
            PiiKind::AbaRouting => 18,
            PiiKind::Phone => 19,
            PiiKind::Custom(_) => 20,
        }
    }

    /// The bit of this kind in a [`PiiKinds`] set (`0` for [`PiiKind::Custom`]).
    pub const fn bit(self) -> u32 {
        let r = self.rank();
        if (r as usize) < PiiKind::ALL.len() { 1 << r as u32 } else { 0 }
    }
}

impl fmt::Display for PiiKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of kinds, one bit each.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PiiKinds(u32);

impl PiiKinds {
    pub const EMPTY: PiiKinds = PiiKinds(0);

    /// The set of `kinds`.
    pub const fn of(kinds: &[PiiKind]) -> PiiKinds {
        let mut bits = 0;
        let mut i = 0;
        while i < kinds.len() {
            bits |= kinds[i].bit();
            i += 1;
        }
        PiiKinds(bits)
    }

    pub const fn contains(self, kind: PiiKind) -> bool {
        self.0 & kind.bit() != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every kind of `self` is in `other`.
    pub const fn is_subset(self, other: PiiKinds) -> bool {
        self.0 & !other.0 == 0
    }

    pub const fn union(self, other: PiiKinds) -> PiiKinds {
        PiiKinds(self.0 | other.0)
    }

    /// The kinds in rank order.
    pub fn iter(self) -> impl Iterator<Item = PiiKind> {
        PiiKind::ALL.into_iter().filter(move |k| self.contains(*k))
    }
}

/// How much the text supports a mention's kind. OpenNLP reports no
/// confidence; this is read from what the detector checked, so it never
/// changes which mentions are found.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PiiConfidence {
    /// The shape alone: the syntax of an address, a key or a token, a
    /// registered domain or calling code, allocated number ranges. No
    /// check digit backs it.
    Format,
    /// A checksum or check digit over the value passes: Luhn, mod-97,
    /// mod-11, ISO 7064, base58check, bech32, EIP-55, the ABA weights.
    Checksum,
    /// A label before the value names the kind (`IMEI:`, `SIN`), and its
    /// check digit passes.
    LabelledChecksum,
}

impl PiiConfidence {
    pub const fn name(self) -> &'static str {
        match self {
            PiiConfidence::Format => "format",
            PiiConfidence::Checksum => "checksum",
            PiiConfidence::LabelledChecksum => "labelled-checksum",
        }
    }
}

/// Every PII mention in a view, one item per mention, ordered as OpenNLP
/// orders them: by start, then longer first, then by kind rank. Equal
/// mentions (span, kind and normalized form) appear once; mentions of
/// different kinds, or of one kind with different normalized forms, may
/// share a span or overlap.
pub struct Pii {
    pub spans: Spans<Pii>,
    pub kind: Vec<PiiKind>,
    pub confidence: Vec<PiiConfidence>,
    /// The value as OpenNLP normalizes it: digits without separators, a
    /// lowercased domain, an RFC 5952 IPv6 address, an EIP-55 address.
    pub normalized: Strs,
}

/// One mention, as handed to [`Pii::push`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PiiMention<'a> {
    pub span: Span,
    pub kind: PiiKind,
    pub confidence: PiiConfidence,
    pub normalized: &'a str,
}

impl Pii {
    /// An empty layer over `view`.
    pub fn new(view: ViewId) -> Pii {
        Pii { spans: Spans::new(view), kind: Vec::new(), confidence: Vec::new(), normalized: Strs::default() }
    }

    /// Appends one mention to every column.
    pub fn push(&mut self, m: PiiMention<'_>) -> Ix<Pii> {
        PiiKind::remember_custom(m.kind);
        self.kind.push(m.kind);
        self.confidence.push(m.confidence);
        self.normalized.push(m.normalized);
        self.spans.push(m.span)
    }

    /// Mention `i`, column by column.
    pub fn get(&self, i: Ix<Pii>) -> PiiMention<'_> {
        PiiMention {
            span: self.spans.get(i),
            kind: self.kind[i.get()],
            confidence: self.confidence[i.get()],
            normalized: self.normalized.get(i.get()),
        }
    }

    /// Every mention, in layer order.
    pub fn mentions(&self) -> impl Iterator<Item = PiiMention<'_>> {
        self.spans.iter().map(|(i, _)| self.get(i))
    }
}

impl Layer for Pii {
    const ID: &'static str = "turbo:pii";
    const OVERLAP: Overlap = Overlap::Any;
    positional!(Pii);

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.kind.len() != n || self.confidence.len() != n || self.normalized.len() != n {
            return Err("every column as long as the spans".into());
        }
        self.spans.check(Self::OVERLAP)?;
        let mut previous: Option<PiiMention<'_>> = None;
        for (i, m) in self.mentions().enumerate() {
            if m.span.is_empty() {
                return Err(format!("pii mention {i} is empty at {}", m.span.start));
            }
            if m.kind.name().is_empty() {
                return Err(format!("pii mention {i} has no kind name"));
            }
            if let PiiKind::Custom(name) = m.kind {
                if name.len() > PiiKind::MAX_CUSTOM_NAME_BYTES {
                    return Err(format!(
                        "pii mention {i} has a custom kind name of {} bytes (max {})",
                        name.len(),
                        PiiKind::MAX_CUSTOM_NAME_BYTES
                    ));
                }
                if PiiKind::parse(name).is_some() {
                    return Err(format!("pii mention {i} uses reserved built-in kind name {name:?} as custom"));
                }
            }
            if m.normalized.is_empty() {
                return Err(format!("pii mention {i} has no normalized form"));
            }
            if let Some(p) = previous {
                let order = (p.span.start, std::cmp::Reverse(p.span.end), p.kind).cmp(&(
                    m.span.start,
                    std::cmp::Reverse(m.span.end),
                    m.kind,
                ));
                if order.is_gt() {
                    return Err(format!("pii mention {i} is out of order: by start, longer first, then by kind"));
                }
            }
            // Mentions with one span and kind are adjacent, so a duplicate
            // is among the run of those just before.
            let duplicate =
                (0..i).rev().map(|j| self.get(Ix::new(j as u32))).take_while(|p| p.span == m.span && p.kind == m.kind);
            if duplicate.into_iter().any(|p| p.normalized == m.normalized) {
                return Err(format!("pii mention {i} repeats an earlier one"));
            }
            previous = Some(m);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "pii_tests.rs"]
mod tests;
