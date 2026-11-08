//! The kind registry: one checked-in table of built-in kinds with their
//! stable id, wire name and priority.
//!
//! The id is identity only. It is append-only: a new kind takes the next
//! free id, and no id is reused or renumbered. The same number is the
//! proto `PiiKindId` value, the C `TURBO_PII_KIND_*` constant, the Java
//! `TurboPiiKind` code and the binary codec's kind code. The priority is a
//! separate column used only to break ties between mentions on one span.

use std::cmp::Reverse;
use std::fmt;

use super::custom::CustomKindId;

macro_rules! pii_kinds {
    ($( $(#[$doc:meta])* $variant:ident = $id:literal, $name:literal, $priority:literal; )*) => {
        /// What a mention is: a built-in kind from the registry table, or a
        /// [`PiiKind::Custom`] kind indexing the custom-kind table of the
        /// layer or policy that holds it.
        ///
        /// There is deliberately no `Ord`: identity ([`PiiKind::id`]) and
        /// tie-break order ([`PiiKind::priority`]) are separate columns.
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum PiiKind {
            $( $(#[$doc])* $variant, )*
            /// A caller-defined kind: an index into a [`super::CustomKinds`]
            /// table.
            Custom(CustomKindId),
        }

        /// One row of the built-in kind registry.
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        pub struct KindRow {
            pub kind: PiiKind,
            /// The stable id, from 1; 0 is the proto's `UNSPECIFIED`.
            pub id: u16,
            /// The wire and display name (`email`, `de-steuer-id`).
            pub name: &'static str,
            /// Tie-break order on one span: a smaller priority sorts first.
            pub priority: u16,
        }

        impl PiiKind {
            /// The built-in kind registry, in id order.
            pub const TABLE: &'static [KindRow] = &[
                $( KindRow { kind: PiiKind::$variant, id: $id, name: $name, priority: $priority }, )*
            ];

            /// The stable id of a built-in kind, or `None` for a custom kind.
            pub const fn builtin_id(self) -> Option<u16> {
                match self {
                    $( PiiKind::$variant => Some($id), )*
                    PiiKind::Custom(_) => None,
                }
            }

            /// The wire name of a built-in kind, or `None` for a custom kind,
            /// whose name lives in its table.
            pub const fn builtin_name(self) -> Option<&'static str> {
                match self {
                    $( PiiKind::$variant => Some($name), )*
                    PiiKind::Custom(_) => None,
                }
            }

            /// The tie-break priority: a smaller one sorts first among
            /// mentions on one span. Every custom kind is after every
            /// built-in kind.
            pub const fn priority(self) -> u16 {
                match self {
                    $( PiiKind::$variant => $priority, )*
                    PiiKind::Custom(_) => u16::MAX,
                }
            }
        }
    };
}

pii_kinds! {
    /// A JSON Web Token in compact serialization.
    Jwt = 1, "jwt", 10;
    /// An AWS access key identifier (`AKIA`, `ASIA`).
    AwsAccessKey = 2, "aws-access-key", 20;
    /// A GitHub access token (`ghp_`, `github_pat_`, `ghs_` and others).
    GithubToken = 3, "github-token", 30;
    /// The `user:password` userinfo of a URL.
    UrlCredential = 4, "url-credential", 40;
    /// An e-mail address.
    Email = 5, "email", 50;
    /// An International Bank Account Number, mod-97 checked.
    Iban = 6, "iban", 60;
    /// A mobile device's IMEI after an `IMEI` label, Luhn checked.
    Imei = 7, "imei", 70;
    /// A payment card number, Luhn checked.
    Card = 8, "card", 80;
    /// A Bitcoin address: base58check or bech32/bech32m.
    BtcAddress = 9, "btc-address", 90;
    /// An Ethereum address, EIP-55 checked when written in mixed case.
    EthAddress = 10, "eth-address", 100;
    /// A 48-bit IEEE 802 MAC address.
    Mac = 11, "mac", 110;
    /// An IPv6 address.
    Ipv6 = 12, "ipv6", 120;
    /// An IPv4 address.
    Ipv4 = 13, "ipv4", 130;
    /// A United States Social Security number.
    UsSsn = 14, "us-ssn", 140;
    /// A United States Individual Taxpayer Identification Number.
    UsItin = 15, "us-itin", 150;
    /// A United Kingdom NHS number, mod-11 checked.
    UkNhs = 16, "uk-nhs", 160;
    /// A German tax identification number (Steuer-ID), ISO 7064 checked.
    DeSteuerId = 17, "de-steuer-id", 170;
    /// A Spanish Documento Nacional de Identidad, mod-23 checked.
    EsDni = 18, "es-dni", 180;
    /// A Spanish Numero de Identidad de Extranjero, mod-23 checked.
    EsNie = 19, "es-nie", 190;
    /// A Spanish Codigo de Identificacion Fiscal, weighted check digit/letter.
    EsCif = 20, "es-cif", 200;
    /// A German Personalausweis document number, ICAO 9303 7-3-1 checked.
    DePersonalausweis = 21, "de-personalausweis", 210;
    /// A French INSEE NIR (numero de securite sociale), mod-97 key checked.
    FrNir = 22, "fr-nir", 220;
    /// A French INSEE SIREN identifier, Luhn checked.
    FrSiren = 23, "fr-siren", 230;
    /// A French INSEE SIRET identifier, Luhn / La Poste checked.
    FrSiret = 24, "fr-siret", 240;
    /// A Brazilian Cadastro de Pessoas Fisicas number, double mod-11 checked.
    BrCpf = 25, "br-cpf", 250;
    /// A Brazilian Cadastro Nacional da Pessoa Juridica number, double mod-11 checked.
    BrCnpj = 26, "br-cnpj", 260;
    /// An ISO 9362 Business Identifier Code (BIC / SWIFT) after a label.
    Bic = 27, "bic", 270;
    /// A Canadian Social Insurance Number after a label, Luhn checked.
    CaSin = 28, "ca-sin", 280;
    /// An ABA routing transit number, check digit verified.
    AbaRouting = 29, "aba-routing", 290;
    /// A telephone number.
    Phone = 30, "phone", 300;
    /// A vehicle identification number (ISO 3779, 49 CFR 565), check digit
    /// verified.
    Vin = 31, "vin", 310;
    /// A UPS, FedEx or USPS parcel tracking number, check digit verified.
    TrackingNumber = 32, "tracking-number", 320;
    /// A United States Employer Identification Number with a valid IRS
    /// campus prefix.
    UsEin = 33, "us-ein", 330;
    /// A web address: an `http`, `https`, `ftp` or `ftps` URL, or a host
    /// starting `www.`.
    Url = 34, "url", 340;
    /// A United States passport book or card number after a passport label.
    UsPassport = 35, "us-passport", 350;
    /// A United States driver's licence number after a licence label.
    UsDriversLicence = 36, "us-drivers-licence", 360;
    /// A United States ZIP or ZIP+4 code after a state code or a ZIP label.
    UsZip = 37, "us-zip", 370;
    /// The extension digits after a telephone number (`ext.`, `x`, `#`).
    PhoneExtension = 38, "phone-extension", 380;
}

/// The first codec kind code of a custom kind: code `CUSTOM_CODE_BASE + i`
/// is entry `i` of the layer's custom-kind table. Built-in ids stay below.
pub const CUSTOM_CODE_BASE: u32 = 1 << 16;

impl PiiKind {
    /// The largest built-in id; a [`PiiKinds`] bit set holds ids up to 127.
    pub const MAX_BUILTIN_ID: u16 = 38;

    /// Every built-in kind, in id order.
    pub fn builtins() -> impl Iterator<Item = PiiKind> {
        PiiKind::TABLE.iter().map(|r| r.kind)
    }

    /// The built-in kind with stable id `id`.
    pub fn from_id(id: u16) -> Option<PiiKind> {
        let at = usize::from(id).checked_sub(1)?;
        PiiKind::TABLE.get(at).filter(|r| r.id == id).map(|r| r.kind)
    }

    /// The built-in kind named `name`.
    pub fn parse(name: &str) -> Option<PiiKind> {
        PiiKind::TABLE.iter().find(|r| r.name == name).map(|r| r.kind)
    }

    /// Whether this is a caller-defined kind.
    pub const fn is_custom(self) -> bool {
        matches!(self, PiiKind::Custom(_))
    }

    /// The codec kind code: the built-in id, or [`CUSTOM_CODE_BASE`] plus
    /// the custom-table index.
    pub const fn code(self) -> u32 {
        match self {
            PiiKind::Custom(c) => CUSTOM_CODE_BASE + c.index() as u32,
            builtin => match builtin.builtin_id() {
                Some(id) => id as u32,
                None => 0,
            },
        }
    }

    /// The sort key of a mention's kind among mentions on one span:
    /// priority, then code, so equal priorities still sort the same way.
    pub const fn order_key(self) -> (u16, u32) {
        (self.priority(), self.code())
    }

    /// The bit of this kind in a [`PiiKinds`] set (`0` for a custom kind).
    pub const fn bit(self) -> u128 {
        match self.builtin_id() {
            Some(id) => 1u128 << id,
            None => 0,
        }
    }
}

/// The sort key of a mention: by start, longer first, then by kind
/// priority.
pub fn mention_order(start: u32, end: u32, kind: PiiKind) -> (u32, Reverse<u32>, (u16, u32)) {
    (start, Reverse(end), kind.order_key())
}

impl fmt::Display for PiiKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PiiKind::Custom(c) => write!(f, "custom-kind-{}", c.index()),
            builtin => f.write_str(builtin.builtin_name().unwrap_or("")),
        }
    }
}

/// A set of built-in kinds, one bit per stable id.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PiiKinds(u128);

impl PiiKinds {
    pub const EMPTY: PiiKinds = PiiKinds(0);

    /// The set of `kinds`; custom kinds are not members of any set.
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

    /// The id bits: bit `id` is set for each member.
    pub const fn bits(self) -> u128 {
        self.0
    }

    /// The kinds in id order.
    pub fn iter(self) -> impl Iterator<Item = PiiKind> {
        PiiKind::builtins().filter(move |k| self.contains(*k))
    }
}

const _: () = {
    assert!(PiiKind::TABLE.len() == PiiKind::MAX_BUILTIN_ID as usize);
    assert!((PiiKind::MAX_BUILTIN_ID as u32) < 128);
    let mut i = 0;
    while i < PiiKind::TABLE.len() {
        assert!(PiiKind::TABLE[i].id as usize == i + 1, "ids are 1.. in table order");
        i += 1;
    }
};
