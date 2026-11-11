//! How much one mention of a kind matters to a document's PII risk
//! ([`PiiSeverity`]), the built-in table of it per kind, the caller's
//! overrides ([`PiiSeverityTable`]), and the rewrite a suggested policy
//! gives each kind ([`PiiSuggestedRewrite`]).

use super::super::pii::{CustomKinds, PiiKind};

/// The severity of a kind. [`PiiSeverity::id`] is the stable number of the
/// proto `PiiSeverity` enum; 0 is its `UNSPECIFIED`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PiiSeverity {
    /// Public or business identifiers: a company number, a routing number,
    /// an EIN, a parcel tracking number, a URL, a ZIP code, a phone
    /// extension, an age, a date, an amount of money, a state, a county.
    Low = 1,
    /// Contact, device and vehicle details: an e-mail address, a phone
    /// number, a street address, an IP address, a wallet address, a VIN.
    Medium = 2,
    /// Financial accounts: an IBAN.
    High = 3,
    /// Credentials, payment cards, national identity numbers, passports
    /// and driver's licences.
    Critical = 4,
}

impl PiiSeverity {
    /// Every severity, in id order.
    pub const ALL: [PiiSeverity; 4] = [PiiSeverity::Low, PiiSeverity::Medium, PiiSeverity::High, PiiSeverity::Critical];

    pub const fn id(self) -> u8 {
        self as u8
    }

    /// The severity with stable id `id`; `None` for 0 and unknown ids.
    pub fn from_id(id: u32) -> Option<PiiSeverity> {
        PiiSeverity::ALL.into_iter().find(|s| u32::from(s.id()) == id)
    }

    pub const fn name(self) -> &'static str {
        match self {
            PiiSeverity::Low => "low",
            PiiSeverity::Medium => "medium",
            PiiSeverity::High => "high",
            PiiSeverity::Critical => "critical",
        }
    }

    /// The weight a kind's mentions are multiplied by in the score:
    /// low 0.5, medium 2, high 5, critical 10.
    pub const fn weight(self) -> f64 {
        match self {
            PiiSeverity::Low => 0.5,
            PiiSeverity::Medium => 2.0,
            PiiSeverity::High => 5.0,
            PiiSeverity::Critical => 10.0,
        }
    }

    /// The built-in severity of `kind`. A custom kind is medium.
    pub const fn default_for(kind: PiiKind) -> PiiSeverity {
        use PiiKind::*;
        match kind {
            Jwt | AwsAccessKey | GithubToken | UrlCredential | Card | UsSsn | UsItin | UkNhs | DeSteuerId | EsDni
            | EsNie | DePersonalausweis | FrNir | BrCpf | CaSin | UsPassport | UsDriversLicence => {
                PiiSeverity::Critical
            }
            Iban => PiiSeverity::High,
            Email | Phone | StreetAddress | Imei | Mac | Ipv4 | Ipv6 | BtcAddress | EthAddress | Vin | Custom(_) => {
                PiiSeverity::Medium
            }
            EsCif | FrSiren | FrSiret | BrCnpj | Bic | AbaRouting | UsEin | TrackingNumber | Url | UsZip
            | PhoneExtension | Age | Date | Currency | State | County | City => PiiSeverity::Low,
        }
    }
}

/// What a suggested redaction policy rewrites a kind with. Each value maps
/// to one `PiiRewriteStrategy` arm of a `RedactionPolicy`; none needs a
/// key, a dictionary or per-document numbering. [`PiiSuggestedRewrite::id`]
/// is the stable number of the proto enum; 0 is its `UNSPECIFIED`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PiiSuggestedRewrite {
    /// Every character becomes `*`.
    Mask = 1,
    /// Letters and digits become `*`; separators stay readable.
    MaskKeepFormat = 2,
    /// The mention becomes its upper-case kind name in brackets (`[EMAIL]`).
    Label = 3,
    /// All but the last four letters and digits become `*`.
    KeepLastFour = 4,
}

impl PiiSuggestedRewrite {
    pub const ALL: [PiiSuggestedRewrite; 4] = [
        PiiSuggestedRewrite::Mask,
        PiiSuggestedRewrite::MaskKeepFormat,
        PiiSuggestedRewrite::Label,
        PiiSuggestedRewrite::KeepLastFour,
    ];

    pub const fn id(self) -> u8 {
        self as u8
    }

    pub fn from_id(id: u32) -> Option<PiiSuggestedRewrite> {
        PiiSuggestedRewrite::ALL.into_iter().find(|s| u32::from(s.id()) == id)
    }

    /// The built-in suggestion for `kind`: a label for credentials,
    /// contact details, street addresses, ages, dates, amounts, states, counties and URLs, the last four for cards and IBANs, a
    /// format-keeping mask for identity, vehicle, parcel and business
    /// numbers.
    pub const fn default_for(kind: PiiKind) -> PiiSuggestedRewrite {
        use PiiKind::*;
        match kind {
            Jwt | AwsAccessKey | GithubToken | UrlCredential | Email | Phone | Mac | Ipv4 | Ipv6 | BtcAddress
            | EthAddress | Url | StreetAddress | Age | Date | Currency | State | County | City | Custom(_) => {
                PiiSuggestedRewrite::Label
            }
            Card | Iban => PiiSuggestedRewrite::KeepLastFour,
            Imei | UsSsn | UsItin | UkNhs | DeSteuerId | EsDni | EsNie | EsCif | DePersonalausweis | FrNir
            | FrSiren | FrSiret | BrCpf | BrCnpj | Bic | CaSin | AbaRouting | Vin | TrackingNumber | UsEin
            | UsPassport | UsDriversLicence | UsZip | PhoneExtension => PiiSuggestedRewrite::MaskKeepFormat,
        }
    }
}

/// The severity of every kind: the built-in table ([`PiiSeverity::default_for`])
/// with the caller's overrides. A custom kind is looked up by name, then
/// takes the custom default (medium unless set).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiiSeverityTable {
    /// Built-in overrides by stable kind id, kept sorted by id.
    builtin: Vec<(u16, PiiSeverity)>,
    /// Custom kind overrides by name, kept sorted by name.
    custom: Vec<(Box<str>, PiiSeverity)>,
    custom_default: PiiSeverity,
}

impl Default for PiiSeverityTable {
    fn default() -> Self {
        PiiSeverityTable { builtin: Vec::new(), custom: Vec::new(), custom_default: PiiSeverity::Medium }
    }
}

impl PiiSeverityTable {
    /// The built-in table with no overrides.
    pub fn new() -> PiiSeverityTable {
        PiiSeverityTable::default()
    }

    /// Overrides the severity of a built-in `kind`; a custom kind is
    /// ignored here, since its index means nothing outside its table: use
    /// [`PiiSeverityTable::with_custom`].
    pub fn with(mut self, kind: PiiKind, severity: PiiSeverity) -> PiiSeverityTable {
        if let Some(id) = kind.builtin_id() {
            match self.builtin.binary_search_by_key(&id, |&(k, _)| k) {
                Ok(at) => self.builtin[at].1 = severity,
                Err(at) => self.builtin.insert(at, (id, severity)),
            }
        }
        self
    }

    /// Overrides the severity of the custom kind named `name`.
    pub fn with_custom(mut self, name: &str, severity: PiiSeverity) -> PiiSeverityTable {
        match self.custom.binary_search_by(|(n, _)| (**n).cmp(name)) {
            Ok(at) => self.custom[at].1 = severity,
            Err(at) => self.custom.insert(at, (name.into(), severity)),
        }
        self
    }

    /// The severity of a custom kind with no override of its own.
    pub fn with_custom_default(mut self, severity: PiiSeverity) -> PiiSeverityTable {
        self.custom_default = severity;
        self
    }

    /// The severity of `kind`, whose custom index (if any) is into `custom_kinds`.
    pub fn severity(&self, kind: PiiKind, custom_kinds: &CustomKinds) -> PiiSeverity {
        match kind.builtin_id() {
            Some(id) => match self.builtin.binary_search_by_key(&id, |&(k, _)| k) {
                Ok(at) => self.builtin[at].1,
                Err(_) => PiiSeverity::default_for(kind),
            },
            None => custom_kinds
                .name(kind)
                .and_then(|name| self.custom.binary_search_by(|(n, _)| (**n).cmp(name)).ok())
                .map_or(self.custom_default, |at| self.custom[at].1),
        }
    }

    /// Whether this is the built-in table.
    pub fn is_default(&self) -> bool {
        *self == PiiSeverityTable::default()
    }

    /// FNV-1a over the overrides in their sorted order, so equal tables
    /// fold equally however they were built.
    pub fn fold_hash(&self, h: &mut turbo_analysis::Fnv1a) {
        h.str("turbo:pii-severity:v1").usize(self.builtin.len());
        for &(id, s) in &self.builtin {
            h.u16(id).u8(s.id());
        }
        h.usize(self.custom.len());
        for (name, s) in &self.custom {
            h.str(name).u8(s.id());
        }
        h.u8(self.custom_default.id());
    }

    /// The table's digest: [`PiiSeverityTable::fold_hash`] finished.
    pub fn digest(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        self.fold_hash(&mut h);
        h.finish()
    }
}
