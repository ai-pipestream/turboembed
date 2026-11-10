//! Built-in terms and their pre-split context labels.

use turbo_analysis_layers::PiiKind;

/// Default context-label terms for `kind` (lowercase ASCII words or phrases).
pub const fn default_labels(kind: PiiKind) -> &'static [&'static str] {
    match kind {
        PiiKind::Email => &["email", "e-mail", "mail"],
        PiiKind::Phone => &["phone", "tel", "telephone", "mobile", "cell", "fax"],
        PiiKind::UsSsn => &["ssn", "social security", "social security number", "social security no"],
        PiiKind::UsItin => &["itin", "taxpayer identification", "taxpayer identification number", "tax id"],
        PiiKind::UkNhs => &["nhs", "nhs number", "nhs no"],
        PiiKind::CaSin => &["social insurance number", "social insurance no", "sin"],
        PiiKind::DeSteuerId => {
            &["steuer-id", "steuer id", "steueridentifikationsnummer", "identifikationsnummer", "idnr"]
        }
        PiiKind::Iban => &["iban"],
        PiiKind::AbaRouting => &["aba", "routing", "routing number", "rtn"],
        PiiKind::Card => &["card", "credit card", "debit card", "pan", "cc"],
        PiiKind::Imei => &["imei"],
        PiiKind::Ipv4 | PiiKind::Ipv6 => &["ip", "ip address", "ipv4", "ipv6"],
        PiiKind::Mac => &["mac", "mac address", "bssid", "hwaddr"],
        PiiKind::BtcAddress => &["btc", "bitcoin", "bitcoin address"],
        PiiKind::EthAddress => &["eth", "ethereum", "ethereum address"],
        PiiKind::AwsAccessKey => &["aws", "access key", "aws access key"],
        PiiKind::GithubToken => &["github", "github token", "gh token", "pat"],
        PiiKind::Jwt => &["jwt", "bearer", "id token", "access token"],
        PiiKind::UrlCredential => &["url", "uri", "dsn", "connection string"],
        PiiKind::EsDni => &["dni"],
        PiiKind::EsNie => &["nie"],
        PiiKind::EsCif => &["cif"],
        PiiKind::DePersonalausweis => &["personalausweis", "ausweisnummer"],
        PiiKind::FrNir => &["nir"],
        PiiKind::FrSiren => &["siren"],
        PiiKind::FrSiret => &["siret"],
        PiiKind::BrCpf => &["cpf"],
        PiiKind::BrCnpj => &["cnpj"],
        PiiKind::Bic => &["bic/swift", "swift/bic", "bic code", "swift code", "swift", "bic"],
        PiiKind::Vin => &["vin", "vehicle identification number", "vehicle id", "chassis number"],
        PiiKind::TrackingNumber => {
            &["tracking number", "tracking no", "tracking", "ups", "fedex", "usps", "shipment", "waybill"]
        }
        PiiKind::UsEin => &["ein", "fein", "employer identification number", "employer id", "tax id"],
        PiiKind::Url => &["url", "link", "website", "web site", "site"],
        PiiKind::UsPassport => &["passport number", "passport no", "passport card", "passport"],
        PiiKind::UsDriversLicence => &[
            "driver's license",
            "driver\u{2019}s license",
            "drivers license",
            "driver license",
            "driver's licence",
            "drivers licence",
            "driver licence",
            "driving licence",
            "license number",
            "licence number",
            "license no",
            "licence no",
            "dl",
            "dln",
        ],
        PiiKind::UsZip => &["zip code", "zip", "zipcode", "postal code"],
        PiiKind::PhoneExtension => &["ext", "extn", "extension"],
        PiiKind::Age => &["age", "aged"],
        PiiKind::StreetAddress => &["address", "street address", "mailing address", "home address"],
        PiiKind::Date => &["date", "date of birth", "dob", "born"],
        PiiKind::Currency => &["amount", "balance", "salary", "payment"],
        PiiKind::Custom(_) => &[],
    }
}

/// Pre-split default label phrases for `kind`, avoiding per-candidate phrase
/// segmentation during detection.
pub(super) const fn default_label_phrases(kind: PiiKind) -> &'static [&'static [&'static str]] {
    match kind {
        PiiKind::Email => &[&["email"], &["e", "mail"], &["mail"]],
        PiiKind::Phone => &[&["phone"], &["tel"], &["telephone"], &["mobile"], &["cell"], &["fax"]],
        PiiKind::UsSsn => {
            &[&["ssn"], &["social", "security"], &["social", "security", "number"], &["social", "security", "no"]]
        }
        PiiKind::UsItin => {
            &[&["itin"], &["taxpayer", "identification"], &["taxpayer", "identification", "number"], &["tax", "id"]]
        }
        PiiKind::UkNhs => &[&["nhs"], &["nhs", "number"], &["nhs", "no"]],
        PiiKind::CaSin => &[&["social", "insurance", "number"], &["social", "insurance", "no"], &["sin"]],
        PiiKind::DeSteuerId => {
            &[&["steuer", "id"], &["steueridentifikationsnummer"], &["identifikationsnummer"], &["idnr"]]
        }
        PiiKind::Iban => &[&["iban"]],
        PiiKind::AbaRouting => &[&["aba"], &["routing"], &["routing", "number"], &["rtn"]],
        PiiKind::Card => &[&["card"], &["credit", "card"], &["debit", "card"], &["pan"], &["cc"]],
        PiiKind::Imei => &[&["imei"]],
        PiiKind::Ipv4 | PiiKind::Ipv6 => &[&["ip"], &["ip", "address"], &["ipv4"], &["ipv6"]],
        PiiKind::Mac => &[&["mac"], &["mac", "address"], &["bssid"], &["hwaddr"]],
        PiiKind::BtcAddress => &[&["btc"], &["bitcoin"], &["bitcoin", "address"]],
        PiiKind::EthAddress => &[&["eth"], &["ethereum"], &["ethereum", "address"]],
        PiiKind::AwsAccessKey => &[&["aws"], &["access", "key"], &["aws", "access", "key"]],
        PiiKind::GithubToken => &[&["github"], &["github", "token"], &["gh", "token"], &["pat"]],
        PiiKind::Jwt => &[&["jwt"], &["bearer"], &["id", "token"], &["access", "token"]],
        PiiKind::UrlCredential => &[&["url"], &["uri"], &["dsn"], &["connection", "string"]],
        PiiKind::EsDni => &[&["dni"]],
        PiiKind::EsNie => &[&["nie"]],
        PiiKind::EsCif => &[&["cif"]],
        PiiKind::DePersonalausweis => &[&["personalausweis"], &["ausweisnummer"]],
        PiiKind::FrNir => &[&["nir"]],
        PiiKind::FrSiren => &[&["siren"]],
        PiiKind::FrSiret => &[&["siret"]],
        PiiKind::BrCpf => &[&["cpf"]],
        PiiKind::BrCnpj => &[&["cnpj"]],
        PiiKind::Bic => {
            &[&["bic", "swift"], &["swift", "bic"], &["bic", "code"], &["swift", "code"], &["swift"], &["bic"]]
        }
        PiiKind::Vin => {
            &[&["vin"], &["vehicle", "identification", "number"], &["vehicle", "id"], &["chassis", "number"]]
        }
        PiiKind::TrackingNumber => &[
            &["tracking", "number"],
            &["tracking", "no"],
            &["tracking"],
            &["ups"],
            &["fedex"],
            &["usps"],
            &["shipment"],
            &["waybill"],
        ],
        PiiKind::UsEin => {
            &[&["ein"], &["fein"], &["employer", "identification", "number"], &["employer", "id"], &["tax", "id"]]
        }
        PiiKind::Url => &[&["url"], &["link"], &["website"], &["web", "site"], &["site"]],
        PiiKind::UsPassport => &[&["passport", "number"], &["passport", "no"], &["passport", "card"], &["passport"]],
        PiiKind::UsDriversLicence => &[
            &["driver's", "license"],
            &["driver\u{2019}s", "license"],
            &["drivers", "license"],
            &["driver", "license"],
            &["driver's", "licence"],
            &["drivers", "licence"],
            &["driver", "licence"],
            &["driving", "licence"],
            &["license", "number"],
            &["licence", "number"],
            &["license", "no"],
            &["licence", "no"],
            &["dl"],
            &["dln"],
        ],
        PiiKind::UsZip => &[&["zip", "code"], &["zip"], &["zipcode"], &["postal", "code"]],
        PiiKind::PhoneExtension => &[&["ext"], &["extn"], &["extension"]],
        PiiKind::Age => &[&["age"], &["aged"]],
        PiiKind::StreetAddress => {
            &[&["address"], &["street", "address"], &["mailing", "address"], &["home", "address"]]
        }
        PiiKind::Date => &[&["date"], &["date", "of", "birth"], &["dob"], &["born"]],
        PiiKind::Currency => &[&["amount"], &["balance"], &["salary"], &["payment"]],
        PiiKind::Custom(_) => &[],
    }
}
