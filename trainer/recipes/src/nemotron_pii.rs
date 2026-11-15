//! NVIDIA `Nemotron-PII` (#1888): 55 fine-grained personal, financial,
//! network, credential and medical/PHI entity types over 200,000 synthetic
//! English records (`us` and `intl` locales), published by NVIDIA Corporation
//! at revision `b70ffaf5ff39e079776134c5bf4381f00a9fd1ed` under CC BY 4.0.
//!
//! Each Parquet split (`train` and `test`) carries 100,000 rows over 50,000
//! persona `uid`s (`us` in rows `0..50_000`, `intl` in rows `50_000..100_000`
//! in matching `uid` order). Because the hub release ships `train` and `test`
//! Parquet files without a separate `dev` file, `read` partitions `train`
//! deterministically by persona `uid` ordinal (`ord % 10 == 9` -> `dev`, else
//! `train`), keeping every persona's `us` and `intl` records on the same side
//! of the boundary (`90,000` `train`, `10,000` `dev`, `100,000` `test`).

pub mod spans;

use std::collections::HashMap;
use std::path::Path;

use turbo_eval::read::parquet::Parquet;

use crate::corpus::{Corpus, CorpusRecord, Counts, Pin, Sentence, Splits, Task, check_pin, read_file};
use crate::{Error, Result};

pub const ID: &str = "nemotron-pii";
pub const NAME: &str = "NVIDIA Nemotron-PII";
pub const REVISION: &str = "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed";
pub const HEAD: &str = "pii";
pub const LAYER: &str = "turboembed:pii";
pub const LICENCE: &str = "CC BY 4.0";
pub const SPDX: &str = "CC-BY-4.0";
pub const TERMS: &str = "https://creativecommons.org/licenses/by/4.0/legalcode";
pub const ATTRIBUTION: &str = "Trained on Nemotron-PII (Amy Steier, Andre Manoel, Alexa Haushalter and Maarten \
    Van Segbroeck, \"Nemotron-PII: Synthesized Data for Privacy-Preserving AI\", NVIDIA Corporation, 2025, \
    https://huggingface.co/datasets/nvidia/Nemotron-PII, revision b70ffaf5ff39e079776134c5bf4381f00a9fd1ed), \
    synthetic data generated with NVIDIA NeMo Data Designer grounded on Nemotron-Personas, licensed under \
    CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode).";

pub const README: Pin =
    Pin { path: "README.md", size: 4_969, sha256: "18c6215fb72c66e5c9fc833130f589e407cae906760f8977409deba0f1f5af4e" };
pub const TRAIN: Pin = Pin {
    path: "data/train-00000-of-00001.parquet",
    size: 156_334_329,
    sha256: "9cb1a88c339ab84ddbc35ccdde4f6bb713300f5585881f9e02b8db1e0dfdbece",
};
pub const TEST: Pin = Pin {
    path: "data/test-00000-of-00001.parquet",
    size: 150_968_398,
    sha256: "1a4b0512ecb5370f0992d29d0f9c07351e6de13f0d7ea33bb18cecb984780247",
};

/// All 55 fine-grained `Nemotron-PII` types in alphabetical order.
pub const TYPES: [&str; 55] = [
    "account_number",
    "age",
    "api_key",
    "bank_routing_number",
    "biometric_identifier",
    "blood_type",
    "certificate_license_number",
    "city",
    "company_name",
    "coordinate",
    "country",
    "county",
    "credit_debit_card",
    "customer_id",
    "cvv",
    "date",
    "date_of_birth",
    "date_time",
    "device_identifier",
    "education_level",
    "email",
    "employee_id",
    "employment_status",
    "fax_number",
    "first_name",
    "gender",
    "health_plan_beneficiary_number",
    "http_cookie",
    "ipv4",
    "ipv6",
    "language",
    "last_name",
    "license_plate",
    "mac_address",
    "medical_record_number",
    "national_id",
    "occupation",
    "password",
    "phone_number",
    "pin",
    "political_view",
    "postcode",
    "race_ethnicity",
    "religious_belief",
    "sexuality",
    "ssn",
    "state",
    "street_address",
    "swift_bic",
    "tax_id",
    "time",
    "unique_id",
    "url",
    "user_name",
    "vehicle_identifier",
];

/// Coarse PII/PHI categories grouping the 55 fine types for summary reporting.
pub const COARSE: [(&str, &[&str]); 9] = [
    ("person", &["first_name", "last_name", "user_name"]),
    ("contact", &["email", "phone_number", "fax_number", "url"]),
    ("location", &["street_address", "city", "county", "state", "postcode", "country", "coordinate"]),
    ("financial", &["credit_debit_card", "cvv", "bank_routing_number", "swift_bic", "account_number", "pin"]),
    (
        "government_id",
        &[
            "ssn",
            "tax_id",
            "national_id",
            "certificate_license_number",
            "license_plate",
            "vehicle_identifier",
            "unique_id",
        ],
    ),
    ("medical_phi", &["medical_record_number", "health_plan_beneficiary_number", "biometric_identifier", "blood_type"]),
    ("credential_network", &["api_key", "password", "http_cookie", "ipv4", "ipv6", "mac_address", "device_identifier"]),
    (
        "organization_work",
        &["company_name", "occupation", "employment_status", "education_level", "employee_id", "customer_id"],
    ),
    (
        "temporal_demographic",
        &[
            "age",
            "date",
            "date_of_birth",
            "date_time",
            "time",
            "gender",
            "sexuality",
            "race_ethnicity",
            "religious_belief",
            "political_view",
            "language",
        ],
    ),
];

/// Returns the static slice from [`TYPES`] matching `raw`, if valid.
pub fn intern_type(raw: &str) -> Option<&'static str> {
    TYPES.binary_search(&raw).ok().map(|i| TYPES[i])
}

/// Returns the coarse PII category for `fine`, or `fine` itself if ungrouped.
pub fn coarse(fine: &str) -> &str {
    COARSE.iter().find(|(_, members)| members.contains(&fine)).map_or(fine, |(group, _)| group)
}

/// Reads the pinned `Nemotron-PII` release from `dir`.
pub fn read(dir: &Path, unrecorded: bool) -> Result<Corpus> {
    let mut files = Vec::with_capacity(3);
    let mut mismatches = Vec::new();

    let (readme_rec, _) = read_file(dir, README.path)?;
    if let Err(m) = check_pin(&readme_rec, &README) {
        mismatches.push(m);
    }
    files.push(readme_rec);

    let (train_rec, train_bytes) = read_file(dir, TRAIN.path)?;
    if let Err(m) = check_pin(&train_rec, &TRAIN) {
        mismatches.push(m);
    }
    files.push(train_rec);

    let (test_rec, test_bytes) = read_file(dir, TEST.path)?;
    if let Err(m) = check_pin(&test_rec, &TEST) {
        mismatches.push(m);
    }
    files.push(test_rec);

    if !mismatches.is_empty() && !unrecorded {
        return Err(Error(format!(
            "{NAME} is not the recorded release (pass --unrecorded-corpus to train on it anyway): {}",
            mismatches.join("; ")
        )));
    }

    let (train, dev, skipped_train) = parse_train_dev(TRAIN.path, &train_bytes)?;
    let (test, skipped_test) = parse_test(TEST.path, &test_bytes)?;
    if [&train, &dev, &test].into_iter().any(Vec::is_empty) {
        return Err(Error(format!("{NAME} requires non-empty train, dev and test splits")));
    }

    let record = CorpusRecord {
        id: ID.into(),
        name: NAME.into(),
        task: Task::Bio,
        licence: LICENCE.into(),
        spdx: SPDX.into(),
        terms: TERMS.into(),
        attribution: ATTRIBUTION.into(),
        recorded: mismatches.is_empty(),
        corpus_sha256: String::new(),
        files,
        split: format!(
            "publisher train/test Parquet at {REVISION}; dev carved from train by persona uid ordinal (uid_ord % 10 == 9, keeping us and intl pairs together), SimpleTokenizer words"
        ),
        train: Counts::of(&train),
        dev: Counts::of(&dev),
        test: Counts::of(&test),
        skipped: skipped_train + skipped_test,
        recipe_only: false,
    };
    Ok(Corpus { record, splits: Splits { train, dev, test } })
}

fn parse_train_dev(file: &str, bytes: &[u8]) -> Result<(Vec<Sentence>, Vec<Sentence>, usize)> {
    let pq = Parquet::open(file, bytes).map_err(Error)?;
    let uids = pq.strings("uid").map_err(Error)?;
    let texts = pq.strings("text").map_err(Error)?;
    let spans = pq.strings("spans").map_err(Error)?;
    let rows = pq.rows();
    let mut uid_ord: HashMap<&str, usize> = HashMap::new();
    let mut train = Vec::new();
    let mut dev = Vec::new();
    let mut skipped = 0;
    for row in 0..rows {
        let uid = uids.at(row).map_err(Error)?;
        let next_ord = uid_ord.len();
        let ord = *uid_ord.entry(uid).or_insert(next_ord);
        let text = texts.at(row).map_err(Error)?;
        let spans_repr = spans.at(row).map_err(Error)?;
        match spans::to_sentence(file, row, text, spans_repr)? {
            Some(sent) if ord % 10 == 9 => dev.push(sent),
            Some(sent) => train.push(sent),
            None => skipped += 1,
        }
    }
    Ok((train, dev, skipped))
}

fn parse_test(file: &str, bytes: &[u8]) -> Result<(Vec<Sentence>, usize)> {
    let pq = Parquet::open(file, bytes).map_err(Error)?;
    let texts = pq.strings("text").map_err(Error)?;
    let spans = pq.strings("spans").map_err(Error)?;
    let rows = pq.rows();
    let mut test = Vec::with_capacity(rows);
    let mut skipped = 0;
    for row in 0..rows {
        let text = texts.at(row).map_err(Error)?;
        let spans_repr = spans.at(row).map_err(Error)?;
        match spans::to_sentence(file, row, text, spans_repr)? {
            Some(sent) => test.push(sent),
            None => skipped += 1,
        }
    }
    Ok((test, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_are_sorted_unique_and_exhaustively_grouped() {
        assert_eq!(TYPES.len(), 55);
        for w in TYPES.windows(2) {
            assert!(w[0] < w[1], "TYPES out of order: {} >= {}", w[0], w[1]);
        }
        let mut grouped = Vec::new();
        for (_, members) in COARSE {
            for &m in members {
                assert!(intern_type(m).is_some(), "unknown coarse member {m}");
                grouped.push(m);
            }
        }
        grouped.sort_unstable();
        assert_eq!(grouped, TYPES);
    }

    #[test]
    fn spans_parser_handles_unicode_char_offsets_escapes_and_overlaps() {
        // "Café owner José O'Neil, age 44: email jose@example.com."
        // char indices:
        // "Café" = 4 chars (5 bytes), " owner " = 7 chars -> "José" at 11..15,
        // " " = 15..16, "O'Neil" at 16..22, ", age " = 22..28, "44" at 28..30,
        // ": email " = 30..38, "jose@example.com" at 38..54.
        let text = "Café owner José O'Neil, age 44: email jose@example.com.";
        let repr = "[{'start': 11, 'end': 15, 'text': 'Jos\\xe9', 'label': 'first_name'}, \
                     {'label': \"last_name\", 'text': \"O'Neil\", 'start': 16, 'end': 22}, \
                     {'start': 28, 'end': 30, 'text': 44, 'label': 'age'}, \
                     {'start': 38, 'end': 54, 'text': 'jose@example.com', 'label': 'email'}, \
                     {'start': 38, 'end': 42, 'text': 'jose', 'label': 'user_name'}]";
        let sent = spans::to_sentence("test.parquet", 0, text, repr).unwrap().unwrap();
        let pairs: Vec<(&str, &str)> =
            sent.words.iter().zip(&sent.labels).map(|(w, l)| (w.as_str(), l.as_str())).collect();
        assert_eq!(
            pairs,
            vec![
                ("Café", "O"),
                ("owner", "O"),
                ("José", "B-first_name"),
                ("O", "B-last_name"),
                ("'", "I-last_name"),
                ("Neil", "I-last_name"),
                (",", "O"),
                ("age", "O"),
                ("44", "B-age"),
                (":", "O"),
                ("email", "O"),
                ("jose", "B-email"),
                ("@", "I-email"),
                ("example", "I-email"),
                (".", "I-email"),
                ("com", "I-email"),
                (".", "O"),
            ]
        );
    }

    #[test]
    fn spans_parser_skips_mid_token_substring_noise_and_rejects_unknown_labels() {
        let text = "Date of birth: 1990-05-01";
        let repr = "[{'start': 8, 'end': 10, 'text': 'bi', 'label': 'sexuality'}, \
                     {'start': 15, 'end': 25, 'text': '1990-05-01', 'label': 'date_of_birth'}]";
        let sent = spans::to_sentence("test.parquet", 1, text, repr).unwrap().unwrap();
        assert_eq!(sent.labels[2], "O");
        assert_eq!(sent.labels[4], "B-date_of_birth");
        assert_eq!(sent.labels[8], "I-date_of_birth");

        let bad = "[{'start': 0, 'end': 4, 'text': 'Date', 'label': 'not_a_nemotron_type'}]";
        assert!(spans::to_sentence("test.parquet", 2, text, bad).is_err());
    }
}
