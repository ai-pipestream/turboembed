//! Document-level country ballots (`turbo:regions`), ported from OpenNLP's
//! `RegionVote` and `DocumentRegionAnnotator.REGIONS` (`opennlp.tools.geo`).
//!
//! Each row is a [`RegionVote`] naming an ISO 3166-1 alpha-2 [`RegionCode`]
//! and its normalized vote share in `(0.0, 1.0]`, ordered by descending
//! share with ties broken by ascending country code.

use std::fmt;

use turbo_analysis::{Anchor, Layer};

/// An ISO 3166-1 alpha-2 region code, two ASCII capital letters.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionCode([u8; 2]);

/// A region code argument that is not two ASCII letters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidRegionCode(pub String);

impl fmt::Display for InvalidRegionCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "isoCountryCode must be an ISO 3166-1 alpha-2 code (two ASCII letters), got: {}", self.0)
    }
}

impl std::error::Error for InvalidRegionCode {}

impl RegionCode {
    /// Creates a region code when both bytes are ASCII capital letters.
    pub const fn new(bytes: [u8; 2]) -> Option<RegionCode> {
        if bytes[0].is_ascii_uppercase() && bytes[1].is_ascii_uppercase() { Some(RegionCode(bytes)) } else { None }
    }

    /// Parses two ASCII letters of either case, upper-casing them as
    /// `Gazetteer.byRegion` and `RegionVote` callers expect.
    pub fn parse(code: &str) -> Result<RegionCode, InvalidRegionCode> {
        match code.as_bytes() {
            &[a, b] if a.is_ascii_alphabetic() && b.is_ascii_alphabetic() => {
                Ok(RegionCode([a.to_ascii_uppercase(), b.to_ascii_uppercase()]))
            }
            _ => Err(InvalidRegionCode(code.to_string())),
        }
    }

    /// Parses a record's country code: strictly two ASCII capital letters.
    pub fn from_uppercase(code: &str) -> Option<RegionCode> {
        let bytes: [u8; 2] = code.as_bytes().try_into().ok()?;
        RegionCode::new(bytes)
    }

    /// The two-letter code as a `&str`.
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).expect("two ASCII capital letters")
    }

    /// The two ASCII capital bytes.
    pub const fn bytes(self) -> [u8; 2] {
        self.0
    }

    /// The code's index in the 26 by 26 table of two-letter codes (`0..676`).
    pub const fn index(self) -> usize {
        (self.0[0] - b'A') as usize * 26 + (self.0[1] - b'A') as usize
    }
}

impl fmt::Display for RegionCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for RegionCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

/// Why a [`RegionVote`] was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum VoteError {
    /// The country code is blank or not two ASCII letters.
    InvalidCountryCode(String),
    /// The share is not in `(0.0, 1.0]`.
    InvalidShare(f64),
}

impl fmt::Display for VoteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoteError::InvalidCountryCode(c) => {
                if c.chars().all(char::is_whitespace) {
                    f.write_str("countryCode must not be null or blank")
                } else {
                    write!(f, "countryCode must be an ISO 3166-1 alpha-2 code (two ASCII letters), got: {c}")
                }
            }
            VoteError::InvalidShare(s) => write!(f, "share must be in (0, 1], got: {s}"),
        }
    }
}

impl std::error::Error for VoteError {}

/// Why a minimum share margin was refused.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MarginError(pub f64);

impl fmt::Display for MarginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "minMargin must be in [0, 1], got: {}", self.0)
    }
}

impl std::error::Error for MarginError {}

/// A single country's share of a document-level region ballot
/// (`opennlp.tools.geo.RegionVote`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RegionVote {
    pub country_code: RegionCode,
    pub share: f64,
}

impl RegionVote {
    /// Creates a checked vote row with `share` in `(0.0, 1.0]`.
    pub fn new(country_code: RegionCode, share: f64) -> Result<RegionVote, VoteError> {
        if !(share > 0.0 && share <= 1.0) {
            return Err(VoteError::InvalidShare(share));
        }
        Ok(RegionVote { country_code, share })
    }

    /// Parses `country_code` and checks `share` in `(0.0, 1.0]`.
    pub fn parse(country_code: &str, share: f64) -> Result<RegionVote, VoteError> {
        if country_code.chars().all(char::is_whitespace) {
            return Err(VoteError::InvalidCountryCode(country_code.to_string()));
        }
        let code =
            RegionCode::parse(country_code).map_err(|_| VoteError::InvalidCountryCode(country_code.to_string()))?;
        RegionVote::new(code, share)
    }

    /// The ISO 3166-1 alpha-2 country code as a `&str`.
    pub fn country_code(&self) -> &str {
        self.country_code.as_str()
    }

    /// The typed [`RegionCode`].
    pub fn region_code(&self) -> RegionCode {
        self.country_code
    }

    /// The share of the document's region evidence, in `(0.0, 1.0]`.
    pub fn share(&self) -> f64 {
        self.share
    }
}

/// The document's region ballot (`turbo:regions`), ordered by descending
/// share with ties broken by ascending country code.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Regions {
    pub country: Vec<RegionCode>,
    pub share: Vec<f64>,
}

impl Regions {
    /// An empty ballot.
    pub fn new() -> Regions {
        Regions::default()
    }

    /// Appends one checked vote row.
    pub fn push(&mut self, vote: RegionVote) {
        self.country.push(vote.country_code);
        self.share.push(vote.share);
    }

    /// Row `i` of the ballot.
    pub fn get(&self, i: usize) -> Option<RegionVote> {
        let &country_code = self.country.get(i)?;
        let &share = self.share.get(i)?;
        Some(RegionVote { country_code, share })
    }

    /// Whether the ballot has no rows.
    pub fn is_empty(&self) -> bool {
        self.country.is_empty()
    }

    /// Iterates over the ballot rows in ranked order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = RegionVote> + '_ {
        self.country.iter().copied().zip(self.share.iter().copied()).map(|(country_code, share)| RegionVote {
            country_code,
            share,
        })
    }

    /// Returns the winning vote when the top row's share leads the runner-up
    /// (or `0.0` when the ballot has one row) by at least `min_margin`
    /// (`DocumentRegionAnnotator.winner`).
    pub fn winner(&self, min_margin: f64) -> Result<Option<RegionVote>, MarginError> {
        if !(0.0..=1.0).contains(&min_margin) {
            return Err(MarginError(min_margin));
        }
        let Some(first) = self.get(0) else {
            return Ok(None);
        };
        let runner_up = self.share.get(1).copied().unwrap_or(0.0);
        if first.share - runner_up >= min_margin { Ok(Some(first)) } else { Ok(None) }
    }
}

impl Layer for Regions {
    const ID: &'static str = "turbo:regions";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.country.len()
    }

    fn check(&self) -> Result<(), String> {
        if self.country.len() != self.share.len() {
            return Err(format!("{} countries and {} shares", self.country.len(), self.share.len()));
        }
        for (i, (&c, &s)) in self.country.iter().zip(self.share.iter()).enumerate() {
            if !(s > 0.0 && s <= 1.0) {
                return Err(format!("region vote {i} ({c}) has share {s} outside (0, 1]"));
            }
            if i > 0 {
                let prev_s = self.share[i - 1];
                let prev_c = self.country[i - 1];
                if s > prev_s || (s == prev_s && c <= prev_c) {
                    return Err(format!(
                        "region votes not in descending share / ascending country order at {i}: ({prev_c}, {prev_s}) then ({c}, {s})"
                    ));
                }
            }
            if self.country[..i].contains(&c) {
                return Err(format!("duplicate country {c} in region ballot"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_vote_validates_country_and_share_bounds() {
        for bad in ["", " ", "\u{00A0}", "A", "AUS", "12"] {
            assert!(RegionVote::parse(bad, 0.5).is_err(), "{bad:?}");
        }
        for bad_share in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(RegionVote::parse("AU", bad_share).is_err(), "{bad_share}");
        }
        assert_eq!(RegionVote::parse("AU", 1.0).unwrap().share(), 1.0);
        assert_eq!(RegionVote::parse("AU", f64::MIN_POSITIVE).unwrap().share(), f64::MIN_POSITIVE);
    }

    #[test]
    fn winner_checks_margin_and_runner_up_lead() {
        let empty = Regions::new();
        assert_eq!(empty.winner(0.0), Ok(None));

        let mut single = Regions::new();
        single.push(RegionVote::parse("AU", 1.0).unwrap());
        assert_eq!(single.winner(0.5).unwrap().unwrap().country_code(), "AU");

        let mut tie = Regions::new();
        tie.push(RegionVote::parse("FR", 0.5).unwrap());
        tie.push(RegionVote::parse("GB", 0.5).unwrap());
        assert_eq!(tie.check(), Ok(()));
        assert_eq!(tie.winner(0.0).unwrap().unwrap().country_code(), "FR");
        assert_eq!(tie.winner(0.1), Ok(None));

        let mut lead = Regions::new();
        lead.push(RegionVote::parse("MX", 0.75).unwrap());
        lead.push(RegionVote::parse("US", 0.25).unwrap());
        assert_eq!(lead.winner(0.5).unwrap().unwrap().country_code(), "MX");
        assert_eq!(lead.winner(0.6), Ok(None));

        for bad_margin in [-0.1, 1.1, f64::NAN] {
            assert!(single.winner(bad_margin).is_err(), "{bad_margin}");
        }
    }
}
