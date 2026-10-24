//! A rule that hides base records from an overlay (`opennlp.geo.Suppression`).

use std::fmt;

use turbo_unicode::java::equals_ignore_case;

use crate::entry::{InvalidRegionCode, RegionCode};
use crate::fold::Folder;
use crate::store::Entry;
use crate::text::is_blank;

/// Hides the base records of an [`OverlayGazetteer`](crate::OverlayGazetteer)
/// with a name, optionally only those of one country or one feature class.
///
/// A rule matches a record when its name and the record's canonical name or
/// one of its alternate names have the same match key (so `Sao Paulo`
/// hides `São Paulo`), its country, when it has one, is the record's, and
/// its feature class, when it has one, equals the record's ignoring case
/// (`String.equalsIgnoreCase`). A record without a country never matches a
/// rule with one, nor a record without a feature class a rule with one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Suppression {
    name: String,
    key: String,
    country: Option<RegionCode>,
    feature_class: Option<String>,
}

/// A rule that could never match, or a filter that is not well formed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SuppressionError {
    /// The name has no word token.
    Unmatchable(String),
    Country(InvalidRegionCode),
    /// The feature class filter is present and blank.
    BlankFeatureClass,
}

impl fmt::Display for SuppressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SuppressionError::Unmatchable(name) => {
                write!(f, "name must contain a word token, otherwise the rule could never match: {name}")
            }
            SuppressionError::Country(e) => e.fmt(f),
            SuppressionError::BlankFeatureClass => f.write_str("featureClass must be null when absent, not blank"),
        }
    }
}

impl std::error::Error for SuppressionError {}

impl Suppression {
    /// A rule for `name`, with an optional country filter (two ASCII
    /// letters of either case) and an optional feature class filter.
    pub fn new(
        name: &str,
        country: Option<&str>,
        feature_class: Option<&str>,
    ) -> Result<Suppression, SuppressionError> {
        let key = Folder::new().fold(name);
        if key.is_empty() {
            return Err(SuppressionError::Unmatchable(name.to_string()));
        }
        let country = country.map(RegionCode::parse).transpose().map_err(SuppressionError::Country)?;
        if feature_class.is_some_and(is_blank) {
            return Err(SuppressionError::BlankFeatureClass);
        }
        Ok(Suppression { name: name.to_string(), key, country, feature_class: feature_class.map(Into::into) })
    }

    /// A rule that hides every record with `name`.
    pub fn name_only(name: &str) -> Result<Suppression, SuppressionError> {
        Suppression::new(name, None, None)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn country(&self) -> Option<RegionCode> {
        self.country
    }

    pub fn feature_class(&self) -> Option<&str> {
        self.feature_class.as_deref()
    }

    /// Whether the rule hides `entry`. `folder` folds the record's names.
    pub fn matches(&self, entry: &Entry<'_>, folder: &mut Folder) -> bool {
        if self.country.is_some() && entry.region() != self.country {
            return false;
        }
        if let Some(class) = &self.feature_class
            && !entry.feature_class().is_some_and(|f| equals_ignore_case(class, f))
        {
            return false;
        }
        let mut key = String::new();
        std::iter::once(entry.name()).chain(entry.alternate_names()).any(|name| {
            folder.fold_into(name, &mut key);
            key == self.key
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_need_a_word_token_and_well_formed_filters() {
        assert!(matches!(Suppression::name_only(" - "), Err(SuppressionError::Unmatchable(_))));
        assert!(matches!(Suppression::new("Paris", Some("FRA"), None), Err(SuppressionError::Country(_))));
        assert_eq!(Suppression::new("Paris", Some("fr"), None).unwrap().country().unwrap().as_str(), "FR");
        assert_eq!(Suppression::new("Paris", None, Some(" ")), Err(SuppressionError::BlankFeatureClass));
    }

    #[test]
    fn feature_classes_compare_ignoring_case() {
        assert!(equals_ignore_case("city", "CITY"));
        assert!(!equals_ignore_case("city", "CITYS"));
        // U+017F LONG S uppercases to S.
        assert!(equals_ignore_case("\u{17F}", "s"));
    }
}
