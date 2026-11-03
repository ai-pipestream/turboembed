//! Checked corpus fetch-and-pin recipes (#1533): the line-based schema of
//! the recipe files under `analysis/formats/recipes`, its parser, and the
//! validation every checked-in recipe must pass.
//!
//! One recipe pins one corpus artifact: the corpus name, the canonical
//! address the bytes are fetched from, the artifact's SHA-256, the address
//! of the licence text as read at the pinned version, and the fetch tier
//! the corpus's class under docs/zoo.md maps to. Every hash and licence
//! claim in a checked-in recipe comes from an actual download at the
//! pinned address (the recipes' header comments say which run verified
//! it), never from a copied catalogue entry alone.
//!
//! The format is line based. `#` starts a comment, blank lines separate
//! blocks, and every other line is `key: value`. The required keys are
//! `name`, `tier`, `url`, `sha256` and `license-url`; `license` names the
//! licence the pinned document states, and `note` (repeatable) records
//! attribution and provenance limits. A file holds one recipe.

use std::fmt;
use std::fs;
use std::path::Path;

/// How a recipe's corpus may be fetched, the fetch side of the class
/// docs/zoo.md gives the corpus under the policy on #326.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// Permissive: fetched by default, the attribution kept.
    Default,
    /// Share-alike or otherwise acceptance-gated: fetched only after the
    /// user accepts the licence.
    AcceptFirst,
    /// Local-only: the user supplies the files and any training stays on
    /// the user's machine; nothing ships.
    UserTrained,
    /// Evaluation only, or download-only until upstream terms are
    /// settled.
    EvalOnly,
}

impl Tier {
    /// The tier a `tier:` line names.
    ///
    /// # Errors
    ///
    /// [`ParseError`] of kind [`Kind::BadTier`] when `text` names none.
    pub fn parse(text: &str) -> Result<Tier, ParseError> {
        match text {
            "default" => Ok(Tier::Default),
            "accept-first" => Ok(Tier::AcceptFirst),
            "user-trained" => Ok(Tier::UserTrained),
            "eval-only" => Ok(Tier::EvalOnly),
            _ => Err(ParseError { line: 0, kind: Kind::BadTier(text.to_owned()) }),
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Tier::Default => "default",
            Tier::AcceptFirst => "accept-first",
            Tier::UserTrained => "user-trained",
            Tier::EvalOnly => "eval-only",
        })
    }
}

/// One checked-in recipe: the pin of one corpus artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    /// The corpus name: ASCII lowercase letters, digits, `-` and `.`,
    /// the sort key of the recipe set.
    pub name: String,
    /// The fetch tier.
    pub tier: Tier,
    /// The canonical address the pinned bytes are fetched from.
    pub url: String,
    /// The artifact's SHA-256, 64 lowercase hex digits.
    pub sha256: String,
    /// The address of the licence text as read at the pinned version.
    pub license_url: String,
    /// The licence the pinned document states, when it states one.
    pub license: Option<String>,
    /// Attribution and provenance limits, in file order.
    pub notes: Vec<String>,
}

impl fmt::Display for Recipe {
    /// The canonical line form, which [`parse`] reads back to an equal
    /// recipe.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "name: {}", self.name)?;
        writeln!(f, "tier: {}", self.tier)?;
        writeln!(f, "url: {}", self.url)?;
        writeln!(f, "sha256: {}", self.sha256)?;
        writeln!(f, "license-url: {}", self.license_url)?;
        if let Some(license) = &self.license {
            writeln!(f, "license: {license}")?;
        }
        for note in &self.notes {
            writeln!(f, "note: {note}")?;
        }
        Ok(())
    }
}

/// What made a recipe text or directory unreadable, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// The 1-based line the failure is at, or 0 when not line-specific.
    pub line: u32,
    /// The failure.
    pub kind: Kind,
}

/// The failure a [`ParseError`] carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A line that is not a comment, blank or `key: value`.
    NoKey(String),
    /// A key the schema has no field for.
    UnknownKey(String),
    /// A key given twice; `note` is the only repeatable key.
    DuplicateKey(&'static str),
    /// A required key no line gave.
    MissingField(&'static str),
    /// A name that is not a letter then lowercase ASCII letters, digits
    /// and `-`.
    BadName(String),
    /// A SHA-256 that is not exactly 64 lowercase hex digits.
    BadSha256(String),
    /// An address that is not `https://` with a host and a path.
    BadUrl(&'static str, String),
    /// A licence address with no pinned version in it: no release tag, no
    /// 40-digit commit or revision, no release asset path.
    UnpinnedLicense(String),
    /// A tier name the schema has no tier for.
    BadTier(String),
    /// A name two recipes of one directory share.
    DuplicateName(String),
    /// The directory itself was unreadable.
    Io(String),
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::NoKey(line) => write!(f, "expected `key: value`, found `{line}`"),
            Kind::UnknownKey(key) => write!(f, "unknown key `{key}`"),
            Kind::DuplicateKey(key) => write!(f, "key `{key}` given twice"),
            Kind::MissingField(key) => write!(f, "required key `{key}` has no line"),
            Kind::BadName(name) => {
                write!(f, "corpus name `{name}` is not lowercase ASCII letters, digits, `-` and `.`")
            }
            Kind::BadSha256(hex) => write!(f, "SHA-256 `{hex}` is not 64 lowercase hex digits"),
            Kind::BadUrl(field, url) => write!(f, "{field} address `{url}` is not https with a host and a path"),
            Kind::UnpinnedLicense(url) => write!(f, "licence address `{url}` pins no version"),
            Kind::BadTier(tier) => write!(f, "unknown tier `{tier}`"),
            Kind::DuplicateName(name) => write!(f, "two recipes are both named `{name}`"),
            Kind::Io(message) => write!(f, "{message}"),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 { write!(f, "{}", self.kind) } else { write!(f, "line {}: {}", self.line, self.kind) }
    }
}

impl std::error::Error for ParseError {}

/// Sets one single-valued key, or errors when it is already set.
fn set(slot: &mut Option<String>, key: &'static str, value: &str, line: u32) -> Result<(), ParseError> {
    if slot.is_some() {
        return Err(ParseError { line, kind: Kind::DuplicateKey(key) });
    }
    *slot = Some(value.to_owned());
    Ok(())
}

/// Parses one recipe from `text` (the whole of one `.recipe` file).
///
/// # Errors
///
/// The first bad line, a missing required key, or a field value the
/// schema rejects.
pub fn parse(text: &str) -> Result<Recipe, ParseError> {
    let mut name = None;
    let mut tier = None;
    let mut url = None;
    let mut sha256 = None;
    let mut license_url = None;
    let mut license = None;
    let mut notes = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line = (index + 1) as u32;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let Some((key, value)) = text.split_once(':') else {
            return Err(ParseError { line, kind: Kind::NoKey(text.to_owned()) });
        };
        let (key, value) = (key.trim(), value.trim());
        match key {
            "name" => {
                if !is_name(value) {
                    return Err(ParseError { line, kind: Kind::BadName(value.to_owned()) });
                }
                set(&mut name, "name", value, line)?;
            }
            "tier" => {
                if tier.is_some() {
                    return Err(ParseError { line, kind: Kind::DuplicateKey("tier") });
                }
                tier = Some(Tier::parse(value).map_err(|mut e| {
                    e.line = line;
                    e
                })?);
            }
            "url" => {
                if !is_https_url(value) {
                    return Err(ParseError { line, kind: Kind::BadUrl("url", value.to_owned()) });
                }
                set(&mut url, "url", value, line)?;
            }
            "sha256" => {
                if !is_sha256(value) {
                    return Err(ParseError { line, kind: Kind::BadSha256(value.to_owned()) });
                }
                set(&mut sha256, "sha256", value, line)?;
            }
            "license-url" => {
                if !is_https_url(value) {
                    return Err(ParseError { line, kind: Kind::BadUrl("license-url", value.to_owned()) });
                }
                if !is_pinned(value) {
                    return Err(ParseError { line, kind: Kind::UnpinnedLicense(value.to_owned()) });
                }
                set(&mut license_url, "license-url", value, line)?;
            }
            "license" => set(&mut license, "license", value, line)?,
            "note" => notes.push(value.to_owned()),
            _ => return Err(ParseError { line, kind: Kind::UnknownKey(key.to_owned()) }),
        }
    }

    let missing = |key| ParseError { line: 0, kind: Kind::MissingField(key) };
    Ok(Recipe {
        name: name.ok_or_else(|| missing("name"))?,
        tier: tier.ok_or_else(|| missing("tier"))?,
        url: url.ok_or_else(|| missing("url"))?,
        sha256: sha256.ok_or_else(|| missing("sha256"))?,
        license_url: license_url.ok_or_else(|| missing("license-url"))?,
        license,
        notes,
    })
}

/// Parses every `.recipe` file of `dir`, sorted by corpus name so the set
/// is in a deterministic order.
///
/// # Errors
///
/// The directory listing or read, two recipes with one name, or any
/// file's first bad line.
pub fn parse_dir(dir: &Path) -> Result<Vec<Recipe>, ParseError> {
    let mut files: Vec<_> = fs::read_dir(dir)
        .map_err(|e| ParseError { line: 0, kind: Kind::Io(format!("{}: {e}", dir.display())) })?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "recipe"))
        .collect();
    files.sort();
    let mut recipes = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file)
            .map_err(|e| ParseError { line: 0, kind: Kind::Io(format!("{}: {e}", file.display())) })?;
        recipes.push(parse(&text)?);
    }
    recipes.sort_by(|a, b| a.name.cmp(&b.name));
    for pair in recipes.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(ParseError { line: 0, kind: Kind::DuplicateName(pair[0].name.clone()) });
        }
    }
    Ok(recipes)
}

/// Whether `text` is a corpus name: a letter, then lowercase ASCII
/// letters, digits, `-` and `.`.
fn is_name(text: &str) -> bool {
    let mut chars = text.chars();
    let Some(first) = chars.next() else { return false };
    first.is_ascii_lowercase() && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
}

/// Whether `text` is exactly 64 lowercase hex digits.
fn is_sha256(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Whether `text` is an `https://` address with a host and a path.
fn is_https_url(text: &str) -> bool {
    let Some(rest) = text.strip_prefix("https://") else { return false };
    match rest.find('/') {
        Some(0) | None => false,
        Some(i) => i + 1 < rest.len(),
    }
}

/// Whether `url` pins a version: a release tag path segment such as
/// `r2.16`, a run of 40 lowercase hex digits (a commit or revision), a
/// 14-digit archive snapshot timestamp, or a release asset path.
fn is_pinned(url: &str) -> bool {
    url.split('/').any(|segment| is_release_tag(segment) || is_snapshot_stamp(segment))
        || has_long_hex_run(url)
        || url.contains("/releases/download/")
}

/// Whether `segment` is an archive snapshot timestamp: 14 ASCII digits.
fn is_snapshot_stamp(segment: &str) -> bool {
    segment.len() == 14 && segment.bytes().all(|b| b.is_ascii_digit())
}

/// Whether `segment` is a release tag such as `r2.16` or `v1.0`.
fn is_release_tag(segment: &str) -> bool {
    let Some(rest) = segment.strip_prefix(|c| c == 'r' || c == 'v') else { return false };
    let Some((major, minor)) = rest.split_once('.') else { return false };
    !major.is_empty()
        && major.bytes().all(|b| b.is_ascii_digit())
        && !minor.is_empty()
        && minor.bytes().all(|b| b.is_ascii_digit())
}

/// Whether `text` has a run of at least 40 lowercase hex digits, as a
/// pinned commit or revision appears in an address.
fn has_long_hex_run(text: &str) -> bool {
    let mut run = 0;
    for b in text.bytes() {
        run = if b.is_ascii_digit() || (b'a'..=b'f').contains(&b) { run + 1 } else { 0 };
        if run >= 40 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = "\
# a comment

name: example-corpus
tier: default
url: https://example.org/data/file.txt
sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
license-url: https://example.org/repo/blob/0123456789abcdef0123456789abcdef01234567/LICENSE
license: CC BY 4.0
note: first note
note: second note
";

    fn recipe(text: &str) -> Result<Recipe, ParseError> {
        parse(text)
    }

    #[test]
    fn parses_a_full_recipe() {
        let parsed = recipe(GOOD).unwrap();
        assert_eq!(parsed.name, "example-corpus");
        assert_eq!(parsed.tier, Tier::Default);
        assert_eq!(parsed.url, "https://example.org/data/file.txt");
        assert_eq!(parsed.sha256.len(), 64);
        assert_eq!(parsed.license.as_deref(), Some("CC BY 4.0"));
        assert_eq!(parsed.notes, ["first note", "second note"]);
        assert_eq!(recipe(&parsed.to_string()).unwrap(), parsed, "display round-trips");
    }

    #[test]
    fn tiers_parse_and_print() {
        for (text, tier) in [
            ("default", Tier::Default),
            ("accept-first", Tier::AcceptFirst),
            ("user-trained", Tier::UserTrained),
            ("eval-only", Tier::EvalOnly),
        ] {
            assert_eq!(Tier::parse(text).unwrap(), tier);
            assert_eq!(tier.to_string(), text);
        }
        for bad in ["DEFAULT", "default ", "accept_first", ""] {
            assert!(matches!(Tier::parse(bad).unwrap_err().kind, Kind::BadTier(_)), "{bad}");
        }
    }

    #[test]
    fn rejects_bad_sha256() {
        for sha in [
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg",
            "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef",
        ] {
            let text = GOOD.replacen("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", sha, 1);
            assert!(matches!(recipe(&text).unwrap_err().kind, Kind::BadSha256(_)), "{sha}");
        }
    }

    #[test]
    fn rejects_bad_names() {
        for name in ["", "Example", "two words", "1abc", "under_score"] {
            let text = GOOD.replacen("name: example-corpus", &format!("name: {name}"), 1);
            assert!(matches!(recipe(&text).unwrap_err().kind, Kind::BadName(_)), "{name}");
        }
    }

    #[test]
    fn rejects_bad_urls_and_unpinned_licenses() {
        let text = GOOD.replacen("https://example.org/data/file.txt", "http://example.org/f", 1);
        assert!(matches!(recipe(&text).unwrap_err().kind, Kind::BadUrl("url", _)));
        let text = GOOD.replacen("https://example.org/data/file.txt", "https://example.org", 1);
        assert!(matches!(recipe(&text).unwrap_err().kind, Kind::BadUrl("url", _)));
        let text = GOOD.replacen(
            "https://example.org/repo/blob/0123456789abcdef0123456789abcdef01234567/LICENSE",
            "https://example.org/terms",
            1,
        );
        assert!(matches!(recipe(&text).unwrap_err().kind, Kind::UnpinnedLicense(_)));
    }

    #[test]
    fn accepts_the_pinned_forms() {
        for url in [
            "https://host/repo/blob/r2.16/README.md",
            "https://host/repo/blob/v1.0/README.md",
            "https://host/resolve/af9c13333eb981300149d5ca60a8e9d659b276b9/README.md",
            "https://github.com/org/repo/releases/download/tag/file.zip",
            "https://web.archive.org/web/20261013191918/https://publisher.example/terms",
        ] {
            assert!(is_pinned(url), "{url}");
        }
        for url in ["https://host/README.md", "https://host/tree/main/README.md", "https://host/r2/README.md"] {
            assert!(!is_pinned(url), "{url}");
        }
    }

    #[test]
    fn rejects_malformed_duplicate_and_missing_lines() {
        let text = GOOD.replacen("name: example-corpus", "name = example-corpus", 1);
        assert!(matches!(recipe(&text).unwrap_err().kind, Kind::NoKey(_)));
        let text = GOOD.replacen("tier: default", "tier: default\ntier: eval-only", 1);
        assert!(matches!(recipe(&text).unwrap_err().kind, Kind::DuplicateKey("tier")));
        let text = GOOD.replacen("url: https://example.org/data/file.txt", "fetch: x", 1);
        assert!(matches!(recipe(&text).unwrap_err().kind, Kind::UnknownKey(_)));
        for field in ["name", "tier", "url", "sha256", "license-url"] {
            let text: String =
                GOOD.lines().filter(|l| !l.starts_with(&format!("{field}:"))).collect::<Vec<_>>().join("\n");
            assert_eq!(recipe(&text).unwrap_err(), ParseError { line: 0, kind: Kind::MissingField(field) }, "{field}");
        }
    }

    #[test]
    fn errors_name_their_line() {
        let text = format!("{GOOD}bogus: x\n");
        assert_eq!(recipe(&text).unwrap_err().line, 11);
    }
}
