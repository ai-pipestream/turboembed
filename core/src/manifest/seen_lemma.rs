//! Seen-form lookup (`seen_lookup`) for `HEAD_EDIT_TREE_LEMMATIZER` (#1823).

use serde::{Deserialize, Serialize};

use super::heads::{HeadKind, HeadSpec};
use super::helpers::{fits, positive, required};
use crate::status::{Result, invalid};

/// The most entries a lemmatizer's seen-form lookup table holds.
pub const MAX_SEEN_LEMMAS: usize = 1 << 16;
/// The longest word form or lemma in a seen-form lookup table, in bytes.
pub const MAX_SEEN_FORM_BYTES: usize = 1024;

/// A lemmatizer's unambiguous seen-form lookup in front of its edit-tree
/// classifier at decode: a word form seen at least `min_freq` times in
/// training with one lemma (or, with a `pos_input`, one lemma for its tag
/// `tag`) takes that lemma, and the edit-tree classifier decides the rest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LemmaLookup {
    pub min_freq: u32,
    pub entries: Vec<SeenLemma>,
}

/// One unambiguous seen-form entry in a [`LemmaLookup`], ordered by
/// `(form, tag)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeenLemma {
    pub form: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<u32>,
    pub lemma: String,
}

impl HeadSpec {
    pub(super) fn validate_seen_lookup(&self, at: &dyn Fn(&str) -> String) -> Result<()> {
        let Some(s) = &self.seen_lookup else { return Ok(()) };
        if self.kind != HeadKind::EditTreeLemmatizer {
            return Err(invalid(format!(
                "manifest.json: {}: only a HEAD_EDIT_TREE_LEMMATIZER takes it",
                at("seen_lookup")
            )));
        }
        positive(&at("seen_lookup.min_freq"), s.min_freq)?;
        if s.entries.len() > MAX_SEEN_LEMMAS {
            return Err(invalid(format!(
                "manifest.json: {}: {} entries, at most {MAX_SEEN_LEMMAS}",
                at("seen_lookup.entries"),
                s.entries.len()
            )));
        }
        let pos_labels = self.pos_input.as_ref().map(|p| p.labels);
        for (j, e) in s.entries.iter().enumerate() {
            let field = |sub: &str| at(&format!("seen_lookup.entries[{j}].{sub}"));
            required(&field("form"), &e.form)?;
            fits(&field("form"), &e.form, MAX_SEEN_FORM_BYTES)?;
            required(&field("lemma"), &e.lemma)?;
            fits(&field("lemma"), &e.lemma, MAX_SEEN_FORM_BYTES)?;
            if let Some(t) = e.tag
                && pos_labels.is_none_or(|n| t >= n)
            {
                return Err(invalid(format!(
                    "manifest.json: {}: tag {t} requires pos_input with more than {t} labels",
                    field("tag")
                )));
            }
            if j > 0 && (s.entries[j - 1].form.as_str(), s.entries[j - 1].tag) >= (e.form.as_str(), e.tag) {
                return Err(invalid(format!(
                    "manifest.json: {}: ({:?}, {:?}) is not strictly after ({:?}, {:?})",
                    at(&format!("seen_lookup.entries[{j}]")),
                    e.form,
                    e.tag,
                    s.entries[j - 1].form,
                    s.entries[j - 1].tag
                )));
            }
        }
        Ok(())
    }
}
