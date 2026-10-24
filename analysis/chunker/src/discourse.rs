//! Sentence-initial discourse markers. These small inventories were
//! authored in-tree under the repository's Apache-2.0 licence; they are
//! not copied from a third-party lexicon. They suggest a cut, not a
//! linguistic guarantee that every connective starts a new topic.

use turbo_analysis::{Annotator, Bound, Error, Layer, Refs, View};
use turbo_analysis_layers::{Boundaries, Cut, Sentences};
use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::{casefold, whitespace::WhitespaceMode};

pub const DISCOURSE: &str = "discourse";

/// Explicit inventory selection, with no English fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscourseLanguage {
    English,
    Spanish,
    Dutch,
}

impl DiscourseLanguage {
    pub fn markers(self) -> &'static [&'static str] {
        match self {
            Self::English => &[
                "however",
                "in contrast",
                "on the other hand",
                "nevertheless",
                "meanwhile",
                "first",
                "second",
                "finally",
                "in conclusion",
                "to summarize",
            ],
            Self::Spanish => &[
                "sin embargo",
                "en cambio",
                "por otro lado",
                "no obstante",
                "mientras tanto",
                "primero",
                "segundo",
                "finalmente",
                "en conclusión",
                "en resumen",
            ],
            Self::Dutch => &[
                "echter",
                "daarentegen",
                "aan de andere kant",
                "niettemin",
                "ondertussen",
                "ten eerste",
                "ten tweede",
                "ten slotte",
                "tot slot",
                "samenvattend",
            ],
        }
    }
}

/// Lowers the gap *before* a sentence whose prefix is a whole marker.
/// Case uses full Unicode folding, whitespace uses Unicode White_Space,
/// and repeated spaces within a marker collapse. Opening quotes are not
/// skipped. A zero discount disables both cost and TopicShift reason.
#[derive(Clone, Copy, Debug)]
pub struct DiscourseBoundaries {
    pub language: DiscourseLanguage,
    /// Cost units removed from the preceding boundary (1000 = one UNIT).
    pub discount: u32,
}

impl DiscourseBoundaries {
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).write(0, DISCOURSE)
    }
}

fn initial_marker(text: &str, markers: &[&str]) -> bool {
    // Only a short prefix is needed. Stop after the longest inventory
    // entry plus one scalar, preserving the boundary character too.
    let limit = markers.iter().map(|s| s.chars().count()).max().unwrap_or(0) + 1;
    let mut prefix = String::new();
    let mut count = 0;
    for c in text.chars() {
        if WhitespaceMode::Unicode.is_whitespace(c as u32) {
            if !prefix.is_empty() && !prefix.ends_with(' ') {
                prefix.push(' ');
                count += 1;
            }
        } else {
            let mut bytes = [0; 4];
            casefold::fold(c.encode_utf8(&mut bytes), &mut prefix);
            count += 1;
        }
        if count >= limit {
            break;
        }
    }
    markers.iter().any(|m| {
        prefix.strip_prefix(m).is_some_and(|rest| {
            rest.chars().next().is_none_or(|c| {
                let cat = general_category(c);
                !(cat.is_letter() || cat.is_mark() || cat.is_number() || cat == GeneralCategory::Pc)
            })
        })
    })
}

impl Annotator for DiscourseBoundaries {
    const ID: &'static str = "turbo:discourse-boundaries";
    type Reads = (View, Sentences);
    type Writes = (Boundaries,);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        sentences.check().map_err(fail)?;
        if sentences.spans.view() != view.id() {
            return Err(fail("sentences and text must share a view".into()));
        }
        let n = sentences.spans.len();
        let mut cost = vec![0; n];
        let mut reason = vec![Cut::Sentence; n];
        for (i, span) in sentences.spans.iter() {
            let text = view.text().get(span.range()).ok_or_else(|| fail("invalid sentence byte span".into()))?;
            if i.get() > 0 && self.discount > 0 && initial_marker(text, self.language.markers()) {
                cost[i.get() - 1] = (-i64::from(self.discount)).max(i64::from(i32::MIN)) as i32;
                reason[i.get() - 1] = Cut::TopicShift;
            }
        }
        Ok((Boundaries { cost, reason },))
    }

    fn config_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in self.language.markers().iter().flat_map(|s| s.bytes().chain([0])).chain(self.discount.to_le_bytes()) {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}
