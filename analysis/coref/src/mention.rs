//! One candidate mention with the attributes the sieves consult
//! (`opennlp.tools.coref.Mention`).

use turbo_analysis::Span;

use crate::layer::{Animacy, Gender, MentionKind, Number, Origin, Person};
use crate::lexicon;

/// A detected mention. Attribute values are `Unknown` when the text gives
/// no evidence; an unknown attribute never blocks a link. Words, head and
/// forms are lowercased as `String.toLowerCase(Locale.ROOT)` lowercases.
#[derive(Clone, Debug)]
pub(crate) struct Mention {
    /// The envelope the mention covers in the view text.
    pub span: Span,
    /// The fragments of a discontinuous mention, `None` for a continuous one.
    pub fragments: Option<Vec<Span>>,
    pub origin: Origin,
    /// The index of the source entity, for an entity mention.
    pub entity: Option<u32>,
    /// The lowercased entity type, `None` for a mention without a known one.
    pub ty: Option<String>,
    pub sentence: usize,
    pub first: usize,
    /// The last token, inclusive.
    pub last: usize,
    pub head_token: usize,
    pub words: Vec<String>,
    pub head: String,
    /// The text from the first word through the head word.
    pub head_prefix: String,
    /// The mention text without a trailing possessive marker.
    pub normalized: String,
    pub proper: bool,
    /// Indefinite or generic, and so never anaphoric.
    pub indefinite: bool,
    pub number: Number,
    pub gender: Gender,
    pub animacy: Animacy,
    pub person: Person,
}

impl Mention {
    pub fn pronoun(&self) -> bool {
        self.origin == Origin::Pronoun
    }

    pub fn named_entity(&self) -> bool {
        self.entity.is_some()
    }

    /// Whether either span lies inside the other, the i-within-i
    /// configuration no sieve may link.
    pub fn nests(&self, other: &Mention) -> bool {
        self.span.contains(other.span) || other.span.contains(self.span)
    }

    pub fn kind(&self) -> MentionKind {
        if self.pronoun() {
            if lexicon::demonstrative(&self.head).is_some() {
                MentionKind::Demonstrative
            } else {
                MentionKind::Pronominal
            }
        } else if self.proper {
            MentionKind::Proper
        } else {
            MentionKind::Nominal
        }
    }
}
