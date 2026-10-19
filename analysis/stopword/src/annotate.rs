//! The lookup as an analysis step: it reads a text view, its tokens and,
//! when the document has it, `opennlp:language`, and writes
//! `turbo:stopwords`, a flag per token.

use turbo_analysis::{Annotator, Error, Refs, View};
use turbo_analysis_layers::{Language, Tokens};
use turbo_stopword_layers::{StopwordLanguage, Stopwords};

use crate::list::StopwordList;

/// Which bundled list the step uses.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ListChoice {
    /// The same list for every document; `opennlp:language` is not
    /// consulted. The step still declares it as an optional input, so a
    /// pipeline runs language detection first and the provenance lists
    /// the language layer whenever the document has one.
    Fixed(StopwordLanguage),
    /// The list for the document's best language in `opennlp:language`,
    /// with its code resolved as `StopwordLists.forLanguage` resolves it.
    /// Where the document has no language layer, or its best language has
    /// no list, the fallback is used, and with no fallback the document is
    /// skipped (`Error::Skipped`).
    Detected { fallback: Option<StopwordLanguage> },
}

/// Flags each token that is a word of the chosen list. It never removes a
/// token.
#[derive(Copy, Clone, Debug)]
pub struct StopwordAnnotator {
    choice: ListChoice,
}

impl StopwordAnnotator {
    pub fn new(choice: ListChoice) -> StopwordAnnotator {
        StopwordAnnotator { choice }
    }

    pub fn choice(&self) -> ListChoice {
        self.choice
    }

    /// The list for one document, or why there is none.
    fn language(&self, detected: Option<&Language>) -> Result<StopwordLanguage, Error> {
        let fallback = match self.choice {
            ListChoice::Fixed(language) => return Ok(language),
            ListChoice::Detected { fallback } => fallback,
        };
        let best = detected.and_then(Language::best).map(|(code, _)| code);
        if let Some(language) = best.and_then(StopwordLanguage::from_code) {
            return Ok(language);
        }
        fallback.ok_or_else(|| {
            Error::Skipped(match best {
                Some(code) => format!("no stopword list for language {code:?}"),
                None => "no language to choose a stopword list by".into(),
            })
        })
    }
}

impl Annotator for StopwordAnnotator {
    const ID: &'static str = "turbo-stopword:bundled";
    type Reads = (View, Tokens, Option<Language>);
    type Writes = (Stopwords,);

    fn annotate(&self, (view, tokens, detected): Refs<'_, Self>) -> Result<(Stopwords,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "the tokens are in another view".into() });
        }
        let language = self.language(detected)?;
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        if starts.len() != ends.len() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("token span start count ({}) does not match end count ({})", starts.len(), ends.len()),
            });
        }
        for i in 0..starts.len() {
            let s = starts[i] as usize;
            let e = ends[i] as usize;
            if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "token {i} span {s}..{e} is not a valid UTF-8 slice of view text of length {}",
                        text.len()
                    ),
                });
            }
        }
        let mut stop = Vec::with_capacity(starts.len());
        let words = starts.iter().zip(ends).map(|(&s, &e)| &text[s as usize..e as usize]);
        StopwordList::bundled(language).flag(words, &mut stop);
        Ok((Stopwords { language, stop },))
    }

    fn config_hash(&self) -> u64 {
        // The discriminants of the choice and its language; 0xFF for no
        // fallback.
        match self.choice {
            ListChoice::Fixed(language) => 0x100 | language as u64,
            ListChoice::Detected { fallback } => 0x200 | fallback.map_or(0xFF, |l| l as u64),
        }
    }
}
