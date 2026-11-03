//! Script evidence without a statistical language model.
//!
//! A script does not identify a language. Until the language head in #1525
//! is available, runs have language `und` and zero language confidence.

use turbo_analysis::{Annotator, Error, Refs, Span, Tag, TagSet, View};
use turbo_analysis_layers::{Language, LanguageSpans, Sentences, Strs, Tokens};
use turbo_unicode::{Script, category::general_category, script::script};

/// Model-free fallback retaining exact source spans and Unicode scripts.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScriptDetector;

impl Annotator for ScriptDetector {
    const ID: &'static str = "turbo-langspan:script-only";
    type Reads = (View, Option<Sentences>, Option<Tokens>);
    type Writes = (LanguageSpans, Language);

    fn annotate(&self, (view, sentences, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let invalid = |detail| Error::Annotator { annotator: Self::ID, detail };
        if u32::try_from(view.text().len()).is_err() {
            return Err(invalid("text exceeds the u32 offset range".into()));
        }
        if let Some(s) = sentences {
            s.check_contiguous_in(view).map_err(invalid)?;
        }
        if let Some(t) = tokens {
            t.check_contiguous_in(view).map_err(invalid)?;
        }
        let mut runs = Vec::<(Span, Script)>::new();
        if let Some(tokens) = tokens {
            for (_, span) in tokens.spans.iter() {
                let mut counts = Vec::<(Script, usize)>::new();
                for c in view.text()[span.range()].chars() {
                    if let Some(s) = letter_script(c) {
                        if let Some((_, count)) = counts.iter_mut().find(|(g, _)| *g == s) {
                            *count += 1;
                        } else {
                            counts.push((s, 1));
                        }
                    }
                }
                // Strict greater-than selects the first script on a tie.
                if let Some(&(mut best, mut count)) = counts.first() {
                    for &(s, n) in &counts[1..] {
                        if n > count {
                            best = s;
                            count = n;
                        }
                    }
                    append(&mut runs, span, best);
                }
            }
        } else {
            for (offset, c) in view.text().char_indices() {
                let end = (offset + c.len_utf8()) as u32;
                if let Some(s) = letter_script(c) {
                    append(&mut runs, Span::new(offset as u32, end), s);
                } else if general_category(c).is_mark()
                    && let Some((span, _)) = runs.last_mut()
                    && span.end == offset as u32
                {
                    span.end = end;
                }
            }
        }
        let mut spans = LanguageSpans::new(view.id(), TagSet::new("languages", &["und"]));
        for (span, s) in runs {
            spans.push(span, Tag(0), s, 0.0);
        }
        let mut language = Language { code: Strs::default(), score: Vec::new() };
        if !spans.spans.is_empty() {
            language.code.push("und");
            language.score.push(0.0);
        }
        Ok((spans, language))
    }
}

pub(crate) fn letter_script(c: char) -> Option<Script> {
    if !general_category(c).is_letter() {
        return None;
    }
    match script(c) {
        Script::Common | Script::Inherited | Script::Unknown => None,
        s => Some(s),
    }
}

pub(crate) fn append(runs: &mut Vec<(Span, Script)>, span: Span, script: Script) {
    if let Some((previous, s)) = runs.last_mut()
        && *s == script
    {
        previous.end = span.end;
        return;
    }
    runs.push((span, script));
}
