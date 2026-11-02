//! Running a bundle's word heads over the features its encoder gives,
//! the same from every caller: the token tagger's labels and their
//! probabilities, the biaffine parser's trees, the edit-tree
//! lemmatizer's lemmas, and the windows a long sequence is read in
//! (`plan`). The analysis annotators (`turbo-tagger`) and the in-process
//! analyzer of the C interface both decode through here, so a head gives
//! one answer whichever asks.

use std::borrow::Cow;

use super::{BiaffineParser, EditTreeLemmatizer, TokenTagger, lemmatizer, parser, tagger};
use crate::status::{Error, INVALID_ARGUMENT, Result};

/// What a sequence decoded to: each word's label and the softmax
/// probability of that label among the word's own scores.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub tags: Vec<u32>,
    pub prob: Vec<f64>,
}

/// One sentence parsed: each word's head (0 the root, else the word's
/// position from 1) and relation label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parse {
    pub heads: Vec<u32>,
    pub relations: Vec<u32>,
}

/// A lemma lookup that overrides a lemmatizer head where it has an answer.
pub trait LemmaDictionary: Send + Sync {
    /// The lemma listed for `form` under `tag` (the tag head's label, None
    /// when the lemmatizer reads no tags), or None to keep the head's.
    fn lemma(&self, form: &str, tag: Option<&str>) -> Option<String>;
}

impl<F: Fn(&str, Option<&str>) -> Option<String> + Send + Sync> LemmaDictionary for F {
    fn lemma(&self, form: &str, tag: Option<&str>) -> Option<String> {
        self(form, tag)
    }
}

/// Whether a head's layer id names the layer `id`. A head's id is always
/// `turboembed:<name>`; until the analysis layer ids move to that
/// namespace too, the names after the namespaces are compared.
pub fn same_layer(head: &str, id: &str) -> bool {
    let name = |s: &str| s.rsplit_once(':').map_or(s, |(_, n)| n).to_owned();
    name(head) == name(id)
}

/// Keeps the last `want` of the `have` layer taps of each sequence's
/// `[have, words, hidden]` features: what a head reading fewer layers
/// than the encoder gave takes.
pub fn tail_taps(features: &[Vec<f32>], have: usize, want: usize) -> Cow<'_, [Vec<f32>]> {
    if have == want {
        return Cow::Borrowed(features);
    }
    Cow::Owned(features.iter().map(|f| f[f.len() - (f.len() / have) * want..].to_vec()).collect())
}

/// Each sequence decoded by `tagger` from its word features `[taps *
/// words * hidden]` and its words' text, which a head with a `hash_embed`
/// reads.
pub fn decode_tagger(tagger: &TokenTagger, features: &[Vec<f32>], forms: &[Vec<&str>]) -> Result<Vec<Decoded>> {
    let (mut scores, mut work) = (Vec::new(), tagger::Work::default());
    let labels = tagger.labels().len();
    let stride = tagger.taps() * tagger.hidden();
    let mut out = Vec::with_capacity(features.len());
    for (f, forms) in features.iter().zip(forms) {
        let n = f.len() / stride;
        let words: Vec<u32> = (0..n as u32).collect();
        tagger.scores_forms(f, &words, Some(forms), &mut scores, &mut work)?;
        let mut tags = Vec::new();
        tagger.decode(&scores, n, &mut tags)?;
        let prob = tags
            .iter()
            .enumerate()
            .map(|(w, &t)| {
                let row = &scores[w * labels..][..labels];
                let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
                let z: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
                (row[t as usize] as f64 - m).exp() / z
            })
            .collect();
        out.push(Decoded { tags, prob });
    }
    Ok(out)
}

/// Each sequence parsed by `parser` from its word features.
pub fn parse_sequences(parser: &BiaffineParser, features: &[Vec<f32>]) -> Result<Vec<Parse>> {
    let (mut arcs, mut scores, mut work) = (Vec::new(), Vec::new(), parser::Work::default());
    let stride = parser.taps() * parser.hidden();
    let mut out = Vec::with_capacity(features.len());
    for f in features {
        let ids: Vec<u32> = (0..(f.len() / stride) as u32).collect();
        let n = parser.arcs(f, &ids, &mut arcs, &mut work)?;
        let mut heads = Vec::new();
        parser.decode(&arcs, n, &mut heads)?;
        parser.relations(&heads, &mut scores, &work)?;
        let mut relations = Vec::new();
        parser.relation_labels(&scores, n, &mut relations)?;
        out.push(Parse { heads, relations });
    }
    Ok(out)
}

/// The tags the lemmatizer `lem` reads for each sequence, when its
/// `pos_input` names a tagger: `given` as they are, else decoded by
/// `pos_tagger` from `pos` (the tagger's features, or the lemmatizer's
/// own when both pool words alike and read the same layer taps).
pub struct LemmaTags<'a> {
    pub pos_tagger: Option<&'a TokenTagger>,
    pub pos: Option<&'a [Vec<f32>]>,
    pub given: Option<&'a [Vec<u32>]>,
}

/// Each sequence's lemmas by `lem`, from its features `head`, `forms` its
/// words' text; `dictionary` over the head where it lists a lemma.
pub fn lemmatize_sequences(
    lem: &EditTreeLemmatizer,
    tags: LemmaTags<'_>,
    dictionary: Option<&dyn LemmaDictionary>,
    head: &[Vec<f32>],
    forms: &[Vec<&str>],
) -> Result<Vec<Vec<String>>> {
    let stride = lem.taps() * lem.hidden();
    let (mut scores, mut decoded, mut trees) = (Vec::new(), Vec::new(), Vec::new());
    let (mut tw, mut lw) = (tagger::Work::default(), lemmatizer::Work::default());
    let mut out = Vec::with_capacity(head.len());
    for (s, f) in head.iter().enumerate() {
        let n = f.len() / stride;
        let ids: Vec<u32> = (0..n as u32).collect();
        let words = &forms[s];
        if words.len() != n {
            return Err(Error::new(INVALID_ARGUMENT, format!("{} forms for {n} words", words.len())));
        }
        let tagged: Option<&[u32]> = match (tags.pos_tagger, tags.given) {
            (Some(_), Some(given)) => Some(&given[s]),
            (Some(t), None) => {
                let pf = tags.pos.map_or(f, |p| &p[s]);
                // A hashed tag head reads each word's text too.
                t.scores_forms(pf, &ids, Some(words), &mut scores, &mut tw)?;
                t.decode(&scores, n, &mut decoded)?;
                Some(&decoded[..])
            }
            (None, _) => None,
        };
        lem.scores(f, &ids, tagged, &mut scores, &mut lw)?;
        lem.decode(&scores, words, &mut trees)?;
        let mut lemmas = lem.lemmas(&trees, words)?;
        if let Some(d) = dictionary {
            let labels = tags.pos_tagger.map(TokenTagger::labels);
            for (k, lemma) in lemmas.iter_mut().enumerate() {
                let tag = labels.zip(tagged).map(|(l, t)| l[t[k] as usize].as_str());
                if let Some(listed) = d.lemma(words[k], tag) {
                    *lemma = listed;
                }
            }
        }
        out.push(lemmas);
    }
    Ok(out)
}

/// The windows a sequence is read in whose words have `lens` pieces each
/// (every one at least 1 and at most `width`), as words `a..b`, and the
/// window each word's features come from. A sequence whose pieces fit
/// one window is read in one; a longer one in windows of whole words,
/// each starting about half a window after the last, and each word's
/// features come from the window where it has the most pieces of context
/// on its nearer side (the earlier on a tie).
pub fn plan(lens: &[usize], width: usize) -> (Vec<(usize, usize)>, Vec<usize>) {
    let n = lens.len();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut a = 0;
    while a < n {
        let (mut b, mut used) = (a, 0);
        while b < n && used + lens[b] <= width {
            used += lens[b];
            b += 1;
        }
        spans.push((a, b));
        if b == n {
            break;
        }
        // The next window starts at the first word of the second half.
        let (mut next, mut seen) = (a, 0);
        while next < b && seen < used / 2 {
            seen += lens[next];
            next += 1;
        }
        a = next.max(a + 1);
    }
    // A word's context in a window: the pieces on its nearer side, a
    // sequence's own ends counting as all the context there is.
    let context = |&(a, b): &(usize, usize), w: usize| -> usize {
        let before = if a == 0 { usize::MAX } else { lens[a..w].iter().sum() };
        let after = if b == n { usize::MAX } else { lens[w + 1..b].iter().sum() };
        before.min(after)
    };
    let owner = (0..n)
        .map(|w| {
            let mut best: Option<(usize, usize)> = None;
            for (i, span) in spans.iter().enumerate() {
                if span.0 <= w && w < span.1 {
                    let c = context(span, w);
                    if best.is_none_or(|(_, bc)| c > bc) {
                        best = Some((i, c));
                    }
                }
            }
            best.map_or(0, |(i, _)| i)
        })
        .collect();
    (spans, owner)
}
