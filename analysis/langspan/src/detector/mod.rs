//! The span detector: script runs first, then the language of each word
//! from the native n-gram model, then the best labelling of the words with
//! a cost for every switch, then the spans.

mod ngram;
mod snap;

pub use ngram::NgramDetector;

use std::sync::Arc;

use turbo_analysis::{Span, TagSet};
use turbo_analysis_layers::UNDETERMINED;
use turbo_langdetect::NgramModel;
use turbo_unicode::Script;

use self::snap::{dominant_script, gap, snap_end, snap_start};
use crate::decode::{Decoder, Gap, SwitchCost};
use crate::scripts::Candidates;
use crate::units::{Unit, units};

/// How the detector weighs a switch of language against the evidence of
/// the words. The scores are natural log probabilities from the model.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Settings {
    /// What a switch costs at each kind of gap between two words.
    pub switch: SwitchCost,
    /// How many letters a clause (words between two gaps stronger than
    /// a space) needs before its best language joins the shortlist.
    pub clause_letters: u32,
    /// How much better (in natural log probability) the best shortlisted
    /// language of a run's whole text must be than the language the
    /// labelling gave it before the run takes it. Infinite turns the
    /// rescoring off.
    pub rescore_margin: f64,
    /// Whether to build a shortlist; without one every word is labelled
    /// against all the languages of its script. Only for measuring what
    /// the shortlist does.
    pub shortlist: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            switch: SwitchCost { space: 0.20, punctuation: 0.10, sentence: 0.05 },
            clause_letters: 10,
            rescore_margin: RESCORE_MARGIN,
            shortlist: true,
        }
    }
}

/// The default [`Settings::rescore_margin`].
const RESCORE_MARGIN: f64 = 0.0;

/// One run of a single language, as [`SpanDetector::detect`] finds it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Run {
    /// The bytes of the text it covers. Runs touch: each starts where the
    /// one before it ends, the first at the first word (with any opening
    /// punctuation before it) and the last after the last word (with any
    /// closing punctuation after it).
    pub span: Span,
    /// The model's outcome index of its language, or `None` when the
    /// model has no language written in its script.
    pub outcome: Option<u32>,
    /// The script most of its letters are in.
    pub script: Script,
    /// The model's probability of the language for the run's whole text,
    /// over the languages written in its script; 1 when its script has
    /// one language, 0 when it has none.
    pub confidence: f64,
    /// How many letters its words hold.
    pub letters: u32,
}

/// Buffers for [`SpanDetector::detect`], kept between calls.
#[derive(Debug, Default)]
pub struct SpanScratch {
    probs: Vec<f64>,
    units: Vec<Unit>,
    gaps: Vec<Gap>,
    emission: Vec<f64>,
    labels: Vec<u32>,
    shortlist: Vec<u32>,
    decoder: Decoder,
    text: String,
    /// Letters per script, for a run's dominant script.
    scripts: Vec<(Script, u32)>,
    /// Runs as (first unit, end unit, outcome), before spans are cut.
    pieces: Vec<(usize, usize, Option<u32>)>,
}

impl SpanScratch {
    pub fn new() -> SpanScratch {
        SpanScratch::default()
    }
}

/// Language by span over a native [`NgramModel`].
#[derive(Debug, Clone)]
pub struct SpanDetector {
    model: NgramModel,
    settings: Settings,
    pub(crate) candidates: Candidates,
    languages: Arc<TagSet>,
    undetermined: u16,
}

impl SpanDetector {
    pub fn new(model: NgramModel, settings: Settings) -> SpanDetector {
        SpanDetector::restricted(model, settings, |_| true)
    }

    /// A detector that answers only languages `keep` accepts, for a
    /// collection known to hold only some of the model's languages. A
    /// script none of the kept languages is written in is `und`.
    pub fn restricted(model: NgramModel, settings: Settings, keep: impl Fn(&str) -> bool) -> SpanDetector {
        assert!(model.num_languages() < usize::from(u16::MAX), "more languages than a tag can hold");
        let candidates = Candidates::new(model.supported_languages(), keep);
        let mut codes: Vec<&str> = model.supported_languages().collect();
        let undetermined = match codes.iter().position(|&c| c == UNDETERMINED) {
            Some(i) => i as u16,
            None => {
                codes.push(UNDETERMINED);
                (codes.len() - 1) as u16
            }
        };
        let languages = TagSet::new("languages", &codes);
        SpanDetector { model, settings, candidates, languages, undetermined }
    }

    pub fn model(&self) -> &NgramModel {
        &self.model
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// The codes a [`LanguageSpans`](turbo_analysis_layers::LanguageSpans)
    /// layer from this detector indexes: the model's outcomes in order,
    /// then `und` unless the model has it.
    pub fn languages(&self) -> &Arc<TagSet> {
        &self.languages
    }

    /// The tag of `und` in [`languages`](Self::languages).
    pub fn undetermined(&self) -> u16 {
        self.undetermined
    }

    /// The language code of a run.
    pub fn code(&self, run: &Run) -> &str {
        match run.outcome {
            Some(o) => self.model.language(o as usize),
            None => UNDETERMINED,
        }
    }

    /// The runs of `text`, in text order, into `out` (cleared first).
    pub fn detect(
        &self,
        text: &str,
        sentence_starts: Option<&[u32]>,
        tokens: Option<(&[u32], &[u32])>,
        scratch: &mut SpanScratch,
        out: &mut Vec<Run>,
    ) {
        self.detect_inner(text, sentence_starts, tokens, scratch, out, None);
    }

    /// Text-only runs plus the uncalibrated distribution over model outcomes
    /// for each run. Unsupported scripts have an all-zero row and no outcome.
    pub fn detect_distributions(
        &self,
        text: &str,
        scratch: &mut SpanScratch,
        out: &mut Vec<Run>,
        rows: &mut Vec<Vec<f64>>,
    ) {
        rows.clear();
        self.detect_inner(text, None, None, scratch, out, Some(rows));
    }

    pub(crate) fn candidate_hash(&self) -> u64 {
        self.candidates.fingerprint()
    }

    fn eval_probs(&self, text: &str, probs: &mut Vec<f64>) -> bool {
        if text.trim().is_empty() {
            probs.clear();
            probs.resize(self.model.num_languages(), 0.0);
            return false;
        }
        let scores = self.model.scores(text);
        probs.clear();
        probs.extend(scores.into_iter().map(f64::from));
        true
    }

    fn detect_inner(
        &self,
        text: &str,
        sentence_starts: Option<&[u32]>,
        tokens: Option<(&[u32], &[u32])>,
        scratch: &mut SpanScratch,
        out: &mut Vec<Run>,
        mut rows: Option<&mut Vec<Vec<f64>>>,
    ) {
        out.clear();
        units(text, tokens, &mut scratch.units);
        let n = scratch.units.len();
        if n == 0 {
            return;
        }
        scratch.gaps.clear();
        scratch.gaps.push(Gap::Sentence);
        for w in scratch.units.windows(2) {
            scratch.gaps.push(gap(text, w[0].end, w[1].start, sentence_starts));
        }

        scratch.pieces.clear();
        let mut a = 0;
        while a < n {
            let g = scratch.units[a].group;
            let b = a + scratch.units[a..].iter().take_while(|u| u.group == g).count();
            self.segment(text, a, b, scratch);
            a = b;
        }

        let pieces = std::mem::take(&mut scratch.pieces);
        let units = &scratch.units;
        for (r, &(first, end, outcome)) in pieces.iter().enumerate() {
            let floor = if first == 0 { 0 } else { units[first - 1].end };
            let start =
                snap_start(text, units[first].start, floor, if first == 0 { None } else { sentence_starts }, tokens);
            if r > 0 {
                out.last_mut().expect("a run before this one").span.end = start;
            }
            let last = units[end - 1].end;
            let letters = units[first..end].iter().map(|u| u.letters).sum();
            let script = dominant_script(text, &units[first..end], &mut scratch.scripts);
            let (confidence, evidence) =
                self.confidence(text, units[first].start, last, units[first].group, outcome, &mut scratch.probs);
            if let Some(rows) = rows.as_deref_mut() {
                let candidates = self.candidates.of(units[first].group);
                let mut row = vec![0.0; self.model.num_languages()];
                if outcome.is_some() {
                    let p = &scratch.probs;
                    let sum: f64 = if evidence {
                        candidates.iter().map(|&i| p[i as usize]).sum()
                    } else {
                        candidates.len() as f64
                    };
                    for &i in candidates {
                        row[i as usize] = if evidence { p[i as usize] / sum } else { 1.0 / sum };
                    }
                }
                rows.push(row);
            }
            out.push(Run { span: Span::new(start, last), outcome, script, confidence, letters });
        }
        if let Some(run) = out.last_mut() {
            run.span.end = snap_end(text, run.span.end, tokens);
        }
        scratch.pieces = pieces;
    }

    fn segment(&self, text: &str, a: usize, b: usize, scratch: &mut SpanScratch) {
        let all = self.candidates.of(scratch.units[a].group);
        match all {
            [] => return scratch.pieces.push((a, b, None)),
            [only] => return scratch.pieces.push((a, b, Some(*only))),
            _ => {}
        }
        let mut shortlist = std::mem::take(&mut scratch.shortlist);
        shortlist.clear();
        if self.settings.shortlist {
            let whole = self.best(text, scratch.units[a].start, scratch.units[b - 1].end, all, &mut scratch.probs);
            shortlist.extend(whole);
            let mut first = a;
            for u in a + 1..=b {
                if u == b || scratch.gaps[u] != Gap::Space {
                    let letters: u32 = scratch.units[first..u].iter().map(|x| x.letters).sum();
                    if letters >= self.settings.clause_letters {
                        let (s, e) = (scratch.units[first].start, scratch.units[u - 1].end);
                        shortlist.extend(self.best(text, s, e, all, &mut scratch.probs));
                    }
                    first = u;
                }
            }
            shortlist.sort_unstable();
            shortlist.dedup();
        }
        if shortlist.is_empty() {
            shortlist.extend_from_slice(all);
        }
        if let [only] = shortlist[..] {
            scratch.pieces.push((a, b, Some(only)));
            scratch.shortlist = shortlist;
            return;
        }
        let k = shortlist.len();
        scratch.emission.clear();
        for u in a..b {
            let unit = scratch.units[u];
            scratch.text.clear();
            scratch.text.push(' ');
            scratch.text.push_str(&text[unit.start as usize..unit.end as usize]);
            scratch.text.push(' ');
            let evidence = self.eval_probs(&scratch.text, &mut scratch.probs);
            for &c in &shortlist {
                scratch.emission.push(if evidence { scratch.probs[c as usize].max(1e-30).ln() } else { 0.0 });
            }
        }
        scratch.decoder.decode(
            b - a,
            k,
            &scratch.emission,
            &scratch.gaps[a..b],
            &self.settings.switch,
            &mut scratch.labels,
        );
        let from = scratch.pieces.len();
        let mut first = a;
        for u in a + 1..=b {
            if u == b || scratch.labels[u - a] != scratch.labels[first - a] {
                let labelled = shortlist[scratch.labels[first - a] as usize];
                let (s, e) = (scratch.units[first].start, scratch.units[u - 1].end);
                let label = self.rescore(text, s, e, labelled, &shortlist, &mut scratch.probs);
                match scratch.pieces[from..].last_mut() {
                    Some(last) if last.2 == Some(label) => last.1 = u,
                    _ => scratch.pieces.push((first, u, Some(label))),
                }
                first = u;
            }
        }
        scratch.shortlist = shortlist;
    }

    fn rescore(&self, text: &str, start: u32, end: u32, labelled: u32, among: &[u32], probs: &mut Vec<f64>) -> u32 {
        if self.settings.rescore_margin == f64::INFINITY {
            return labelled;
        }
        let Some(best) = self.best(text, start, end, among, probs) else { return labelled };
        if probs[best as usize].max(1e-30).ln() - probs[labelled as usize].max(1e-30).ln()
            > self.settings.rescore_margin
        {
            best
        } else {
            labelled
        }
    }

    fn best(&self, text: &str, start: u32, end: u32, among: &[u32], probs: &mut Vec<f64>) -> Option<u32> {
        if !self.eval_probs(&text[start as usize..end as usize], probs) {
            return None;
        }
        let mut best = *among.first()?;
        for &c in among {
            if probs[c as usize] > probs[best as usize] {
                best = c;
            }
        }
        Some(best)
    }

    fn confidence(
        &self,
        text: &str,
        start: u32,
        end: u32,
        group: Script,
        outcome: Option<u32>,
        probs: &mut Vec<f64>,
    ) -> (f64, bool) {
        let Some(outcome) = outcome else { return (0.0, false) };
        let candidates = self.candidates.of(group);
        if candidates.len() == 1 {
            return (1.0, false);
        }
        let evidence = self.eval_probs(&text[start as usize..end as usize], probs);
        if !evidence {
            return (1.0 / candidates.len() as f64, false);
        }
        let total: f64 = candidates.iter().map(|&c| probs[c as usize]).sum();
        if total <= 0.0 {
            return (1.0 / candidates.len() as f64, false);
        }
        ((probs[outcome as usize] / total).clamp(0.0, 1.0), true)
    }
}
