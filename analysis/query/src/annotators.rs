//! Query-tuned annotator steps:
//! - [`default_span_detector`]: an in-memory [`turbo_langspan::SpanDetector`] covering
//!   major Latin languages (`deu`, `eng`, `fra`, `ita`, `nld`, `por`, `spa`)
//!   and script-distinct languages (`ara`, `ell`, `heb`, `jpn`, `kor`, `rus`, `zho`).
//! - [`GazetteerCandidateAnnotator`]: fast multi-token gazetteer phrase
//!   matcher producing candidate location entities for [`turbo_geo::GeocodeAnnotator`].
//! - [`QueryGrammarAnnotator`] and [`QueryTimeAnnotator`]: trigger-gated
//!   wrappers around [`GrammarAnnotator`] and [`TimeAnnotator`] that skip
//!   chart parsing when no grammar/time lexical triggers are present.

use std::sync::Arc;

use turbo_analysis::{Annotator, Bound, Error, Ix, IxRange, Refs, Span, Spans, View};
use turbo_analysis_layers::{
    DateTime, DocumentDate, Entities, Language, Money, Numbers, Ordinals, Quantities, Tag, TagSet, Times, Tokens,
};
use turbo_geo::{DEFAULT_LOCATION_TYPE, Entry, Gazetteer, GeoResolution, Locations, by_prior_view};
use turbo_grammar::{GRAMMAR, GrammarAnnotator, TimeAnnotator};
use turbo_stopword::Stopwords;

pub(crate) use crate::lang_model::SequentialBuilder;
use crate::lang_model::{
    COMMON_EN_TIME_PHRASES, COMMON_ES_TIME_PHRASES, EN_TIME_TRIGGERS, ES_TIME_TRIGGERS, is_loc_preposition,
    is_non_place_word, is_numeric_time_token, is_place_connector, is_time_ctx, with_lowercase,
};
pub use crate::lang_model::{QuerySpanDetector, default_span_detector};

/// Scans 1-to-4 token phrases against a [`Gazetteer`] and emits candidate
/// `"location"` entities in `opennlp:entities` for [`turbo_geo::GeocodeAnnotator`].
pub struct GazetteerCandidateAnnotator<G> {
    gazetteer: Arc<G>,
    kinds: Arc<TagSet>,
    min_lowercase_population: u64,
    config_hash: u64,
}

impl<G: Gazetteer> GazetteerCandidateAnnotator<G> {
    pub fn new(gazetteer: Arc<G>) -> Self {
        let min_lowercase_population = 50_000u64;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in std::any::type_name::<G>()
            .bytes()
            .chain(min_lowercase_population.to_le_bytes())
            .chain(gazetteer.fingerprint().to_le_bytes())
        {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        Self {
            gazetteer,
            kinds: TagSet::new("opennlp:entities", &[DEFAULT_LOCATION_TYPE]),
            min_lowercase_population,
            config_hash: h,
        }
    }

    pub fn bound(self, instance: &'static str) -> Bound<Self>
    where
        G: Send + Sync + 'static,
    {
        Bound::new(self).write(0, instance)
    }

    fn scan_candidates(
        &self,
        text: &str,
        tokens: &Tokens,
        stopwords: Option<&Stopwords>,
        mut on_match: impl FnMut(Span, IxRange<Tokens>, Vec<Entry<'_>>) -> Result<bool, Error>,
    ) -> Result<(), Error> {
        let (starts, ends) = tokens.spans.columns();
        let n = starts.len();
        // Bitflags: 1=valid, 2=non_place, 4=connector, 8=upper, 16=loc_prep, 32=checked_non_place
        let mut stack_flags = [0u8; 64];
        let mut heap_flags = Vec::new();
        let flags: &mut [u8] = if n <= 64 {
            &mut stack_flags[..n]
        } else {
            heap_flags.resize(n, 0);
            &mut heap_flags
        };
        let (mut any_upper, mut any_lower) = (false, false);
        for k in 0..n {
            let Some(tok) = text.get(starts[k] as usize..ends[k] as usize) else {
                flags[k] = 2 | 32;
                continue;
            };
            let (is_up, is_lo, valid) = if tok.is_ascii() {
                let b0 = tok.as_bytes()[0];
                let has_alpha = tok.bytes().any(|b| b.is_ascii_alphabetic());
                let has_digit = tok.bytes().any(|b| b.is_ascii_digit());
                (b0.is_ascii_uppercase(), b0.is_ascii_lowercase(), has_alpha && !has_digit)
            } else {
                let first_ch = tok.chars().next();
                (
                    first_ch.is_some_and(char::is_uppercase),
                    first_ch.is_some_and(char::is_lowercase),
                    tok.chars().any(char::is_alphabetic) && !tok.bytes().any(|b| b.is_ascii_digit()),
                )
            };
            any_upper |= is_up;
            any_lower |= is_lo;
            if !valid {
                flags[k] = 2 | 32;
                continue;
            }
            let is_sw = stopwords.is_some_and(|sw| sw.stop.get(k).copied().unwrap_or(false));
            let (conn, prep) = with_lowercase(tok, |lower| (is_place_connector(lower), is_loc_preposition(lower)));
            flags[k] = 1
                | (if is_sw || prep { 2 | 32 } else { 0 })
                | (u8::from(conn) << 2)
                | (u8::from(is_up) << 3)
                | (u8::from(prep) << 4);
        }
        let mixed_case = any_upper && any_lower;
        let is_non_place_idx = |flags: &mut [u8], k: usize| -> bool {
            if flags[k] & 32 == 0 {
                let tok = &text[starts[k] as usize..ends[k] as usize];
                let np = with_lowercase(tok, is_non_place_word);
                flags[k] |= 32 | (u8::from(np) << 1);
            }
            flags[k] & 2 != 0
        };

        let mut i = 0usize;
        while i < n {
            let fi = flags[i];
            if (fi & 1 == 0)
                || (mixed_case && (fi & 8 == 0) && !(i > 0 && (flags[i - 1] & 16 != 0)))
                || is_non_place_idx(flags, i)
            {
                i += 1;
                continue;
            }
            let max_w = 4.min(n - i);
            let mut matched = false;
            for w in (1..=max_w).rev() {
                let j = i + w;
                let fj = flags[j - 1];
                if (fj & 1 == 0)
                    || (mixed_case && w > 1 && (fj & 8 == 0))
                    || is_non_place_idx(flags, j - 1)
                    || !(i..j).all(|k| (flags[k] & 1 != 0) && (!is_non_place_idx(flags, k) || (flags[k] & 4 != 0)))
                {
                    continue;
                }
                let Some(surface) = text.get(starts[i] as usize..ends[j - 1] as usize) else { continue };
                if w == 1 && surface.chars().count() < 2 {
                    continue;
                }
                let entries = self.gazetteer.lookup(surface);
                let Some(best) = entries.first() else { continue };
                if w == 1 && (fi & 8 == 0) && best.population() < self.min_lowercase_population {
                    continue;
                }
                if on_match(Span::new(starts[i], ends[j - 1]), IxRange::new(i as u32, j as u32), entries)? {
                    i = j;
                    matched = true;
                    break;
                }
            }
            if !matched {
                i += 1;
            }
        }
        Ok(())
    }
}

impl<G: Gazetteer + Send + Sync + 'static> Annotator for GazetteerCandidateAnnotator<G> {
    const ID: &'static str = "turbo-query:gazetteer-candidates";
    type Reads = (View, Tokens, Option<Stopwords>);
    type Writes = (Entities,);

    fn annotate(&self, (view, tokens, stopwords): Refs<'_, Self>) -> Result<(Entities,), Error> {
        let mut out = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: Arc::clone(&self.kinds),
            score: Vec::new(),
        };
        self.scan_candidates(view.text(), tokens, stopwords, |span, tok_range, _entries| {
            out.spans.push(span);
            out.tokens.push(tok_range);
            out.kind.push(Tag(0));
            out.score.push(1.0);
            Ok(true)
        })?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        self.config_hash
    }
}

/// Combined single-pass query gazetteer scanner and population-prior geocoder
/// writing `turbo:locations` without a duplicate lookup pass.
pub struct QueryGeocodeAnnotator<G> {
    candidates: GazetteerCandidateAnnotator<G>,
}

impl<G: Gazetteer> QueryGeocodeAnnotator<G> {
    pub fn new(gazetteer: Arc<G>) -> Self {
        Self { candidates: GazetteerCandidateAnnotator::new(gazetteer) }
    }
}

impl<G: Gazetteer + Send + Sync + 'static> Annotator for QueryGeocodeAnnotator<G> {
    const ID: &'static str = "turbo-geo:geocode";
    type Reads = (View, Tokens, Option<Stopwords>);
    type Writes = (Locations,);

    fn annotate(&self, (view, tokens, stopwords): Refs<'_, Self>) -> Result<(Locations,), Error> {
        let mut out = Locations::new(view.id());
        self.candidates.scan_candidates(view.text(), tokens, stopwords, |span, _toks, mut entries| {
            if entries.len() > 1 {
                entries.sort_by(by_prior_view);
            }
            let confidence: f64 = if entries.len() == 1 {
                0.9
            } else {
                let a = entries[0].population() as f64;
                let b = entries[1].population() as f64;
                if a + b == 0.0 { 0.5 } else { (0.5 + 0.4 * ((a - b) / (a + b))).clamp(0.0, 1.0) }
            };
            let res = GeoResolution::new(span, entries[0].to_entry(), confidence)
                .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
            out.push(res);
            Ok(true)
        })?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        self.candidates.config_hash()
    }
}

/// Query wrapper around [`GrammarAnnotator`] with pre-warmed parser state and
/// forwarded configuration hash.
pub struct QueryGrammarAnnotator {
    inner: GrammarAnnotator,
    config_hash: u64,
}

impl QueryGrammarAnnotator {
    pub fn english() -> Self {
        let inner = GrammarAnnotator::english();
        let _ = inner.find("", turbo_analysis::ViewId::SOURCE);
        let config_hash = inner.config_hash();
        Self { inner, config_hash }
    }

    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).write(0, GRAMMAR).write(1, GRAMMAR).write(3, GRAMMAR)
    }
}

impl Annotator for QueryGrammarAnnotator {
    const ID: &'static str = "turbo-query:grammar";
    type Reads = (View, Tokens);
    type Writes = (Numbers, Quantities, Ordinals, Money);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        if !crate::model::has_grammar_trigger(view.text(), tokens) {
            return Ok((
                Numbers::new(view.id()),
                Quantities::new(view.id()),
                Ordinals::new(view.id()),
                Money::new(view.id()),
            ));
        }
        self.inner.annotate((view,))
    }

    fn config_hash(&self) -> u64 {
        self.config_hash
    }
}

fn build_common_phrases(
    inner: &TimeAnnotator,
    phrases: &'static [&'static str],
    reference: Option<DateTime>,
) -> (Box<[(u32, u32)]>, Times) {
    let mut ranges = Vec::with_capacity(phrases.len());
    let mut out = Times::new(turbo_analysis::ViewId::SOURCE);
    for &p in phrases {
        let start = out.spans.len() as u32;
        let times = inner.find(p, turbo_analysis::ViewId::SOURCE, reference);
        for (ix, _) in times.spans.iter() {
            out.push(times.get(ix));
        }
        let end = out.spans.len() as u32;
        ranges.push((start, end));
    }
    (ranges.into_boxed_slice(), out)
}

/// Query wrapper around [`TimeAnnotator`] with pre-warmed parser state and
/// forwarded configuration hash covering both the grammar language and reference [`DateTime`].
pub struct QueryTimeAnnotator {
    inner: TimeAnnotator,
    reference: Option<DateTime>,
    triggers: &'static [&'static str],
    phrases: &'static [&'static str],
    phrase_ranges: Box<[(u32, u32)]>,
    common_phrases: Times,
    is_spanish: bool,
    config_hash: u64,
}

impl QueryTimeAnnotator {
    pub fn english(reference: Option<DateTime>) -> Self {
        Self::build(TimeAnnotator::english(), reference, EN_TIME_TRIGGERS, COMMON_EN_TIME_PHRASES, false)
    }

    pub fn spanish(reference: Option<DateTime>) -> Self {
        Self::build(TimeAnnotator::spanish(), reference, ES_TIME_TRIGGERS, COMMON_ES_TIME_PHRASES, true)
    }

    fn build(
        mut inner: TimeAnnotator,
        reference: Option<DateTime>,
        triggers: &'static [&'static str],
        phrases: &'static [&'static str],
        is_spanish: bool,
    ) -> Self {
        if let Some(r) = reference {
            inner = inner.with_reference(r);
        }
        let (phrase_ranges, common_phrases) = build_common_phrases(&inner, phrases, reference);
        let salt = (triggers.len() as u64) | if is_spanish { 0x8000 } else { 0 };
        let config_hash = inner.config_hash() ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        Self { inner, reference, triggers, phrases, phrase_ranges, common_phrases, is_spanish, config_hash }
    }
}

impl Annotator for QueryTimeAnnotator {
    const ID: &'static str = "turbo-query:times";
    type Reads = (View, Tokens, Option<Language>, Option<DocumentDate>);
    type Writes = (Times,);

    fn annotate(&self, (view, tokens, lang, doc_date): Refs<'_, Self>) -> Result<(Times,), Error> {
        let text = view.text();
        if u32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the u32 offset range", text.len()),
            });
        }
        let (starts, ends) = tokens.spans.columns();
        let n = starts.len();
        if n == 0 {
            return Ok((Times::new(view.id()),));
        }
        let sole_english = self.is_spanish && lang.is_some_and(|l| l.code.len() == 1 && l.code.get(0) == "eng");
        let doc_ref = doc_date.and_then(|d| d.date).map(DateTime::midnight);
        let r = doc_ref.or(self.reference);

        let mut stack_trig = [false; 64];
        let mut heap_trig = Vec::new();
        let is_trig: &mut [bool] = if n <= 64 {
            &mut stack_trig[..n]
        } else {
            heap_trig.resize(n, false);
            &mut heap_trig
        };
        let mut any_trigger = false;
        for (idx, (&s, &e)) in starts.iter().zip(ends).enumerate() {
            let (s, e) = (s as usize, e as usize);
            let Some(tok) = text.get(s..e) else { continue };
            let has_digit = tok.bytes().any(|b| b.is_ascii_digit());
            let is_currency_num = has_digit && text.as_bytes().get(s.wrapping_sub(1)) == Some(&b'$');
            let num_time =
                !self.is_spanish && !is_currency_num && (is_numeric_time_token(text, s, e) || (n <= 6 && has_digit));
            let trig = num_time
                || (!has_digit
                    && (2..=16).contains(&tok.len())
                    && with_lowercase(tok, |l| {
                        self.triggers.binary_search(&l).is_ok()
                            && !(sole_english
                                && matches!(
                                    l,
                                    "ago"
                                        | "feb"
                                        | "hr"
                                        | "hrs"
                                        | "jul"
                                        | "jun"
                                        | "mar"
                                        | "may"
                                        | "min"
                                        | "mins"
                                        | "nov"
                                        | "oct"
                                        | "sep"
                                        | "sept"
                                ))
                    }));
            any_trigger |= trig;
            is_trig[idx] = trig;
        }
        if !any_trigger {
            return Ok((Times::new(view.id()),));
        }

        let mut windows: Vec<(usize, usize, usize, usize)> = Vec::new();
        for i in 0..n {
            if !is_trig[i] {
                continue;
            }
            let (mut lo, mut hi) = (i, i + 1);
            while lo > 0
                && i - lo < 4
                && is_time_ctx(text, starts[lo - 1] as usize, ends[lo - 1] as usize, is_trig[lo - 1])
            {
                lo -= 1;
            }
            while hi < n && hi - i <= 4 && is_time_ctx(text, starts[hi] as usize, ends[hi] as usize, is_trig[hi]) {
                hi += 1;
            }
            if let Some(last) = windows.last_mut()
                && lo <= last.1
            {
                last.1 = last.1.max(hi);
                last.3 = i + 1;
            } else {
                windows.push((lo, hi, i, i + 1));
            }
        }

        let mut out = Times::new(view.id());
        for (tok_lo, tok_hi, trig_lo, trig_hi) in windows {
            let (byte_lo, byte_hi) = (starts[tok_lo] as usize, ends[tok_hi - 1] as usize);
            let sub_text = &text[byte_lo..byte_hi];
            let cached = doc_ref
                .is_none()
                .then(|| {
                    with_lowercase(sub_text, |lower| self.phrases.binary_search(&lower).ok())
                        .map(|idx| (idx, byte_lo as u32))
                        .or_else(|| {
                            let outer_digit = (tok_lo..trig_lo).chain(trig_hi..tok_hi).any(|k| {
                                text[starts[k] as usize..ends[k] as usize].bytes().any(|b| b.is_ascii_digit())
                            });
                            (!outer_digit).then(|| {
                                let (t_lo, t_hi) = (starts[trig_lo] as usize, ends[trig_hi - 1] as usize);
                                with_lowercase(&text[t_lo..t_hi], |lower| self.phrases.binary_search(&lower).ok())
                                    .map(|idx| (idx, t_lo as u32))
                            })?
                        })
                })
                .flatten();
            if let Some((idx, offset)) = cached {
                let (p_start, p_end) = self.phrase_ranges[idx];
                for raw_ix in p_start..p_end {
                    let mut m = self.common_phrases.get(Ix::new(raw_ix));
                    m.span = Span::new(m.span.start + offset, m.span.end + offset);
                    out.push(m);
                }
                continue;
            }
            let offset = byte_lo as u32;
            let sub = self.inner.find(sub_text, view.id(), r);
            for (ix, raw_span) in sub.spans.iter() {
                let mut m = sub.get(ix);
                m.span = Span::new(raw_span.start + offset, raw_span.end + offset);
                out.push(m);
            }
        }
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        self.config_hash
    }
}
