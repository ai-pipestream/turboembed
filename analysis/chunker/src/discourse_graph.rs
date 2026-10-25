//! `DiscourseAnnotator`: Elementary Discourse Units ([`Edus`]) and the
//! rhetorical edges between them ([`Discourse`]).
//!
//! With `Dependencies`, a sentence splits at the clause heads below its
//! root (`advcl`, `ccomp`, `parataxis`, and a `conj` with a subject,
//! object, oblique, auxiliary or marker of its own): every token belongs
//! to its nearest clause head, and a run of tokens with one owner is a
//! unit. Each clause is linked to its governor's unit; a sentence whose
//! root unit opens with an adverbial or coordinating connective
//! (`However`, `But`) is linked to the previous sentence's root unit.
//!
//! Without dependencies, a `turbo-matcher` connective automaton cuts
//! before every subordinator (`because`, `although`, `if`), before an
//! adverbial or coordinator that follows a comma or semicolon, after a
//! semicolon, and after the first comma of a sentence that opens with a
//! subordinator. A marked unit links to the unit before it, a fronted
//! subordinate unit to the one after it, and a sentence-initial
//! adverbial to the previous unit.
//!
//! The connective decides the relation. A subordinate or adverbial
//! unit is the satellite (`Contrast` and `Joint` are multi-nuclear),
//! every edge runs from the earlier unit to the later one, and an
//! unmarked clause is an `Elaboration` (a conjunct a `Joint`). Scores:
//! 0.9 for a marked edge from dependencies, 0.7 from the fallback, 0.5
//! unmarked.

use std::collections::BTreeMap;

use turbo_analysis::{Annotator, Document, Edge, Error, Ix, Key, Refs, Span, View};
use turbo_analysis_layers::{
    Dependencies, Discourse, DiscourseEdgeLabel, DiscourseRel as R, Edus, Nuclearity, Sentences, Tokens,
};
use turbo_matcher::{Attr, Field, Input, PhraseMatcher};

/// How a connective attaches its unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Introduces a subordinate clause: `because`, `although`.
    Sub,
    /// Links to the preceding text: `however`, `therefore`.
    Adv,
    /// Coordinates: `and`, `but`, `so`.
    Coord,
}

const LEXICON: &[(R, Kind, &[&str])] = &[
    (R::Contrast, Kind::Sub, &["whereas", "while"]),
    (R::Contrast, Kind::Adv, &["however", "nevertheless", "nonetheless", "conversely", "instead", "on the other hand"]),
    (R::Contrast, Kind::Coord, &["but", "yet"]),
    (R::CauseResult, Kind::Sub, &["because", "since", "so that"]),
    (R::CauseResult, Kind::Adv, &["therefore", "consequently", "thus", "hence", "as a result", "accordingly"]),
    (R::CauseResult, Kind::Coord, &["so"]),
    (R::Condition, Kind::Sub, &["if", "unless", "provided that", "as long as", "in case"]),
    (R::Condition, Kind::Adv, &["otherwise"]),
    (R::TemporalSequence, Kind::Sub, &["when", "after", "before", "once", "until"]),
    (
        R::TemporalSequence,
        Kind::Adv,
        &["then", "subsequently", "meanwhile", "previously", "later", "afterwards", "afterward"],
    ),
    (R::Concession, Kind::Sub, &["although", "though", "even though", "even if", "despite"]),
    (R::Concession, Kind::Adv, &["still"]),
    (R::Evidence, Kind::Adv, &["for example", "for instance", "specifically", "in fact", "indeed"]),
    (R::Elaboration, Kind::Adv, &["moreover", "furthermore", "in addition", "additionally", "also"]),
    (R::Joint, Kind::Coord, &["and", "or"]),
];

/// A connective found in the text: tokens `start..end`.
#[derive(Debug, Clone, Copy)]
struct Connective {
    start: usize,
    end: usize,
    rel: R,
    kind: Kind,
}

/// Writes [`Edus`] and [`Discourse`] from sentences, tokens and, when
/// present, dependencies.
pub struct DiscourseAnnotator {
    matcher: PhraseMatcher,
}

impl Default for DiscourseAnnotator {
    fn default() -> Self {
        let mut matcher = PhraseMatcher::new(Field::Attr(Attr::Lower));
        for (i, (_, _, phrases)) in LEXICON.iter().enumerate() {
            for p in *phrases {
                let words: Vec<&str> = p.split(' ').collect();
                matcher.add(&i.to_string(), &words).expect("the built-in connectives are valid phrases");
            }
        }
        DiscourseAnnotator { matcher }
    }
}

impl std::fmt::Debug for DiscourseAnnotator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscourseAnnotator").finish_non_exhaustive()
    }
}

/// One unit while building: tokens `lo..=hi`, its sentence, the clause
/// head that owns it (dependencies only) and its opening connective.
struct Unit {
    lo: usize,
    hi: usize,
    sentence: usize,
    owner: Option<usize>,
    connective: Option<Connective>,
}

struct Cx<'a> {
    text: &'a str,
    spans: Vec<Span>,
    deps: Option<&'a Dependencies>,
    connectives: Vec<Connective>,
}

impl Cx<'_> {
    fn lower(&self, t: usize) -> String {
        let s = self.spans[t];
        self.text[s.start as usize..s.end as usize].to_lowercase()
    }

    fn punct(&self, t: usize) -> bool {
        self.lower(t).chars().all(|c| c.is_ascii_punctuation())
    }

    fn base(&self, t: usize) -> &str {
        let d = self.deps.expect("dependency mode");
        let r = d.relations.label(d.relation[t]);
        r.split(':').next().unwrap_or(r)
    }

    fn head(&self, t: usize) -> Option<usize> {
        self.deps.and_then(|d| d.head[t].map(|h| h.get()))
    }

    /// The connective opening tokens `lo..=hi`: one starting within its
    /// first two words (`, however`, `and then`).
    fn opening(&self, lo: usize, hi: usize) -> Option<Connective> {
        let mut first = lo;
        while first < hi && self.punct(first) {
            first += 1;
        }
        self.connectives.iter().find(|c| c.start >= first && c.start <= (first + 1).min(hi)).copied()
    }
}

impl DiscourseAnnotator {
    pub fn new() -> Self {
        Self::default()
    }

    /// The units and edges of the default layers of `doc`, without
    /// adding them.
    pub fn discourse(&self, doc: &Document) -> Result<(Edus, Discourse), Error> {
        self.annotate((
            doc.require(&Key::<View>::DEFAULT)?,
            doc.require(&Key::<Sentences>::DEFAULT)?,
            doc.require(&Key::<Tokens>::DEFAULT)?,
            doc.get_key(&Key::<Dependencies>::DEFAULT),
        ))
    }

    fn fail(detail: String) -> Error {
        Error::Annotator { annotator: <Self as Annotator>::ID, detail }
    }

    fn connectives(&self, text: &str, tokens: &Tokens) -> Result<Vec<Connective>, Error> {
        let input = Input::new(text, tokens).map_err(|e| Self::fail(e.to_string()))?;
        let mut found: Vec<Connective> = Vec::new();
        for m in self.matcher.find(&input).map_err(|e| Self::fail(e.to_string()))? {
            let i: usize = self.matcher.label_name(m.label).parse().expect("labels are lexicon rows");
            let (rel, kind, _) = LEXICON[i];
            found.push(Connective { start: m.tokens.start.get(), end: m.tokens.end.get(), rel, kind });
        }
        found.sort_by_key(|c| (c.start, std::cmp::Reverse(c.end)));
        let mut out: Vec<Connective> = Vec::new();
        for c in found {
            if out.last().is_none_or(|l| c.start >= l.end) {
                out.push(c);
            }
        }
        Ok(out)
    }
}

/// The clause heads of the dependency tree.
fn clause_heads(cx: &Cx<'_>) -> Vec<bool> {
    (0..cx.spans.len())
        .map(|t| match cx.head(t) {
            None => true,
            Some(_) => match cx.base(t) {
                "advcl" | "ccomp" | "parataxis" => true,
                "conj" => {
                    let d = cx.deps.expect("dependency mode");
                    (0..cx.spans.len()).any(|c| {
                        d.head[c].map(|h| h.get()) == Some(t)
                            && matches!(cx.base(c), "nsubj" | "csubj" | "expl" | "obj" | "obl" | "aux" | "cop" | "mark")
                    })
                }
                _ => false,
            },
        })
        .collect()
}

/// Splits `lo..=hi` into units at changes of `owner`, folding a
/// punctuation-only run into the unit before it (or after it, first).
fn runs(
    cx: &Cx<'_>,
    lo: usize,
    hi: usize,
    owner: impl Fn(usize) -> Option<usize>,
) -> Vec<(usize, usize, Option<usize>)> {
    let mut out: Vec<(usize, usize, Option<usize>)> = Vec::new();
    let mut pending: Option<usize> = None;
    for t in lo..=hi {
        if cx.punct(t) {
            match out.last_mut() {
                Some(last) if last.1 + 1 == t => last.1 = t,
                _ => {
                    pending.get_or_insert(t);
                }
            }
            continue;
        }
        let o = owner(t);
        match out.last_mut() {
            Some(last) if last.2 == o && last.1 + 1 == t => last.1 = t,
            _ => {
                let start = pending.take().unwrap_or(t);
                out.push((start, t, o));
            }
        }
    }
    if out.is_empty() {
        out.push((lo, hi, None));
    }
    out
}

/// The label of an edge between a unit marked by `c` and `other`, as
/// (from, to, label): from the earlier unit to the later one.
fn link(marked: usize, other: usize, c: Option<Connective>, conj: bool) -> (usize, usize, DiscourseEdgeLabel) {
    let (relation, multi) = match c {
        Some(c) => (c.rel, c.kind != Kind::Sub && matches!(c.rel, R::Contrast | R::Joint)),
        None if conj => (R::Joint, true),
        None => (R::Elaboration, false),
    };
    let nuclearity = if multi {
        Nuclearity::MultiNuclear
    } else if marked < other {
        Nuclearity::SatelliteToNucleus
    } else {
        Nuclearity::NucleusToSatellite
    };
    (marked.min(other), marked.max(other), DiscourseEdgeLabel { relation, nuclearity })
}

type Link = (usize, usize, DiscourseEdgeLabel, f64);

/// Units and links from the dependency tree.
fn by_dependencies(cx: &Cx<'_>, sentences: &[(usize, usize)]) -> (Vec<Unit>, Vec<Link>) {
    let heads = clause_heads(cx);
    let n = cx.spans.len();
    let owner = |t: usize| {
        let mut x = t;
        for _ in 0..=n {
            if heads[x] {
                return Some(x);
            }
            x = cx.head(x)?;
        }
        None
    };
    let mut units: Vec<Unit> = Vec::new();
    for (s, &(lo, hi)) in sentences.iter().enumerate() {
        for (a, b, o) in runs(cx, lo, hi, owner) {
            units.push(Unit { lo: a, hi: b, sentence: s, owner: o, connective: cx.opening(a, b) });
        }
    }
    let main = |h: usize| units.iter().position(|u| u.owner == Some(h) && u.lo <= h && h <= u.hi);
    let mut links = Vec::new();
    let mut last_root: Option<(usize, usize)> = None;
    for (i, u) in units.iter().enumerate() {
        let Some(o) = u.owner else { continue };
        if main(o) != Some(i) {
            continue;
        }
        match cx.head(o) {
            Some(h) => {
                let Some(g) = owner(h).and_then(main) else { continue };
                let conj = cx.base(o) == "conj";
                let (from, to, label) = link(i, g, u.connective, conj);
                links.push((from, to, label, if u.connective.is_some() { 0.9 } else { 0.5 }));
            }
            None => {
                if let (Some(c), Some((prev, s))) = (u.connective, last_root)
                    && c.kind != Kind::Sub
                    && s < u.sentence
                {
                    let (from, to, label) = link(i, prev, Some(c), false);
                    links.push((from, to, label, 0.9));
                }
                last_root = Some((i, u.sentence));
            }
        }
    }
    (units, links)
}

/// Units and links from connectives and punctuation alone.
fn by_connectives(cx: &Cx<'_>, sentences: &[(usize, usize)]) -> (Vec<Unit>, Vec<Link>) {
    let mut units: Vec<Unit> = Vec::new();
    let mut links = Vec::new();
    for (s, &(lo, hi)) in sentences.iter().enumerate() {
        let mut cuts: Vec<usize> = Vec::new();
        let mut first = lo;
        while first < hi && cx.punct(first) {
            first += 1;
        }
        for c in cx.connectives.iter().filter(|c| c.start > first && c.end - 1 <= hi) {
            let after_comma = matches!(cx.lower(c.start - 1).as_str(), "," | ";");
            if c.kind == Kind::Sub || after_comma {
                cuts.push(c.start);
            }
        }
        for t in lo..hi {
            if cx.lower(t) == ";" {
                cuts.push(t + 1);
            }
        }
        let fronted = cx.connectives.iter().any(|c| c.start == first && c.kind == Kind::Sub);
        if fronted && let Some(k) = (first..hi).find(|&t| cx.lower(t) == ",") {
            cuts.push(k + 1);
        }
        cuts.sort_unstable();
        cuts.dedup();
        let base = units.len();
        let mut start = lo;
        for &k in cuts.iter().chain(std::iter::once(&(hi + 1))) {
            if k > start && (start..k).any(|t| !cx.punct(t)) {
                units.push(Unit {
                    lo: start,
                    hi: k - 1,
                    sentence: s,
                    owner: None,
                    connective: cx.opening(start, k - 1),
                });
                start = k;
            }
        }
        if units.len() > base
            && start <= hi
            && let Some(last) = units.last_mut()
        {
            last.hi = hi;
        }
        let mut forward = false;
        for i in base..units.len() {
            let u = &units[i];
            let (other, score) = match u.connective {
                Some(c) if i == base && c.kind == Kind::Sub => {
                    if i + 1 >= units.len() {
                        continue;
                    }
                    forward = true;
                    (i + 1, 0.7)
                }
                Some(_) if i > 0 => (i - 1, 0.7),
                None if i > base && !(forward && i == base + 1) => (i - 1, 0.5),
                _ => continue,
            };
            let (from, to, label) = link(i, other, u.connective, false);
            links.push((from, to, label, score));
        }
    }
    (units, links)
}

impl Annotator for DiscourseAnnotator {
    const ID: &'static str = "turbo:discourse-annotator";
    const VERSION: &'static str = "1";
    type Reads = (View, Sentences, Tokens, Option<Dependencies>);
    type Writes = (Edus, Discourse);

    fn annotate(&self, (view, sentences, tokens, deps): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let n = tokens.spans.len();
        if deps.is_some_and(|d| d.head.len() != n) {
            return Err(Self::fail(format!("dependencies not aligned to the {n} tokens")));
        }
        let spans: Vec<Span> = tokens.spans.iter().map(|(_, s)| s).collect();
        let sentence_of: Vec<usize> = (0..n)
            .map(|t| match &tokens.sentence {
                Some(col) => col[t].get(),
                None => sentences
                    .spans
                    .iter()
                    .position(|(_, s)| s.start <= spans[t].start && spans[t].start < s.end)
                    .unwrap_or(0),
            })
            .collect();
        let mut ranges: Vec<(usize, usize, usize)> = Vec::new();
        for (t, &s) in sentence_of.iter().enumerate() {
            match ranges.last_mut() {
                Some(r) if r.2 == s => r.1 = t,
                _ => ranges.push((t, t, s)),
            }
        }
        let connectives = if n == 0 { Vec::new() } else { self.connectives(view.text(), tokens)? };
        let cx = Cx { text: view.text(), spans, deps, connectives };
        let bounds: Vec<(usize, usize)> = ranges.iter().map(|r| (r.0, r.1)).collect();
        let (units, links) = if deps.is_some() { by_dependencies(&cx, &bounds) } else { by_connectives(&cx, &bounds) };
        let mut edus = Edus::new(view.id());
        for u in &units {
            let span = Span::new(cx.spans[u.lo].start, cx.spans[u.hi].end);
            edus.push(span, Ix::new(ranges[u.sentence].2 as u32));
        }
        let mut rows: BTreeMap<(usize, usize, u16), f64> = BTreeMap::new();
        for (from, to, label, score) in links {
            if from != to {
                let s = rows.entry((from, to, label.tag().0)).or_insert(score);
                *s = s.max(score);
            }
        }
        let edges = Discourse::new(
            DiscourseEdgeLabel::tagset(),
            rows.into_iter()
                .map(|((f, t, k), s)| Edge::new(Ix::new(f as u32), Ix::new(t as u32), turbo_analysis::Tag(k), s)),
        );
        Ok((edus, edges))
    }
}
