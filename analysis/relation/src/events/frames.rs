//! Predicates, their dependents' roles and the anchoring of
//! [`EventExtractor`](super::EventExtractor).

use turbo_analysis::{Ix, Span};
use turbo_analysis_layers::{
    AnchoredLayerKind, ArgumentRole, Dependencies, Entities, EventArgument, EventFrame, Lemmas, Money, PosTags,
    Quantities, Temporals,
};

use super::EventConfig;

/// A mention of another layer an argument can be anchored to.
#[derive(Clone, Copy)]
pub(super) struct Anchor {
    span: Span,
    layer: AnchoredLayerKind,
    index: u32,
    /// A location entity.
    place: bool,
}

/// The mentions of the anchoring layers, in priority order.
pub(super) fn anchors(
    temporals: Option<&Temporals>,
    money: Option<&Money>,
    quantities: Option<&Quantities>,
    entities: Option<&Entities>,
) -> Vec<Anchor> {
    let mut out = Vec::new();
    let mut add = |spans: Vec<Span>, layer, place: &dyn Fn(usize) -> bool| {
        for (i, span) in spans.into_iter().enumerate() {
            out.push(Anchor { span, layer, index: i as u32, place: place(i) });
        }
    };
    if let Some(t) = temporals {
        add(t.spans.iter().map(|(_, s)| s).collect(), AnchoredLayerKind::Temporals, &|_| false);
    }
    if let Some(m) = money {
        add(m.spans.iter().map(|(_, s)| s).collect(), AnchoredLayerKind::Money, &|_| false);
    }
    if let Some(q) = quantities {
        add(q.spans.iter().map(|(_, s)| s).collect(), AnchoredLayerKind::Quantities, &|_| false);
    }
    if let Some(e) = entities {
        let place = |i: usize| {
            let label = e.kinds.label(e.kind[i]).to_ascii_lowercase();
            ["loc", "gpe", "place", "city", "country"].iter().any(|k| label.contains(k))
        };
        add(e.spans.iter().map(|(_, s)| s).collect(), AnchoredLayerKind::Names, &place);
    }
    out
}

pub(super) struct Input<'a> {
    text: &'a str,
    spans: Vec<Span>,
    deps: &'a Dependencies,
    pos: Option<&'a PosTags>,
    lemmas: Option<&'a Lemmas>,
    sentence_of: Vec<u32>,
    anchors: Vec<Anchor>,
    children: Vec<Vec<usize>>,
}

/// An extracted event before it is pushed.
pub(super) struct Frame {
    trigger: Span,
    lemma: Option<String>,
    sentence: u32,
    args: Vec<EventArgument>,
    confidence: f32,
}

impl Frame {
    pub(super) fn as_frame(&self) -> EventFrame<'_> {
        EventFrame {
            trigger: self.trigger,
            trigger_lemma: self.lemma.as_deref(),
            sentence_ix: self.sentence,
            arguments: &self.args,
            confidence: self.confidence,
        }
    }
}

const BE: [&str; 9] = ["be", "is", "are", "was", "were", "been", "being", "am", "'s"];

impl<'a> Input<'a> {
    pub(super) fn new(
        text: &'a str,
        spans: Vec<Span>,
        deps: &'a Dependencies,
        pos: Option<&'a PosTags>,
        lemmas: Option<&'a Lemmas>,
        sentence_of: Vec<u32>,
        anchors: Vec<Anchor>,
    ) -> Self {
        let mut children = vec![Vec::new(); spans.len()];
        for (t, h) in deps.head.iter().enumerate() {
            if let Some(h) = h {
                children[h.get()].push(t);
            }
        }
        Input { text, spans, deps, pos, lemmas, sentence_of, anchors, children }
    }

    fn lower(&self, t: usize) -> String {
        let s = self.spans[t];
        self.text[s.start as usize..s.end as usize].to_lowercase()
    }

    fn lemma(&self, t: usize) -> String {
        self.lemmas.map(|l| l.lemma.get(t).to_lowercase()).filter(|l| !l.is_empty()).unwrap_or_else(|| self.lower(t))
    }

    fn rel(&self, t: usize) -> &str {
        self.deps.relations.label(self.deps.relation[t])
    }

    fn base(&self, t: usize) -> &str {
        let r = self.rel(t);
        r.split(':').next().unwrap_or(r)
    }

    fn tag(&self, t: usize) -> Option<&str> {
        self.pos.map(|p| p.tagset.label(p.tags[t]))
    }

    fn has_child(&self, t: usize, f: impl Fn(&str) -> bool) -> bool {
        self.children[t].iter().any(|&c| f(self.rel(c)))
    }

    fn is_verb(&self, t: usize) -> bool {
        if matches!(self.base(t), "aux" | "cop" | "case" | "mark" | "det" | "punct") {
            return false;
        }
        match self.tag(t) {
            Some(tag) => tag == "VERB" || tag.starts_with("VB"),
            None => self.children[t]
                .iter()
                .any(|&c| matches!(self.base(c), "nsubj" | "csubj" | "obj" | "iobj" | "obl" | "aux")),
        }
    }

    fn passive(&self, t: usize) -> bool {
        self.has_child(t, |r| matches!(r, "nsubj:pass" | "aux:pass" | "obl:agent" | "csubj:pass"))
            || (self.tag(t) == Some("VBN")
                && self.children[t].iter().any(|&c| self.base(c) == "aux" && BE.contains(&self.lower(c).as_str())))
    }

    /// The first and last token of `t`'s subtree.
    fn subtree(&self, t: usize) -> (usize, usize) {
        let (mut lo, mut hi, mut stack) = (t, t, vec![t]);
        while let Some(x) = stack.pop() {
            lo = lo.min(x);
            hi = hi.max(x);
            stack.extend(&self.children[x]);
        }
        (lo, hi)
    }

    /// The function words (`case`, `mark`, `fixed`) that introduce `c`.
    fn marker(&self, c: usize) -> String {
        let mut words: Vec<usize> = Vec::new();
        for &m in &self.children[c] {
            if matches!(self.base(m), "case" | "mark") {
                words.push(m);
                words.extend(self.children[m].iter().filter(|&&f| self.base(f) == "fixed"));
            }
        }
        words.sort_unstable();
        words.iter().map(|&w| self.lower(w)).collect::<Vec<_>>().join(" ")
    }

    /// `c`'s subtree as bytes, without leading markers or trailing
    /// punctuation and possessive markers.
    fn arg_span(&self, c: usize) -> Span {
        let (mut lo, mut hi) = self.subtree(c);
        while lo < c && matches!(self.base(lo), "case" | "mark" | "punct" | "fixed" | "cc") {
            lo += 1;
        }
        while hi > c && matches!(self.base(hi), "punct" | "case") {
            hi -= 1;
        }
        Span::new(self.spans[lo].start, self.spans[hi].end)
    }

    fn inside(span: Span, outer: Span) -> bool {
        outer.start <= span.start && span.end <= outer.end
    }

    /// The first anchor (in priority order) inside `span` that covers
    /// head word `c`; a mention deeper in the subtree is left to
    /// [`governed`].
    fn anchor_of(&self, c: usize, span: Span) -> Option<Anchor> {
        let word = self.spans[c];
        self.anchors
            .iter()
            .find(|a| a.span.start <= word.start && word.end <= a.span.end && Self::inside(a.span, span))
            .copied()
    }

    /// The role of dependent `c` of predicate `p`, if it has one.
    fn role(&self, p: usize, c: usize, nominal: bool) -> Option<ArgumentRole> {
        use ArgumentRole::*;
        let passive = !nominal && self.passive(p);
        let marker = self.marker(c);
        Some(match (self.rel(c), self.base(c)) {
            ("nsubj:pass" | "csubj:pass", _) => Patient,
            (_, "nsubj" | "csubj") => {
                if passive || self.has_child(p, |r| r == "cop") {
                    Patient
                } else {
                    Agent
                }
            }
            (_, "obj") => Patient,
            (_, "iobj") => Recipient,
            ("obl:agent", _) => Agent,
            ("obl:tmod", _) => Time,
            ("obl:npmod", _) => Measure,
            ("nmod:poss", _) if nominal => Agent,
            (_, "obl") | (_, "nmod") if nominal || self.base(c) == "obl" => match marker.as_str() {
                "by" if passive || nominal => Agent,
                "by" => Measure,
                "of" if nominal => Patient,
                "to" | "for" => Recipient,
                "with" | "using" => Instrument,
                "in" | "at" | "from" | "near" | "on" | "across" | "inside" | "throughout" => Location,
                "because of" | "due to" | "owing to" | "because" => Cause,
                "during" | "after" | "before" | "since" | "until" => Time,
                _ => return None,
            },
            (_, "advcl") => match marker.as_str() {
                "because" | "since" | "as" => Cause,
                "to" | "so" | "so that" | "in order to" => Purpose,
                _ => return None,
            },
            (_, "xcomp") if marker == "in order to" => Purpose,
            (_, "advmod") if self.anchor_of(c, self.arg_span(c)).is_some() => Time,
            _ => return None,
        })
    }

    fn argument(&self, p: usize, c: usize, role: ArgumentRole) -> EventArgument {
        use ArgumentRole::*;
        let span = self.arg_span(c);
        let mut arg = EventArgument { span, role, anchored_layer: AnchoredLayerKind::None, anchored_index: None };
        if let Some(a) = self.anchor_of(c, span) {
            arg.anchored_layer = a.layer;
            arg.anchored_index = Some(a.index);
            let subject = matches!(self.base(c), "nsubj" | "csubj") && p != c;
            arg.role = match a.layer {
                AnchoredLayerKind::Temporals if !subject => Time,
                AnchoredLayerKind::Money | AnchoredLayerKind::Quantities if !subject => Measure,
                AnchoredLayerKind::Names if a.place && matches!(role, Location | Recipient | Time) => Location,
                _ => role,
            };
        }
        arg
    }
}

/// Every event in `inp`, in trigger order.
pub(super) fn extract(inp: &Input<'_>, config: &EventConfig) -> Vec<Frame> {
    let n = inp.spans.len();
    let nominal = |t: usize| {
        !inp.is_verb(t)
            && inp.tag(t).is_none_or(|g| g == "NOUN" || g.starts_with("NN"))
            && config.nominal_triggers.iter().any(|w| *w == inp.lemma(t))
    };
    let copular = |t: usize| inp.children[t].iter().copied().find(|&c| inp.base(c) == "cop");
    let predicate = |t: usize| inp.is_verb(t) || nominal(t) || copular(t).is_some();
    let is_pred: Vec<bool> = (0..n).map(predicate).collect();
    // The subject dependent each predicate has or inherits.
    let mut subject: Vec<Option<usize>> = vec![None; n];
    let mut frames = Vec::new();
    let mut by_token: Vec<Option<usize>> = vec![None; n];
    for p in 0..n {
        if !is_pred[p] {
            continue;
        }
        let nom = nominal(p);
        let mut args = Vec::new();
        for &c in &inp.children[p] {
            if let Some(role) = inp.role(p, c, nom) {
                if matches!(inp.base(c), "nsubj" | "csubj") {
                    subject[p] = Some(c);
                }
                args.push(inp.argument(p, c, role));
            }
        }
        if config.propagate_conj
            && subject[p].is_none()
            && inp.base(p) == "conj"
            && let Some(h) = inp.deps.head[p].map(Ix::get)
            && is_pred[h]
            && let Some(s) = subject[h]
        {
            subject[p] = Some(s);
            let role = if inp.passive(p) { ArgumentRole::Patient } else { ArgumentRole::Agent };
            args.push(inp.argument(p, s, role));
        }
        let trigger_token = copular(p).unwrap_or(p);
        if let Some(cop) = copular(p) {
            let (lo, hi) = (p.min(cop + 1), p);
            let lo = (lo..=hi).find(|&t| t > cop && inp.base(t) != "punct").unwrap_or(p);
            let span = Span::new(inp.spans[lo].start, inp.spans[hi].end);
            args.push(EventArgument {
                span,
                role: ArgumentRole::Attribute,
                anchored_layer: AnchoredLayerKind::None,
                anchored_index: None,
            });
        }
        if args.is_empty() {
            continue;
        }
        let core = args.iter().any(|a| matches!(a.role, ArgumentRole::Agent | ArgumentRole::Patient));
        let confidence = match (nom, core) {
            (true, _) => 0.6,
            (false, true) => 0.85,
            (false, false) => 0.6,
        };
        by_token[p] = Some(frames.len());
        frames.push(Frame {
            trigger: inp.spans[trigger_token],
            lemma: inp.lemmas.map(|_| inp.lemma(trigger_token)),
            sentence: inp.sentence_of[trigger_token],
            args,
            confidence,
        });
    }
    if !inp.anchors.is_empty() {
        governed(inp, &is_pred, &by_token, &mut frames);
    }
    for f in &mut frames {
        f.args.sort_by_key(|a| (a.span.start, a.span.end));
    }
    frames.sort_by_key(|f| (f.trigger.start, f.trigger.end));
    frames
}

/// Adds each temporal, money, quantity or location mention to the
/// nearest predicate governing it, when no argument there anchors it.
fn governed(inp: &Input<'_>, is_pred: &[bool], by_token: &[Option<usize>], frames: &mut [Frame]) {
    for a in &inp.anchors {
        if a.layer == AnchoredLayerKind::Names && !a.place {
            continue;
        }
        let Some(head) = (0..inp.spans.len()).find(|&t| {
            Input::inside(inp.spans[t], a.span)
                && inp.deps.head[t].is_none_or(|h| !Input::inside(inp.spans[h.get()], a.span))
        }) else {
            continue;
        };
        let mut at = inp.deps.head[head].map(Ix::get);
        for _ in 0..inp.spans.len() {
            match at {
                Some(p) if is_pred[p] => break,
                Some(p) => at = inp.deps.head[p].map(Ix::get),
                None => break,
            }
        }
        let Some(f) = at.and_then(|p| by_token[p]) else { continue };
        let frame = &mut frames[f];
        if frame.args.iter().any(|x| x.anchored_layer == a.layer && x.anchored_index == Some(a.index)) {
            continue;
        }
        let role = match a.layer {
            AnchoredLayerKind::Temporals => ArgumentRole::Time,
            AnchoredLayerKind::Names => ArgumentRole::Location,
            _ => ArgumentRole::Measure,
        };
        frame.args.push(EventArgument { span: a.span, role, anchored_layer: a.layer, anchored_index: Some(a.index) });
    }
}
