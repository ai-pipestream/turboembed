//! Predicate-argument events (`turbo:events`): a trigger word with its
//! N-ary arguments, each argument with a role and, when it is a mention
//! of another layer (a temporal expression, an amount of money, a
//! quantity, a named entity), the index of that mention.

use turbo_analysis::{Document, Ix, Key, KeyId, Layer, Overlap, Span, Spans, TextSpans, ViewId};

use crate::{Entities, Money, Quantities, Sentences, Strs, Temporals};

/// What an argument does in its event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArgumentRole {
    /// The active subject (`nsubj`) or the passive agent (`by` obl).
    Agent = 0,
    /// The direct object (`obj`) or the passive subject (`nsubj:pass`).
    Patient = 1,
    /// The indirect object (`iobj`) or a `to`/`for` obl.
    Recipient = 2,
    /// A temporal expression or temporal modifier.
    Time = 3,
    /// A place: a location entity or a locative obl (`in`, `at`, `from`).
    Location = 4,
    /// An amount of money, a quantity or a numeric modifier.
    Measure = 5,
    /// A `with` or `using` obl.
    Instrument = 6,
    /// A causal clause or obl (`because`, `due to`).
    Cause = 7,
    /// A purpose clause (`to` advcl or xcomp).
    Purpose = 8,
    /// The predicative complement of a copula.
    Attribute = 9,
}

impl ArgumentRole {
    pub const ALL: [ArgumentRole; 10] = [
        ArgumentRole::Agent,
        ArgumentRole::Patient,
        ArgumentRole::Recipient,
        ArgumentRole::Time,
        ArgumentRole::Location,
        ArgumentRole::Measure,
        ArgumentRole::Instrument,
        ArgumentRole::Cause,
        ArgumentRole::Purpose,
        ArgumentRole::Attribute,
    ];

    pub fn from_byte(b: u8) -> Option<ArgumentRole> {
        Self::ALL.get(b as usize).copied()
    }
}

/// The layer an argument is a mention of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum AnchoredLayerKind {
    #[default]
    None = 0,
    /// `turbo:temporal`.
    Temporals = 1,
    /// `turbo:money`.
    Money = 2,
    /// `turbo:quantities`.
    Quantities = 3,
    /// Named entities (`opennlp:entities`).
    Names = 4,
}

impl AnchoredLayerKind {
    pub const ALL: [AnchoredLayerKind; 5] = [
        AnchoredLayerKind::None,
        AnchoredLayerKind::Temporals,
        AnchoredLayerKind::Money,
        AnchoredLayerKind::Quantities,
        AnchoredLayerKind::Names,
    ];

    pub fn from_byte(b: u8) -> Option<AnchoredLayerKind> {
        Self::ALL.get(b as usize).copied()
    }
}

/// One argument of an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventArgument {
    pub span: Span,
    pub role: ArgumentRole,
    /// [`AnchoredLayerKind::None`] exactly when `anchored_index` is `None`.
    pub anchored_layer: AnchoredLayerKind,
    /// The row of the default instance of `anchored_layer`.
    pub anchored_index: Option<u32>,
}

/// One event, as [`Events::push`] takes it and [`Events::get`] gives it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EventFrame<'a> {
    pub trigger: Span,
    pub trigger_lemma: Option<&'a str>,
    /// The row of the default `Sentences` the trigger is in.
    pub sentence_ix: u32,
    pub arguments: &'a [EventArgument],
    /// In `[0, 1]`.
    pub confidence: f32,
}

/// Events in trigger start order, by column; the arguments of event `i`
/// are `arguments[arg_end[i - 1]..arg_end[i]]`.
#[derive(Debug, Clone)]
pub struct Events {
    pub spans: Spans<Events>,
    /// The trigger's lemma; empty when unknown.
    pub lemma: Strs,
    pub sentence: Vec<u32>,
    pub confidence: Vec<f32>,
    pub arg_end: Vec<u32>,
    pub arguments: Vec<EventArgument>,
}

impl Default for Events {
    fn default() -> Self {
        Events::new(ViewId(0))
    }
}

impl PartialEq for Events {
    fn eq(&self, other: &Self) -> bool {
        self.spans.view() == other.spans.view() && self.iter().eq(other.iter())
    }
}

impl Events {
    pub fn new(view: ViewId) -> Events {
        Events {
            spans: Spans::new(view),
            lemma: Strs::default(),
            sentence: Vec::new(),
            confidence: Vec::new(),
            arg_end: Vec::new(),
            arguments: Vec::new(),
        }
    }

    pub fn push(&mut self, f: EventFrame<'_>) -> Ix<Events> {
        self.lemma.push(f.trigger_lemma.unwrap_or(""));
        self.sentence.push(f.sentence_ix);
        self.confidence.push(f.confidence);
        self.arguments.extend_from_slice(f.arguments);
        self.arg_end.push(self.arguments.len() as u32);
        self.spans.push(f.trigger)
    }

    /// Event `i`.
    ///
    /// # Panics
    ///
    /// When `i` is out of bounds.
    pub fn get(&self, i: usize) -> EventFrame<'_> {
        let start = if i == 0 { 0 } else { self.arg_end[i - 1] as usize };
        let lemma = self.lemma.get(i);
        EventFrame {
            trigger: self.spans.get(Ix::new(i as u32)),
            trigger_lemma: (!lemma.is_empty()).then_some(lemma),
            sentence_ix: self.sentence[i],
            arguments: &self.arguments[start..self.arg_end[i] as usize],
            confidence: self.confidence[i],
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = EventFrame<'_>> {
        (0..self.spans.len()).map(|i| self.get(i))
    }
}

impl Layer for Events {
    const ID: &'static str = "turbo:events";
    const OVERLAP: Overlap = Overlap::Any;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let mut out = Vec::new();
        if let Some(bound) = self.sentence.iter().map(|&i| (i as usize).saturating_add(1)).max() {
            out.push((Key::<Sentences>::DEFAULT.id(), bound));
        }
        for (kind, key) in [
            (AnchoredLayerKind::Temporals, Key::<Temporals>::DEFAULT.id()),
            (AnchoredLayerKind::Money, Key::<Money>::DEFAULT.id()),
            (AnchoredLayerKind::Quantities, Key::<Quantities>::DEFAULT.id()),
            (AnchoredLayerKind::Names, Key::<Entities>::DEFAULT.id()),
        ] {
            if let Some(bound) = self
                .arguments
                .iter()
                .filter(|a| a.anchored_layer == kind)
                .filter_map(|a| a.anchored_index)
                .map(|i| (i as usize).saturating_add(1))
                .max()
            {
                out.push((key, bound));
            }
        }
        out
    }

    fn check_document(&self, doc: &Document, _on: &[KeyId]) -> Result<(), String> {
        let text = doc.text(self.spans.view());
        for a in &self.arguments {
            if text.get(a.span.range()).is_none() {
                return Err(format!(
                    "event argument {}..{} outside view or off UTF-8 boundaries",
                    a.span.start, a.span.end
                ));
            }
        }
        Ok(())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let lens = [self.lemma.len(), self.sentence.len(), self.confidence.len(), self.arg_end.len()];
        if lens.iter().any(|&l| l != n) {
            return Err(format!("columns of lengths {lens:?} for {n} events"));
        }
        if self.arg_end.windows(2).any(|w| w[0] > w[1])
            || self.arg_end.last().map_or(0, |&e| e as usize) != self.arguments.len()
        {
            return Err("argument ends must rise to the argument count".into());
        }
        self.spans.check(Self::OVERLAP)?;
        if !self.spans.is_start_ordered() {
            return Err("events must be in trigger start order".into());
        }
        for f in self.iter() {
            if f.trigger.is_empty() {
                return Err(format!("an empty trigger at {}", f.trigger.start));
            }
            if !(0.0..=1.0).contains(&f.confidence) {
                return Err(format!("event confidence {} outside [0, 1]", f.confidence));
            }
            for a in f.arguments {
                if a.span.is_empty() || a.span.start > a.span.end {
                    return Err(format!("an empty argument on the event at {}", f.trigger.start));
                }
                if (a.anchored_layer == AnchoredLayerKind::None) != a.anchored_index.is_none() {
                    return Err(format!("an anchor without a layer or index on the event at {}", f.trigger.start));
                }
            }
        }
        Ok(())
    }
}
