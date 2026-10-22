//! What one request extracts: entity types, classification tasks and
//! structured record types, checked against named limits before anything
//! is tokenized.

use std::collections::BTreeSet;
use std::fmt;

/// How many values a field or entity type keeps.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Cardinality {
    /// At most one: the highest-scoring span.
    Single,
    /// Every span above the threshold that does not overlap a better one.
    Repeated,
}

/// A deterministic layer already on the document that a model candidate
/// can be checked against.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Source {
    Glossary,
    Pii,
    Numbers,
    Locations,
}

/// What a deterministic layer does to a candidate span.
#[derive(Clone, Debug, PartialEq)]
pub enum Check {
    /// Keep the candidate only if the layer has a span overlapping it.
    Require(Source),
    /// Add `boost` to the candidate's probability (capped at 1) when the
    /// layer has a span overlapping it, before the threshold applies.
    Boost(Source, f32),
}

/// One entity type.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityType {
    pub name: String,
    pub description: Option<String>,
    /// Overrides the request's threshold.
    pub threshold: Option<f32>,
    pub cardinality: Cardinality,
    pub checks: Vec<Check>,
}

/// One candidate label of a classification task.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub name: String,
    pub description: Option<String>,
}

/// One document classification task.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationTask {
    pub name: String,
    pub labels: Vec<Label>,
    /// Independent sigmoid scores, else one softmax over the labels.
    pub multi_label: bool,
    /// For a multi-label task, the score a label needs to be kept.
    pub threshold: f32,
    /// For a multi-label task, the most labels kept, best first.
    pub top_k: Option<usize>,
}

/// One field of a record type.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldSpec {
    pub name: String,
    pub description: Option<String>,
    pub threshold: Option<f32>,
    pub cardinality: Cardinality,
    /// A closed set of values: the field picks among them instead of
    /// pointing at a span of the text.
    pub choices: Option<Vec<String>>,
    pub checks: Vec<Check>,
}

/// One record type and its fields.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureSpec {
    pub name: String,
    pub fields: Vec<FieldSpec>,
}

/// One extraction request's schema.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ExtractionSchema {
    pub entities: Vec<EntityType>,
    pub classifications: Vec<ClassificationTask>,
    pub structures: Vec<StructureSpec>,
}

/// The named limits a schema is checked against.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_entity_types: usize,
    pub max_classification_tasks: usize,
    pub max_labels_per_task: usize,
    pub max_structures: usize,
    pub max_fields_per_structure: usize,
    /// The longest name, description or choice, in bytes.
    pub max_text_bytes: usize,
    /// The subwords the schema prompt may take of the encoder's input.
    pub max_prompt_subwords: usize,
    /// The words of text read; longer text is refused, not cut.
    pub max_words: usize,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            max_entity_types: 32,
            max_classification_tasks: 8,
            max_labels_per_task: 32,
            max_structures: 8,
            max_fields_per_structure: 16,
            max_text_bytes: 256,
            max_prompt_subwords: 384,
            max_words: 512,
        }
    }
}

/// Why a schema was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum SchemaError {
    Empty,
    /// A limit exceeded: its name, its value and what the schema held.
    OverLimit {
        limit: &'static str,
        max: usize,
        got: usize,
    },
    /// A blank name, description, label or choice, at this path.
    Blank(String),
    /// The same name twice among its siblings, at this path.
    Duplicate(String),
    /// A model control token such as `[P]` inside a string, at this path.
    Reserved(String),
    /// A threshold, boost or `top_k` outside its range, at this path.
    BadValue(String),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaError::Empty => f.write_str("an empty schema"),
            SchemaError::OverLimit { limit, max, got } => write!(f, "{limit} is {max}, the schema has {got}"),
            SchemaError::Blank(p) => write!(f, "{p} is blank"),
            SchemaError::Duplicate(p) => write!(f, "{p} is given twice"),
            SchemaError::Reserved(p) => write!(f, "{p} holds a model control token"),
            SchemaError::BadValue(p) => write!(f, "{p} is out of range"),
        }
    }
}

impl std::error::Error for SchemaError {}

/// The control tokens of a GLiNER2 prompt, which no caller string may hold.
pub const RESERVED: [&str; 11] = [
    "[P]",
    "[C]",
    "[E]",
    "[R]",
    "[L]",
    "[SEP_STRUCT]",
    "[SEP_TEXT]",
    "[DESCRIPTION]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "[MASK]",
];

fn over(limit: &'static str, max: usize, got: usize) -> Result<(), SchemaError> {
    if got > max { Err(SchemaError::OverLimit { limit, max, got }) } else { Ok(()) }
}

fn text(path: &str, s: &str, limits: &Limits) -> Result<(), SchemaError> {
    if s.trim().is_empty() {
        return Err(SchemaError::Blank(path.to_owned()));
    }
    over("max_text_bytes", limits.max_text_bytes, s.len())?;
    if RESERVED.iter().any(|r| s.contains(r)) {
        return Err(SchemaError::Reserved(path.to_owned()));
    }
    Ok(())
}

fn unique<'a>(path: &str, names: impl Iterator<Item = &'a str>) -> Result<(), SchemaError> {
    let mut seen = BTreeSet::new();
    for n in names {
        if !seen.insert(n) {
            return Err(SchemaError::Duplicate(format!("{path} {n:?}")));
        }
    }
    Ok(())
}

fn unit(path: &str, t: Option<f32>) -> Result<(), SchemaError> {
    match t {
        Some(t) if !(t.is_finite() && (0.0..=1.0).contains(&t)) => Err(SchemaError::BadValue(path.to_owned())),
        _ => Ok(()),
    }
}

fn checks(path: &str, checks: &[Check]) -> Result<(), SchemaError> {
    for c in checks {
        if let Check::Boost(_, b) = c {
            unit(&format!("{path} boost"), Some(*b))?;
        }
    }
    Ok(())
}

impl ExtractionSchema {
    /// Every limit, name and value checked, before any tokenization. Task,
    /// record and entity names share one namespace, as they share the
    /// result.
    pub fn validate(&self, limits: &Limits) -> Result<(), SchemaError> {
        if self.entities.is_empty() && self.classifications.is_empty() && self.structures.is_empty() {
            return Err(SchemaError::Empty);
        }
        over("max_entity_types", limits.max_entity_types, self.entities.len())?;
        over("max_classification_tasks", limits.max_classification_tasks, self.classifications.len())?;
        over("max_structures", limits.max_structures, self.structures.len())?;
        let tasks = self.classifications.iter().map(|c| c.name.as_str());
        unique("task or record", tasks.chain(self.structures.iter().map(|s| s.name.as_str())))?;
        // The entity types' task is named `entities` in the prompt.
        if self
            .classifications
            .iter()
            .map(|c| &c.name)
            .chain(self.structures.iter().map(|s| &s.name))
            .any(|n| n == "entities")
        {
            return Err(SchemaError::Reserved("task or record \"entities\"".into()));
        }
        unique("entity type", self.entities.iter().map(|e| e.name.as_str()))?;
        for e in &self.entities {
            let p = format!("entity {:?}", e.name);
            text(&p, &e.name, limits)?;
            if let Some(d) = &e.description {
                text(&format!("{p} description"), d, limits)?;
            }
            unit(&format!("{p} threshold"), e.threshold)?;
            checks(&p, &e.checks)?;
        }
        for c in &self.classifications {
            let p = format!("task {:?}", c.name);
            text(&p, &c.name, limits)?;
            if c.labels.is_empty() {
                return Err(SchemaError::Blank(format!("{p} labels")));
            }
            over("max_labels_per_task", limits.max_labels_per_task, c.labels.len())?;
            unique(&format!("{p} label"), c.labels.iter().map(|l| l.name.as_str()))?;
            for l in &c.labels {
                text(&format!("{p} label"), &l.name, limits)?;
                if let Some(d) = &l.description {
                    text(&format!("{p} label {:?} description", l.name), d, limits)?;
                }
            }
            unit(&format!("{p} threshold"), Some(c.threshold))?;
            if c.top_k == Some(0) {
                return Err(SchemaError::BadValue(format!("{p} top_k")));
            }
        }
        for s in &self.structures {
            let p = format!("record {:?}", s.name);
            text(&p, &s.name, limits)?;
            if s.fields.is_empty() {
                return Err(SchemaError::Blank(format!("{p} fields")));
            }
            over("max_fields_per_structure", limits.max_fields_per_structure, s.fields.len())?;
            unique(&format!("{p} field"), s.fields.iter().map(|f| f.name.as_str()))?;
            for f in &s.fields {
                let fp = format!("{p} field {:?}", f.name);
                text(&fp, &f.name, limits)?;
                if let Some(d) = &f.description {
                    text(&format!("{fp} description"), d, limits)?;
                }
                unit(&format!("{fp} threshold"), f.threshold)?;
                checks(&fp, &f.checks)?;
                if let Some(choices) = &f.choices {
                    if choices.is_empty() {
                        return Err(SchemaError::Blank(format!("{fp} choices")));
                    }
                    over("max_labels_per_task", limits.max_labels_per_task, choices.len())?;
                    unique(&format!("{fp} choice"), choices.iter().map(String::as_str))?;
                    for c in choices {
                        text(&format!("{fp} choice"), c, limits)?;
                    }
                }
            }
        }
        Ok(())
    }
}
