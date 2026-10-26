//! Read spaCy token patterns and EntityRuler/SpanRuler JSON or JSONL rule
//! files. Parsing is separate from adding rules: a malformed file never
//! partially mutates a matcher. Phrase rules retain their text for the
//! caller's tokenizer. Unsupported attributes/operators are named errors.
mod json;
mod token;
use crate::{Matcher, PatternError, PatternId, Tok};
use json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportError {
    Json { byte: usize },
    Schema { field: String },
    Unsupported { operator: String },
    Pattern(PatternError),
}
impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json { byte } => write!(f, "invalid pattern JSON at byte {byte}"),
            Self::Schema { field } => write!(f, "invalid spaCy pattern field {field}"),
            Self::Unsupported { operator } => write!(f, "unsupported spaCy operator or attribute {operator}"),
            Self::Pattern(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for ImportError {}
fn invalid(field: &str) -> ImportError {
    ImportError::Schema { field: field.into() }
}
fn unsupported(operator: &str) -> ImportError {
    ImportError::Unsupported { operator: operator.into() }
}
fn string<'a>(v: &'a Value, field: &str) -> Result<&'a str, ImportError> {
    if let Value::String(s) = v { Ok(s) } else { Err(invalid(field)) }
}
fn object<'a>(v: &'a Value, field: &str) -> Result<&'a [(String, Value)], ImportError> {
    if let Value::Object(o) = v { Ok(o) } else { Err(invalid(field)) }
}
fn strings(v: &Value, field: &str) -> Result<Box<[Box<str>]>, ImportError> {
    let Value::Array(v) = v else {
        return Err(invalid(field));
    };
    v.iter().map(|v| string(v, field).map(Into::into)).collect()
}
fn unsigned(v: &Value, field: &str) -> Result<u32, ImportError> {
    let Value::Number(n) = v else {
        return Err(invalid(field));
    };
    n.parse().map_err(|_| invalid(field))
}

/// One token pattern, the JSON array passed to spaCy `Matcher.add`.
pub fn read_pattern(text: &str) -> Result<Vec<Tok>, ImportError> {
    if text.len() > 16 * 1024 * 1024 {
        return Err(invalid("file exceeds 16 MiB"));
    }
    pattern(&json::parse(text)?)
}
fn pattern(value: &Value) -> Result<Vec<Tok>, ImportError> {
    let Value::Array(tokens) = value else {
        return Err(invalid("pattern"));
    };
    let toks: Vec<Tok> = tokens.iter().map(token::read).collect::<Result<_, _>>()?;
    // Use the compiler's state/repeat bounds before returning any rules.
    Matcher::new().add("validation", &toks).map_err(ImportError::Pattern)?;
    Ok(toks)
}

#[derive(Debug, Clone)]
pub enum RulePattern {
    Tokens(Vec<Tok>),
    /// Tokenize with the same tokenizer as the input before adding it to
    /// a PhraseMatcher or SpanRuler. Splitting on spaces is not equivalent.
    Phrase(String),
}
#[derive(Debug, Clone)]
pub struct Rule {
    pub label: String,
    pub id: Option<String>,
    pub pattern: RulePattern,
}
impl Rule {
    /// Add a token pattern. Phrase rules need the caller's tokenizer.
    /// `id` remains metadata on this rule; Matcher returns a PatternId.
    pub fn add_to(&self, matcher: &mut Matcher) -> Result<PatternId, ImportError> {
        match &self.pattern {
            RulePattern::Tokens(p) => matcher.add(&self.label, p).map_err(ImportError::Pattern),
            RulePattern::Phrase(_) => Err(unsupported("phrase pattern requires tokenization")),
        }
    }
}

/// JSON arrays of rules or JSONL, as exported by spaCy's rulers. Rules
/// accept `label`, `pattern` and optional `id`; all other keys refuse.
pub fn read_rules(text: &str) -> Result<Vec<Rule>, ImportError> {
    if text.len() > 16 * 1024 * 1024 {
        return Err(invalid("file exceeds 16 MiB"));
    }
    let values = if text.trim_start_matches([' ', '\r', '\n', '\t']).starts_with('[') {
        let Value::Array(v) = json::parse(text)? else { unreachable!() };
        v
    } else {
        let mut values = Vec::new();
        for line in text.lines().filter(|s| !s.trim_matches([' ', '\r', '\t']).is_empty()) {
            values.push(json::parse(line)?);
        }
        values
    };
    values.iter().map(rule).collect()
}
fn rule(value: &Value) -> Result<Rule, ImportError> {
    let fields = object(value, "rule")?;
    let mut label = None;
    let mut id = None;
    let mut body = None;
    for (key, value) in fields {
        match key.as_str() {
            "label" => label = Some(string(value, key)?.to_owned()),
            "id" => id = Some(string(value, key)?.to_owned()),
            "pattern" => {
                body = Some(match value {
                    Value::String(s) => RulePattern::Phrase(s.clone()),
                    _ => RulePattern::Tokens(pattern(value)?),
                })
            }
            _ => return Err(unsupported(key)),
        }
    }
    Ok(Rule { label: label.ok_or_else(|| invalid("label"))?, id, pattern: body.ok_or_else(|| invalid("pattern"))? })
}

mod dependency;
pub use dependency::read_dependency_pattern;
