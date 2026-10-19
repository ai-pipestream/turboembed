//! The parameters of one `generator` element, as
//! `AbstractXmlFeatureGeneratorFactory.init` collects them, and the typed
//! getters the factories read them with.

use super::tree::{Element, Node};
use crate::error::{Error, Result};
use crate::java::{parse_int, parse_long};

/// A parameter value. `generator` children are not here; the builder
/// keeps their elements.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Value {
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Str(String),
    Bool(bool),
}

/// The parameters by name. A name given twice keeps its first place and
/// its last value, as the factory's `LinkedHashMap` does.
#[derive(Debug, Default)]
pub(super) struct Args {
    values: Vec<(String, Value)>,
}

fn error<T>(message: String) -> Result<T> {
    Err(Error::Descriptor(message))
}

impl Args {
    /// Reads every parameter child of `element`; `generator` children are
    /// skipped (the builder reads them).
    pub fn read(element: &Element) -> Result<Args> {
        let mut args = Args::default();
        for child in element.elements() {
            let kind = child.name.as_str();
            if kind == "generator" {
                continue;
            }
            let name = child.attribute("name");
            // `(Text) elem.getFirstChild()`: Java throws when the element
            // is empty or starts with another element.
            let text = match child.children.first() {
                Some(Node::Text(t)) => t.as_str(),
                _ => return error(format!("the {kind} parameter {name:?} has no text")),
            };
            let bad = || Error::Descriptor(format!("the {kind} parameter {name:?} cannot be read from {text:?}"));
            let value = match kind {
                "int" => Value::Int(parse_int(text).ok_or_else(bad)?),
                "long" => Value::Long(parse_long(text).ok_or_else(bad)?),
                "float" => Value::Float(parse_float(text).ok_or_else(bad)? as f32),
                "double" => Value::Double(parse_float(text).ok_or_else(bad)?),
                "str" => Value::Str(text.to_owned()),
                "bool" => Value::Bool(text.eq_ignore_ascii_case("true")),
                _ => {
                    return error(
                        "child element must be one of generator, int, long, float, double, str or bool".into(),
                    );
                }
            };
            match args.values.iter_mut().find(|(k, _)| k == name) {
                Some(slot) => slot.1 = value,
                None => args.values.push((name.to_owned(), value)),
            }
        }
        Ok(args)
    }

    fn get(&self, name: &str) -> Option<&Value> {
        self.values.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    /// `getInt(name)`.
    pub fn int(&self, name: &str) -> Result<i32> {
        match self.get(name) {
            None => error(format!("parameter {name} must be set!")),
            Some(Value::Int(i)) => Ok(*i),
            Some(_) => error(format!("parameter {name} must be integer!")),
        }
    }

    /// `getInt(name, default)`.
    pub fn int_or(&self, name: &str, default: i32) -> Result<i32> {
        match self.get(name) {
            None => Ok(default),
            Some(Value::Int(i)) => Ok(*i),
            Some(_) => error(format!("parameter {name} must be integer!")),
        }
    }

    /// `getBool(name)`.
    pub fn bool(&self, name: &str) -> Result<bool> {
        match self.get(name) {
            None => error(format!("parameter {name} must be set!")),
            Some(Value::Bool(b)) => Ok(*b),
            Some(_) => error(format!("parameter {name} must be boolean!")),
        }
    }

    /// `getBool(name, default)`.
    pub fn bool_or(&self, name: &str, default: bool) -> Result<bool> {
        match self.get(name) {
            None => Ok(default),
            Some(Value::Bool(b)) => Ok(*b),
            Some(_) => error(format!("parameter {name} must be boolean!")),
        }
    }

    /// `getStr(name)`. Java's message for a value of another type says
    /// "double"; it is kept.
    pub fn str(&self, name: &str) -> Result<&str> {
        match self.get(name) {
            None => error(format!("parameter {name} must be set!")),
            Some(Value::Str(s)) => Ok(s),
            Some(_) => error(format!("parameter {name} must be double!")),
        }
    }
}

/// `Double.parseDouble` for the forms a descriptor would hold: leading
/// and trailing characters up to U+0020 are trimmed, a final `f`, `F`,
/// `d` or `D` is allowed, and `NaN` and `Infinity` with an optional sign
/// are read. Hexadecimal floating point literals are not. No factory of
/// OpenNLP reads a `float` or `double` parameter, so only whether the
/// value parses matters.
fn parse_float(text: &str) -> Option<f64> {
    let t = text.trim_matches(|c: char| c <= ' ');
    let (sign, body) = match t.strip_prefix('-') {
        Some(rest) => (-1.0, rest),
        None => (1.0, t.strip_prefix('+').unwrap_or(t)),
    };
    match body {
        "NaN" => return Some(f64::NAN),
        "Infinity" => return Some(sign * f64::INFINITY),
        _ => {}
    }
    let body = body.strip_suffix(['f', 'F', 'd', 'D']).unwrap_or(body);
    if body.is_empty() || !body.bytes().all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-')) {
        return None;
    }
    body.parse::<f64>().ok().map(|v| sign * v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_parse_the_usual_forms() {
        assert_eq!(parse_float(" 1.5 "), Some(1.5));
        assert_eq!(parse_float("2f"), Some(2.0));
        assert_eq!(parse_float("-Infinity"), Some(f64::NEG_INFINITY));
        assert!(parse_float("NaN").unwrap().is_nan());
        assert_eq!(parse_float("x"), None);
        assert_eq!(parse_float(""), None);
    }
}
