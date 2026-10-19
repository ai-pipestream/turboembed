//! `opennlp.tools.util.TrainingParameters`: the named settings a trainer
//! and an indexer read, and the algorithms and trainer types they name.
//!
//! OpenNLP keeps the parameters in a `TreeMap<String, Object>` ordered by
//! `String.CASE_INSENSITIVE_ORDER`, so `Cutoff` and `cutoff` are one key,
//! and a value is whatever Java object was put: a `String`, an `Integer`,
//! a `Double` or a `Boolean`. Each getter casts to its type and, when the
//! value is a `String`, parses it; a value of any other type is a
//! `ClassCastException`. [`TrainingParameters`] keeps those rules with a
//! [`Value`] per entry, and reads a parameters file the way the
//! `InputStream` constructor does (`java.util.Properties`, every value a
//! string).
//!
//! A namespaced parameter is the key `namespace.key`; `parameters(ns)`
//! picks the entries of one namespace, as `getParameters` does.

mod factory;
#[cfg(test)]
mod tests;

use std::cmp::Ordering;
use std::fmt;

use crate::error::{Error, Result};
use turbo_jvm::text;
use turbo_maxent::Properties;
use turbo_unicode::java::{to_lower_char, to_upper_char};

pub use self::factory::{Algorithm, TrainerType, is_valid, trainer_type};

/// The parameter names OpenNLP defines, with their spelling: the map is
/// case-insensitive, but a serialized file keeps the spelling put.
pub struct Key;

impl Key {
    /// `Parameters.ALGORITHM_PARAM`.
    pub const ALGORITHM: &'static str = "Algorithm";
    /// `Parameters.TRAINER_TYPE_PARAM`.
    pub const TRAINER_TYPE: &'static str = "TrainerType";
    /// `Parameters.ITERATIONS_PARAM`.
    pub const ITERATIONS: &'static str = "Iterations";
    /// `Parameters.CUTOFF_PARAM`.
    pub const CUTOFF: &'static str = "Cutoff";
    /// `Parameters.THREADS_PARAM`.
    pub const THREADS: &'static str = "Threads";
    /// `AbstractEventTrainer.DATA_INDEXER_PARAM`.
    pub const DATA_INDEXER: &'static str = "DataIndexer";
    /// `AbstractDataIndexer.SORT_PARAM`.
    pub const SORT: &'static str = "sort";
    /// `GISTrainer.LOG_LIKELIHOOD_THRESHOLD_PARAM`.
    pub const LL_THRESHOLD: &'static str = "LLThreshold";
    /// The perceptron trainer's tolerance, read by `IterDeltaAccuracyUnderTolerance`.
    pub const TOLERANCE: &'static str = "Tolerance";
    /// `GISTrainer`'s `Smoothing`.
    pub const SMOOTHING: &'static str = "Smoothing";
    /// `GISTrainer`'s `SmoothingObservation`.
    pub const SMOOTHING_OBSERVATION: &'static str = "SmoothingObservation";
    /// `GISTrainer`'s `GaussianSmoothing`.
    pub const GAUSSIAN_SMOOTHING: &'static str = "GaussianSmoothing";
    /// `GISTrainer`'s `GaussianSmoothingSigma`.
    pub const GAUSSIAN_SMOOTHING_SIGMA: &'static str = "GaussianSmoothingSigma";
    /// `QNTrainer.L1COST_PARAM`.
    pub const L1_COST: &'static str = "L1Cost";
    /// `QNTrainer.L2COST_PARAM`.
    pub const L2_COST: &'static str = "L2Cost";
    /// `QNTrainer.M_PARAM`: the corrections L-BFGS keeps.
    pub const NUM_OF_UPDATES: &'static str = "NumOfUpdates";
    /// `QNTrainer.MAX_FCT_EVAL_PARAM`.
    pub const MAX_FCT_EVAL: &'static str = "MaxFctEval";
    /// `PerceptronTrainer`'s `UseAverage`.
    pub const USE_AVERAGE: &'static str = "UseAverage";
    /// `PerceptronTrainer`'s `UseSkippedAveraging`.
    pub const USE_SKIPPED_AVERAGING: &'static str = "UseSkippedAveraging";
    /// `PerceptronTrainer`'s `StepSizeDecrease`.
    pub const STEP_SIZE_DECREASE: &'static str = "StepSizeDecrease";
}

/// `Parameters.ITERATIONS_DEFAULT_VALUE`.
pub const DEFAULT_ITERATIONS: i32 = 100;
/// `Parameters.CUTOFF_DEFAULT_VALUE`.
pub const DEFAULT_CUTOFF: i32 = 5;
/// `GISTrainer.LOG_LIKELIHOOD_THRESHOLD_DEFAULT`.
pub const DEFAULT_LL_THRESHOLD: f64 = 0.0001;
/// `PerceptronTrainer.TOLERANCE_DEFAULT`.
pub const DEFAULT_TOLERANCE: f64 = 0.00001;

/// A parameter's value: the Java object OpenNLP holds.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Int(i32),
    Double(f64),
    Bool(bool),
}

impl Value {
    /// The Java class of the value, for messages.
    pub fn java_class(&self) -> &'static str {
        match self {
            Value::Str(_) => "String",
            Value::Int(_) => "Integer",
            Value::Double(_) => "Double",
            Value::Bool(_) => "Boolean",
        }
    }

    /// `String.valueOf` of the value: the text a report or a manifest
    /// holds.
    pub fn java_string(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::Int(i) => i.to_string(),
            Value::Double(d) => text::double_to_string(*d),
            Value::Bool(b) => b.to_string(),
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Str(s.to_owned())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::Str(s)
    }
}

impl From<i32> for Value {
    fn from(i: i32) -> Value {
        Value::Int(i)
    }
}

impl From<f64> for Value {
    fn from(d: f64) -> Value {
        Value::Double(d)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}

/// `String.CASE_INSENSITIVE_ORDER` as JDK 21 computes it: code point by
/// code point, two code points equal when they are, or their upper case
/// forms are, or the lower case forms of those are; then the shorter
/// string first.
pub fn case_insensitive_compare(a: &str, b: &str) -> Ordering {
    let mut ca = a.chars();
    let mut cb = b.chars();
    loop {
        match (ca.next(), cb.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) => {
                if x == y {
                    continue;
                }
                let (ux, uy) = (to_upper_char(x), to_upper_char(y));
                if ux == uy {
                    continue;
                }
                let (lx, ly) = (to_lower_char(ux), to_lower_char(uy));
                if lx != ly {
                    return lx.cmp(&ly);
                }
            }
        }
    }
}

/// `Parameters.getKey(namespace, key)`: `namespace.key`, or `key` alone.
fn namespaced(namespace: Option<&str>, key: &str) -> String {
    match namespace {
        Some(ns) => format!("{ns}.{key}"),
        None => key.to_owned(),
    }
}

/// The parameters, ordered as the Java map iterates them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainingParameters {
    /// Sorted by [`case_insensitive_compare`] on the key; a key is stored
    /// with the spelling it was first put with.
    entries: Vec<(String, Value)>,
}

impl TrainingParameters {
    /// No parameters.
    pub fn new() -> TrainingParameters {
        TrainingParameters::default()
    }

    /// `TrainingParameters.defaultParams()`: `Algorithm=MAXENT`,
    /// `TrainerType=Event`, `Iterations=100` and `Cutoff=5`, the two
    /// numbers as integers.
    pub fn defaults() -> TrainingParameters {
        let mut p = TrainingParameters::new();
        p.put(Key::ALGORITHM, Algorithm::Maxent.name());
        p.put(Key::TRAINER_TYPE, TrainerType::Event.value());
        p.put(Key::ITERATIONS, DEFAULT_ITERATIONS);
        p.put(Key::CUTOFF, DEFAULT_CUTOFF);
        p
    }

    /// `new TrainingParameters(InputStream)`: a Java properties file, every
    /// value a string.
    pub fn load(bytes: &[u8]) -> Result<TrainingParameters> {
        Ok(TrainingParameters::from_properties(&Properties::load(bytes)?))
    }

    /// The entries of a loaded properties file, every value a string.
    pub fn from_properties(properties: &Properties) -> TrainingParameters {
        let mut p = TrainingParameters::new();
        for (k, v) in properties.iter() {
            p.put(k, v);
        }
        p
    }

    fn position(&self, key: &str) -> std::result::Result<usize, usize> {
        self.entries.binary_search_by(|(k, _)| case_insensitive_compare(k, key))
    }

    /// `put(key, value)`: sets the value, keeping the spelling of a key
    /// already present (as `TreeMap.put` keeps its key).
    pub fn put(&mut self, key: &str, value: impl Into<Value>) {
        match self.position(key) {
            Ok(i) => self.entries[i].1 = value.into(),
            Err(i) => self.entries.insert(i, (key.to_owned(), value.into())),
        }
    }

    /// `put(namespace, key, value)`.
    pub fn put_in(&mut self, namespace: &str, key: &str, value: impl Into<Value>) {
        self.put(&namespaced(Some(namespace), key), value);
    }

    /// `putIfAbsent(key, value)`.
    pub fn put_if_absent(&mut self, key: &str, value: impl Into<Value>) {
        if let Err(i) = self.position(key) {
            self.entries.insert(i, (key.to_owned(), value.into()));
        }
    }

    /// `putIfAbsent(namespace, key, value)`.
    pub fn put_if_absent_in(&mut self, namespace: &str, key: &str, value: impl Into<Value>) {
        self.put_if_absent(&namespaced(Some(namespace), key), value);
    }

    /// `put` of a string, for callers that hold `&str` values.
    pub fn put_str(&mut self, key: &str, value: &str) {
        self.put(key, value);
    }

    pub fn put_int(&mut self, key: &str, value: i32) {
        self.put(key, value);
    }

    pub fn put_double(&mut self, key: &str, value: f64) {
        self.put(key, value);
    }

    pub fn put_bool(&mut self, key: &str, value: bool) {
        self.put(key, value);
    }

    /// The value under a key, whatever its type.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.position(key).ok().map(|i| &self.entries[i].1)
    }

    /// The value under `namespace.key`.
    pub fn get_in(&self, namespace: &str, key: &str) -> Option<&Value> {
        self.get(&namespaced(Some(namespace), key))
    }

    /// `getStringParameter`: the value when it is a string; `None` when
    /// absent (the caller's default); an error when it is another type.
    pub fn get_str(&self, key: &str) -> Result<Option<&str>> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Str(s)) => Ok(Some(s)),
            Some(other) => {
                Err(Error::ParameterType { key: key.to_owned(), expected: "String", found: other.java_class() })
            }
        }
    }

    /// `getIntParameter`: an integer, or a string read with
    /// `Integer.parseInt`; `None` when absent.
    pub fn get_int(&self, key: &str) -> Result<Option<i32>> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Int(i)) => Ok(Some(*i)),
            Some(Value::Str(s)) => text::parse_int(s).map(Some).map_err(|_| Error::ParameterFormat {
                key: key.to_owned(),
                expected: "Integer",
                value: s.clone(),
            }),
            Some(other) => {
                Err(Error::ParameterType { key: key.to_owned(), expected: "Integer", found: other.java_class() })
            }
        }
    }

    /// `getDoubleParameter`: a double, or a string read with
    /// `Double.parseDouble`; `None` when absent.
    pub fn get_double(&self, key: &str) -> Result<Option<f64>> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Double(d)) => Ok(Some(*d)),
            Some(Value::Str(s)) => text::parse_double(s).map(Some).map_err(|_| Error::ParameterFormat {
                key: key.to_owned(),
                expected: "Double",
                value: s.clone(),
            }),
            Some(other) => {
                Err(Error::ParameterType { key: key.to_owned(), expected: "Double", found: other.java_class() })
            }
        }
    }

    /// `getBooleanParameter`: a boolean, or a string read with
    /// `Boolean.parseBoolean` (true only for `true` in any case); `None`
    /// when absent.
    pub fn get_bool(&self, key: &str) -> Result<Option<bool>> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(Value::Str(s)) => Ok(Some(text::parse_boolean(s))),
            Some(other) => {
                Err(Error::ParameterType { key: key.to_owned(), expected: "Boolean", found: other.java_class() })
            }
        }
    }

    /// `getStringParameter(namespace, key, ...)`.
    pub fn get_str_in(&self, namespace: &str, key: &str) -> Result<Option<&str>> {
        self.get_str(&namespaced(Some(namespace), key))
    }

    pub fn get_int_in(&self, namespace: &str, key: &str) -> Result<Option<i32>> {
        self.get_int(&namespaced(Some(namespace), key))
    }

    pub fn get_double_in(&self, namespace: &str, key: &str) -> Result<Option<f64>> {
        self.get_double(&namespaced(Some(namespace), key))
    }

    pub fn get_bool_in(&self, namespace: &str, key: &str) -> Result<Option<bool>> {
        self.get_bool(&namespaced(Some(namespace), key))
    }

    /// `algorithm()`: the `Algorithm` parameter as a string.
    pub fn algorithm(&self) -> Result<Option<&str>> {
        self.get_str(Key::ALGORITHM)
    }

    /// `algorithm(namespace)`.
    pub fn algorithm_in(&self, namespace: &str) -> Result<Option<&str>> {
        self.get_str_in(namespace, Key::ALGORITHM)
    }

    /// `getParameters(namespace)`: with a namespace, the entries whose
    /// stored key starts with `namespace.` (matched with case, unlike the
    /// map's lookups), without the prefix; with none, the entries whose key
    /// has no `.`.
    pub fn parameters(&self, namespace: Option<&str>) -> TrainingParameters {
        let mut out = TrainingParameters::new();
        match namespace {
            Some(ns) => {
                let prefix = format!("{ns}.");
                for (k, v) in &self.entries {
                    if let Some(rest) = k.strip_prefix(prefix.as_str()) {
                        out.put(rest, v.clone());
                    }
                }
            }
            None => {
                for (k, v) in &self.entries {
                    if !k.contains('.') {
                        out.put(k, v.clone());
                    }
                }
            }
        }
        out
    }

    /// The entries in the map's order, keys as first put.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `serialize(OutputStream)`: `Properties.store(out, null)` as JDK 21
    /// writes it, in ISO-8859-1 with `\uXXXX` escapes. The first line is
    /// `#` and `date`, where the JDK writes `new Date().toString()`; the
    /// entries follow sorted by `String.compareTo` of the key, each
    /// `key=value` with the escapes of `saveConvert`. Every value must be a
    /// string: the JDK casts, and OpenNLP's method throws otherwise.
    pub fn store(&self, date: &str) -> Result<String> {
        let mut keys: Vec<&(String, Value)> = self.entries.iter().collect();
        keys.sort_by(|a, b| turbo_jvm::chars::compare(&a.0, &b.0));
        let mut out = String::new();
        out.push('#');
        out.push_str(date);
        out.push('\n');
        for (k, v) in keys {
            let Value::Str(v) = v else {
                return Err(Error::SerializeType { key: k.clone(), found: v.java_class() });
            };
            save_convert(k, true, &mut out);
            out.push('=');
            save_convert(v, false, &mut out);
            out.push('\n');
        }
        Ok(out)
    }
}

/// `Properties.saveConvert(s, escapeSpace, escUnicode = true)`.
pub(crate) fn save_convert(s: &str, escape_space: bool, out: &mut String) {
    for (i, c) in s.chars().enumerate() {
        match c {
            '\\' => out.push_str("\\\\"),
            ' ' if i == 0 || escape_space => out.push_str("\\ "),
            ' ' => out.push(' '),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\u{C}' => out.push_str("\\f"),
            '=' | ':' | '#' | '!' => {
                out.push('\\');
                out.push(c);
            }
            '\u{20}'..='\u{7E}' => out.push(c),
            _ => {
                // Outside printable ASCII: one escape per UTF-16 unit.
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04X}"));
                }
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.java_string())
    }
}
