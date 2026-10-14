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

use std::cmp::Ordering;
use std::fmt;

use crate::error::{Error, Result};
use turbo_jvm::text;
use turbo_maxent::Properties;
use turbo_unicode::java::{to_lower_char, to_upper_char};

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
    /// `Double.parseDouble`; `None` when absent. A hexadecimal literal,
    /// which Java would read, is refused.
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

/// `opennlp.tools.ml.AlgorithmType`: the trainers OpenNLP has built in,
/// as the `Algorithm` parameter names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// `MAXENT`: `GISTrainer`, model type `GIS`.
    Maxent,
    /// `MAXENT_QN`: `QNTrainer`, model type `QN`.
    MaxentQn,
    /// `PERCEPTRON`: `PerceptronTrainer`, model type `Perceptron`.
    Perceptron,
    /// `PERCEPTRON_SEQUENCE`: `SimplePerceptronSequenceTrainer`, a sequence
    /// trainer, model type `Perceptron`.
    PerceptronSequence,
    /// `NAIVEBAYES`: `NaiveBayesTrainer`, model type `NaiveBayes`.
    NaiveBayes,
}

impl Algorithm {
    pub const ALL: [Algorithm; 5] = [
        Algorithm::Maxent,
        Algorithm::MaxentQn,
        Algorithm::Perceptron,
        Algorithm::PerceptronSequence,
        Algorithm::NaiveBayes,
    ];

    /// `getAlgorithmType()`: the `Algorithm` parameter value.
    pub fn name(self) -> &'static str {
        match self {
            Algorithm::Maxent => "MAXENT",
            Algorithm::MaxentQn => "MAXENT_QN",
            Algorithm::Perceptron => "PERCEPTRON",
            Algorithm::PerceptronSequence => "PERCEPTRON_SEQUENCE",
            Algorithm::NaiveBayes => "NAIVEBAYES",
        }
    }

    /// `getModelType()`: the type string at the start of the model file.
    pub fn model_type(self) -> &'static str {
        match self {
            Algorithm::Maxent => "GIS",
            Algorithm::MaxentQn => "QN",
            Algorithm::Perceptron | Algorithm::PerceptronSequence => "Perceptron",
            Algorithm::NaiveBayes => "NaiveBayes",
        }
    }

    /// `fromAlgorithmType`, matched exactly.
    pub fn from_name(name: &str) -> Result<Algorithm> {
        Algorithm::ALL.into_iter().find(|a| a.name() == name).ok_or_else(|| Error::UnknownAlgorithm(name.to_owned()))
    }

    /// `fromModelType`: the first algorithm with this model type
    /// (`Perceptron` is [`Algorithm::Perceptron`]).
    pub fn from_model_type(model_type: &str) -> Option<Algorithm> {
        Algorithm::ALL.into_iter().find(|a| a.model_type() == model_type)
    }

    /// Which kind of trainer the algorithm's trainer class is.
    pub fn trainer_type(self) -> TrainerType {
        match self {
            Algorithm::PerceptronSequence => TrainerType::Sequence,
            _ => TrainerType::Event,
        }
    }
}

/// `TrainerFactory.TrainerType`, with the `TrainerType` parameter value
/// each trainer interface writes into the report (`EventTrainer.EVENT_VALUE`
/// and the others).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainerType {
    /// `EVENT_MODEL_TRAINER`, value `Event`.
    Event,
    /// `EVENT_MODEL_SEQUENCE_TRAINER`, value `EventModelSequence`.
    EventModelSequence,
    /// `SEQUENCE_TRAINER`, value `Sequence`.
    Sequence,
}

impl TrainerType {
    /// The `TrainerType` parameter value.
    pub fn value(self) -> &'static str {
        match self {
            TrainerType::Event => "Event",
            TrainerType::EventModelSequence => "EventModelSequence",
            TrainerType::Sequence => "Sequence",
        }
    }
}

/// `TrainerFactory.getTrainerType`: `Event` with no `Algorithm` parameter,
/// the algorithm's type when it names a built in one, and `None` for a name
/// OpenNLP would try to load as an extension class. A non-string value is
/// an error, as the string getter throws.
pub fn trainer_type(params: &TrainingParameters) -> Result<Option<TrainerType>> {
    Ok(match params.algorithm()? {
        None => Some(TrainerType::Event),
        Some(name) => Algorithm::from_name(name).ok().map(Algorithm::trainer_type),
    })
}

/// `TrainerFactory.isValid`: false when `Algorithm` names no built in
/// trainer, or `Cutoff` or `Iterations` is a string that is not an
/// integer. Reads `DataIndexer` as a string too, so a non-string value
/// there, or under `Algorithm`, is an error.
pub fn is_valid(params: &TrainingParameters) -> Result<bool> {
    if let Some(name) = params.algorithm()?
        && Algorithm::from_name(name).is_err()
    {
        return Ok(false);
    }
    for key in [Key::CUTOFF, Key::ITERATIONS] {
        match params.get_int(key) {
            Ok(_) => {}
            Err(Error::ParameterFormat { .. }) => return Ok(false),
            Err(e) => return Err(e),
        }
    }
    params.get_str(Key::DATA_INDEXER)?;
    Ok(true)
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.java_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_case_insensitive_and_keep_their_first_spelling() {
        let mut p = TrainingParameters::new();
        p.put("Cutoff", 5);
        p.put("CUTOFF", 7);
        assert_eq!(p.len(), 1);
        assert_eq!(p.get("cutoff"), Some(&Value::Int(7)));
        assert_eq!(p.iter().next().unwrap().0, "Cutoff");
        p.put_if_absent("cutOFF", 9);
        assert_eq!(p.get_int("Cutoff").unwrap(), Some(7));
        p.put_if_absent("Iterations", 3);
        assert_eq!(p.get_int("iterations").unwrap(), Some(3));
    }

    #[test]
    fn entries_iterate_in_the_case_insensitive_order() {
        let mut p = TrainingParameters::new();
        for k in ["b", "A", "_x", "C", "a.b", "ab"] {
            p.put(k, k);
        }
        // Letters compare by their lower case forms, so "_" (0x5F) sorts
        // before every letter, and a prefix before the longer key.
        let keys: Vec<&str> = p.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, ["_x", "A", "a.b", "ab", "b", "C"]);
        assert_eq!(case_insensitive_compare("straße", "STRASSE"), Ordering::Greater);
        assert_eq!(case_insensitive_compare("ǅ", "ǆ"), Ordering::Equal);
        assert_eq!(case_insensitive_compare("abc", "ab"), Ordering::Greater);
    }

    #[test]
    fn getters_cast_then_parse_as_java_does() {
        let mut p = TrainingParameters::new();
        p.put("i", 4);
        p.put("d", 0.5);
        p.put("b", true);
        p.put("s", "12");
        p.put("t", "true");
        p.put("w", "x");
        assert_eq!(p.get_int("i").unwrap(), Some(4));
        assert_eq!(p.get_int("s").unwrap(), Some(12));
        assert_eq!(p.get_double("d").unwrap(), Some(0.5));
        assert_eq!(p.get_double("s").unwrap(), Some(12.0));
        assert_eq!(p.get_bool("b").unwrap(), Some(true));
        assert_eq!(p.get_bool("t").unwrap(), Some(true));
        assert_eq!(p.get_bool("w").unwrap(), Some(false));
        assert_eq!(p.get_str("s").unwrap(), Some("12"));
        assert_eq!(p.get_str("missing").unwrap(), None);
        assert_eq!(p.get_int("missing").unwrap(), None);
        assert_eq!(p.get_str("i"), Err(Error::ParameterType { key: "i".into(), expected: "String", found: "Integer" }));
        assert_eq!(p.get_int("d"), Err(Error::ParameterType { key: "d".into(), expected: "Integer", found: "Double" }));
        assert_eq!(
            p.get_double("i"),
            Err(Error::ParameterType { key: "i".into(), expected: "Double", found: "Integer" })
        );
        assert_eq!(
            p.get_bool("i"),
            Err(Error::ParameterType { key: "i".into(), expected: "Boolean", found: "Integer" })
        );
        assert_eq!(
            p.get_int("w"),
            Err(Error::ParameterFormat { key: "w".into(), expected: "Integer", value: "x".into() })
        );
        assert_eq!(
            p.get_double("w"),
            Err(Error::ParameterFormat { key: "w".into(), expected: "Double", value: "x".into() })
        );
        // parseInt refuses what parseDouble reads.
        p.put("f", "1.5");
        assert!(p.get_int("f").is_err());
        assert_eq!(p.get_double("f").unwrap(), Some(1.5));
    }

    #[test]
    fn namespaces_join_with_a_dot_and_are_matched_with_case() {
        let mut p = TrainingParameters::new();
        p.put_in("ns", "Cutoff", 3);
        p.put("Cutoff", 5);
        p.put("other.x", "y");
        assert_eq!(p.get_int_in("ns", "cutoff").unwrap(), Some(3));
        assert_eq!(p.get_int("cutoff").unwrap(), Some(5));
        let ns = p.parameters(Some("ns"));
        assert_eq!(ns.iter().collect::<Vec<_>>(), [("Cutoff", &Value::Int(3))]);
        // getObjectSettings compares the stored key with case.
        assert!(p.parameters(Some("NS")).is_empty());
        let top = p.parameters(None);
        assert_eq!(top.iter().map(|(k, _)| k).collect::<Vec<_>>(), ["Cutoff"]);
        assert_eq!(p.algorithm_in("ns").unwrap(), None);
    }

    #[test]
    fn defaults_are_default_params() {
        let p = TrainingParameters::defaults();
        assert_eq!(p.algorithm().unwrap(), Some("MAXENT"));
        assert_eq!(p.get_str(Key::TRAINER_TYPE).unwrap(), Some("Event"));
        assert_eq!(p.get(Key::ITERATIONS), Some(&Value::Int(100)));
        assert_eq!(p.get(Key::CUTOFF), Some(&Value::Int(5)));
        assert_eq!(p.len(), 4);
    }

    #[test]
    fn a_properties_file_loads_as_strings() {
        let p = TrainingParameters::load(b"# comment\nIterations=10\ncutoff : 2\nAlgorithm=PERCEPTRON\nns.Cutoff=1\n")
            .unwrap();
        assert_eq!(p.get(Key::ITERATIONS), Some(&Value::Str("10".into())));
        assert_eq!(p.get_int(Key::ITERATIONS).unwrap(), Some(10));
        assert_eq!(p.get_int(Key::CUTOFF).unwrap(), Some(2));
        assert_eq!(p.get_int_in("ns", "cutoff").unwrap(), Some(1));
        assert_eq!(trainer_type(&p).unwrap(), Some(TrainerType::Event));
    }

    #[test]
    fn store_writes_sorted_escaped_properties_and_refuses_non_strings() {
        let mut p = TrainingParameters::new();
        p.put("b key", "v=1 #x");
        p.put("a", " lead é\u{10000}");
        p.put("Z", "tab\tend\\");
        let text = p.store("Tue Oct 14 00:00:00 UTC 2026").unwrap();
        assert_eq!(
            text,
            "#Tue Oct 14 00:00:00 UTC 2026\nZ=tab\\tend\\\\\na=\\ lead \\u00E9\\uD800\\uDC00\nb\\ key=v\\=1 \\#x\n"
        );
        p.put("n", 3);
        assert_eq!(p.store("d"), Err(Error::SerializeType { key: "n".into(), found: "Integer" }));
    }

    #[test]
    fn algorithms_and_trainer_types() {
        assert_eq!(Algorithm::from_name("MAXENT").unwrap(), Algorithm::Maxent);
        assert_eq!(Algorithm::from_name("maxent"), Err(Error::UnknownAlgorithm("maxent".into())));
        assert_eq!(Algorithm::Maxent.model_type(), "GIS");
        assert_eq!(Algorithm::from_model_type("Perceptron"), Some(Algorithm::Perceptron));
        assert_eq!(Algorithm::from_model_type("gis"), None);
        assert_eq!(Algorithm::PerceptronSequence.trainer_type(), TrainerType::Sequence);
        assert_eq!(Algorithm::NaiveBayes.trainer_type(), TrainerType::Event);
        for a in Algorithm::ALL {
            assert_eq!(Algorithm::from_name(a.name()).unwrap(), a);
        }
        assert_eq!(TrainerType::EventModelSequence.value(), "EventModelSequence");
    }

    #[test]
    fn trainer_type_and_validity_follow_the_factory() {
        let mut p = TrainingParameters::new();
        assert_eq!(trainer_type(&p).unwrap(), Some(TrainerType::Event));
        assert!(is_valid(&p).unwrap());
        p.put(Key::ALGORITHM, "PERCEPTRON_SEQUENCE");
        assert_eq!(trainer_type(&p).unwrap(), Some(TrainerType::Sequence));
        p.put(Key::ALGORITHM, "com.example.MyTrainer");
        assert_eq!(trainer_type(&p).unwrap(), None);
        assert!(!is_valid(&p).unwrap());
        p.put(Key::ALGORITHM, "MAXENT");
        p.put(Key::CUTOFF, "many");
        assert!(!is_valid(&p).unwrap());
        p.put(Key::CUTOFF, 2.5);
        assert!(is_valid(&p).is_err());
        p.put(Key::CUTOFF, 2);
        p.put(Key::DATA_INDEXER, 1);
        assert!(is_valid(&p).is_err());
        p.put(Key::ALGORITHM, 1);
        assert!(trainer_type(&p).is_err());
    }

    #[test]
    fn values_print_as_java_strings() {
        assert_eq!(Value::Double(0.0001).java_string(), "1.0E-4");
        assert_eq!(Value::Int(-3).to_string(), "-3");
        assert_eq!(Value::Bool(false).to_string(), "false");
        assert_eq!(Value::from("x").java_class(), "String");
    }
}
