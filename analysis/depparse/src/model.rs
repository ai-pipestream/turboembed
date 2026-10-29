use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::{
    ArcStandardState, DependencyContextGenerator, DependencyEventStream, DependencyGraph, DependencyParser,
    DependencySample, Error, LabelEncoding, ParserInput, Result, TrainingMode, Transition, error,
};
use turbo_analysis::ModelRef;
use turbo_analysis_layers::{Tokens, WordOrTokenAnchor};
use turbo_train::{Algorithm, TrainedModel, TrainingConfiguration, TrainingParameters};

const MAGIC: &[u8] = b"TURBO-DEPPARSE-LIN-V1\n";

#[derive(Clone, Debug)]
struct LinearClassifier {
    algorithm: Algorithm,
    outcomes: Vec<String>,
    predicates: HashMap<String, usize>,
    patterns: Vec<Vec<u32>>,
    parameters: Vec<Vec<f64>>,
}

impl LinearClassifier {
    fn from_trained(trained: &TrainedModel) -> Result<Self> {
        let writable = trained.as_writable();
        let outcomes: Vec<String> = writable
            .outcome_labels()
            .iter()
            .map(|s| s.to_str().map(std::borrow::Cow::into_owned).ok_or_else(|| error("non-UTF-8 outcome label")))
            .collect::<Result<_>>()?;
        if outcomes.is_empty() {
            return Err(error("classifier has no outcomes"));
        }
        let n_preds = writable.num_predicates();
        let mut predicates = HashMap::with_capacity(n_preds);
        let mut patterns = Vec::with_capacity(n_preds);
        let mut parameters = Vec::with_capacity(n_preds);
        for p in 0..n_preds {
            let label =
                writable.pred_labels()[p].to_str().ok_or_else(|| error("non-UTF-8 predicate label"))?.into_owned();
            let pat = writable.pattern(p).to_vec();
            let params = writable.parameters(p).to_vec();
            if pat.len() != params.len() {
                return Err(error("predicate pattern and parameter lengths differ"));
            }
            if pat.iter().any(|&o| o as usize >= outcomes.len()) {
                return Err(error("predicate pattern outcome out of bounds"));
            }
            let idx = patterns.len();
            predicates.insert(label, idx);
            patterns.push(pat);
            parameters.push(params);
        }
        Ok(Self { algorithm: trained.algorithm(), outcomes, predicates, patterns, parameters })
    }

    fn num_outcomes(&self) -> usize {
        self.outcomes.len()
    }

    fn eval(&self, context: &[String], out: &mut [f64]) {
        out.fill(0.0);
        if self.algorithm == Algorithm::NaiveBayes
            && let Some(&default_idx) = self.predicates.get("default")
        {
            for (&o, &w) in self.patterns[default_idx].iter().zip(&self.parameters[default_idx]) {
                out[o as usize] += w;
            }
        }
        for feat in context {
            if let Some(&p) = self.predicates.get(feat.as_str()) {
                for (&o, &w) in self.patterns[p].iter().zip(&self.parameters[p]) {
                    out[o as usize] += w;
                }
            }
        }
        let max = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max.is_finite() {
            let mut sum = 0.0;
            for v in out.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in out.iter_mut() {
                    *v /= sum;
                }
            }
        }
    }
}

/// A trained linear arc-standard dependency model with native binary persistence.
#[derive(Clone, Debug)]
pub struct DependencyModel {
    classifier: Arc<LinearClassifier>,
    transitions: Arc<[Transition]>,
    language: Option<Arc<str>>,
    encoding: LabelEncoding,
    pub(crate) identity: ModelRef,
}

impl DependencyModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path).map_err(error)?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(error)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        r.expect(MAGIC)?;
        let encoding = match r.u8()? {
            0 => LabelEncoding::Plain,
            1 => LabelEncoding::EscapedV1,
            other => return Err(Error(format!("unknown label encoding tag {other}"))),
        };
        let lang_str = r.str()?;
        let language = (!lang_str.is_empty()).then(|| Arc::<str>::from(lang_str));
        let algorithm = match r.u8()? {
            0 => Algorithm::Maxent,
            1 => Algorithm::MaxentQn,
            2 => Algorithm::Perceptron,
            3 => Algorithm::NaiveBayes,
            other => return Err(Error(format!("unknown linear algorithm tag {other}"))),
        };
        let n_outcomes = r.len()?;
        if n_outcomes == 0 {
            return Err(error("linear dependency model has no outcomes"));
        }
        let mut outcomes = Vec::with_capacity(n_outcomes);
        let mut transitions = Vec::with_capacity(n_outcomes);
        for _ in 0..n_outcomes {
            let label = r.str()?.to_owned();
            transitions.push(Transition::decode(&label)?);
            outcomes.push(label);
        }
        let n_preds = r.len()?;
        let mut predicates = HashMap::with_capacity(n_preds);
        let mut patterns = Vec::with_capacity(n_preds);
        let mut parameters = Vec::with_capacity(n_preds);
        for i in 0..n_preds {
            let pred = r.str()?.to_owned();
            let pat_len = r.len()?;
            let mut pat = Vec::with_capacity(pat_len);
            for _ in 0..pat_len {
                let o = r.u32()?;
                if o as usize >= n_outcomes {
                    return Err(error("outcome index out of bounds in linear dependency model"));
                }
                pat.push(o);
            }
            let mut params = Vec::with_capacity(pat_len);
            for _ in 0..pat_len {
                let w = r.f64()?;
                if !w.is_finite() {
                    return Err(error("non-finite weight in linear dependency model"));
                }
                params.push(w);
            }
            if predicates.insert(pred, i).is_some() {
                return Err(error("duplicate predicate in linear dependency model"));
            }
            patterns.push(pat);
            parameters.push(params);
        }
        r.finish()?;
        validate_transitions(&transitions)?;
        let identity = model_ref_from_bytes(bytes);
        Ok(Self {
            classifier: Arc::new(LinearClassifier { algorithm, outcomes, predicates, patterns, parameters }),
            transitions: transitions.into(),
            language,
            encoding,
            identity,
        })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::from(MAGIC);
        out.push(match self.encoding {
            LabelEncoding::Plain => 0,
            LabelEncoding::EscapedV1 => 1,
        });
        write_str(&mut out, self.language.as_deref().unwrap_or(""))?;
        out.push(match self.classifier.algorithm {
            Algorithm::Maxent => 0,
            Algorithm::MaxentQn => 1,
            Algorithm::Perceptron | Algorithm::PerceptronSequence => 2,
            Algorithm::NaiveBayes => 3,
        });
        write_len(&mut out, self.classifier.outcomes.len())?;
        for label in &self.classifier.outcomes {
            write_str(&mut out, label)?;
        }
        let mut entries: Vec<(&str, usize)> =
            self.classifier.predicates.iter().map(|(k, &v)| (k.as_str(), v)).collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        write_len(&mut out, entries.len())?;
        for (pred, idx) in entries {
            write_str(&mut out, pred)?;
            let pat = &self.classifier.patterns[idx];
            let params = &self.classifier.parameters[idx];
            write_len(&mut out, pat.len())?;
            for &o in pat {
                out.extend_from_slice(&o.to_le_bytes());
            }
            for &w in params {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        Ok(out)
    }

    pub fn train<A: WordOrTokenAnchor>(
        language: &str,
        samples: &[DependencySample<A>],
        params: &TrainingParameters,
    ) -> Result<Self> {
        Self::train_with_mode(language, samples, params, TrainingMode::PseudoProjective)
    }

    pub fn train_projective<A: WordOrTokenAnchor>(
        language: &str,
        samples: &[DependencySample<A>],
        params: &TrainingParameters,
    ) -> Result<Self> {
        Self::train_with_mode(language, samples, params, TrainingMode::Projective)
    }

    pub fn train_swap<A: WordOrTokenAnchor>(
        language: &str,
        samples: &[DependencySample<A>],
        params: &TrainingParameters,
    ) -> Result<Self> {
        Self::train_with_mode(language, samples, params, TrainingMode::Swap)
    }

    pub fn train_with_mode<A: WordOrTokenAnchor>(
        language: &str,
        samples: &[DependencySample<A>],
        params: &TrainingParameters,
        mode: TrainingMode,
    ) -> Result<Self> {
        let stream = DependencyEventStream::generate_with_mode(samples, mode)?;
        if stream.events.is_empty() {
            return Err(error("no training events generated"));
        }
        let mut config = TrainingConfiguration::default();
        let (trained, _) = turbo_train::trainer::train_events(&stream.events, params, &mut config).map_err(error)?;
        let classifier = LinearClassifier::from_trained(&trained)?;
        let transitions: Vec<Transition> =
            classifier.outcomes.iter().map(|s| Transition::decode(s)).collect::<Result<_>>()?;
        validate_transitions(&transitions)?;
        let encoding = mode.encoding();
        let mut model = Self {
            classifier: Arc::new(classifier),
            transitions: transitions.into(),
            language: Some(Arc::from(language)),
            encoding,
            identity: ModelRef { id: "turbo:depparse-linear".into(), revision: "1".into(), sha256: String::new() },
        };
        let bytes = model.to_bytes()?;
        model.identity = model_ref_from_bytes(&bytes);
        Ok(model)
    }

    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    pub fn encoding(&self) -> LabelEncoding {
        self.encoding
    }

    pub fn identity(&self) -> ModelRef {
        self.identity.clone()
    }

    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }
}

/// Greedy arc-standard dependency parser backed by a [`DependencyModel`].
#[derive(Clone, Debug)]
pub struct DependencyParserME {
    pub(crate) model: DependencyModel,
}

impl DependencyParserME {
    pub fn new(model: DependencyModel) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &DependencyModel {
        &self.model
    }

    pub fn parse<A: WordOrTokenAnchor>(&self, tokens: &[&str], tags: &[&str]) -> Result<DependencyGraph<A>> {
        ParserInput::check(tokens, tags)?;
        let mut state = ArcStandardState::<A>::new(tokens.len())?;
        let mut scores = vec![0.0; self.model.classifier.num_outcomes()];
        while !state.is_terminal() {
            let context = DependencyContextGenerator::context(&state, tokens, tags)?;
            self.model.classifier.eval(&context, &mut scores);
            let transition = self.best_transition(&state, &scores)?;
            state.apply(&transition)?;
        }
        state.to_graph()?.deprojectivize_with(self.model.encoding)
    }

    pub fn parse_tokens(&self, tokens: &[&str], tags: &[&str]) -> Result<DependencyGraph<Tokens>> {
        self.parse(tokens, tags)
    }

    fn best_transition<A: WordOrTokenAnchor>(&self, state: &ArcStandardState<A>, scores: &[f64]) -> Result<Transition> {
        let mut best: Option<(usize, f64)> = None;
        for (idx, t) in self.model.transitions.iter().enumerate() {
            if state.can_apply(t) {
                let s = scores[idx];
                if best.is_none_or(|(_, b)| s > b) {
                    best = Some((idx, s));
                }
            }
        }
        if let Some((idx, _)) = best {
            return Ok(self.model.transitions[idx].clone());
        }
        for candidate in [Transition::Shift, Transition::RightArc("dep".into()), Transition::LeftArc("dep".into())] {
            if state.can_apply(&candidate) {
                return Ok(candidate);
            }
        }
        Err(error("no legal transition available"))
    }
}

impl<A: WordOrTokenAnchor> DependencyParser<A> for DependencyParserME {
    fn parse(&self, tokens: &[&str], tags: &[&str]) -> Result<DependencyGraph<A>> {
        Self::parse(self, tokens, tags)
    }
}

fn validate_transitions(transitions: &[Transition]) -> Result<()> {
    let has_shift = transitions.iter().any(|t| matches!(t, Transition::Shift));
    let has_arc = transitions.iter().any(|t| matches!(t, Transition::LeftArc(_) | Transition::RightArc(_)));
    if !has_shift || !has_arc {
        return Err(error("dependency model must include SHIFT and at least one arc transition"));
    }
    Ok(())
}

fn model_ref_from_bytes(bytes: &[u8]) -> ModelRef {
    ModelRef { id: "turbo:depparse-linear".into(), revision: "1".into(), sha256: turbo_sha256::sha256(bytes).to_hex() }
}

fn write_len(out: &mut Vec<u8>, n: usize) -> Result<()> {
    let n = u32::try_from(n).map_err(|_| error("count exceeds u32"))?;
    out.extend_from_slice(&n.to_le_bytes());
    Ok(())
}

fn write_str(out: &mut Vec<u8>, s: &str) -> Result<()> {
    write_len(out, s.len())?;
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
    fn expect(&mut self, prefix: &[u8]) -> Result<()> {
        if !self.bytes.starts_with(prefix) {
            return Err(error("invalid linear dependency model header"));
        }
        self.bytes = &self.bytes[prefix.len()..];
        Ok(())
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.bytes.len() < n {
            return Err(error("truncated linear dependency model"));
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn len(&mut self) -> Result<usize> {
        Ok(self.u32()? as usize)
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<&'a str> {
        let n = self.len()?;
        std::str::from_utf8(self.take(n)?).map_err(error)
    }
    fn finish(self) -> Result<()> {
        if !self.bytes.is_empty() {
            return Err(error("trailing bytes in linear dependency model"));
        }
        Ok(())
    }
}
