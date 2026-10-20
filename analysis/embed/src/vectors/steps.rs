use super::*;
/// The provenance every vector step gives: the model, where it ran and
/// the settings, with the layer that lays out the step's windows.
macro_rules! provenance {
    ($layout:expr) => {
        const VERSION: &'static str = "1";
        fn model(&self) -> Option<ModelRef> {
            Some(self.0.model.clone())
        }

        fn execution(&self) -> Option<Execution> {
            Some(self.0.execution.clone())
        }

        fn config_hash(&self) -> u64 {
            self.0.config($layout)
        }
    };
}

/// Writes `opennlp:embeddings<opennlp:sentences>`: a vector per sentence.
#[derive(Clone)]
pub struct SentenceVectors(pub Arc<VectorAnnotator>);

impl Annotator for SentenceVectors {
    const ID: &'static str = "turbo:sentence-vectors";
    type Reads = (Subwords, Sentences);
    type Writes = (Vectors<Sentences>,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let inputs: Vec<Input<'_>> = inputs
            .into_iter()
            .map(|(sub, s)| Input { subwords: sub, sentences: Some(&s.spans), chunks: None, levels: None })
            .collect();
        self.0.embed(Self::ID, &inputs).into_iter().map(|r| r.map(|o| (o.sentences.expect("asked for"),))).collect()
    }

    provenance!(Which::Sentences);
}

/// Writes `opennlp:embeddings<turbo:chunks>`: a vector per chunk, at
/// every level of the chunk hierarchy.
#[derive(Clone)]
pub struct ChunkVectors(pub Arc<VectorAnnotator>);

impl Annotator for ChunkVectors {
    const ID: &'static str = "turbo:chunk-vectors";
    type Reads = (Subwords, Chunks);
    type Writes = (Vectors<Chunks>,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let inputs: Vec<Input<'_>> = inputs
            .into_iter()
            .map(|(sub, c)| Input { subwords: sub, sentences: None, chunks: Some(&c.spans), levels: None })
            .collect();
        self.0.embed(Self::ID, &inputs).into_iter().map(|r| r.map(|o| (o.chunks.expect("asked for"),))).collect()
    }

    provenance!(Which::Chunks);
}

/// Writes both, from the same windows: a sentence and the chunk it is in
/// are pooled from one encoder pass when one window holds them.
#[derive(Clone)]
pub struct SentenceChunkVectors(pub Arc<VectorAnnotator>);

impl Annotator for SentenceChunkVectors {
    const ID: &'static str = "turbo:sentence-chunk-vectors";
    type Reads = (Subwords, Sentences, Chunks);
    type Writes = (Vectors<Sentences>, Vectors<Chunks>);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let inputs: Vec<Input<'_>> = inputs
            .into_iter()
            .map(|(sub, s, c)| Input { subwords: sub, sentences: Some(&s.spans), chunks: Some(&c.spans), levels: None })
            .collect();
        self.0
            .embed(Self::ID, &inputs)
            .into_iter()
            .map(|r| r.map(|o| (o.sentences.expect("asked for"), o.chunks.expect("asked for"))))
            .collect()
    }

    provenance!(Which::Sentences);
}

/// Writes `opennlp:embeddings<opennlp:subwords>`: a contextual vector per
/// subword piece, extracted from its most-interior window via `#263` and `#709`.
#[derive(Clone)]
pub struct SubwordVectors(pub Arc<VectorAnnotator>);

impl Annotator for SubwordVectors {
    const ID: &'static str = "turbo:subword-vectors";
    type Reads = (Subwords, Option<Sentences>);
    type Writes = (Vectors<Subwords>,);

    fn annotate(&self, (subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        Ok((self.0.embed_subwords(Self::ID, subwords, sentences.map(|s| &s.spans))?,))
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let batch: Vec<(&Subwords, Option<&Spans<Sentences>>)> =
            inputs.into_iter().map(|(sub, s)| (sub, s.map(|s| &s.spans))).collect();
        self.0
            .embed_subword_tokens_batch(Self::ID, &batch, self.0.stride())
            .into_iter()
            .map(|r| r.map(|t| (t.into_subword_vectors(),)))
            .collect()
    }

    const VERSION: &'static str = "1";

    fn model(&self) -> Option<ModelRef> {
        Some(self.0.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.0.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        self.0.token_config()
    }
}

/// Offers the vector steps to pipelines built from configuration, as
/// "vectors". The spec's location is the bundle directory; its options:
///
/// - `layers`: `sentences`, `chunks`, `both` (the default), `levels`
///   for the local and contextual vectors of `Levels` (`LevelVectors`),
///   or `subwords` for per-token contextual vectors (`SubwordVectors`);
/// - `device`: `cpu` (the default), `select`, a device index, or a backend
///   name such as `cuda`;
/// - `precision`: `model` (the default), `fastest` or `exact`;
/// - `prompt`: `document` (the default), `query` or `none`;
/// - `stride`: sliding-window stride in subword pieces for `subwords` (`0`
///   by default, selecting half the window width).
pub struct VectorsProvider;

fn parse_options(spec: &Spec) -> Result<VectorOptions, String> {
    let device = match spec.option("device", "cpu") {
        "cpu" => Device::Cpu,
        "select" => Device::Select,
        d => match d.parse() {
            Ok(i) => Device::Index(i),
            Err(_) => Device::Backend(d.to_owned()),
        },
    };
    let precision = match spec.option("precision", "model") {
        "model" => Precision::Model,
        "fastest" => Precision::Fastest,
        "exact" => Precision::Exact,
        p => return Err(format!("precision {p:?}: model, fastest or exact")),
    };
    let prompt = match spec.option("prompt", "document") {
        "document" => PromptRole::Document,
        "query" => PromptRole::Query,
        "none" => PromptRole::None,
        p => return Err(format!("prompt {p:?}: document, query or none")),
    };
    let stride = match spec.option("stride", "0").parse::<u32>() {
        Ok(s) => s,
        Err(_) => return Err(format!("stride {:?}: expected a non-negative integer", spec.option("stride", "0"))),
    };
    Ok(VectorOptions { encoder: EncoderOptions { device, precision, ..EncoderOptions::default() }, prompt, stride })
}

impl Provider<dyn Step> for VectorsProvider {
    fn name(&self) -> &'static str {
        "vectors"
    }

    fn supports(&self, spec: &Spec) -> bool {
        spec.location.is_some()
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let dir = spec.location.as_deref().ok_or_else(|| Error::Pipeline("vectors: no bundle location".into()))?;
        let o = parse_options(spec).map_err(|e| Error::Pipeline(format!("vectors: {e}")))?;
        let layers = spec.option("layers", "both");
        if !matches!(layers, "sentences" | "chunks" | "both" | "levels" | "subwords") {
            return Err(Error::Pipeline(format!(
                "vectors: layers {layers:?}: sentences, chunks, both, levels or subwords"
            )));
        }
        let a =
            VectorAnnotator::open(dir, &o).map_err(|e| Error::Pipeline(format!("vectors: {}: {e}", dir.display())))?;
        let a = Arc::new(a);
        Ok(match layers {
            "sentences" => Box::new(Bound::new(SentenceVectors(a))),
            "chunks" => Box::new(Bound::new(ChunkVectors(a))),
            "levels" => Box::new(crate::levels::LevelVectors(a).bound()),
            "subwords" => Box::new(Bound::new(SubwordVectors(a))),
            _ => Box::new(Bound::new(SentenceChunkVectors(a))),
        })
    }
}
