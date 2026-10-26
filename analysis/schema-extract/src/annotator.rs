//! The extraction as an annotator over a document's text: entity types into
//! `Entities`, classification tasks into the document's `Categories`, and
//! record types into [`Records`], from one encoder pass. The deterministic
//! layers a schema's checks name are read when the document has them.
//!
//! The document's `Categories` hold every task's labels as `task: label`,
//! task by task: each label's score is its probability within its task,
//! whether or not it was chosen (a single-label pass's scores sum to 1),
//! the selected set is every task's chosen labels, and the best category
//! the highest-scoring of them.
//!
//! Text longer than one encoder pass is read in sentence-aligned windows
//! (the document's `Sentences`, or its tokens when it has none) and the
//! windows' results merged; each label keeps its maximum probability
//! across passes without renormalizing. [`crate::window`] gives the rules.

use turbo_analysis::{Annotator, Error, ModelRef, Refs, Span, Spans, Tag, TagSet, View};
use turbo_analysis_layers::{Categories, Entities, Numbers, Pii, Sentences, Tokens};
use turbo_geo::Locations;
use turbo_glossary_layers::GlossaryTerms;

use crate::config_hash::config_hash;
use crate::decode::{Corroborate, Extraction, FieldValue, decode};
use crate::layer::{Records, source_code};
use crate::model::Gliner2Extractor;
use crate::schema::{ExtractionSchema, Source};
use crate::window;

/// A schema and the model that fills it.
pub struct SchemaAnnotator {
    extractor: Gliner2Extractor,
    schema: ExtractionSchema,
    threshold: f32,
    hash: u64,
}

impl SchemaAnnotator {
    /// The annotator for `schema` at `threshold`, the schema checked against
    /// the model's limits now.
    pub fn new(extractor: Gliner2Extractor, schema: ExtractionSchema, threshold: f32) -> Result<Self, String> {
        schema.validate(extractor.limits()).map_err(|e| e.to_string())?;
        if !(threshold.is_finite() && (0.0..=1.0).contains(&threshold)) {
            return Err(format!("a threshold outside [0, 1]: {threshold}"));
        }
        if extractor.model().id.is_empty() {
            return Err("a model with no id".into());
        }
        let hash = config_hash(&schema, threshold, extractor.limits(), extractor.max_width());
        Ok(SchemaAnnotator { extractor, schema, threshold, hash })
    }

    pub fn schema(&self) -> &ExtractionSchema {
        &self.schema
    }

    /// The byte ranges of `text` this annotator reads in one pass each:
    /// the whole text when it fits, else its sentence-aligned windows.
    pub fn windows(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
    ) -> Result<Vec<std::ops::Range<usize>>, String> {
        let words = prompt_words(view.text(), tokens, 0)?;
        let (measure, max_sequence) =
            self.extractor.measure_words(view.text(), &self.schema, words).map_err(|e| e.to_string())?;
        window::plan_measured(
            view.text(),
            &window::units(view, tokens, sentences)?,
            &measure,
            self.extractor.limits().max_words,
            max_sequence,
        )
    }

    /// Text too long for one pass: each window read on its own, merged.
    fn windowed(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        layers: &dyn Corroborate,
    ) -> Result<Extraction, String> {
        let mut out = Extraction { entities: vec![Vec::new(); self.schema.entities.len()], ..Extraction::default() };
        for at in self.windows(view, tokens, sentences)? {
            let text = &view.text()[at.clone()];
            let words = prompt_words(text, tokens, at.start)?;
            let prompt = self.extractor.prompt_words(text, &self.schema, words).map_err(|e| e.to_string())?;
            let pass = self.extractor.pass_prompt(prompt).map_err(|e| e.to_string())?;
            let shifted = window::Shifted { layers, by: at.start };
            window::merge(&mut out, decode(&pass, &self.schema, self.threshold, &shifted)?, at);
        }
        window::finish(&mut out, &self.schema);
        Ok(out)
    }
}

/// The document's deterministic layers, where it has them.
pub struct DocumentLayers<'a> {
    pub glossary: Option<&'a GlossaryTerms>,
    pub pii: Option<&'a Pii>,
    pub numbers: Option<&'a Numbers>,
    pub locations: Option<&'a Locations>,
}

impl Corroborate for DocumentLayers<'_> {
    fn supporting(&self, source: Source, start: usize, end: usize) -> Vec<u32> {
        let span = Span::new(start as u32, end as u32);
        let rows = |s: Option<Vec<usize>>| s.unwrap_or_default().into_iter().map(|i| i as u32).collect();
        rows(match source {
            Source::Glossary => self.glossary.map(|l| l.spans.overlapping(span).iter().map(|i| i.get()).collect()),
            Source::Pii => self.pii.map(|l| l.spans.overlapping(span).iter().map(|i| i.get()).collect()),
            Source::Numbers => self.numbers.map(|l| l.spans.overlapping(span).iter().map(|i| i.get()).collect()),
            Source::Locations => self.locations.map(|l| l.spans.overlapping(span).iter().map(|i| i.get()).collect()),
        })
    }
}

fn prompt_words<'a>(
    text: &'a str,
    tokens: &Tokens,
    base: usize,
) -> Result<Vec<turbo_relation::gliner2::words::Word<'a>>, String> {
    if tokens.spans.has_discontinuous() {
        return Ok(turbo_relation::gliner2::words::split(text));
    }
    let mut words = Vec::new();
    for (_, span) in tokens.spans.iter() {
        let (start, end) = (span.start as usize, span.end as usize);
        if start < base || end > base + text.len() {
            continue;
        }
        let raw = text.get(start - base..end - base).ok_or_else(|| "a token outside the view".to_owned())?;
        let folded = turbo_relation::gliner2::words::lowercase(raw);
        words.push(turbo_relation::gliner2::words::Word {
            text: folded,
            start: start - base,
            end: end - base,
            subword_len: 0,
        });
    }
    Ok(words)
}

/// A value's bytes in the view's text: the model reads the text with a
/// final `.` added when it lacks one, and a span reaching into it is cut
/// back to the text. `None` when nothing is left.
fn clip(start: usize, end: usize, len: usize) -> Option<Span> {
    let end = end.min(len);
    (start < end).then(|| Span::new(start as u32, end as u32))
}

fn entities(view: &View, tokens: &Tokens, schema: &ExtractionSchema, out: &Extraction) -> Entities {
    let names: Vec<&str> = schema.entities.iter().map(|e| e.name.as_str()).collect();
    let kinds = TagSet::new("schema-entities", &names);
    let mut rows: Vec<(Span, Tag, f64)> = Vec::new();
    for (k, values) in out.entities.iter().enumerate() {
        for v in values {
            if let Some(s) = clip(v.start, v.end, view.text().len()) {
                rows.push((s, Tag(k as u16), f64::from(v.score)));
            }
        }
    }
    rows.sort_by_key(|r| (r.0.start, r.0.end, r.1));
    let mut e =
        Entities { spans: Spans::new(view.id()), tokens: Vec::new(), kind: Vec::new(), kinds, score: Vec::new() };
    for (s, k, score) in rows {
        e.spans.push(s);
        e.tokens.push(tokens.spans.within(s));
        e.kind.push(k);
        e.score.push(score);
    }
    e
}

fn categories(schema: &ExtractionSchema, out: &Extraction) -> Categories {
    let mut names = Vec::new();
    let mut first = Vec::new();
    for t in &schema.classifications {
        first.push(names.len());
        names.extend(t.labels.iter().map(|l| format!("{}: {}", t.name, l.name)));
    }
    let set = TagSet::new("schema-classifications", &names.iter().map(String::as_str).collect::<Vec<_>>());
    if names.is_empty() {
        return Categories::new(set);
    }
    let mut scores = vec![0.0; names.len()];
    let mut selected = Vec::new();
    for c in &out.classifications {
        for (l, &p) in c.probs.iter().enumerate() {
            scores[first[c.task] + l] = f64::from(p);
        }
        for &(l, _, _) in &c.labels {
            selected.push(Tag((first[c.task] + l) as u16));
        }
    }
    selected.sort();
    let best = selected.iter().copied().fold(None::<Tag>, |b, t| match b {
        Some(b) if scores[b.0 as usize] >= scores[t.0 as usize] => Some(b),
        _ => Some(t),
    });
    Categories::document_selected(set, best.unwrap_or(Tag::NONE), &selected, &scores)
}

fn records(view: &View, schema: &ExtractionSchema, out: &Extraction) -> Records {
    let types: Vec<&str> = schema.structures.iter().map(|s| s.name.as_str()).collect();
    let mut paths = Vec::new();
    let mut first = Vec::new();
    let mut choice_names: Vec<String> = Vec::new();
    for s in &schema.structures {
        first.push(paths.len());
        paths.extend(s.fields.iter().map(|f| format!("{}.{}", s.name, f.name)));
        for c in s.fields.iter().filter_map(|f| f.choices.as_ref()).flatten() {
            if !choice_names.contains(c) {
                choice_names.push(c.clone());
            }
        }
    }
    let mut r = Records {
        spans: Spans::new(view.id()),
        record: Vec::new(),
        path: Vec::new(),
        score: Vec::new(),
        logit: Vec::new(),
        support_end: Vec::new(),
        support_source: Vec::new(),
        support_row: Vec::new(),
        record_type: Vec::new(),
        instance: Vec::new(),
        choice_record: Vec::new(),
        choice_path: Vec::new(),
        choice: Vec::new(),
        choice_score: Vec::new(),
        choice_logit: Vec::new(),
        types: TagSet::new("schema-records", &types),
        paths: TagSet::new("schema-fields", &paths.iter().map(String::as_str).collect::<Vec<_>>()),
        choices: TagSet::new("schema-choices", &choice_names.iter().map(String::as_str).collect::<Vec<_>>()),
    };
    let len = view.text().len();
    for rec in &out.records {
        let spec = &schema.structures[rec.structure];
        let at = r.record_type.len() as u32;
        let before = (r.spans.len(), r.choice.len());
        for (f, values) in rec.fields.iter().enumerate() {
            let path = Tag((first[rec.structure] + f) as u16);
            for v in values {
                match v {
                    FieldValue::Span(v) => {
                        let Some(s) = clip(v.start, v.end, len) else { continue };
                        r.spans.push(s);
                        r.record.push(at);
                        r.path.push(path);
                        r.score.push(f64::from(v.score));
                        r.logit.push(f64::from(v.logit));
                        for &(src, row) in &v.supported_by {
                            r.support_source.push(source_code(src));
                            r.support_row.push(row);
                        }
                        r.support_end.push(r.support_source.len() as u32);
                    }
                    FieldValue::Choice { choice, score, logit } => {
                        let name = &spec.fields[f].choices.as_ref().expect("a choice field")[*choice];
                        let tag = choice_names.iter().position(|c| c == name).expect("every choice named");
                        r.choice_record.push(at);
                        r.choice_path.push(path);
                        r.choice.push(Tag(tag as u16));
                        r.choice_score.push(f64::from(*score));
                        r.choice_logit.push(f64::from(*logit));
                    }
                }
            }
        }
        // A record whose only values fell past the text is left out.
        if (r.spans.len(), r.choice.len()) != before {
            r.record_type.push(Tag(rec.structure as u16));
            r.instance.push(rec.instance as u32);
        }
    }
    r
}

impl Annotator for SchemaAnnotator {
    const ID: &'static str = "turbo:schema-extract";
    type Reads =
        (View, Tokens, Option<GlossaryTerms>, Option<Pii>, Option<Numbers>, Option<Locations>, Option<Sentences>);
    type Writes = (Entities, Categories, Records);

    fn annotate(
        &self,
        (view, tokens, glossary, pii, numbers, locations, sentences): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let id = view.id();
        if tokens.spans.view() != id
            || glossary.is_some_and(|l| l.spans.view() != id)
            || pii.is_some_and(|l| l.spans.view() != id)
            || numbers.is_some_and(|l| l.spans.view() != id)
            || locations.is_some_and(|l| l.spans.view() != id)
        {
            return Err(fail("a layer read is not in the view read".into()));
        }
        let layers = DocumentLayers { glossary, pii, numbers, locations };
        let words = prompt_words(view.text(), tokens, 0).map_err(fail)?;
        let out = match self.extractor.prompt_words(view.text(), &self.schema, words) {
            Ok(prompt) => {
                let pass = self.extractor.pass_prompt(prompt).map_err(|e| fail(e.to_string()))?;
                decode(&pass, &self.schema, self.threshold, &layers).map_err(fail)?
            }
            Err(e) if window::too_long(&e) => self.windowed(view, tokens, sentences, &layers).map_err(fail)?,
            Err(e) => return Err(fail(e.to_string())),
        };
        let written = (
            entities(view, tokens, &self.schema, &out),
            categories(&self.schema, &out),
            records(view, &self.schema, &out),
        );
        turbo_analysis::Layer::check(&written.0).map_err(fail)?;
        turbo_analysis::Layer::check(&written.1).map_err(fail)?;
        turbo_analysis::Layer::check(&written.2).map_err(fail)?;
        Ok(written)
    }

    fn config_hash(&self) -> u64 {
        self.hash
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.extractor.model().clone())
    }
}
