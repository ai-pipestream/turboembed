//! turbo-chunker's optimizer as a pipeline chunker.

use turbo_analysis::{Analyzer, Bound, Document, Ix, Key, Provenance, View};
use turbo_analysis_layers::{Aspects, Categories, Chunks, Incivility, Sentences, Structure, Subwords, Tokens, Vectors};
use turbo_chunker::{
    Budget, Chunker as Optimizer, Cohesion, CohesiveChunker, Costs, Drift, DriftBoundaries, STRUCTURE, Score,
    SentimentBoundaries, SentimentShift, StructureBoundaries, WeighAllBoundaries, WeighBoundaries, Weight,
};

use super::{Chunker, chunk_span, need, setting};
use crate::Result;
use crate::embedder::{Embedder, Role};
use crate::record::Setting;
use crate::spans::Span;

/// The boundary signals the optimizer reads.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Signals {
    /// The structure's costs alone (`StructureBoundaries`); with no
    /// structure layer every gap is a plain sentence boundary.
    Structure,
    /// The structure's and the drift's costs, weighed
    /// (`WeighBoundaries`) with these weights.
    StructureDrift { drift: Drift, structure: Weight, weight: Weight },
    /// The structure's, optional drift's, and sentiment/aspect shift's
    /// costs (`SentimentBoundaries`), weighed with `WeighAllBoundaries`.
    StructureSentiment {
        drift: Option<(Drift, Weight)>,
        sentiment: SentimentShift,
        structure: Weight,
        sentiment_weight: Weight,
    },
}

/// The chunks turbo-chunker's `Chunker` (or `CohesiveChunker`, with a
/// cohesion term) writes, at level 0: the leaves, not the sections above
/// them.
///
/// Drift and cohesion read sentence vectors. Here each sentence is
/// embedded alone, in the document role, rather than late pooled from
/// the document's windows as `SentenceVectors` pools them; the record
/// says so.
#[derive(Copy, Clone, Debug)]
pub struct TurboChunker {
    pub budget: Budget,
    pub costs: Costs,
    pub signals: Signals,
    pub cohesion: Option<Cohesion>,
}

impl TurboChunker {
    fn needs_vectors(&self) -> bool {
        matches!(self.signals, Signals::StructureDrift { .. } | Signals::StructureSentiment { drift: Some(_), .. })
            || self.cohesion.is_some()
    }

    /// Each sentence's vector, as the layer drift and cohesion read.
    fn sentence_vectors(doc: &Document, embedder: &mut dyn Embedder) -> Result<Vectors<Sentences>> {
        let sentences = need::<Sentences>(doc, "turbo-chunker")?;
        let texts: Vec<&str> = sentences.spans.iter().map(|(_, s)| &doc.source()[s.range()]).collect();
        let mut data = Vec::new();
        if !texts.is_empty() {
            embedder.embed(&texts, Role::Document, &mut data)?;
        }
        let dim = embedder.dim();
        let empty = data
            .chunks_exact(dim)
            .enumerate()
            .filter(|(_, v)| v.iter().all(|&x| x == 0.0))
            .map(|(i, _)| Ix::new(i as u32))
            .collect();
        let dim = u32::try_from(dim).map_err(|_| format!("vectors of {dim} values"))?;
        Ok(Vectors::new(dim, data, empty, embedder.identity().model_ref()))
    }
}

fn score_name(s: Score) -> &'static str {
    match s {
        Score::Depth => "depth",
        Score::Distance => "distance",
    }
}

impl Chunker for TurboChunker {
    fn name(&self) -> &'static str {
        "turbo-chunker"
    }

    fn config(&self) -> Vec<Setting> {
        let (b, c) = (self.budget, self.costs);
        let mut out = vec![
            setting("max", b.max),
            setting("target", b.target),
            setting("size_weight", b.size_weight),
            setting("overhead", b.overhead),
            setting("cost.section", c.section),
            setting("cost.heading", c.heading),
            setting("cost.paragraph", c.paragraph),
            setting("cost.item", c.item),
            setting("cost.sentence", c.sentence),
            setting("cost.after_heading", c.after_heading),
        ];
        match self.signals {
            Signals::Structure => out.push(setting("signals", "structure")),
            Signals::StructureDrift { drift: d, structure, weight } => out.extend([
                setting("signals", "structure+drift"),
                setting("drift.window", d.window),
                setting("drift.smoothing", d.smoothing),
                setting("drift.score", score_name(d.score)),
                setting("drift.floor", d.floor),
                setting("drift.scale", d.scale),
                setting("drift.shift", d.shift),
                setting("weigh.structure", structure.0),
                setting("weigh.drift", weight.0),
            ]),
            Signals::StructureSentiment { drift, sentiment: s, structure, sentiment_weight } => {
                if let Some((d, dw)) = drift {
                    out.extend([
                        setting("signals", "structure+drift+sentiment"),
                        setting("drift.window", d.window),
                        setting("drift.smoothing", d.smoothing),
                        setting("drift.score", score_name(d.score)),
                        setting("drift.floor", d.floor),
                        setting("drift.scale", d.scale),
                        setting("drift.shift", d.shift),
                        setting("weigh.drift", dw.0),
                    ]);
                } else {
                    out.push(setting("signals", "structure+sentiment"));
                }
                out.extend([
                    setting("sentiment.window", s.window),
                    setting("sentiment.polarity_weight", s.polarity_weight),
                    setting("sentiment.aspect_weight", s.aspect_weight),
                    setting("sentiment.incivility_weight", s.incivility_weight),
                    setting("sentiment.floor", s.floor),
                    setting("sentiment.scale", s.scale),
                    setting("sentiment.shift", s.shift),
                    setting("weigh.structure", structure.0),
                    setting("weigh.sentiment", sentiment_weight.0),
                ]);
            }
        }
        out.push(setting("cohesion.weight", self.cohesion.map_or_else(|| "none".into(), |c| c.weight.to_string())));
        if self.needs_vectors() {
            out.push(setting("sentence_vectors", "each sentence embedded alone, document role"));
        }
        out
    }

    fn chunk(&self, doc: &Document, embedder: &mut dyn Embedder) -> Result<Vec<Span>> {
        let fail = |e: turbo_analysis::Error| format!("turbo-chunker: {e}");
        let mut doc = doc.clone();
        let mut b = Analyzer::builder().given(Key::<Sentences>::DEFAULT).given(Key::<Subwords>::DEFAULT);
        if doc.get::<Structure>().is_some() {
            b = b.given(Key::<Structure>::DEFAULT);
        }
        if self.needs_vectors() {
            let v = TurboChunker::sentence_vectors(&doc, embedder)?;
            doc.add(&Key::DEFAULT, v, Provenance::by("turbo-eval")).map_err(fail)?;
            b = b.given(Key::<Vectors<Sentences>>::DEFAULT);
        }
        b = match self.signals {
            Signals::Structure => b.add(StructureBoundaries { costs: self.costs }),
            Signals::StructureDrift { drift, structure, weight } => b
                .step(Bound::new(StructureBoundaries { costs: self.costs }).write(0, STRUCTURE))
                .step(DriftBoundaries { drift }.bound())
                .step(WeighBoundaries { structure, drift: weight }.bound()),
            Signals::StructureSentiment { drift, sentiment, structure, sentiment_weight } => {
                if doc.get::<View>().is_some() {
                    b = b.given(Key::<View>::DEFAULT);
                }
                if doc.get::<Tokens>().is_some() {
                    b = b.given(Key::<Tokens>::DEFAULT);
                }
                if doc.get::<Categories<Sentences>>().is_some() {
                    b = b.given(Key::<Categories<Sentences>>::DEFAULT);
                }
                if doc.get::<Aspects>().is_some() {
                    b = b.given(Key::<Aspects>::DEFAULT);
                }
                if doc.get::<Incivility>().is_some() {
                    b = b.given(Key::<Incivility>::DEFAULT);
                }
                b = b.step(Bound::new(StructureBoundaries { costs: self.costs }).write(0, STRUCTURE));
                let drift_weight = if let Some((d, dw)) = drift {
                    b = b.step(DriftBoundaries { drift: d }.bound());
                    dw
                } else {
                    Weight(0)
                };
                b.step(SentimentBoundaries { shift: sentiment }.bound())
                    .step(WeighAllBoundaries { structure, drift: drift_weight, sentiment: sentiment_weight }.bound())
            }
        };
        let optimizer = Optimizer { budget: self.budget };
        b = match self.cohesion {
            Some(cohesion) => b.add(CohesiveChunker { chunker: optimizer, cohesion }),
            None => b.add(optimizer),
        };
        let analyzer = b.build().map_err(fail)?;
        let mut docs = vec![doc];
        analyzer.run(&mut docs).remove(0).map_err(fail)?;
        let chunks = need::<Chunks>(&docs[0], self.name())?;
        Ok(chunks.spans.iter().filter(|(i, _)| chunks.level[i.get()] == 0).filter_map(|(_, s)| chunk_span(s)).collect())
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{Tag, TagSet};
    use turbo_structure::Syntax;

    use super::*;
    use crate::chunkers::tests::{analyzed, texts};
    use crate::embedder::stub::Stub;

    fn plain(max: u32, target: u32) -> TurboChunker {
        TurboChunker {
            budget: Budget { max, target, size_weight: 1000, overhead: 2 },
            costs: Costs::default(),
            signals: Signals::Structure,
            cohesion: None,
        }
    }

    #[test]
    fn the_optimizer_cuts_at_the_paragraph_by_hand() {
        // Four sentences of 3 words, in two paragraphs. With the overhead
        // of 2, one sentence is 5 tokens, two are 8, three 11.
        // Budget: max 9, target 8, so a chunk holds at most two sentences.
        // Cuts: after 1 (a sentence boundary, 1000) and after 3 (1000),
        // or after 2 (a paragraph end, 400). Sizes 8 and 8 hit the target
        // exactly, so cutting at the paragraph costs 400 and wins.
        let text = "cat purrs loud.\ncat naps long.\n\nrust compiles fast.\nrust ships now.\n";
        let mut e = Stub::new(&["cat"]);
        let doc = analyzed(text, &e, Some(Syntax::Plain));
        let spans = plain(9, 8).chunk(&doc, &mut e).unwrap();
        assert_eq!(texts(text, &spans), ["cat purrs loud.\ncat naps long.", "rust compiles fast.\nrust ships now."]);
        // Level 1 (the sections) is left out: with no section headings the
        // plain text has none, and every chunk is a leaf.
        let c = plain(9, 8).config();
        assert_eq!(c[0], setting("max", 9));
        assert!(c.contains(&setting("signals", "structure")));
        assert!(c.contains(&setting("cohesion.weight", "none")));
    }

    #[test]
    fn cohesion_embeds_the_sentences_and_cuts_where_they_part() {
        // One paragraph, no structure: every gap is a plain sentence
        // boundary. A budget of 20 holds all four (4 x 3 + 2 = 14), and
        // the size penalty with target 14 favours one chunk; cohesion
        // with a large weight splits it where the vectors turn.
        let text = "cat purrs loud\ncat naps long\nrust compiles fast\nrust ships now\n";
        let mut e = Stub::new(&["cat", "rust"]);
        let doc = analyzed(text, &e, None);
        let mut c =
            TurboChunker { budget: Budget { max: 20, target: 14, size_weight: 1000, overhead: 2 }, ..plain(0, 0) };
        assert_eq!(texts(text, &c.chunk(&doc, &mut e).unwrap()), [text.trim_end()]);
        c.cohesion = Some(Cohesion { weight: 100_000 });
        assert_eq!(
            texts(text, &c.chunk(&doc, &mut e).unwrap()),
            ["cat purrs loud\ncat naps long", "rust compiles fast\nrust ships now"]
        );
        assert!(c.config().contains(&setting("cohesion.weight", 100_000)));
    }

    #[test]
    fn drift_is_weighed_in_when_asked_for() {
        // Eight sentences, four on each topic, all within one budget.
        // Drift (window 3) finds the valley at the turn; with the drift
        // signal the optimizer cuts there, where without it the size
        // penalty keeps one chunk.
        let text = "cat a b\ncat c d\ncat e f\ncat g h\nrust a b\nrust c d\nrust e f\nrust g h\n";
        let mut e = Stub::new(&["cat", "rust"]);
        let doc = analyzed(text, &e, None);
        let mut c =
            TurboChunker { budget: Budget { max: 40, target: 26, size_weight: 1000, overhead: 2 }, ..plain(0, 0) };
        assert_eq!(c.chunk(&doc, &mut e).unwrap().len(), 1);
        c.signals = Signals::StructureDrift { drift: Drift::default(), structure: Weight::ONE, weight: Weight(4000) };
        assert_eq!(
            texts(text, &c.chunk(&doc, &mut e).unwrap()),
            ["cat a b\ncat c d\ncat e f\ncat g h", "rust a b\nrust c d\nrust e f\nrust g h"]
        );
        let config = c.config();
        assert!(config.contains(&setting("signals", "structure+drift")));
        assert!(config.contains(&setting("weigh.drift", 4000)));
        assert!(config.contains(&setting("sentence_vectors", "each sentence embedded alone, document role")));
    }

    #[test]
    fn sentiment_signal_is_off_at_zero_weight_and_splits_when_weighed() {
        let text = "good food here\ngreat meal today\nawful service now\nrude staff here\n";
        let mut e = Stub::new(&["food"]);
        let mut doc = analyzed(text, &e, None);
        let cat_set = TagSet::new("sentiment", &["positive", "negative"]);
        let mut cats = Categories::<Sentences>::with_capacity(cat_set, 4);
        cats.push(Tag(0), &[0.95, 0.05]);
        cats.push(Tag(0), &[0.92, 0.08]);
        cats.push(Tag(1), &[0.06, 0.94]);
        cats.push(Tag(1), &[0.04, 0.96]);
        doc.add(&Key::DEFAULT, cats, Provenance::by("test")).unwrap();

        let mut c = TurboChunker {
            budget: Budget { max: 20, target: 14, size_weight: 1000, overhead: 2 },
            signals: Signals::StructureSentiment {
                drift: None,
                sentiment: SentimentShift::default(),
                structure: Weight::ONE,
                sentiment_weight: Weight(0),
            },
            ..plain(0, 0)
        };
        assert_eq!(c.chunk(&doc, &mut e).unwrap().len(), 1);

        c.signals = Signals::StructureSentiment {
            drift: None,
            sentiment: SentimentShift::default(),
            structure: Weight::ONE,
            sentiment_weight: Weight(3000),
        };
        assert_eq!(
            texts(text, &c.chunk(&doc, &mut e).unwrap()),
            ["good food here\ngreat meal today", "awful service now\nrude staff here"]
        );
        let config = c.config();
        assert!(config.contains(&setting("signals", "structure+sentiment")));
        assert!(config.contains(&setting("weigh.sentiment", 3000)));
    }
}
