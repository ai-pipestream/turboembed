//! What the chunking and tree-insertion event streams share: the `TAG` events of
//! `AbstractParserEventStream.addTagEvents`, and the events the parser's chunker is trained on
//! (`ChunkerEventStream` over `ChunkSampleStream`).

use turbo_parser::{ChunkContextGenerator, Parse, ParserChunkSample};
use turbo_sequence::Features;
use turbo_sequence::chunker::DEFAULT_BEAM_SIZE as CHUNKER_DEFAULT_BEAM_SIZE;
use turbo_sequence::postag::DefaultPosContext;
use turbo_train::{Event, JavaString};

/// An event with `outcome` and the context strings `ctx`, in order.
pub(crate) fn event_from_strings(outcome: &str, ctx: &[String]) -> Event {
    let preds: Vec<JavaString> = ctx.iter().map(|s| JavaString::new(s.as_str())).collect();
    Event::new(outcome, preds)
}

/// Adds the part-of-speech events of `p`, a parse pruned and with its heads updated, to `events`
/// (`AbstractParserEventStream.addTagEvents`): one per token of its initial chunks, the outcome
/// the token's tag and the context what `DefaultPOSContextGenerator(null)` makes of the tokens and
/// all of the sentence's tags.
pub(crate) fn add_tag_events(p: &Parse, events: &mut Vec<Event>) {
    let mut tokens = Vec::new();
    let mut tags = Vec::new();
    for chunk in p.initial_chunks() {
        let nodes = if chunk.is_pos_tag() { std::slice::from_ref(chunk) } else { chunk.children() };
        for node in nodes {
            tokens.push(node.covered_text());
            tags.push(node.kind());
        }
    }
    let mut generator = DefaultPosContext::new(None);
    let mut features = Features::new();
    for (i, tag) in tags.iter().enumerate() {
        features.clear();
        generator.features(&mut features, i, &tokens, &tags);
        let context =
            features.iter().map(|f| JavaString::from_modified_utf8(f).expect("a feature is modified UTF-8")).collect();
        events.push(Event::new(*tag, context));
    }
}

/// The chunker training events (`ChunkerEventStream` over `ChunkSampleStream`), one per token.
pub(crate) fn parser_chunker_events(samples: &[Parse]) -> Vec<Event> {
    let mut cg = ChunkContextGenerator::with_cache(CHUNKER_DEFAULT_BEAM_SIZE);
    let mut events = Vec::new();
    for p in samples {
        let cs = ParserChunkSample::from_parse(p);
        let words: Vec<&str> = cs.sentence.iter().map(String::as_str).collect();
        let tags: Vec<&str> = cs.tags.iter().map(String::as_str).collect();
        let preds: Vec<&str> = cs.preds.iter().map(String::as_str).collect();
        for i in 0..words.len() {
            let ctx = cg.get_context(i, &words, &tags, &preds);
            events.push(event_from_strings(preds[i], &ctx));
        }
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_events_use_each_sample_s_own_words() {
        // Same tags and outcomes, so the cache keys match; only the words differ.
        let samples: Vec<Parse> = ["(TOP (NP (DT The) (NN dog)))", "(TOP (NP (DT A) (NN cat)))"]
            .iter()
            .map(|s| Parse::parse(s).unwrap())
            .collect();
        let events = parser_chunker_events(&samples);
        assert_eq!(events.len(), 4);
        let has = |e: &Event, f: &str| e.context().iter().any(|c| c == f);
        assert!(has(&events[0], "0=The|DT"));
        assert!(has(&events[2], "0=A|DT"), "{:?}", events[2].context());
        assert!(has(&events[3], "0=cat|NN"), "{:?}", events[3].context());
    }
}
