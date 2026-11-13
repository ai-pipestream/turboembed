//! Writes the graphs the demo browser tests open: the named/default
//! positional token graph, and, when a second path is given, a graph with
//! entities and the relations between them.
use turbo_analysis::{Document, Ix, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_kserve::results::document::codecs;
use turbo_relation::Relations;

type Failure = Box<dyn std::error::Error>;

fn write(path: &str, doc: &Document) -> Result<(), Failure> {
    let graph = turbo_analysis::export::export(doc, codecs())?.to_bytes()?;
    std::fs::write(path, graph)?;
    Ok(())
}

/// `é 😀` with a default token on `é` and a `gold` token on `😀`.
fn tokens() -> Result<Document, Failure> {
    let mut doc = Document::new("é 😀");
    for (instance, span) in [("", Span::new(0, 2)), ("gold", Span::new(3, 7))] {
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(span);
        doc.add(&Key::named(instance), Tokens { spans, sentence: None }, Provenance::by("browser-test"))?;
    }
    Ok(doc)
}

const RELATION_TEXT: &str = "Ada Lovelace worked for Charles Babbage in London. Babbage was born in London.";
/// The entities of [`RELATION_TEXT`]: byte span, token range, kind.
const ENTITIES: [(u32, u32, u32, u32, u16); 5] =
    [(0, 12, 0, 2, 0), (24, 39, 4, 6, 0), (43, 49, 7, 8, 1), (51, 58, 9, 10, 0), (71, 77, 13, 14, 1)];
/// The relations between them: subject, object, kind, score.
const RELATIONS: [(u32, u32, u16, f64); 3] = [(0, 1, 0, 0.9), (1, 2, 1, 0.75), (3, 4, 2, 0.5)];

fn spans<L>(all: impl IntoIterator<Item = (u32, u32)>) -> Spans<L> {
    let mut spans = Spans::new(ViewId::SOURCE);
    for (start, end) in all {
        spans.push(Span::new(start, end));
    }
    spans
}

/// Two sentences, three people and two places in five mentions, and three
/// relations, as a relation model would write them.
fn relations() -> Result<Document, Failure> {
    let mut doc = Document::new(RELATION_TEXT);
    let by = || Provenance::by("browser-test");
    doc.add(&Key::DEFAULT, Sentences { spans: spans([(0, 50), (51, 78)]) }, by())?;
    // A token per word and per full stop.
    let mut words = Vec::new();
    let mut start = None;
    for (i, c) in RELATION_TEXT.char_indices() {
        if c.is_alphanumeric() {
            start.get_or_insert(i as u32);
            continue;
        }
        if let Some(s) = start.take() {
            words.push((s, i as u32));
        }
        if c == '.' {
            words.push((i as u32, i as u32 + 1));
        }
    }
    doc.add(&Key::DEFAULT, Tokens { spans: spans(words), sentence: None }, by())?;
    doc.add(
        &Key::DEFAULT,
        Entities {
            spans: spans(ENTITIES.map(|e| (e.0, e.1))),
            tokens: ENTITIES.map(|e| IxRange::new(e.2, e.3)).to_vec(),
            kind: ENTITIES.map(|e| Tag(e.4)).to_vec(),
            kinds: TagSet::new("entity", &["person", "location"]),
            score: vec![1.0; ENTITIES.len()],
        },
        by(),
    )?;
    let ends = |r: &(u32, u32, u16, f64)| (ENTITIES[r.0 as usize].0, ENTITIES[r.1 as usize].1);
    doc.add(
        &Key::DEFAULT,
        Relations {
            spans: spans(RELATIONS.iter().map(ends)),
            subject: RELATIONS.map(|r| Ix::new(r.0)).to_vec(),
            object: RELATIONS.map(|r| Ix::new(r.1)).to_vec(),
            kind: RELATIONS.map(|r| Tag(r.2)).to_vec(),
            kinds: TagSet::new("relation", &["works for", "located in", "born in"]),
            score: RELATIONS.map(|r| r.3).to_vec(),
        },
        by(),
    )?;
    Ok(doc)
}

fn main() -> Result<(), Failure> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("expected output graph path")?;
    write(&path, &tokens()?)?;
    if let Some(path) = args.next() {
        write(&path, &relations()?)?;
    }
    Ok(())
}
