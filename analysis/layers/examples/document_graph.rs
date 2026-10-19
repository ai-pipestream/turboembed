//! Hands a document across the graph export boundary and reads it back
//! the way a search indexer or a coordinator would: register a codec
//! per layer type, export to bytes, import in a fresh scope, then
//! resolve nodes, edge endpoints and group members by `NodeRef` and map
//! a span in the normalized view back to source bytes.
//!
//! Run with: cargo run -p turbo-analysis-layers --example document_graph

use turbo_analysis::export::{Codecs, DocumentGraph, export, import, node_ref};
use turbo_analysis::{Document, Edge, Edges, Groups, Ix, Key, Provenance, Span, Tag, View, ViewBuilder};
use turbo_analysis_layers::{AbbreviationForm, Abbreviations, abbreviation_links};

fn main() {
    let text = "The  World Health Organization (WHO) met. The WHO met again.";
    let mut doc = Document::new(text);

    // A normalized view with the double space collapsed.
    let gap = text.find("  ").unwrap();
    let mut builder = ViewBuilder::new(text);
    builder.keep(gap);
    builder.replace(gap + 2, " ");
    let (clean, alignment) = builder.finish();
    let source = doc.get::<View>().unwrap().clone();
    doc.add(
        &Key::<View>::named("clean"),
        View::derived(&source, clean.clone(), alignment),
        Provenance::by("example:normalize"),
    )
    .unwrap();
    let clean_id = doc.get_key(&Key::<View>::named("clean")).unwrap().id();

    let span = |needle: &str, from: usize| {
        let at = clean[from..].find(needle).unwrap() + from;
        Span::new(at as u32, (at + needle.len()) as u32)
    };
    let mut abbrev = Abbreviations::new(clean_id);
    let long = abbrev.push(span("World Health Organization", 0), AbbreviationForm::Long);
    let short = abbrev.push(span("WHO", 0), AbbreviationForm::Short);
    let mention = abbrev.push(span("WHO", span("WHO", 0).end as usize), AbbreviationForm::Mention);
    doc.add(&Key::DEFAULT, abbrev, Provenance::by("example:abbrev")).unwrap();
    let links =
        Edges::new(abbreviation_links(), [Edge::new(short, long, Tag(0), 0.95), Edge::new(mention, long, Tag(1), 0.5)]);
    doc.add(&Key::DEFAULT, links, Provenance::by("example:abbrev")).unwrap();
    doc.add(&Key::DEFAULT, Groups::<Abbreviations>::new([[short, mention]]), Provenance::by("example:groups")).unwrap();

    // Out through the boundary and back, as bytes only.
    let mut codecs = Codecs::new();
    codecs
        .register::<Abbreviations>()
        .register::<Edges<Abbreviations, Abbreviations>>()
        .register::<Groups<Abbreviations>>();
    let bytes = export(&doc, &codecs).unwrap().to_bytes().unwrap();
    println!("exported {} bytes (format {})", bytes.len(), turbo_analysis::export::FORMAT_VERSION);
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();

    // Resolve everything by node identity on the imported document.
    let abbrev = back.get::<Abbreviations>().unwrap();
    let links = back.get::<Edges<Abbreviations, Abbreviations>>().unwrap();
    let view = back.text(abbrev.spans.view());
    for (_, edge) in links.iter() {
        let from = node_ref(&links.from_key, edge.from);
        let to = node_ref(&links.to_key, edge.to);
        let source_span = back.to_source(abbrev.spans.view(), abbrev.spans.get(edge.to));
        println!(
            "{from} -[{} {:.2}]-> {to} (\"{}\", in source bytes \"{}\")",
            links.kinds.label(edge.kind),
            edge.score,
            &view[abbrev.spans.get(edge.to).range()],
            &back.source()[source_span.range()],
        );
    }
    let groups = back.get::<Groups<Abbreviations>>().unwrap();
    for (group, members) in groups.iter() {
        let names: Vec<String> = members.iter().map(|&m| node_ref(&groups.of_key, m).to_string()).collect();
        println!("group {}: {}", group.get(), names.join(", "));
    }
    let provenance = back.provenance(Key::<Edges<Abbreviations, Abbreviations>>::DEFAULT.id()).unwrap();
    println!("links made by {} (config {:016x})", provenance.annotator, provenance.config);
    assert_eq!(abbrev.form(Ix::new(0)), AbbreviationForm::Long);
}
