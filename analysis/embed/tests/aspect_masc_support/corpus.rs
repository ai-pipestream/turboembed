use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use turbo_analysis::{Document, IxRange, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Sentences, Tokens};

pub type Aspect = (IxRange<Tokens>, &'static str, &'static str);
pub type Opinion = (IxRange<Tokens>, &'static str);

pub struct CorpusDoc {
    pub id: String,
    pub view: View,
    pub sentences: Sentences,
    pub tokens: Tokens,
    pub aspects: Vec<Aspect>,
    pub opinions: Vec<Opinion>,
}

#[derive(Default)]
struct Annotation {
    label: String,
    node: String,
    id: String,
    target_link: String,
    attitude_link: String,
    polarity: String,
    attitude_type: String,
}

pub fn load(root: &Path) -> Vec<CorpusDoc> {
    let mut paths = Vec::new();
    collect(root, &mut paths);
    paths.sort();
    let mut docs: Vec<CorpusDoc> = paths.iter().filter_map(|p| read_doc(p)).collect();
    docs.retain(|d| !d.aspects.is_empty());
    docs
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with("-mpqa.xml")) {
            out.push(path);
        }
    }
}

fn attr<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("{key}=\"");
    let (_, rest) = line.split_once(&marker)?;
    rest.split('"').next()
}

fn char_to_byte(text: &str) -> Vec<usize> {
    text.char_indices().map(|(b, _)| b).chain(std::iter::once(text.len())).collect()
}

fn span(text: &str, line: &str) -> Option<Span> {
    let mut anchors = attr(line, "anchors")?.split_whitespace();
    let start = anchors.next()?.parse::<usize>().ok()?;
    let end = anchors.next()?.parse::<usize>().ok()?;
    if anchors.next().is_some() || start >= end {
        return None;
    }
    let offsets = char_to_byte(text);
    if end >= offsets.len() {
        return None;
    }
    Some(Span::new(offsets[start] as u32, offsets[end] as u32))
}

fn regions(text: &str, xml: &str) -> (BTreeMap<String, Span>, Vec<Span>) {
    let mut regions = BTreeMap::new();
    let mut ordered = Vec::new();
    for line in xml.lines().map(str::trim) {
        if line.starts_with("<region ")
            && let (Some(id), Some(span)) = (attr(line, "xml:id"), span(text, line))
        {
            regions.insert(id.to_owned(), span);
            ordered.push(span);
        }
    }
    (regions, ordered)
}

fn linked_spans(text: &str, links: &str, regions_xml: &str) -> Vec<Span> {
    let (regions, _) = regions(text, regions_xml);
    let mut out = Vec::new();
    for line in links.lines().map(str::trim) {
        if line.starts_with("<link ")
            && let Some(targets) = attr(line, "targets")
        {
            let mut targets = targets.split_whitespace();
            let (Some(target), None) = (targets.next(), targets.next()) else { continue };
            if let Some(span) = regions.get(target) {
                out.push(*span);
            }
        }
    }
    out
}

fn node_regions(text: &str, xml: &str) -> (BTreeMap<String, Span>, Vec<Span>) {
    let (regions, ordered) = regions(text, xml);
    let mut nodes = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in xml.lines().map(str::trim) {
        if line.starts_with("<node ") {
            current = attr(line, "xml:id").map(str::to_owned);
        } else if line.starts_with("<link ")
            && let (Some(node), Some(targets)) = (&current, attr(line, "targets"))
        {
            let mut targets = targets.split_whitespace();
            let (Some(target), None) = (targets.next(), targets.next()) else { continue };
            if let Some(span) = regions.get(target) {
                nodes.insert(node.clone(), *span);
            }
        } else if line == "</node>" {
            current = None;
        }
    }
    (nodes, ordered)
}

fn annotations(xml: &str) -> Vec<Annotation> {
    let mut out = Vec::new();
    let mut current: Option<Annotation> = None;
    for line in xml.lines().map(str::trim) {
        if line.starts_with("<a ") {
            let label = attr(line, "label").unwrap_or("");
            if matches!(label, "target" | "attitude" | "direct-subjective") {
                current = Some(Annotation {
                    label: label.to_owned(),
                    node: attr(line, "ref").unwrap_or("").to_owned(),
                    ..Annotation::default()
                });
            }
        } else if line.starts_with("<f ")
            && let Some(a) = &mut current
        {
            let name = attr(line, "name").unwrap_or("");
            let value = attr(line, "value").unwrap_or("");
            match name {
                "id" => a.id = value.to_owned(),
                "target-link" => a.target_link = value.to_owned(),
                "attitude-link" => a.attitude_link = value.to_owned(),
                "polarity" => a.polarity = value.to_owned(),
                "attitude-type" => a.attitude_type = value.to_owned(),
                _ => {}
            }
        } else if line == "</a>"
            && let Some(a) = current.take()
        {
            out.push(a);
        }
    }
    out
}

fn token_range(tokens: &Tokens, span: Span) -> Option<IxRange<Tokens>> {
    let mut start = None;
    let mut end = None;
    for (i, (_, token)) in tokens.spans.iter().enumerate() {
        if token.start == span.start {
            start = Some(i as u32);
        }
        if token.end == span.end {
            end = Some(i as u32 + 1);
        }
    }
    let range = IxRange::new(start?, end?);
    (range.start < range.end).then_some(range)
}

fn polarity(attitude: &str) -> Option<&'static str> {
    match attitude {
        "sentiment-pos" | "agree-pos" | "arguing-pos" | "intention-pos" | "positive" => Some("positive"),
        "sentiment-neg" | "agree-neg" | "arguing-neg" | "intention-neg" | "negative" => Some("negative"),
        "sentiment-neutral" | "neutral" => Some("neutral"),
        _ => None,
    }
}

fn read_doc(mpqa_path: &Path) -> Option<CorpusDoc> {
    let stem = mpqa_path.file_name()?.to_str()?.strip_suffix("-mpqa.xml")?;
    let text = std::fs::read_to_string(mpqa_path.with_file_name(format!("{stem}.txt"))).ok()?;
    let mpqa = std::fs::read_to_string(mpqa_path).ok()?;
    let penn = std::fs::read_to_string(mpqa_path.with_file_name(format!("{stem}-penn.xml"))).ok()?;
    let seg = std::fs::read_to_string(mpqa_path.with_file_name(format!("{stem}-seg.xml"))).ok()?;
    let sent_xml = std::fs::read_to_string(mpqa_path.with_file_name(format!("{stem}-s.xml"))).ok()?;
    let (nodes, _) = node_regions(&text, &mpqa);
    let mut token_regions = linked_spans(&text, &penn, &seg);
    let (_, mut sent_regions) = node_regions(&text, &sent_xml);
    token_regions.sort_by_key(|span| (span.start, span.end));
    sent_regions.sort_by_key(|span| (span.start, span.end));
    if token_regions.is_empty() {
        return None;
    }
    let mut token_spans = Spans::new(ViewId::SOURCE);
    for span in token_regions {
        token_spans.push(span);
    }
    let tokens = Tokens { spans: token_spans, sentence: None };
    let mut sentence_spans = Spans::new(ViewId::SOURCE);
    for span in sent_regions {
        sentence_spans.push(span);
    }
    let sentences = Sentences { spans: sentence_spans };
    let annotations = annotations(&mpqa);
    let targets: BTreeMap<&str, Span> = annotations
        .iter()
        .filter(|a| a.label == "target")
        .filter_map(|a| Some((a.id.as_str(), *nodes.get(&a.node)?)))
        .collect();
    let attitudes: BTreeMap<&str, &Annotation> =
        annotations.iter().filter(|a| a.label == "attitude").map(|a| (a.id.as_str(), a)).collect();
    let mut aspects = Vec::new();
    let mut seen = BTreeSet::new();
    for attitude in attitudes.values() {
        let Some(label) = polarity(&attitude.attitude_type) else { continue };
        let Some(target) = targets.get(attitude.target_link.as_str()) else { continue };
        let Some(range) = token_range(&tokens, *target) else { continue };
        if seen.insert((range.start.get(), range.end.get(), label)) {
            aspects.push((range, "general", label));
        }
    }
    aspects.sort_by_key(|a| (a.0.start.get(), a.0.end.get()));
    let mut opinions = Vec::new();
    for a in annotations.iter().filter(|a| a.label == "direct-subjective") {
        let Some(attitude) = attitudes.get(a.attitude_link.as_str()) else { continue };
        let Some(label) = polarity(&a.polarity).or_else(|| polarity(&attitude.attitude_type)) else { continue };
        let Some(span) = nodes.get(&a.node) else { continue };
        let Some(range) = token_range(&tokens, *span) else { continue };
        opinions.push((range, label));
    }
    opinions.sort_by_key(|o| (o.0.start.get(), o.0.end.get()));
    opinions.dedup();
    let document = Document::new(text);
    let view = document.get::<View>().unwrap().clone();
    Some(CorpusDoc { id: stem.to_owned(), view, sentences, tokens, aspects, opinions })
}
