use turbo_analysis::{Annotator, Document, Key, Layer, Provenance, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{Block, Structure};
use turbo_structure::{StructureProjector, Syntax, detect};

fn decode(hex: &str) -> String {
    String::from_utf8((0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect())
        .unwrap()
}

/// Block topology, heading levels and list tightness, not inline HTML
/// rendering. Section wrappers and thematic breaks are outside this layer.
#[test]
fn commonmark_block_trees() {
    let mut pass = 0;
    let mut failures = Vec::new();
    let mut differences = Vec::new();
    for line in include_str!("data/commonmark.tsv").lines() {
        let fields: Vec<_> = line.split('\t').collect();
        let text = decode(fields[2]);
        let tree = detect(&text, Syntax::Markdown, ViewId::SOURCE);
        tree.check().unwrap();
        let mut indexes = vec![None; tree.kind.len()];
        let mut nodes = Vec::new();
        for (i, kind) in tree.kind.iter().enumerate() {
            let mut parent = tree.parent[i];
            while parent.is_some_and(|p| tree.kind[p.get()] == Block::Section) {
                parent = tree.parent[parent.unwrap().get()];
            }
            if *kind == Block::Section {
                continue;
            }
            indexes[i] = Some(nodes.len());
            let parent = parent.map_or(-1, |p| indexes[p.get()].unwrap() as i32);
            let level = if *kind == Block::Heading { tree.depth[i] } else { 0 };
            nodes.push(format!("{kind:?}:{parent}:{level}"));
        }
        if nodes.join(",") == fields[3] {
            pass += 1;
        } else {
            failures.push(fields[0].parse::<u32>().unwrap());
            let reason = if fields[0] == "96" { "front-matter" } else { "reference-definition" };
            let actual = nodes.join(",");
            let actual = if actual.is_empty() { "<empty>" } else { &actual };
            let expected = if fields[3].is_empty() { "<empty>" } else { fields[3] };
            differences.push(format!("{}\t{}\t{reason}\t{actual}\t{expected}", fields[0], fields[1]));
        }
    }
    println!("CommonMark block trees: {pass} pass; failures {failures:?}");
    let ledger = differences.join("\n") + "\n";
    assert_eq!(ledger, include_str!("data/commonmark-differences.tsv"));
    // Known gaps: reference definitions stay paragraphs, plus front matter
    // example 96. Pin every gap so new regressions cannot hide in a count.
    assert_eq!(pass, 577);
    assert_eq!(
        failures,
        [
            23, 33, 96, 192, 193, 194, 195, 196, 198, 200, 202, 203, 204, 205, 206, 207, 214, 216, 217, 218, 317, 527,
            528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 549, 550, 553,
            554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 573, 576, 577,
            582, 583, 584, 585, 586, 587, 588, 589, 591, 592, 593
        ]
    );
}

/// Exercise the actual projector on all published spec inputs, including
/// the documented block differences. Newline expansion changes byte offsets
/// without deleting a block or making its endpoints ambiguous.
#[test]
fn commonmark_blocks_project_through_crlf_and_back() {
    let mut examples = 0;
    let mut blocks = 0;
    let mut shifted = 0;
    for line in include_str!("data/commonmark.tsv").lines() {
        let fields: Vec<_> = line.split('\t').collect();
        let text = decode(fields[2]);
        let mut doc = Document::new(text.clone());
        let original = detect(&text, Syntax::Markdown, ViewId::SOURCE);
        let mut builder = ViewBuilder::new(&text);
        for (i, c) in text.char_indices() {
            if c == '\n' {
                builder.keep(i);
                builder.replace(i + 1, "\r\n");
            }
        }
        let (expanded, align) = builder.finish();
        let child = View::derived(doc.get::<View>().unwrap(), expanded, align);
        doc.add(&Key::<View>::named("crlf"), child, Provenance::by("test")).unwrap();
        let child = doc.get_key(&Key::<View>::named("crlf")).unwrap();
        let (projected,) = StructureProjector.annotate((child, &original)).unwrap();
        assert_eq!(projected.kind, original.kind, "example {}", fields[0]);
        assert_eq!(projected.parent, original.parent, "example {}", fields[0]);
        assert_eq!(projected.depth, original.depth, "example {}", fields[0]);
        for ((_, source), (_, target)) in original.spans.iter().zip(projected.spans.iter()) {
            assert_eq!(child.alignment().span_to_child(source), target, "example {}", fields[0]);
            assert_eq!(doc.to_source(child.id(), target), source, "example {}", fields[0]);
            assert_eq!(child.text()[target.range()].replace("\r\n", "\n"), text[source.range()]);
            shifted += usize::from(source != target);
            blocks += 1;
        }
        doc.add(&Key::<Structure>::named("source"), original, Provenance::by("test")).unwrap();
        doc.add(&Key::<Structure>::named("crlf"), projected, Provenance::by("test")).unwrap();
        examples += 1;
    }
    assert_eq!(examples, 652);
    assert!(blocks > 652);
    assert!(shifted > 100);
    println!("CommonMark projection: {examples} examples, {blocks} blocks, {shifted} shifted spans");
}
