use turbo_analysis::{Layer, ViewId};
use turbo_analysis_layers::Block;
use turbo_structure::{Syntax, detect};

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
        }
    }
    println!("CommonMark block trees: {pass} pass; failures {failures:?}");
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
