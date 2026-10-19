//! Boilerplate across a batch of documents and inside one, and its
//! projection from structure blocks to sentences.

use turbo_analysis::*;
use turbo_analysis_layers::{Block, Boilerplate, BoilerplateKind, Sentences, Structure};
use turbo_group::{BoilerplateDetector, BoilerplateProjector, Sketcher};
use turbo_structure::{Syntax, detect};

const DISCLAIMER: &str = "This message and any attachments are confidential and intended solely for the addressee. \
     If you received it in error, notify the sender and delete it. Opinions expressed are those of the author \
     and not of the company.";

/// A page of the site: header, navigation, its own text, a numbered
/// footer and the disclaimer.
fn page(n: usize, body: &str) -> String {
    format!(
        "ACME intranet home\n\n- Home page\n- About us\n- Contact us\n\n{body}\n\nPage {n} of 3, ACME Corp 2024\n\n{DISCLAIMER}"
    )
}

fn document(text: &str) -> Document {
    let mut doc = Document::new(text.to_owned());
    doc.add(&Key::DEFAULT, detect(text, Syntax::Markdown, ViewId::SOURCE), Provenance::by("t")).unwrap();
    doc
}

fn run(docs: &mut [Document]) {
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(Sketcher::<Structure>::default())
        .add(BoilerplateDetector::<Structure>::default())
        .build()
        .unwrap();
    assert!(pipeline.run(docs).iter().all(Result::is_ok));
}

/// Each block's kind and text, flagged blocks only, leaves only (no
/// sections or lists around them).
fn flagged(doc: &Document) -> Vec<(Block, Option<BoilerplateKind>, f32, String)> {
    let st = doc.get::<Structure>().unwrap();
    let b = doc.get::<Boilerplate<Structure>>().unwrap();
    st.spans
        .iter()
        .filter(|(i, _)| matches!(st.kind[i.get()], Block::Paragraph | Block::ListItem))
        .map(|(i, s)| {
            let text: String = doc.source()[s.range()].chars().take(18).collect();
            (st.kind[i.get()], b.kind[i.get()], b.score[i.get()], text)
        })
        .collect()
}

#[test]
fn blocks_repeated_across_a_batch_are_headers_navigation_footers_and_disclaimers() {
    let mut docs = [
        document(&page(1, "Quarterly results were strong across every region we serve.")),
        document(&page(2, "The new office opens in March with room for two hundred people.")),
        document(&page(3, "Security training is due for everyone by the end of the month.")),
    ];
    run(&mut docs);
    use BoilerplateKind::*;
    for doc in &docs {
        let f = flagged(doc);
        let kinds: Vec<(Block, Option<BoilerplateKind>)> = f.iter().map(|r| (r.0, r.1)).collect();
        assert_eq!(
            kinds,
            [
                (Block::Paragraph, Some(Header)),
                (Block::ListItem, Some(Navigation)),
                (Block::Paragraph, Some(Navigation)),
                (Block::ListItem, Some(Navigation)),
                (Block::Paragraph, Some(Navigation)),
                (Block::ListItem, Some(Navigation)),
                (Block::Paragraph, Some(Navigation)),
                (Block::Paragraph, None),
                (Block::Paragraph, Some(Footer)),
                (Block::Paragraph, Some(Disclaimer)),
            ],
            "{f:?}"
        );
        // Every page holds them: the whole batch's share.
        assert!(f.iter().all(|r| r.1.is_none() || r.2 == 1.0), "{f:?}");
        assert!(f.iter().all(|r| r.1.is_some() || r.2 == 0.0), "{f:?}");
    }
}

#[test]
fn a_block_held_by_too_few_documents_is_content() {
    let mut docs = [
        document("Shared notice of the week\n\nOne."),
        document("Shared notice of the week\n\nTwo."),
        document("Unrelated text here\n\nThree."),
        document("More unrelated text\n\nFour."),
        document("Still unrelated text\n\nFive."),
    ];
    run(&mut docs);
    // Two of five documents is under half the batch.
    assert!(docs.iter().all(|d| d.get::<Boilerplate<Structure>>().unwrap().flagged().next().is_none()));
}

#[test]
fn spans_repeated_inside_one_document_are_templates() {
    let text = "Intro text of the page.\n\nBack to top link\n\nMiddle text here.\n\nBack to top link\n\nEnd text.\n\nBack to top link";
    let mut docs = [document(text)];
    run(&mut docs);
    let f = flagged(&docs[0]);
    let templates: Vec<_> = f.iter().filter(|r| r.1.is_some()).map(|r| (r.1, r.2, r.3.as_str())).collect();
    let two_thirds = 1.0 - 1.0 / 3.0;
    assert_eq!(templates, [(Some(BoilerplateKind::Template), two_thirds, "Back to top link"); 3]);
}

#[test]
fn a_document_alone_finds_templates_but_no_cross_document_boilerplate() {
    let mut docs = [document(&page(1, "Only page."))];
    run(&mut docs);
    assert!(flagged(&docs[0]).iter().all(|r| r.1.is_none()));
}

#[test]
fn flags_project_from_blocks_to_the_sentences_over_them() {
    let mut docs =
        [document(&page(1, "Own text one. More own text.")), document(&page(2, "Own text two. Still more."))];
    run(&mut docs);
    let doc = &mut docs[0];
    let text = doc.source().to_owned();
    // Sentences by hand: the body's two, one inside the footer, and an
    // empty one inside it.
    let at = |s: &str| text.find(s).unwrap() as u32;
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(at("Own text one."), at("Own text one.") + 13));
    spans.push(Span::new(at("More own"), at("More own") + 15));
    spans.push(Span::new(at("Page 1"), at("Page 1") + 6));
    spans.push(Span::new(at("Page 1") + 7, at("Page 1") + 7));
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<Structure>::DEFAULT)
        .given(Key::<Boilerplate<Structure>>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .add(BoilerplateProjector::<Sentences>::default())
        .build()
        .unwrap();
    assert_eq!(pipeline.run(std::slice::from_mut(doc)), [Ok(())]);
    let b = doc.get::<Boilerplate<Sentences>>().unwrap();
    assert_eq!(b.kind, [None, None, Some(BoilerplateKind::Footer), Some(BoilerplateKind::Footer)]);
    assert_eq!(b.score, [0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn a_bad_share_is_refused_for_every_document() {
    let mut docs = [document("a b"), document("c d")];
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(Sketcher::<Structure>::default())
        .add(BoilerplateDetector::<Structure>::new(2, 1.5, 2, 25))
        .build()
        .unwrap();
    assert!(pipeline.run(&mut docs).iter().all(|r| r.as_ref().unwrap_err().to_string().contains("1.5")));
}
