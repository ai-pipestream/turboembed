//! Integration and linguistic tests for `GrammarDiagnostics` and `GrammarChecker` (#1464).

use std::sync::Arc;

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{
    Analyzer, Annotator, CodePointOffset, CodePointSpan, Document, Ix, Key, Layer, Provenance, Span, Spans, TagSet,
    Utf16Offset, Utf16Span, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{
    Dependencies, DependencyGraph, GrammarCategory, GrammarChecker, GrammarDiagnostic, GrammarDiagnostics, Lemmas,
    PosTags, Sentences, Strs, Tokens, builtin_rule_id,
};

/// Builds a `Document` from `text` and whitespace-delimited `"surface/POS"` or `"surface/POS/lemma"` specs.
fn build_doc(text: &str, tagged_tokens: &str) -> Document {
    let mut doc = Document::new(text);
    let mut sentences = Sentences { spans: Spans::new(ViewId::SOURCE) };
    if !text.is_empty() {
        sentences.spans.push(Span::new(0, text.len() as u32));
    }

    let parsed: Vec<(&str, &str, Option<&str>)> = tagged_tokens
        .split_whitespace()
        .map(|entry| {
            let mut parts = entry.splitn(3, '/');
            let surf = parts.next().unwrap();
            let pos = parts.next().expect("missing POS in token spec");
            let lemma = parts.next();
            (surf, pos, lemma)
        })
        .collect();

    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut tok_sentence = Vec::with_capacity(parsed.len());
    let mut tag_names = Vec::new();
    for &(_, pos, _) in &parsed {
        if !tag_names.contains(&pos) {
            tag_names.push(pos);
        }
    }
    if tag_names.is_empty() {
        tag_names.push("NN");
    }
    let tagset = TagSet::new("penn", &tag_names);
    let mut tags = Vec::with_capacity(parsed.len());
    let mut lemmas = Strs::default();
    let mut has_lemmas = false;

    let mut cursor = 0usize;
    for &(surf, pos, lemma) in &parsed {
        let rel =
            text[cursor..].find(surf).unwrap_or_else(|| panic!("token {surf:?} not found after {cursor} in {text:?}"));
        let start = cursor + rel;
        let end = start + surf.len();
        cursor = end;
        tok_spans.push(Span::new(start as u32, end as u32));
        tok_sentence.push(Ix::new(0));
        tags.push(tagset.tag(pos).unwrap());
        if let Some(l) = lemma {
            has_lemmas = true;
            lemmas.push(l);
        } else {
            lemmas.push(&surf.to_ascii_lowercase());
        }
    }

    doc.add(&Key::<Sentences>::DEFAULT, sentences, Provenance::by("test:sent")).unwrap();
    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: tok_spans, sentence: Some(tok_sentence) },
        Provenance::by("test:tok"),
    )
    .unwrap();
    doc.add(&Key::<PosTags>::DEFAULT, PosTags { tagset, tags, prob: None }, Provenance::by("test:pos")).unwrap();
    if has_lemmas {
        doc.add(&Key::<Lemmas>::DEFAULT, Lemmas { lemma: lemmas }, Provenance::by("test:lemma")).unwrap();
    }
    doc
}

#[test]
fn detects_all_seven_grammar_rule_categories_with_exact_spans_and_repairs() {
    // 1. SVA-001 across intervening PP ("The box of chocolates are" -> "is")
    let doc = build_doc(
        "The box of chocolates are on the table.",
        "The/DT box/NN of/IN chocolates/NNS are/VBP/be on/IN the/DT table/NN ./.",
    );
    let diags = GrammarChecker::new().check_document(&doc).unwrap();
    assert_eq!(diags.len(), 1, "unexpected diagnostics: {diags:?}");
    let d = &diags.diagnostics[0];
    assert_eq!(d.rule_id, "SVA-001");
    assert_eq!(d.category, GrammarCategory::SubjectVerbAgreement);
    assert_eq!(&doc.text(ViewId::SOURCE)[d.span.range()], "are");
    assert_eq!(d.replacements.iter().map(AsRef::as_ref).collect::<Vec<_>>(), vec!["is"]);

    // 2. SVA-001 lexical verb ("The dogs barks" -> "bark", "The dog bark" -> "barks")
    let doc = build_doc("The dogs barks loudly.", "The/DT dogs/NNS barks/VBZ/bark loudly/RB ./.");
    let diags = GrammarChecker::new().check_document(&doc).unwrap();
    assert_eq!(diags.len(), 1);
    assert_eq!(diags.diagnostics[0].rule_id, "SVA-001");
    assert_eq!(diags.diagnostics[0].replacements[0].as_ref(), "bark");

    let doc = build_doc("The dog bark loudly.", "The/DT dog/NN bark/VBP/bark loudly/RB ./.");
    let diags = GrammarChecker::new().check_document(&doc).unwrap();
    assert_eq!(diags.len(), 1);
    assert_eq!(diags.diagnostics[0].rule_id, "SVA-001");
    assert_eq!(diags.diagnostics[0].replacements[0].as_ref(), "barks");

    // 3. DET-001 ("these kind", "much books", "one of the student")
    let doc = build_doc(
        "These kind of errors and much books confuse one of the student.",
        "These/DT kind/NN of/IN errors/NNS and/CC much/JJ books/NNS confuse/VBP one/CD of/IN the/DT student/NN ./.",
    );
    let diags = GrammarChecker::new().check_document(&doc).unwrap();
    let det_reps: Vec<(&str, &str)> = diags
        .iter()
        .filter(|d| d.rule_id == "DET-001")
        .map(|d| (&doc.text(ViewId::SOURCE)[d.span.range()], d.replacements[0].as_ref()))
        .collect();
    assert_eq!(det_reps, vec![("These", "This"), ("much", "many"), ("student", "students")]);

    // 4. ART-001 ("a hour", "an university", "a MRI", "an URL")
    let doc = build_doc(
        "Wait a hour at an university for a MRI or an URL.",
        "Wait/VB a/DT hour/NN at/IN an/DT university/NN for/IN a/DT MRI/NNP or/CC an/DT URL/NNP ./.",
    );
    let diags = GrammarChecker::new().check_document(&doc).unwrap();
    let art_reps: Vec<(u32, &str, &str)> = diags
        .iter()
        .filter(|d| d.rule_id == "ART-001")
        .map(|d| (d.span.start, &doc.text(ViewId::SOURCE)[d.span.range()], d.replacements[0].as_ref()))
        .collect();
    assert_eq!(art_reps, vec![(5, "a", "an"), (15, "an", "a"), (33, "a", "an"), (42, "an", "a")]);

    // 5. AUX-001 ("should went", "did not bought")
    let doc = build_doc(
        "They should went home and did not bought tickets.",
        "They/PRP should/MD went/VBD/go home/NN and/CC did/VBD not/RB bought/VBD/buy tickets/NNS ./.",
    );
    let diags = GrammarChecker::new().check_document(&doc).unwrap();
    let aux_reps: Vec<(&str, &str)> = diags
        .iter()
        .filter(|d| d.rule_id == "AUX-001")
        .map(|d| (&doc.text(ViewId::SOURCE)[d.span.range()], d.replacements[0].as_ref()))
        .collect();
    assert_eq!(aux_reps, vec![("went", "go"), ("bought", "buy")]);

    // 6. REP-001 ("the the")
    let doc = build_doc("Read the the manual carefully.", "Read/VB the/DT the/DT manual/NN carefully/RB ./.");
    let diags = GrammarChecker::new().check_document(&doc).unwrap();
    assert_eq!(diags.len(), 1);
    assert_eq!(diags.diagnostics[0].rule_id, "REP-001");
    assert_eq!(&doc.text(ViewId::SOURCE)[diags.diagnostics[0].span.range()], "the the");
    assert_eq!(diags.diagnostics[0].replacements[0].as_ref(), "the");

    // 7. CASE-001 ("between you and I", "with he")
    let doc = build_doc(
        "Between you and I, speak with he.",
        "Between/IN you/PRP and/CC I/PRP ,/, speak/VB with/IN he/PRP ./.",
    );
    let diags = GrammarChecker::new().check_document(&doc).unwrap();
    let case_reps: Vec<(&str, &str)> = diags
        .iter()
        .filter(|d| d.rule_id == "CASE-001")
        .map(|d| (&doc.text(ViewId::SOURCE)[d.span.range()], d.replacements[0].as_ref()))
        .collect();
    assert_eq!(case_reps, vec![("I", "me"), ("he", "him")]);

    // 8. PUNCT-001 ("..", ", ,")
    let doc = build_doc("Wait , , now..", "Wait/VB ,/, ,/, now/RB ../.");
    let diags = GrammarChecker::new().check_document(&doc).unwrap();
    let punct_reps: Vec<(&str, &str)> = diags
        .iter()
        .filter(|d| d.rule_id == "PUNCT-001")
        .map(|d| (&doc.text(ViewId::SOURCE)[d.span.range()], d.replacements[0].as_ref()))
        .collect();
    assert_eq!(punct_reps, vec![(", ,", ","), ("..", ".")]);
}

#[test]
fn tricky_grammatical_sentences_produce_zero_false_positives() {
    let valid_cases: &[(&str, &str)] = &[
        (
            "The key to the cabinets is on the table.",
            "The/DT key/NN to/IN the/DT cabinets/NNS is/VBZ on/IN the/DT table/NN ./.",
        ),
        (
            "The keys to the cabinet are on the table.",
            "The/DT keys/NNS to/IN the/DT cabinet/NN are/VBP on/IN the/DT table/NN ./.",
        ),
        (
            "The scientist who studies these planets is presenting today.",
            "The/DT scientist/NN who/WP studies/VBZ these/DT planets/NNS is/VBZ presenting/VBG today/NN ./.",
        ),
        (
            "The scientists who study this planet are presenting today.",
            "The/DT scientists/NNS who/WP study/VBP this/DT planet/NN are/VBP presenting/VBG today/NN ./.",
        ),
        (
            "There is a problem and there are two solutions.",
            "There/EX is/VBZ a/DT problem/NN and/CC there/EX are/VBP two/CD solutions/NNS ./.",
        ),
        ("Alice and Bob are ready.", "Alice/NNP and/CC Bob/NNP are/VBP ready/JJ ./."),
        (
            "She had had enough time to notice that that example was valid.",
            "She/PRP had/VBD had/VBN enough/JJ time/NN to/TO notice/VB that/IN that/DT example/NN was/VBD valid/JJ ./.",
        ),
        (
            "A university in a European country issued a one-way permit for a URL.",
            "A/DT university/NN in/IN a/DT European/JJ country/NN issued/VBD a/DT one-way/JJ permit/NN for/IN a/DT URL/NNP ./.",
        ),
        (
            "An hour later, an honest engineer filed an MRI report over an HTTP link.",
            "An/DT hour/NN later/RB ,/, an/DT honest/JJ engineer/NN filed/VBD an/DT MRI/NNP report/NN over/IN an/DT HTTP/NNP link/NN ./.",
        ),
    ];

    let checker = GrammarChecker::new();
    for &(text, specs) in valid_cases {
        let doc = build_doc(text, specs);
        let diags = checker.check_document(&doc).unwrap();
        assert!(diags.is_empty(), "false positive on {text:?}: {diags:?}");
    }
}

#[test]
fn dependency_graph_guides_agreement_and_projection_and_wire_round_trip() {
    let mut doc = build_doc(
        "😀 The report on the servers are ready.",
        "The/DT report/NN on/IN the/DT servers/NNS are/VBP/be ready/JJ ./.",
    );
    let rels: Arc<TagSet> = TagSet::new("ud", &["root", "det", "nsubj", "case", "nmod", "cop", "punct"]);
    let deps: DependencyGraph = Dependencies {
        head: vec![
            Some(Ix::new(1)),
            Some(Ix::new(6)),
            Some(Ix::new(4)),
            Some(Ix::new(4)),
            Some(Ix::new(1)),
            Some(Ix::new(6)),
            None,
            Some(Ix::new(6)),
        ],
        relation: vec![
            rels.tag("det").unwrap(),
            rels.tag("nsubj").unwrap(),
            rels.tag("case").unwrap(),
            rels.tag("det").unwrap(),
            rels.tag("nmod").unwrap(),
            rels.tag("cop").unwrap(),
            rels.tag("root").unwrap(),
            rels.tag("punct").unwrap(),
        ],
        relations: rels,
    };
    doc.add(&Key::<Dependencies>::DEFAULT, deps, Provenance::by("test:deps")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .given(Key::<Lemmas>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .add(GrammarChecker::new())
        .build()
        .unwrap();
    let res = analyzer.run(std::slice::from_mut(&mut doc));
    assert!(res[0].is_ok());

    let diags = doc.get::<GrammarDiagnostics>().unwrap().clone();
    assert_eq!(diags.len(), 1);
    let d0 = &diags.diagnostics[0];
    assert_eq!(d0.span, Span::new(31, 34));
    assert_eq!(d0.rule_id, "SVA-001");
    assert_eq!(d0.replacements[0].as_ref(), "is");

    // Verify UTF-16 and Unicode scalar (code-point) span offsets across the 4-byte astral emoji prefix.
    let u16_span = doc.view_span_to_utf16(ViewId::SOURCE, d0.span).unwrap();
    let cp_span = doc.view_span_to_code_point(ViewId::SOURCE, d0.span).unwrap();
    assert_eq!(u16_span, Utf16Span::new(Utf16Offset(29), Utf16Offset(32)));
    assert_eq!(cp_span, CodePointSpan::new(CodePointOffset(28), CodePointOffset(31)));

    // Derived trimmed view (`ViewId(1)`): project source -> trimmed -> source.
    let source = doc.get::<View>().unwrap().clone();
    let mut vb = ViewBuilder::new(source.text());
    vb.replace(5, "");
    let (trimmed_text, align) = vb.finish();
    doc.add(&Key::<View>::named("trimmed"), View::derived(&source, trimmed_text, align), Provenance::by("test:trim"))
        .unwrap();
    let trimmed_view = doc.get_key(&Key::<View>::named("trimmed")).unwrap().id();
    let projected = doc.project_layer(&diags, trimmed_view).unwrap();
    assert_eq!(projected.view(), trimmed_view);
    assert_eq!(projected.diagnostics[0].span, Span::new(26, 29));
    assert_eq!(&doc.text(trimmed_view)[projected.diagnostics[0].span.range()], "are");

    let back_to_source = doc.project_layer(&projected, ViewId::SOURCE).unwrap();
    assert_eq!(back_to_source, diags);
    doc.add(&Key::<GrammarDiagnostics>::named("trimmed"), projected, Provenance::by("test:proj")).unwrap();

    // Binary DocumentGraph wire round-trip.
    let reg = codecs();
    let graph = export(&doc, &reg).unwrap();
    let bytes = graph.to_bytes().unwrap();
    let restored_graph = DocumentGraph::from_bytes(&bytes).unwrap();
    let restored_doc = import(&restored_graph, &reg).unwrap();
    assert_eq!(restored_doc.get::<GrammarDiagnostics>().unwrap(), &diags);
    assert_eq!(
        restored_doc.get_key(&Key::<GrammarDiagnostics>::named("trimmed")).unwrap().diagnostics[0].span,
        Span::new(26, 29)
    );
}

#[test]
fn config_hash_and_malformed_wire_import_and_rule_id_helpers() {
    let base = GrammarChecker::new();
    let hashes = [
        base.config_hash(),
        base.with_subject_verb_agreement(false).config_hash(),
        base.with_determiner_noun_agreement(false).config_hash(),
        base.with_article_phonetics(false).config_hash(),
        base.with_auxiliary_verb_form(false).config_hash(),
        base.with_repeated_tokens(false).config_hash(),
        base.with_case_government(false).config_hash(),
        base.with_clause_punctuation(false).config_hash(),
        base.with_american_silent_h_herb(false).config_hash(),
    ];
    for i in 0..hashes.len() {
        for j in (i + 1)..hashes.len() {
            assert_ne!(hashes[i], hashes[j], "config_hash collision between {i} and {j}");
        }
    }
    assert_eq!(builtin_rule_id("SVA-001"), Some("SVA-001"));
    assert_eq!(builtin_rule_id("UNKNOWN"), None);

    // Verify Layer::check and DocumentGraph import reject malformed payloads.
    let mut doc = Document::new("hello world");
    let mut bad_layer = GrammarDiagnostics::new(ViewId::SOURCE);
    bad_layer.push(GrammarDiagnostic::new(
        Span::new(0, 5),
        "SVA-001",
        GrammarCategory::SubjectVerbAgreement,
        "bad confidence",
        ["hi"],
        1.5,
    ));
    assert!(bad_layer.check().is_err());

    let mut good_layer = GrammarDiagnostics::new(ViewId::SOURCE);
    good_layer.push(GrammarDiagnostic::new(
        Span::new(0, 5),
        "SVA-001",
        GrammarCategory::SubjectVerbAgreement,
        "valid diagnostic",
        ["hi"],
        0.95,
    ));
    doc.add(&Key::<GrammarDiagnostics>::DEFAULT, good_layer, Provenance::by("test:good")).unwrap();
    let reg = codecs();
    let mut graph = export(&doc, &reg).unwrap();
    let body = graph
        .layers
        .iter_mut()
        .find(|r| r.layer == GrammarDiagnostics::NAME.to_string())
        .unwrap()
        .body
        .as_mut()
        .unwrap();
    let end = body.len();
    body[end - 4..].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    assert!(import(&graph, &reg).is_err());
}
