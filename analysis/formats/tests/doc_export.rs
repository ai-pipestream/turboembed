use std::sync::Arc;

use turbo_analysis::{
    Document, Edge, Edges, Ix, IxRange, Key, Layer as _, Provenance, Span, Spans, TagSet, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Categories, Chunks, Cut, EmotionDimension, EmotionItem, EmotionLayer, EmotionScores,
    EmotionTarget, Entities, Language, Lemmas, MorphFeatures, Opinion, Opinions, Pii, PiiConfidence, PiiKind,
    PiiMention, PolarityScores, PosTags, Sentences, Stems, Strs, Tokens, WordDependencies, WordLemmas, WordPosTags,
    Words,
};
use turbo_formats::conllu::{ConlluStream, ConlluWordReader};
use turbo_formats::doc_export::{
    BratFormatter, ConlluFormatter, CsvFormatter, DocumentFormatter, ExportOffsetMode, FormatExportError,
    FormatterRegistry, JsonLdFormatter, MarkdownFormatter, OutputFormatDescriptor, TsvFormatter, WarcFormatter,
    parse_brat_with_offsets, parse_warc_records,
};
use turbo_formats::ner::brat::{AnnotationValue, BratDocument};
use turbo_stopword_layers::{StopwordLanguage, Stopwords};
use turbo_unicode::WhitespaceMode;

#[test]
fn registry_standard_orders_and_validates_format_ids() {
    let mut reg = FormatterRegistry::standard();
    assert_eq!(reg.len(), 7);
    assert!(!reg.is_empty());

    let ids: Vec<&str> = reg.list().into_iter().map(|d| d.format_id).collect();
    assert_eq!(ids, vec!["brat", "conllu", "csv", "jsonld", "markdown", "tsv", "warc"]);

    for id in &ids {
        let fmt = reg.get(id).expect("standard format exists");
        assert_eq!(fmt.descriptor().format_id, *id);
    }

    assert!(matches!(reg.get("unknown-fmt"), Err(FormatExportError::UnknownFormatId(_))));
    assert!(matches!(reg.get("CONLLU"), Err(FormatExportError::InvalidFormatId(_))));
    assert!(matches!(reg.get("conllu v2"), Err(FormatExportError::InvalidFormatId(_))));
    assert!(matches!(reg.get(""), Err(FormatExportError::InvalidFormatId(_))));

    // Refuses duplicate registration.
    let dup_err = reg.register(Arc::new(ConlluFormatter)).unwrap_err();
    assert!(matches!(dup_err, FormatExportError::DuplicateFormatId(id) if id == "conllu"));
}

#[test]
fn conllu_round_trips_multiword_tokens_lemmas_pos_deps_and_space_after_no() {
    let text = "Wait, don't go!";
    let mut doc = Document::new(text);

    let mut sent_spans = Spans::<Sentences>::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test")).unwrap();

    let mut tok_spans = Spans::<Tokens>::new(ViewId::SOURCE);
    tok_spans.push(Span::new(0, 4)); // Wait
    tok_spans.push(Span::new(4, 5)); // ,
    tok_spans.push(Span::new(6, 11)); // don't
    tok_spans.push(Span::new(12, 14)); // go
    tok_spans.push(Span::new(14, 15)); // !
    let tokens = Tokens { spans: tok_spans, sentence: Some(vec![Ix::new(0); 5]) };
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();

    let mut words = Words::new();
    words.push_token(Ix::new(0));
    words.push_token(Ix::new(1));
    let mwt_range = IxRange::new(2, 3);
    words.push_word(mwt_range, None, Some(Span::new(6, 8)));
    words.push_word(mwt_range, None, Some(Span::new(8, 11)));
    words.push_token(Ix::new(3));
    words.push_token(Ix::new(4));
    doc.add(&Key::DEFAULT, words, Provenance::by("test")).unwrap();

    let upos_set = TagSet::new("upos", &["VERB", "PUNCT", "AUX", "PART"]);
    let pos_tags = vec![
        upos_set.tag("VERB").unwrap(),
        upos_set.tag("PUNCT").unwrap(),
        upos_set.tag("AUX").unwrap(),
        upos_set.tag("PART").unwrap(),
        upos_set.tag("VERB").unwrap(),
        upos_set.tag("PUNCT").unwrap(),
    ];
    doc.add(&Key::DEFAULT, WordPosTags { tagset: upos_set, tags: pos_tags, prob: None }, Provenance::by("test"))
        .unwrap();

    let mut lemmas = Strs::default();
    for l in ["wait", ",", "do", "not", "go", "!"] {
        lemmas.push(l);
    }
    doc.add(&Key::DEFAULT, WordLemmas { lemma: lemmas }, Provenance::by("test")).unwrap();

    let mut feats = Strs::default();
    for f in ["Mood=Imp|VerbForm=Fin", "_", "Mood=Ind|Tense=Pres|VerbForm=Fin", "Polarity=Neg", "VerbForm=Inf", "_"] {
        feats.push(f);
    }
    doc.add(&Key::DEFAULT, MorphFeatures::<Words>::new(feats), Provenance::by("test")).unwrap();

    let dep_set = TagSet::new("deprel", &["root", "punct", "aux", "advmod", "parataxis"]);
    let head = vec![None, Some(Ix::new(0)), Some(Ix::new(4)), Some(Ix::new(4)), Some(Ix::new(0)), Some(Ix::new(4))];
    let relation = vec![
        dep_set.tag("root").unwrap(),
        dep_set.tag("punct").unwrap(),
        dep_set.tag("aux").unwrap(),
        dep_set.tag("advmod").unwrap(),
        dep_set.tag("parataxis").unwrap(),
        dep_set.tag("punct").unwrap(),
    ];
    doc.add(&Key::DEFAULT, WordDependencies { head, relation, relations: dep_set }, Provenance::by("test")).unwrap();

    let reg = FormatterRegistry::standard();
    let (_, conllu_bytes) = reg.format_to_vec("conllu", "doc-mwt", &doc, ExportOffsetMode::Utf8).unwrap();
    let conllu_str = String::from_utf8(conllu_bytes.clone()).unwrap();

    assert!(conllu_str.contains("# newdoc id = doc-mwt\n"));
    assert!(conllu_str.contains("# sent_id = doc-mwt-1\n"));
    assert!(conllu_str.contains("# text = Wait, don't go!\n"));
    assert!(conllu_str.contains("1\tWait\twait\tVERB\t_\tMood=Imp|VerbForm=Fin\t0\troot\t_\tSpaceAfter=No\n"));
    assert!(conllu_str.contains("3-4\tdon't\t_\t_\t_\t_\t_\t_\t_\t_\n"));
    assert!(conllu_str.contains("3\tdo\tdo\tAUX\t_\tMood=Ind|Tense=Pres|VerbForm=Fin\t5\taux\t_\t_\n"));
    assert!(conllu_str.contains("4\tn't\tnot\tPART\t_\tPolarity=Neg\t5\tadvmod\t_\t_\n"));
    assert!(conllu_str.contains("5\tgo\tgo\tVERB\t_\tVerbForm=Inf\t1\tparataxis\t_\tSpaceAfter=No\n"));

    // Round-trip through ConlluStream and ConlluWordReader.
    let mut stream = ConlluStream::new(&conllu_str).unwrap();
    let item = stream.next().expect("one sentence").unwrap();
    assert_eq!(item.text.as_ref().map(|f| f.as_str()), Some("Wait, don't go!"));
    assert_eq!(item.word_lines.len(), 5);

    let reader = ConlluWordReader::new(&conllu_str).unwrap();
    let sentences: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(sentences.len(), 1);
    let round_doc = sentences[0].to_document().unwrap();
    assert_eq!(round_doc.source(), text);

    let rt_words = round_doc.get::<Words>().unwrap();
    assert_eq!(rt_words.len(), 6);
    let rt_lemmas = round_doc.get::<WordLemmas>().unwrap();
    assert_eq!(rt_lemmas.lemma.iter().collect::<Vec<_>>(), vec!["wait", ",", "do", "not", "go", "!"]);
    let rt_pos = round_doc.get::<WordPosTags>().unwrap();
    let rt_pos_labels: Vec<&str> = rt_pos.tags.iter().map(|&t| rt_pos.tagset.label(t)).collect();
    assert_eq!(rt_pos_labels, vec!["VERB", "PUNCT", "AUX", "PART", "VERB", "PUNCT"]);
    let rt_deps = round_doc.get::<WordDependencies>().unwrap();
    assert_eq!(
        rt_deps.head,
        vec![None, Some(Ix::new(0)), Some(Ix::new(4)), Some(Ix::new(4)), Some(Ix::new(0)), Some(Ix::new(4))]
    );
}

#[test]
fn brat_preserves_continuous_and_discontinuous_entities_and_relations_across_offset_modes() {
    // Multibyte UTF-8 text with BMP accented char ("Café") and astral emoji ("🚀"):
    let text = "Café 🚀 New York and Jersey";
    let mut doc = Document::new(text);

    let mut tok_spans = Spans::<Tokens>::new(ViewId::SOURCE);
    for (s, e) in [(0, 5), (6, 10), (11, 14), (15, 19), (20, 23), (24, 30)] {
        tok_spans.push(Span::new(s, e));
    }
    doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("test")).unwrap();

    // Entity 0 (continuous): "New York" (11..19)
    // Entity 1 (discontinuous): "New ... Jersey" ([11..14, 24..30])
    let kinds = TagSet::new("ner", &["LOC"]);
    let loc = kinds.tag("LOC").unwrap();
    let mut ent_spans = Spans::<Entities>::new(ViewId::SOURCE);
    ent_spans.try_push_fragments(&[Span::new(11, 19)]).unwrap();
    ent_spans.try_push_fragments(&[Span::new(11, 14), Span::new(24, 30)]).unwrap();
    let entities = Entities {
        spans: ent_spans,
        tokens: vec![IxRange::new(2, 4), IxRange::new(2, 6)],
        kind: vec![loc, loc],
        kinds,
        score: vec![0.95, 0.875],
    };
    doc.add(&Key::DEFAULT, entities, Provenance::by("test")).unwrap();

    let rel_kinds = TagSet::new("rel", &["Near"]);
    let near = rel_kinds.tag("Near").unwrap();
    let edges = Edges::<Entities, Entities>::new(rel_kinds, [Edge::new(Ix::new(0), Ix::new(1), near, 0.9)]);
    doc.add(&Key::DEFAULT, edges, Provenance::by("test")).unwrap();

    for mode in [ExportOffsetMode::Utf8, ExportOffsetMode::Utf16, ExportOffsetMode::CodePoint] {
        let ann_str = BratFormatter.format_to_string("brat-doc", &doc, mode).unwrap();
        let (brat_doc, round_ents) = parse_brat_with_offsets("brat-doc", &doc, ann_str.as_bytes(), mode).unwrap();

        assert_eq!(round_ents.spans.len(), 2, "failed entity count in mode {mode:?}");
        let frags0: Vec<Span> = round_ents.spans.fragments(Ix::new(0)).collect();
        let frags1: Vec<Span> = round_ents.spans.fragments(Ix::new(1)).collect();
        // BratDocument::to_entities sorts by start, then longer envelope end first:
        // Entity [11..14, 24..30] has envelope 11..30, so it sorts before [11..19].
        assert_eq!(frags0, vec![Span::new(11, 14), Span::new(24, 30)], "discontinuous fragments in mode {mode:?}");
        assert_eq!(frags1, vec![Span::new(11, 19)], "continuous fragment in mode {mode:?}");

        // Verify relation R1 and confidence attributes A1/A2 were parsed by BratDocument::parse.
        let has_rel = brat_doc.annotations().iter().any(|a| {
            matches!(&a.value, AnnotationValue::Relation { arg1, arg2 } if a.kind == "Near" && arg1 == "T1" && arg2 == "T2")
        });
        assert!(has_rel, "missing relation in mode {mode:?}");

        if mode == ExportOffsetMode::Utf16 {
            assert!(ann_str.contains("T1\tLOC 8 16\tNew York\n"));
            assert!(ann_str.contains("T2\tLOC 8 11;21 27\tNew Jersey\n"));
            let direct_brat =
                BratDocument::parse("brat-doc", text.as_bytes(), ann_str.as_bytes(), WhitespaceMode::Unicode).unwrap();
            let direct_ents = direct_brat.to_entities(doc.get::<Tokens>().unwrap()).unwrap();
            assert_eq!(direct_ents.spans.len(), 2);
        } else if mode == ExportOffsetMode::Utf8 {
            assert!(ann_str.contains("T1\tLOC 11 19\tNew York\n"));
            assert!(ann_str.contains("T2\tLOC 11 14;24 30\tNew Jersey\n"));
        } else {
            assert!(ann_str.contains("T1\tLOC 7 15\tNew York\n"));
            assert!(ann_str.contains("T2\tLOC 7 10;20 26\tNew Jersey\n"));
        }
    }
    let tsv = TsvFormatter.format_to_string("brat-doc", &doc, ExportOffsetMode::Utf8).unwrap();
    assert!(tsv.contains("0\t4\t20\t23\tand\t\t\t\tO\tfalse\n"));
    assert!(tsv.contains("0\t5\t24\t30\tJersey\t\t\t\tB-LOC\tfalse\n"));
}

#[test]
fn tsv_csv_markdown_jsonld_and_warc_handle_special_chars_and_child_view_projection() {
    // Source text has a prefix ">>> " (4 bytes) stripped in child view 1, plus quotes, tabs, newlines,
    // Markdown characters, and multibyte UTF-8.
    let source_text = ">>> \"Hello\tworld\",\n*Dr._Smith* [🚀] <admin@example.com>!";
    let child_text = "\"Hello\tworld\",\n*Dr._Smith* [🚀] <admin@example.com>!";
    let mut doc = Document::new(source_text);

    // Build child view 1 that deletes the leading 4 bytes ">>> ".
    let mut vb = ViewBuilder::new(source_text);
    vb.replace(4, "");
    let (derived_text, alignment) = vb.finish();
    assert_eq!(derived_text, child_text);
    let source_view = doc.get::<View>().unwrap().clone();
    let view_key = Key::<View>::named("stripped");
    doc.add(&view_key, View::derived(&source_view, derived_text, alignment), Provenance::by("test")).unwrap();
    let child_view = doc.get_key(&view_key).unwrap().id();

    // Add Sentences and Tokens on child_view so exporters must project back to ViewId::SOURCE.
    let mut sent_spans = Spans::<Sentences>::new(child_view);
    sent_spans.push(Span::new(0, child_text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test")).unwrap();

    let mut tok_spans = Spans::<Tokens>::new(child_view);
    // Token 0 in child_view: "\"Hello\tworld\"," (0..14) -> in source: 4..18
    // Token 1 in child_view: "*Dr._Smith*" (15..26) -> in source: 19..30
    // Token 2 in child_view: "[🚀]" (27..33) -> in source: 31..37
    // Token 3 in child_view: "<admin@example.com>!" (34..54) -> in source: 38..58
    tok_spans.push(Span::new(0, 14));
    tok_spans.push(Span::new(15, 26));
    tok_spans.push(Span::new(27, 33));
    tok_spans.push(Span::new(34, 54));
    let tokens = Tokens { spans: tok_spans, sentence: Some(vec![Ix::new(0); 4]) };
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();

    let pos_set = TagSet::new("pos", &["NNP", "SYM", "NN"]);
    doc.add(
        &Key::DEFAULT,
        PosTags {
            tagset: pos_set.clone(),
            tags: vec![
                pos_set.tag("NN").unwrap(),
                pos_set.tag("NNP").unwrap(),
                pos_set.tag("SYM").unwrap(),
                pos_set.tag("NN").unwrap(),
            ],
            prob: Some(vec![0.9, 0.95, 0.8, 0.99]),
        },
        Provenance::by("test"),
    )
    .unwrap();

    let mut lemmas = Strs::default();
    for l in ["hello world", "dr. smith", "🚀", "admin@example.com"] {
        lemmas.push(l);
    }
    doc.add(&Key::DEFAULT, Lemmas { lemma: lemmas }, Provenance::by("test")).unwrap();

    let mut stems = Strs::default();
    for s in ["hello", "smith", "🚀", "admin"] {
        stems.push(s);
    }
    doc.add(&Key::DEFAULT, Stems { stem: stems }, Provenance::by("test")).unwrap();

    doc.add(
        &Key::DEFAULT,
        Stopwords { language: StopwordLanguage::English, stop: vec![false, false, true, false] },
        Provenance::by("test"),
    )
    .unwrap();

    let ent_kinds = TagSet::new("ner", &["PER"]);
    let mut ent_spans = Spans::<Entities>::new(child_view);
    ent_spans.push(Span::new(15, 26)); // *Dr._Smith* -> source 19..30
    doc.add(
        &Key::DEFAULT,
        Entities {
            spans: ent_spans,
            tokens: vec![IxRange::new(1, 2)],
            kind: vec![ent_kinds.tag("PER").unwrap()],
            kinds: ent_kinds,
            score: vec![0.98],
        },
        Provenance::by("test"),
    )
    .unwrap();

    let mut lang_codes = Strs::default();
    lang_codes.push("eng");
    doc.add(&Key::DEFAULT, Language { code: lang_codes, score: vec![0.995] }, Provenance::by("test")).unwrap();

    let cat_set = TagSet::new("cats", &["science_tech", "general"]);
    let cats = Categories::document(cat_set.clone(), cat_set.tag("science_tech").unwrap(), &[0.85, 0.15]);
    doc.add(&Key::DEFAULT, cats, Provenance::by("test")).unwrap();

    let mut pii = Pii::new(child_view);
    pii.push(PiiMention {
        span: Span::new(35, 52), // admin@example.com -> source 39..56
        kind: PiiKind::Email,
        confidence: PiiConfidence::Format,
        normalized: "admin@example.com",
    });
    doc.add(&Key::DEFAULT, pii, Provenance::by("test")).unwrap();

    let mut opinions = Opinions::new();
    opinions.push(Opinion { tokens: IxRange::new(0, 1), polarity: 0.75, intensity: 1.2, negated: false });
    doc.add(&Key::DEFAULT, opinions, Provenance::by("test")).unwrap();

    let asp_cats = TagSet::new("aspect", &["greeting"]);
    let mut aspects = Aspects::new(asp_cats.clone());
    aspects
        .try_push(
            doc.get::<Tokens>().unwrap(),
            doc.get::<Sentences>(),
            doc.get::<Opinions>(),
            Aspect {
                target: AspectTarget::Term { tokens: IxRange::new(0, 1), sentence: Some(Ix::new(0)) },
                category: asp_cats.tag("greeting"),
                polarity: PolarityScores::new(0.8, 0.1, 0.1),
                opinions: &[Ix::new(0)],
            },
        )
        .unwrap();
    doc.add(&Key::DEFAULT, aspects, Provenance::by("test")).unwrap();

    let mut emotions = EmotionLayer::new();
    let mut em_scores = [0.0_f64; 28];
    em_scores[EmotionDimension::Joy.index()] = 0.92;
    emotions
        .try_push(
            doc.get::<Tokens>().unwrap(),
            doc.get::<Sentences>(),
            None,
            EmotionItem {
                target: EmotionTarget::Sentence(Ix::new(0)),
                scores: EmotionScores(em_scores),
                evidence: &[(EmotionDimension::Joy, IxRange::new(0, 1))],
            },
        )
        .unwrap();
    doc.add(&Key::DEFAULT, emotions, Provenance::by("test")).unwrap();

    let mut chunk_spans = Spans::<Chunks>::new(child_view);
    chunk_spans.push(Span::new(0, child_text.len() as u32));
    doc.add(
        &Key::DEFAULT,
        Chunks {
            spans: chunk_spans,
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart],
            cost: vec![0],
            model_tokens: vec![4],
            over_budget: vec![false],
            prompt_overhead: 0,
        },
        Provenance::by("test"),
    )
    .unwrap();

    // 1. TSV: 1 header + 4 token rows, each with exactly 10 tab-separated columns and source byte offsets.
    let tsv = TsvFormatter.format_to_string("doc-special", &doc, ExportOffsetMode::Utf8).unwrap();
    let tsv_lines: Vec<&str> = tsv.lines().collect();
    assert_eq!(tsv_lines.len(), 5);
    assert_eq!(tsv_lines[0], "sentence\ttoken\tstart\tend\ttext\tpos\tlemma\tstem\tentity\tstopword");
    for line in &tsv_lines[1..] {
        assert_eq!(line.split('\t').count(), 10, "TSV row must have 10 columns: {line}");
    }
    // Token 0 source span is 4..18; token 1 is 19..30 with B-PER.
    assert!(tsv_lines[1].starts_with("0\t0\t4\t18\t\"Hello world\",\tNN\thello world\thello\tO\tfalse"));
    assert!(tsv_lines[2].starts_with("0\t1\t19\t30\t*Dr._Smith*\tNNP\tdr. smith\tsmith\tB-PER\tfalse"));

    // 2. CSV: RFC 4180 CRLF framing with doubled quotes inside Token 0 (`"""Hello\tworld"","`).
    let csv = CsvFormatter.format_to_string("doc-special", &doc, ExportOffsetMode::Utf8).unwrap();
    assert!(csv.starts_with("sentence,token,start,end,text,pos,lemma,stem,entity,stopword\r\n"));
    assert!(csv.contains("0,0,4,18,\"\"\"Hello\tworld\"\",\",NN,hello world,hello,O,false\r\n"));

    // 3. Markdown: escapes Markdown control characters and summarizes language, categories, entities, sentiment, PII, and chunks.
    let md = MarkdownFormatter.format_to_string("doc*1", &doc, ExportOffsetMode::Utf8).unwrap();
    assert!(md.contains("# Document doc\\*1\n"));
    assert!(md.contains("- **Detected Language**: `eng` (0.9950)"));
    assert!(md.contains("- **Best Category**: `science\\_tech` (0.8500)"));
    assert!(md.contains("- **PER** [19..30]: \\*Dr.\\_Smith\\* (score: 0.9800)"));
    assert!(md.contains("## Sentiment & Emotion Summary"));
    assert!(md.contains("- **Emotion Target 1**: `joy` (0.9200)"));
    assert!(md.contains("- **email** [39..56]: `admin@example.com` (confidence: `format`)"));
    assert!(md.contains("- **Chunk 1** [4..58]"));

    // 4. JSON-LD: valid JSON containing projected spans in all three offset modes.
    for mode in [ExportOffsetMode::Utf8, ExportOffsetMode::Utf16, ExportOffsetMode::CodePoint] {
        let jsonld = JsonLdFormatter.format_to_string("doc-special", &doc, mode).unwrap();
        assert!(jsonld.contains("\"@context\": \"https://turboembed.ai/ns/jsonld/v1\""));
        assert!(jsonld.contains(&format!("\"offsetMode\": \"{}\"", mode.as_str())));
        assert!(jsonld.contains("\"opennlp:tokens\": ["));
        assert!(jsonld.contains("\"opennlp:entities\": ["));
        assert!(jsonld.contains("\"turbo:pii\": ["));
    }

    // 5. WARC/1.1: parses cleanly into warcinfo + resource records with exact Content-Length and CRLF trailer.
    let warc_bytes = WarcFormatter.format_to_vec("doc-special", &doc, ExportOffsetMode::Utf8).unwrap();
    let records = parse_warc_records(&warc_bytes).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].header("WARC-Type"), Some("warcinfo"));
    assert_eq!(records[1].header("WARC-Type"), Some("resource"));
    assert_eq!(records[1].header("WARC-Target-URI"), Some("urn:turbo:document:doc-special"));
    let resource_str = std::str::from_utf8(&records[1].block).unwrap();
    assert!(resource_str.contains("\"@type\": \"turbo:AnnotatedDocument\""));
}

#[test]
fn all_builtin_descriptors_stay_well_formed() {
    let reg = FormatterRegistry::standard();
    let descriptors: Vec<OutputFormatDescriptor> = reg.list();
    assert_eq!(descriptors.len(), 7);
    for d in descriptors {
        assert!(!d.format_id.is_empty());
        assert!(!d.display_name.is_empty());
        assert!(!d.media_type.is_empty());
        assert!(!d.file_extension.is_empty());
        assert!(!d.description.is_empty());
        assert!(!d.lossless);
    }
}

#[test]
fn conllu_projects_child_view_multiword_subspans_across_offset_modes() {
    let source = ">>> é don't\ngo";
    let mut doc = Document::new(source);
    let mut vb = ViewBuilder::new(source);
    vb.replace(4, "");
    let (derived_text, alignment) = vb.finish();
    assert_eq!(derived_text, "é don't\ngo");
    let src_view = doc.get::<View>().unwrap().clone();
    let view_key = Key::<View>::named("stripped");
    doc.add(&view_key, View::derived(&src_view, derived_text.clone(), alignment), Provenance::by("test")).unwrap();
    let child_view = doc.get_key(&view_key).unwrap().id();

    let mut sent_spans = Spans::<Sentences>::new(child_view);
    sent_spans.push(Span::new(0, derived_text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test")).unwrap();

    let mut tok_spans = Spans::<Tokens>::new(child_view);
    for (s, e) in [(0, 2), (3, 8), (9, 11)] {
        tok_spans.push(Span::new(s, e));
    }
    doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: Some(vec![Ix::new(0); 3]) }, Provenance::by("test"))
        .unwrap();

    let mut words = Words::new();
    words.push_token(Ix::new(0));
    words.push_word(IxRange::new(1, 2), None, Some(Span::new(3, 5)));
    words.push_word(IxRange::new(1, 2), None, Some(Span::new(5, 8)));
    words.push_token(Ix::new(2));
    doc.add(&Key::DEFAULT, words, Provenance::by("test")).unwrap();

    let u8_out = ConlluFormatter.format_to_string("mwt-view", &doc, ExportOffsetMode::Utf8).unwrap();
    assert!(u8_out.contains("2-3\tdon't\t_\t_\t_\t_\t_\t_\t_\tSpacesAfter=\\n\n"));
    let u16_out = ConlluFormatter.format_to_string("mwt-view", &doc, ExportOffsetMode::Utf16).unwrap();
    assert!(u16_out.contains("2\tdo\t_\t_\t_\t_\t_\t_\t_\tTokenRangeUtf16=6:8\n"));
    assert!(u16_out.contains("3\tn't\t_\t_\t_\t_\t_\t_\t_\tTokenRangeUtf16=8:11\n"));
    let cp_out = ConlluFormatter.format_to_string("mwt-view", &doc, ExportOffsetMode::CodePoint).unwrap();
    assert!(cp_out.contains("2\tdo\t_\t_\t_\t_\t_\t_\t_\tTokenRangeCp=6:8\n"));
    assert!(cp_out.contains("3\tn't\t_\t_\t_\t_\t_\t_\t_\tTokenRangeCp=8:11\n"));
}
