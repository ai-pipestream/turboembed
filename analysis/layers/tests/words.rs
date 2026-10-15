use turbo_analysis::{
    Analyzer, Annotator, Bound, Document, Ix, IxRange, Key, Layer, Provenance, Span, Spans, Tag, TagSet, Utf16Offset,
    ViewId,
};
use turbo_analysis_layers::{
    Dependencies, Lemmas, MorphFeatures, MwtLanguage, PosTags, RuleMwtExpander, Strs, Tokens, WordLemmas, WordPosTags,
    Words,
};

fn make_tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = None;
    for (i, ch) in text.char_indices() {
        if ch.is_whitespace() || ch == '.' || ch == ',' {
            if let Some(s) = start.take() {
                spans.push(Span::new(s as u32, i as u32));
            }
            if !ch.is_whitespace() {
                spans.push(Span::new(i as u32, (i + ch.len_utf8()) as u32));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        spans.push(Span::new(s as u32, text.len() as u32));
    }
    Tokens { spans, sentence: None }
}

#[test]
fn identity_words_cost_one_index_column_and_slice_view_text() {
    let text = "The quick brown fox.";
    let tokens = make_tokens(text);
    let words = Words::from_tokens(&tokens);
    assert!(words.is_identity());
    assert!(words.form.is_none());
    assert!(words.subspan.is_none());
    assert_eq!(words.len(), 5);
    assert_eq!(words.check(), Ok(()));

    let forms: Vec<&str> = words.iter(text, &tokens).map(|w| w.form).collect();
    assert_eq!(forms, ["The", "quick", "brown", "fox", "."]);
    for i in 0..5u32 {
        assert_eq!(words.of_token(Ix::new(i)), IxRange::new(i, i + 1));
    }
}

#[test]
fn rule_mwt_expander_splits_spanish_catalan_french_portuguese_italian_german_and_english() {
    // Spanish (del -> de + el, al -> a + el)
    let es_text = "Vamos al mar del norte.";
    let es_tok = make_tokens(es_text);
    let es_words = RuleMwtExpander::new(MwtLanguage::Spanish).expand(es_text, &es_tok);
    assert_eq!(es_words.check(), Ok(()));
    let es_forms: Vec<&str> = es_words.iter(es_text, &es_tok).map(|w| w.form).collect();
    assert_eq!(es_forms, ["Vamos", "a", "el", "mar", "de", "el", "norte", "."]);
    assert_eq!(es_words.of_token(Ix::new(1)), IxRange::new(1, 3));
    assert_eq!(es_words.of_token(Ix::new(3)), IxRange::new(4, 6));

    // Catalan (pels -> per + els, dels -> de + els)
    let ca_text = "Passa pels camins dels boscos.";
    let ca_tok = make_tokens(ca_text);
    let ca_words = RuleMwtExpander::new(MwtLanguage::Catalan).expand(ca_text, &ca_tok);
    let ca_forms: Vec<&str> = ca_words.iter(ca_text, &ca_tok).map(|w| w.form).collect();
    assert_eq!(ca_forms, ["Passa", "per", "els", "camins", "de", "els", "boscos", "."]);

    // French (au -> à + le, duquel -> de + lequel; des stays whole)
    let fr_text = "Il va au marché des fleurs duquel il parle.";
    let fr_tok = make_tokens(fr_text);
    let fr_words = RuleMwtExpander::new(MwtLanguage::French).expand(fr_text, &fr_tok);
    let fr_forms: Vec<&str> = fr_words.iter(fr_text, &fr_tok).map(|w| w.form).collect();
    assert_eq!(fr_forms, ["Il", "va", "à", "le", "marché", "des", "fleurs", "de", "lequel", "il", "parle", "."]);

    // Catalan `cal` (the verb "it is necessary") is not a contraction and
    // stays whole; `al` next to it still splits.
    let ca_text = "Cal dir que va al mar.";
    let ca_tok = make_tokens(ca_text);
    let ca_words = RuleMwtExpander::new(MwtLanguage::Catalan).expand(ca_text, &ca_tok);
    let ca_forms: Vec<&str> = ca_words.iter(ca_text, &ca_tok).map(|w| w.form).collect();
    assert_eq!(ca_forms, ["Cal", "dir", "que", "va", "a", "el", "mar", "."]);

    // Portuguese (pelos -> por + os, na -> em + a, daquele -> de + aquele)
    let pt_text = "Andou pelos campos na sombra daquele monte.";
    let pt_tok = make_tokens(pt_text);
    let pt_words = RuleMwtExpander::new(MwtLanguage::Portuguese).expand(pt_text, &pt_tok);
    let pt_forms: Vec<&str> = pt_words.iter(pt_text, &pt_tok).map(|w| w.form).collect();
    assert_eq!(pt_forms, ["Andou", "por", "os", "campos", "em", "a", "sombra", "de", "aquele", "monte", "."]);

    // Italian (nella -> in + la, degli -> di + gli, sul -> su + il)
    let it_text = "Siamo nella casa degli amici sul colle.";
    let it_tok = make_tokens(it_text);
    let it_words = RuleMwtExpander::new(MwtLanguage::Italian).expand(it_text, &it_tok);
    let it_forms: Vec<&str> = it_words.iter(it_text, &it_tok).map(|w| w.form).collect();
    assert_eq!(it_forms, ["Siamo", "in", "la", "casa", "di", "gli", "amici", "su", "il", "colle", "."]);

    // German (zum -> zu + dem, im -> in + dem, fürs -> für + das)
    let de_text = "Wir gehen zum Bahnhof im Dorf fürs Fest.";
    let de_tok = make_tokens(de_text);
    let de_words = RuleMwtExpander::new(MwtLanguage::German).expand(de_text, &de_tok);
    let de_forms: Vec<&str> = de_words.iter(de_text, &de_tok).map(|w| w.form).collect();
    assert_eq!(de_forms, ["Wir", "gehen", "zu", "dem", "Bahnhof", "in", "dem", "Dorf", "für", "das", "Fest", "."]);

    // English substring clitics (can't -> ca + n't with subspans, form: None)
    let en_text = "We can't go.";
    let en_tok = make_tokens(en_text);
    let en_words = RuleMwtExpander::new(MwtLanguage::English).expand(en_text, &en_tok);
    assert!(en_words.form.is_none());
    assert!(en_words.subspan.is_some());
    let en_forms: Vec<&str> = en_words.iter(en_text, &en_tok).map(|w| w.form).collect();
    assert_eq!(en_forms, ["We", "ca", "n't", "go", "."]);
    assert_eq!(en_words.subspan_of(Ix::new(1)), Some(Span::new(3, 5)));
    assert_eq!(en_words.subspan_of(Ix::new(2)), Some(Span::new(5, 8)));
    // Both words still map to the full token span 3..8 at the edge.
    assert_eq!(en_words.token_span(Ix::new(1), &en_tok), Span::new(3, 8));
    assert_eq!(en_words.token_span(Ix::new(2), &en_tok), Span::new(3, 8));

    // A typographic apostrophe, a host outside ASCII, and an upper case
    // contraction: the split lands on the apostrophe's bytes.
    let en_text = "José\u{2019}s DOG can\u{2019}t";
    let en_tok = make_tokens(en_text);
    let en_words = RuleMwtExpander::new(MwtLanguage::English).expand(en_text, &en_tok);
    assert_eq!(en_words.check_against(en_text, &en_tok), Ok(()));
    let en_forms: Vec<&str> = en_words.iter(en_text, &en_tok).map(|w| w.form).collect();
    assert_eq!(en_forms, ["José", "\u{2019}s", "DOG", "ca", "n\u{2019}t"]);
    assert_eq!(en_words.subspan_of(Ix::new(0)), Some(Span::new(0, 5)));
    assert_eq!(en_words.subspan_of(Ix::new(1)), Some(Span::new(5, 9)));
    assert_eq!(en_words.of_token(Ix::new(1)), IxRange::new(2, 3));
    // A token past the words, and an upper case Spanish contraction.
    assert_eq!(en_words.of_token(Ix::new(7)), IxRange::new(5, 5));
    let es_text = "DEL Norte";
    let es_tok = make_tokens(es_text);
    let es_words = RuleMwtExpander::new(MwtLanguage::Spanish).expand(es_text, &es_tok);
    let es_forms: Vec<&str> = es_words.iter(es_text, &es_tok).map(|w| w.form).collect();
    assert_eq!(es_forms, ["DE", "EL", "Norte"]);
    assert_ne!(
        RuleMwtExpander::new(MwtLanguage::Spanish).config_hash(),
        RuleMwtExpander::new(MwtLanguage::French).config_hash()
    );
}

#[test]
fn the_layer_names_tell_word_layers_from_token_layers() {
    assert_eq!(Words::NAME.to_string(), "turbo:words");
    assert_eq!(WordPosTags::NAME.to_string(), "opennlp:pos<turbo:words>");
    assert_eq!(WordLemmas::NAME.to_string(), "opennlp:lemmas<turbo:words>");
    assert_eq!(MorphFeatures::<Words>::NAME.to_string(), "turbo:feats<turbo:words>");
    assert_eq!(MorphFeatures::<Tokens>::NAME.to_string(), "turbo:feats");
    assert_eq!(Dependencies::<Tokens>::NAME.to_string(), "opennlp:dependencies");
    assert_eq!(Dependencies::<Words>::NAME.to_string(), "opennlp:dependencies<turbo:words>");
    assert_eq!(Words::from_tokens(&make_tokens("")).of_token(Ix::new(0)), IxRange::new(0, 0));
}

#[test]
fn utf16_conversion_maps_words_to_token_spans_without_splitting_tokens() {
    let text = "😀 Vamos al mar.";
    let mut doc = Document::new(text);
    let tokens = make_tokens(text);
    doc.add(&Key::DEFAULT, tokens, Provenance::by("tok")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(RuleMwtExpander::new(MwtLanguage::Spanish)))
        .build()
        .unwrap();
    let mut batch = vec![doc];
    analyzer.run(&mut batch)[0].as_ref().unwrap();
    let doc = &batch[0];
    let tok = doc.get::<Tokens>().unwrap();
    let words = doc.get::<Words>().unwrap();
    let offsets = doc.offsets(ViewId::SOURCE);

    // Token 2 is "al" (bytes 11..13, UTF-16 units 9..11 after 2-unit emoji + space + "Vamos ").
    // Words 2 ("a") and 3 ("el") both belong to token 2 and must map to the exact same UTF-16 token span.
    let u_a = words.utf16_span(Ix::new(2), text, tok, offsets);
    let u_el = words.utf16_span(Ix::new(3), text, tok, offsets);
    assert_eq!(u_a, (Utf16Offset(9), Utf16Offset(11)));
    assert_eq!(u_el, (Utf16Offset(9), Utf16Offset(11)));
}

#[test]
fn word_anchored_and_token_anchored_layers_coexist_on_one_document() {
    let text = "Vamos al mar.";
    let mut doc = Document::new(text);
    let tokens = make_tokens(text);
    let words = RuleMwtExpander::new(MwtLanguage::Spanish).expand(text, &tokens);
    assert_eq!(tokens.len(), 4);
    assert_eq!(words.len(), 5);

    doc.add(&Key::DEFAULT, tokens, Provenance::by("tok")).unwrap();
    doc.add(&Key::DEFAULT, words, Provenance::by("mwt")).unwrap();

    let upos = TagSet::new("upos", &["VERB", "ADP", "DET", "NOUN", "PUNCT", "ADP+DET"]);
    // Token-anchored POS (4 tokens)
    let tok_pos = PosTags { tagset: upos.clone(), tags: vec![Tag(0), Tag(5), Tag(3), Tag(4)], prob: None };
    // Word-anchored POS (5 words)
    let word_pos = WordPosTags { tagset: upos, tags: vec![Tag(0), Tag(1), Tag(2), Tag(3), Tag(4)], prob: None };

    let mut tok_lemma = Strs::default();
    for l in ["ir", "a+el", "mar", "."] {
        tok_lemma.push(l);
    }
    let mut word_lemma = Strs::default();
    for l in ["ir", "a", "el", "mar", "."] {
        word_lemma.push(l);
    }

    let rels = TagSet::new("ud", &["root", "case", "det", "obl", "punct"]);
    let word_deps: Dependencies<Words> = Dependencies {
        head: vec![None, Some(Ix::new(3)), Some(Ix::new(3)), Some(Ix::new(0)), Some(Ix::new(0))],
        relation: vec![Tag(0), Tag(1), Tag(2), Tag(3), Tag(4)],
        relations: rels,
    };

    doc.add(&Key::DEFAULT, tok_pos, Provenance::by("tok-pos")).unwrap();
    doc.add(&Key::DEFAULT, word_pos, Provenance::by("word-pos")).unwrap();
    doc.add(&Key::DEFAULT, Lemmas { lemma: tok_lemma }, Provenance::by("tok-lemma")).unwrap();
    doc.add(&Key::DEFAULT, WordLemmas { lemma: word_lemma }, Provenance::by("word-lemma")).unwrap();
    doc.add(&Key::DEFAULT, word_deps, Provenance::by("word-dep")).unwrap();

    assert_eq!(doc.get::<PosTags>().unwrap().len(), 4);
    assert_eq!(doc.get::<WordPosTags>().unwrap().len(), 5);
    assert_eq!(doc.get::<Lemmas>().unwrap().len(), 4);
    assert_eq!(doc.get::<WordLemmas>().unwrap().len(), 5);
    assert_eq!(doc.get::<Dependencies<Words>>().unwrap().len(), 5);
}

#[test]
fn words_check_rejects_broken_invariants() {
    let mut w = Words::new();
    w.tokens.push(IxRange::new(1, 1));
    assert!(w.check().unwrap_err().contains("empty or backwards token range"));

    let mut w = Words::new();
    w.tokens.push(IxRange::new(1, 2));
    w.tokens.push(IxRange::new(0, 1));
    assert!(w.check().unwrap_err().contains("out of token order"));

    let mut w = Words::new();
    w.tokens.push(IxRange::new(0, 2));
    w.tokens.push(IxRange::new(1, 3));
    assert!(w.check().unwrap_err().contains("partially overlaps"));

    let mut w = Words::new();
    w.push_word(IxRange::new(0, 1), None, Some(Span::new(0, 3)));
    w.push_word(IxRange::new(0, 1), None, Some(Span::new(2, 5)));
    assert!(w.check().unwrap_err().contains("overlaps previous subspan"));

    let mut w = Words::new();
    w.push_word(IxRange::new(0, 1), Some("de"), Some(Span::new(0, 1)));
    assert!(w.check().unwrap_err().contains("both a form and a subspan"));

    // What only the tokens can tell: a subspan outside its token, or
    // inside a character, and a word past the tokens.
    let text = "ñu ok";
    let tokens = make_tokens(text);
    let mut w = Words::new();
    w.push_word(IxRange::new(0, 1), None, Some(Span::new(1, 2)));
    w.push_word(IxRange::new(0, 1), None, Some(Span::new(2, 3)));
    assert!(w.check_against(text, &tokens).unwrap_err().contains("not on character boundaries"));
    let mut w = Words::new();
    w.push_word(IxRange::new(0, 1), None, Some(Span::new(0, 2)));
    w.push_word(IxRange::new(0, 1), None, Some(Span::new(2, 4)));
    assert!(w.check_against(text, &tokens).unwrap_err().contains("outside its token"));
    let mut w = Words::from_tokens(&tokens);
    w.push_token(Ix::new(5));
    assert!(w.check_against(text, &tokens).unwrap_err().contains("points at token 5 of 2"));
}
