use super::*;

#[test]
fn normalized_unigram_added_tokens_match_upstream_ids_and_spans() {
    let mut json: serde_json::Value =
        serde_json::from_str(include_str!("../../../testdata/bge-m3/tokenizer.json")).unwrap();
    for token in json["added_tokens"].as_array_mut().unwrap() {
        if token["content"] == "<mask>" {
            token["normalized"] = true.into();
        }
    }
    let recipe: serde_json::Value = serde_json::from_str(include_str!("../../../bundle/recipes/bge-m3.json")).unwrap();
    let settings = serde_json::from_value(recipe["manifest"]["tokenizer"]["unigram"].clone()).unwrap();
    let u = Unigram::parse("tokenizer.json", &json, &settings, 3).unwrap();
    let mut upstream = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap();
    upstream.with_padding(None);
    upstream.with_truncation(None).unwrap();
    let added = [Special { content: "<mask>".into(), id: 28232, lstrip: true, single_word: false }];
    for text in ["<mask>", "a <mask> b", "a<mask>b", "a  <ｍａｓｋ> 東京", "Ｃａｆé 東京", "<mask><mask>", ""]
    {
        let expected = upstream.encode(text, false).unwrap();
        let mut ids = Vec::new();
        u.encode_added(text, 0, text, &added, &mut ids);
        assert_eq!(ids, expected.get_ids().iter().map(|&v| v as i32).collect::<Vec<_>>(), "{text:?}");
        let (mut spanned, mut spans) = (Vec::new(), Vec::new());
        u.encode_added(text, 0, text, &added, &mut Spanned { ids: &mut spanned, spans: &mut spans });
        assert_eq!(ids, spanned);
        assert!(spans.iter().all(|s| text.get(s[0] as usize..s[1] as usize).is_some()));
    }
}
