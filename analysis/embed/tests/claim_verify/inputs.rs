use super::*;
use turbo_analysis_layers::Tokens;

#[test]
fn real_encoder_claim_candidates_match_with_punctuation_tokens() {
    let dir = bundle();
    for (sents, expected_runs) in [(["It rained.", "Cats purr."], 0), (["Acme grew.", "Acme grew fast."], 1)] {
        let text = sents.join(" ");
        let mut d = Document::new(text.as_str());
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, sents[0].len() as u32));
        spans.push(Span::new((sents[0].len() + 1) as u32, text.len() as u32));
        d.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test:real-nli")).unwrap();
        let mut spans = Spans::new(ViewId::SOURCE);
        let mut start = None;
        for (at, c) in text.char_indices().chain([(text.len(), ' ')]) {
            if c.is_alphanumeric() {
                start.get_or_insert(at);
                continue;
            }
            if let Some(start) = start.take() {
                spans.push(Span::new(start as u32, at as u32));
            }
            if !c.is_whitespace() {
                spans.push(Span::new(at as u32, (at + c.len_utf8()) as u32));
            }
        }
        let tokens = Tokens { spans, sentence: None };
        let mut results = Vec::new();
        for provided in [None, Some(&tokens)] {
            let config = ClaimVerifyConfig { entail_threshold: 0.0, contradict_threshold: 0.0, ..Default::default() };
            let verifier = ClaimVerificationAnnotator::<_, Sentences>::with_config(scorer(&dir, 64), config);
            let (edges,) = verifier
                .annotate((
                    d.require(&Key::<View>::DEFAULT).unwrap(),
                    d.get::<Sentences>().unwrap(),
                    None,
                    None,
                    provided,
                    None,
                ))
                .unwrap();
            assert_eq!(verifier.scorer().run_count(), expected_runs, "{text}");
            results.push(edges.iter().map(|(_, e)| (e.from.get(), e.to.get(), e.score.to_bits())).collect::<Vec<_>>());
        }
        assert_eq!(results[0], results[1], "{text}");
    }
}
