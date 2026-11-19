use turbo_formats::native::TokenReader;
use turbo_tokenize::TokenizerOptions;
use turbo_tokenize::me::{TokenizerMe, TokenizerModel};
use turbo_train::{Key, TrainingConfiguration, TrainingParameters};
use turbo_train_components::dictionary::AbbreviationDictionary;
use turbo_train_components::{Stamp, TokenizerFactory, train_tokenizer_model};

#[test]
fn after_an_abbreviation_jump_the_next_decision_is_after_the_first_character() {
    // The training text teaches splits after "x" and after a dot before
    // "zzzz", but never inside "zzzz". For the listed abbreviation the
    // first split emits the whole unit and jumps its two dots, landing
    // the cursor on the first "z" (byte 4). Deciding there would emit
    // the remainder at once and then its last character again; the
    // remainder comes out once, at the end.
    let mut text = String::new();
    for _ in 0..30 {
        text.push_str(
            "x<SPLIT>.aaaa\nx<SPLIT>.bbbb\nq.<SPLIT>zzzz\nw.<SPLIT>zzzz\nplain words here\nno splits at all\n",
        );
    }
    let samples: Vec<_> = TokenReader::new(&text).unwrap().map(Result::unwrap).collect();
    let dict = AbbreviationDictionary::new(false, vec![vec!["x.y.zzzz".to_string()]]).unwrap();
    let factory = TokenizerFactory::new("en", Some(dict), false, None).unwrap();
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 100);
    params.put(Key::CUTOFF, 0);
    let trained =
        train_tokenizer_model(&samples, &factory, &params, &mut TrainingConfiguration::default(), Stamp::at(0))
            .unwrap();
    let model = TokenizerModel::from_trained_archive(trained.package.to_bytes()).unwrap();
    let tok = TokenizerMe::new(model, TokenizerOptions::default());
    let mut out = Vec::new();
    tok.tokenize("see x.y.zzzz now", &mut out);
    let spans: Vec<(u32, u32)> = out.iter().map(|(s, _)| (s.start, s.end)).collect();
    assert_eq!(spans, [(0, 3), (4, 12), (8, 12), (13, 16)]);
}
