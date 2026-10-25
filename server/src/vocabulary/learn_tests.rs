use super::*;

fn header() -> LearnVocabularyHeader {
    LearnVocabularyHeader {
        min_term_frequency: 1,
        max_terms: 0,
        lowercase: true,
        filter_stopwords: false,
        include_ngrams: true,
        min_ngram_size: 2,
        max_ngram_size: 2,
        min_pmi: -100.0,
        dictionary_artifact_id: String::new(),
        language: "en".into(),
    }
}

fn learn(h: &LearnVocabularyHeader, docs: &[&str], seeds: &[&str]) -> (Learned, Vec<Vec<String>>) {
    let mut l = Learner::new(Options::new(h, 5).unwrap());
    for d in docs {
        l.add(d);
    }
    let seeds: Vec<String> = seeds.iter().map(|s| s.to_string()).collect();
    let learned = l.finish(&seeds);
    let text = String::from_utf8(learned.bytes.clone()).unwrap();
    let rows = text.lines().skip(3).map(|l| l.split('\t').map(str::to_owned).collect()).collect();
    (learned, rows)
}

fn row<'a>(rows: &'a [Vec<String>], term: &str) -> Option<&'a [String]> {
    rows.iter().find(|r| r[0] == term).map(Vec::as_slice)
}

#[test]
fn counts_terms_and_documents_and_scores_adjacent_pairs() {
    let docs = ["New York is big. New York is old.", "York"];
    let (learned, rows) = learn(&header(), &docs, &[]);
    assert_eq!((learned.documents, learned.tokens), (2, 9));
    // tf 3 over two documents, so df 2.
    assert_eq!(row(&rows, "york").unwrap()[1..4], ["term", "3", "2"]);
    // Eight words in the first document, two runs of four: six pair
    // positions. pmi(new york) = log2((2/6) / ((2/9) * (3/9))).
    let want = ((2.0f64 / 6.0) / ((2.0 / 9.0) * (3.0 / 9.0))).log2();
    let ny = row(&rows, "new york").unwrap();
    assert_eq!(ny[1..4], ["ngram", "2", "1"]);
    assert_eq!(ny[5], format!("{want:.6}"));
    // The full stop parts "big" from the next "New".
    assert!(row(&rows, "big new").is_none());
    // Most frequent first, ties in byte order.
    let terms: Vec<&str> = rows.iter().filter(|r| r[1] == "term").map(|r| r[0].as_str()).collect();
    assert_eq!(terms, ["york", "is", "new", "big", "old"]);
    assert_eq!((learned.terms, learned.ngrams), (5, 4));
}

#[test]
fn stopwords_leave_terms_and_pair_edges_but_not_pair_middles() {
    let h = LearnVocabularyHeader { filter_stopwords: true, max_ngram_size: 3, ..header() };
    let (_, rows) = learn(&h, &["The bank of the river", "a bank of fish"], &[]);
    assert!(row(&rows, "the").is_none() && row(&rows, "of").is_none());
    assert_eq!(row(&rows, "bank").unwrap()[2], "2");
    assert!(row(&rows, "bank of").is_none(), "ends with a stopword");
    assert!(row(&rows, "of the river").is_none(), "starts with a stopword");
    assert_eq!(row(&rows, "bank of fish").unwrap()[1..4], ["ngram", "1", "1"]);
}

#[test]
fn thresholds_caps_and_seeds() {
    let h = LearnVocabularyHeader { min_term_frequency: 2, max_terms: 1, ..header() };
    let (learned, rows) = learn(&h, &["red red blue", "red green"], &["Green", "sky blue", "red"]);
    let terms: Vec<&[String]> = rows.iter().filter(|r| r[1] == "term").map(Vec::as_slice).collect();
    // red passes the frequency and the cap; green is a seed under both.
    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0][0..4], ["red", "term", "3", "2"]);
    assert_eq!(terms[0][6], "1");
    assert_eq!(terms[1][0..4], ["green", "term", "1", "1"]);
    // A seed phrase the corpus never has: no counts and no pmi.
    let sky = row(&rows, "sky blue").unwrap();
    assert_eq!(sky[1..4], ["ngram", "0", "0"]);
    assert_eq!((sky[5].as_str(), sky[6].as_str()), ("", "1"));
    assert_eq!(learned.terms, 2);
}

#[test]
fn the_same_documents_give_the_same_bytes() {
    let docs = ["one two three, two three four", "three four five"];
    let (a, _) = learn(&header(), &docs, &[]);
    let (b, _) = learn(&header(), &docs, &[]);
    assert_eq!(a.bytes, b.bytes);
    let text = String::from_utf8(a.bytes).unwrap();
    assert!(text.starts_with("# turbo-vocabulary 1\n# language=en\tdocuments=2\ttokens=9\t"));
    assert_eq!(text.lines().nth(2), Some("term\tkind\ttf\tdf\tidf\tpmi\tseed"));
}

#[test]
fn refused_settings() {
    let bad = [
        LearnVocabularyHeader { min_ngram_size: 1, ..header() },
        LearnVocabularyHeader { min_ngram_size: 3, max_ngram_size: 2, ..header() },
        LearnVocabularyHeader { max_ngram_size: 6, ..header() },
        LearnVocabularyHeader { min_pmi: f64::NAN, ..header() },
        LearnVocabularyHeader { filter_stopwords: true, language: "xx".into(), ..header() },
    ];
    for h in &bad {
        assert!(Options::new(h, 5).is_err(), "{h:?}");
    }
    // Without n-grams their sizes are not read, and a region subtag
    // finds the language's list.
    assert!(Options::new(&LearnVocabularyHeader { include_ngrams: false, min_ngram_size: 0, ..header() }, 5).is_ok());
    assert!(
        Options::new(&LearnVocabularyHeader { filter_stopwords: true, language: "en-US".into(), ..header() }, 5)
            .is_ok()
    );
}
