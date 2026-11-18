use turbo_analysis::{Alternatives, Analyzer, Document, SplitMode};
use turbo_analysis_layers::{TokenGraph, Tokens};
use turbo_tokenize::lattice::{DictionaryFiles, DictionaryLimits};
use turbo_tokenize::{LatticeGraphAnnotator, LatticeTokenizer, MecabDictionary};

const LEXICON: &str = "\
東京都,0,0,5000,noun,proper\n\
東京,0,0,2000,noun,proper\n\
都,0,0,1500,noun,suffix\n\
選挙管理委員会,0,0,2500,noun,compound\n\
選挙,0,0,2000,noun,common\n\
管理,0,0,2000,noun,common\n\
委員会,0,0,2000,noun,common\n\
委員,0,0,1500,noun,common\n\
会,0,0,1500,noun,suffix\n\
に,0,0,1000,particle,case\n\
行く,0,0,2000,verb,base\n";

const CHAR_DEF: &str = "\
DEFAULT 0 1 0\n\
KANJI 0 0 2\n\
HIRAGANA 0 1 0\n\n\
0x3041..0x3096 HIRAGANA\n\
0x4E00..0x9FFF KANJI\n";

const UNK_DEF: &str = "\
DEFAULT,0,0,10000,symbol,unknown\n\
KANJI,0,0,8000,noun,unknown\n\
HIRAGANA,0,0,9000,particle,unknown\n";

fn sample_tokenizer() -> LatticeTokenizer {
    let lexicons = [("lexicon.csv", LEXICON.as_bytes())];
    let files = DictionaryFiles {
        matrix: b"1 1\n0 0 0\n",
        lexicons: &lexicons,
        char_def: CHAR_DEF.as_bytes(),
        unk_def: UNK_DEF.as_bytes(),
    };
    let dict = MecabDictionary::from_files(files, DictionaryLimits::default()).unwrap();
    LatticeTokenizer::new(dict)
}

#[test]
fn mecab_lattice_graph_exposes_finest_viterbi_and_compound_paths() {
    let text = "東京都選挙管理委員会に行く";
    let tokenizer = sample_tokenizer();
    let analyzer = Analyzer::builder().add(LatticeGraphAnnotator::new(tokenizer)).build().unwrap();

    let mut doc = Document::new(text);
    analyzer.run(std::slice::from_mut(&mut doc)).remove(0).unwrap();

    let tokens = doc.get::<Tokens>().unwrap();
    let alt = doc.get::<Alternatives<Tokens>>().unwrap();
    let graph = doc.get::<TokenGraph>().unwrap();

    // SplitMode::Best matches 1-best Viterbi Tokens bit-for-bit.
    let from_best = Tokens::from_alternatives(alt, SplitMode::Best).unwrap();
    assert_eq!(from_best.spans.columns(), tokens.spans.columns());

    // SplitMode::A yields finest lexicon units, SplitMode::C yields maximal compounds.
    let path_a: Vec<&str> =
        alt.path_for_mode(SplitMode::A).unwrap().into_iter().map(|ix| &text[alt.spans.get(ix).range()]).collect();
    assert_eq!(path_a, ["東京", "都", "選挙", "管理", "委員", "会", "に", "行く"]);

    let path_c: Vec<&str> =
        alt.path_for_mode(SplitMode::C).unwrap().into_iter().map(|ix| &text[alt.spans.get(ix).range()]).collect();
    assert_eq!(path_c, ["東京都", "選挙管理委員会", "に", "行く"]);

    // Both granularities match exact slop=0 phrases in TokenGraph without ordinal drift!
    assert_eq!(graph.matches_phrase(&["東京都", "選挙管理委員会", "に", "行く"]).len(), 1);
    assert_eq!(graph.matches_phrase(&["東京", "都", "選挙", "管理", "委員", "会", "に", "行く"]).len(), 1);
}
