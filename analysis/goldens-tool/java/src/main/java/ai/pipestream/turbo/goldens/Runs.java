package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import opennlp.tools.postag.POSTagFormat;
import opennlp.tools.util.WhitespaceMode;

/**
 * Every annotator run the generator makes goldens of, with the model and fixture files each
 * loads and the input sets each runs on, and the recorders that make the decision point dumps.
 */
final class Runs {

  /** The model files the generator loads from the models directory. */
  static final String SENTENCE_MODEL_EN = "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin";
  static final String SENTENCE_MODEL_DE = "opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin";
  static final String SENTENCE_MODEL_PT = "opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin";
  static final String TOKEN_MODEL_EN = "opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin";
  static final String TOKEN_MODEL_DE = "opennlp-de-ud-gsd-tokens-1.3-2.5.4.bin";
  static final String TOKEN_MODEL_ES = "opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin";
  static final String TOKEN_MODEL_FR = "opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin";
  static final String TOKEN_MODEL_PT = "opennlp-pt-ud-gsd-tokens-1.3-2.5.4.bin";
  static final String LANGUAGE_MODEL = "langdetect-183.bin";
  static final String POS_MODEL_EN = "opennlp-en-ud-ewt-pos-1.3-2.5.4.bin";
  static final String POS_MODEL_DE = "opennlp-de-ud-gsd-pos-1.3-2.5.4.bin";
  static final String LEMMA_MODEL_EN = "opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin";
  static final String LEMMA_MODEL_DE = "opennlp-de-ud-gsd-lemmas-1.3-2.5.4.bin";

  /**
   * The MeCab dictionaries the generator loads from the models directory, each unpacked from its
   * published archive, and the UTF-8 copy of the one that is not UTF-8.
   */
  static final String IPADIC = "mecab-ipadic-2.7.0-20070801";
  static final String IPADIC_UTF8 = "mecab-ipadic-2.7.0-20070801.utf8";
  static final String IPADIC_CHARSET = "EUC-JP";
  static final String KO_DIC = "mecab-ko-dic-2.1.1-20180720";

  /**
   * The files the generator loads from the fixtures directory: models and dictionaries made for
   * the tests or copied from OpenNLP, which are checked in (analysis/goldens/data/fixtures).
   */
  static final String SENTENCE_FIXTURE_ABBREVIATIONS = "en-abbrev.bin";
  static final String SENTENCE_FIXTURE_PLAIN = "en-plain.bin";
  /** The sentence model the Rust trainer wrote from the events of en-plain (analysis/train/tests/roundtrip.rs). */
  static final String SENTENCE_FIXTURE_RUST = "en-plain-rust.bin";
  static final String TOKEN_FIXTURE = "tokenizer-abbreviations.bin";
  static final String POS_FIXTURE = "pos-fixture.bin";
  static final String CHUNKER_FIXTURE = "chunker-fixture.bin";
  static final String LEMMA_DICTIONARY = "lemmatizer-dictionary.txt";
  static final String ENGLISH_ABBREVIATIONS = "english-dictionary.xml";
  static final List<String> DETOKENIZER_DICTIONARIES = List.of("en-detokenizer.xml", "fr-detokenizer.xml",
      "latin-detokenizer.xml", "pt-detokenizer.xml", "quotes-detokenizer.xml");
  static final String WORDPIECE_VOCABULARY = "all-minilm-l6-v2-vocab.txt";
  static final String UNIGRAM_FIXTURE = "zh-unigram-fixture.txt";
  static final String UNIGRAM_JIEBA = "jieba-dict-subset.txt";
  static final String UNIGRAM_MODES = "zh-unigram-modes.txt";
  static final String LATTICE_JA = "mecab-fixture-ja";
  static final String LATTICE_KO = "mecab-fixture-ko";
  static final String LATTICE_PARTIAL = "mecab-fixture-partial";
  static final String FORMAT_DETOKENIZER = "en-detokenizer.xml";

  /** The input sets, the directories of the inputs directory. */
  static final String EN = "en";
  static final String MULTI = "multi";
  static final String EDGE = "edge";
  static final String SURROGATE = "surrogate";
  static final String ABBREVIATIONS = "abbreviations";
  static final String ALPHANUMERIC = "alphanumeric";
  static final String SENTENCE_FIXTURE = "sentence-fixture";
  static final String DETOKENIZER = "detokenizer";
  static final String SUBWORDS = "subwords";
  static final String BPE = "bpe";
  static final String ZH = "zh";
  static final String JA = "ja";
  static final String KO = "ko";
  /** The sentences the fixture model en-plain.bin was trained on, one per line, for the training goldens. */
  static final String TRAIN = "train";
  static final String FORMATS = "formats";

  /** The numbers of merges the BPE trainer runs with: few, so the tokens are pieces of words, and many. */
  static final List<Integer> BPE_MERGES = List.of(40, 400);

  /** The beam sizes the English part-of-speech model is rebuilt with, besides its own 3. */
  static final List<Integer> POS_BEAM_SIZES = List.of(1, 10);

  private static final List<String> TOKEN_MODELS = List.of(TOKEN_MODEL_EN, TOKEN_MODEL_DE, TOKEN_MODEL_ES,
      TOKEN_MODEL_FR, TOKEN_MODEL_PT);
  private static final List<WhitespaceMode> MODES = List.of(WhitespaceMode.UNICODE, WhitespaceMode.LEGACY);

  /** The annotators, and the recorders of the ones that ask a maxent model. */
  record All(List<AnnotatorGolden> annotators, List<AnnotatorDump> recorders) {}

  private Runs() {}

  static All load(Path models, Path fixtures, Path moneySnapshot) throws IOException {
    List<AnnotatorGolden> annotators = new ArrayList<>();
    List<AnnotatorDump> recorders = new ArrayList<>();

    // Sentence detection: the published models, the two fixture models and the model the Rust
    // trainer wrote in both whitespace modes, and the newline detector in both.
    List<SentenceGolden> sentences = new ArrayList<>();
    sentences.add(SentenceGolden.load(models.resolve(SENTENCE_MODEL_EN), WhitespaceMode.UNICODE,
        Set.of(EN, MULTI, EDGE)));
    for (String m : List.of(SENTENCE_MODEL_DE, SENTENCE_MODEL_PT)) {
      sentences.add(SentenceGolden.load(models.resolve(m), WhitespaceMode.UNICODE, Set.of(EDGE)));
    }
    for (String m : List.of(SENTENCE_FIXTURE_ABBREVIATIONS, SENTENCE_FIXTURE_PLAIN, SENTENCE_FIXTURE_RUST)) {
      for (WhitespaceMode mode : MODES) {
        sentences.add(SentenceGolden.load(fixtures.resolve(m), mode, Set.of(SENTENCE_FIXTURE)));
      }
    }
    for (SentenceGolden s : sentences) {
      annotators.add(s);
      recorders.add(SentenceDump.of(s));
    }
    for (WhitespaceMode mode : MODES) {
      annotators.add(new NewlineSentenceGolden(mode, Set.of(EDGE)));
    }

    // Tokenization: the published models as they are, the fixture model with its own
    // abbreviation dictionary, the English model with one added, and every published model with
    // the alphanumeric optimization on.
    List<TokenGolden> tokens = new ArrayList<>();
    tokens.add(TokenGolden.load(models.resolve(TOKEN_MODEL_EN), Set.of(EN, MULTI, EDGE, SURROGATE)));
    for (String m : TOKEN_MODELS.subList(1, TOKEN_MODELS.size())) {
      tokens.add(TokenGolden.load(models.resolve(m), Set.of(EDGE, SURROGATE)));
    }
    tokens.add(TokenGolden.load(fixtures.resolve(TOKEN_FIXTURE), Set.of(ABBREVIATIONS, EDGE)));
    tokens.add(TokenGolden.withAbbreviations(models.resolve(TOKEN_MODEL_EN), fixtures.resolve(ENGLISH_ABBREVIATIONS),
        Set.of(ABBREVIATIONS, EDGE)));
    for (String m : TOKEN_MODELS) {
      tokens.add(TokenGolden.withAlphaNumericOptimization(models.resolve(m), Set.of(EDGE, ALPHANUMERIC)));
    }
    for (TokenGolden t : tokens) {
      annotators.add(t);
      recorders.add(TokenDump.of(t));
    }
    for (PlainTokenGolden.Kind kind : PlainTokenGolden.Kind.values()) {
      for (boolean keepNewLines : new boolean[] {false, true}) {
        for (WhitespaceMode mode : MODES) {
          annotators.add(new PlainTokenGolden(kind, keepNewLines, mode, Set.of(EDGE)));
        }
      }
    }
    annotators.add(new WordTokenGolden(Set.of(EDGE)));

    for (String d : DETOKENIZER_DICTIONARIES) {
      annotators.add(DetokenizerGolden.load(fixtures.resolve(d), Set.of(DETOKENIZER)));
    }

    annotators.add(WordpieceGolden.load(fixtures.resolve(WORDPIECE_VOCABULARY),
        Set.of(EN, MULTI, EDGE, SURROGATE, SUBWORDS)));

    for (int merges : BPE_MERGES) {
      annotators.add(new BpeGolden(merges, Set.of(BPE)));
    }

    // The unigram segmenter: the fixture lexicon, the jieba subset and a lexicon whose words
    // depend on the whitespace mode, each in both modes.
    for (WhitespaceMode mode : MODES) {
      annotators.add(UnigramGolden.load(fixtures.resolve(UNIGRAM_FIXTURE), mode, Set.of(ZH, MULTI, EDGE, SURROGATE)));
      annotators.add(UnigramGolden.load(fixtures.resolve(UNIGRAM_JIEBA), mode, Set.of(ZH, MULTI, EDGE)));
      annotators.add(UnigramGolden.load(fixtures.resolve(UNIGRAM_MODES), mode, Set.of(ZH)));
    }

    // The lattice tokenizer: the fixture dictionaries (the Japanese one in both modes, and one
    // whose unk.def has no DEFAULT template, which throws on some inputs), IPADIC loaded in its
    // own encoding beside its UTF-8 copy, and mecab-ko-dic.
    for (WhitespaceMode mode : MODES) {
      annotators.add(LatticeGolden.load(fixtures.resolve(LATTICE_JA), mode, Set.of(JA, MULTI, EDGE, SURROGATE)));
    }
    annotators.add(LatticeGolden.load(fixtures.resolve(LATTICE_KO), WhitespaceMode.UNICODE, Set.of(KO, MULTI, EDGE)));
    annotators.add(LatticeGolden.load(fixtures.resolve(LATTICE_PARTIAL), WhitespaceMode.UNICODE, Set.of(JA, EDGE)));
    annotators.add(LatticeGolden.loadRecoded(models.resolve(IPADIC), Charset.forName(IPADIC_CHARSET),
        models.resolve(IPADIC_UTF8), WhitespaceMode.UNICODE, Set.of(JA, MULTI, EDGE, SURROGATE)));
    annotators.add(LatticeGolden.load(models.resolve(KO_DIC), WhitespaceMode.UNICODE,
        Set.of(KO, MULTI, EDGE, SURROGATE)));

    // Part-of-speech tagging, on the sentences and tokens of the published models of the same
    // language: the English model as it is, with Penn Treebank tags asked for, and rebuilt with
    // other beam sizes; the German model; and the fixture model, whose Penn outcomes the tagger
    // converts to UD and whose tag dictionary restricts the tags of the words it lists.
    List<PosGolden> taggers = new ArrayList<>();
    PosGolden.Pipeline english = PosGolden.Pipeline.load(models.resolve(SENTENCE_MODEL_EN), models.resolve(TOKEN_MODEL_EN));
    PosGolden.Pipeline german = PosGolden.Pipeline.load(models.resolve(SENTENCE_MODEL_DE), models.resolve(TOKEN_MODEL_DE));
    taggers.add(PosGolden.load(models.resolve(POS_MODEL_EN), english, Set.of(EN, MULTI, EDGE)));
    taggers.add(PosGolden.withFormat(models.resolve(POS_MODEL_EN), english, POSTagFormat.PENN, Set.of(EN)));
    for (int beam : POS_BEAM_SIZES) {
      taggers.add(PosGolden.withBeamSize(models.resolve(POS_MODEL_EN), english, beam, Set.of(EN)));
    }
    taggers.add(PosGolden.load(models.resolve(POS_MODEL_DE), german, Set.of(EDGE)));
    taggers.add(PosGolden.load(fixtures.resolve(POS_FIXTURE), english, Set.of(EN, EDGE)));
    for (PosGolden t : taggers) {
      annotators.add(t);
      recorders.add(PosDump.of(t));
    }

    // Lemmatization, on the sentences, tokens and tags of the published models of the same
    // language: the English and German maxent lemmatizers, and the dictionary lemmatizer with the
    // fixture dictionary over the English pipeline.
    LemmaGolden.Pipeline taggedEnglish = LemmaGolden.Pipeline.load(models.resolve(SENTENCE_MODEL_EN),
        models.resolve(TOKEN_MODEL_EN), models.resolve(POS_MODEL_EN));
    LemmaGolden.Pipeline taggedGerman = LemmaGolden.Pipeline.load(models.resolve(SENTENCE_MODEL_DE),
        models.resolve(TOKEN_MODEL_DE), models.resolve(POS_MODEL_DE));
    for (LemmaGolden l : List.of(LemmaGolden.load(models.resolve(LEMMA_MODEL_EN), taggedEnglish, Set.of(EN, MULTI, EDGE)),
        LemmaGolden.load(models.resolve(LEMMA_MODEL_DE), taggedGerman, Set.of(EDGE)))) {
      annotators.add(l);
      recorders.add(LemmaDump.of(l));
    }
    annotators.add(DictionaryLemmaGolden.load(fixtures.resolve(LEMMA_DICTIONARY), taggedEnglish, Set.of(EN, MULTI, EDGE)));

    // Phrase chunking, on the sentences and tokens of the English models and the Penn Treebank
    // tags the English part-of-speech model is asked for, since the fixture chunker was trained
    // on Penn Treebank tags: the fixture model as it is, and rebuilt with a beam of one.
    ChunkGolden.Pipeline pennEnglish = ChunkGolden.Pipeline.load(models.resolve(SENTENCE_MODEL_EN),
        models.resolve(TOKEN_MODEL_EN), models.resolve(POS_MODEL_EN), POSTagFormat.PENN);
    for (ChunkGolden c : List.of(ChunkGolden.load(fixtures.resolve(CHUNKER_FIXTURE), pennEnglish, Set.of(EN, MULTI, EDGE)),
        ChunkGolden.withBeamSize(fixtures.resolve(CHUNKER_FIXTURE), pennEnglish, 1, Set.of(EN)))) {
      annotators.add(c);
      recorders.add(ChunkDump.of(c));
    }

    LanguageGolden languages = LanguageGolden.load(models.resolve(LANGUAGE_MODEL), Set.of(EN, MULTI));
    annotators.add(languages);
    recorders.add(LanguageDump.of(languages));

    annotators.addAll(NormalizerGolden.all());
    annotators.addAll(NumberGolden.all());
    annotators.addAll(MoneyGolden.all(moneySnapshot));
    annotators.addAll(QuantityGolden.all());

    // The training stack: the events of the sentence detector, the tokenizer and a real valued
    // source on the texts, indexed and trained; the sentence detector runs on the fixture
    // sentences too.
    annotators.addAll(TrainGolden.all(Set.of(EN, MULTI, EDGE), Set.of(TRAIN)));
    // The sample streams of OpenNLP's native formats, and the conversion streams, which
    // detokenize with the English dictionary.
    annotators.addAll(FormatGolden.all(fixtures.resolve(FORMAT_DETOKENIZER)));
    // The evaluators, over pairs of a reference sample and a prediction in a native format.
    annotators.addAll(EvalGolden.all());
    return new All(annotators, recorders);
  }
}
