/// A model file the generator loads, where it is published and its
/// SHA-512: the one published beside it, or for the OpenNLP 1.5 models,
/// which are published with no hash, the hash of the download, whose
/// SHA-256 matched the one the sequence crate's README lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub file: &'static str,
    pub url: &'static str,
    pub sha512: &'static str,
}

/// The published models, in the order the generator loads them. The file
/// names are the ones the generator's `Runs` class opens in the models
/// directory.
pub const MODELS: [Model; 13] = [
    Model {
        file: "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin",
        sha512: "fed5d53ea87bf66afeb08c084f89ca9b95a96de0920dca40487b2ab9972ee6b7\
                 ef211e477db480b7253431de079fd66dfd14329cd7d19ad1f565155eb0cf5191",
    },
    Model {
        file: "opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin",
        sha512: "c811604c24cdbe490ab42dcffdbae63f9a74b39622f21b7b31f63eea566ffe35\
                 ef871f0da905e61180656b877cf0be4d626588c6312fcc18df5ab9a35b4183fa",
    },
    Model {
        file: "opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin",
        sha512: "0c0390b36530ce9807e9f3d69cfd09524d965506dbb6e2aab2f34556c8858903\
                 d538433b828e1f6ed7eaca92f336a04be83461a45b28b177d01ba372fafa35f8",
    },
    Model {
        file: "opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin",
        sha512: "147d1cdcb89ca5d9bd0ead8fddf2cc14ae543a9151803fe56b3c1fa152b98660\
                 5e34fb6dda4819c68aa22fc9bffa6bfab14f916b7d0bf4892d452686af34778a",
    },
    Model {
        file: "opennlp-de-ud-gsd-tokens-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-de-ud-gsd-tokens-1.3-2.5.4.bin",
        sha512: "343c9fe93ff301b736f30fa2e0d954b270994a1535ec2bd424553cbd82b13eb0\
                 d86aa2042946db62322430a8b5a8f29a8a2d882f107cb26c6d4f6d494d97debf",
    },
    Model {
        file: "opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin",
        sha512: "ba5280ca74b190515f630ed2b2257a758e8baa876a8da3c02735f81a5543c37f\
                 7bc57614fd1ecb2850b4a2df98b2d2e452718de43f676f9b63b064206f86c467",
    },
    Model {
        file: "opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin",
        sha512: "18e56e96eae9741935c5f54f16d3c6933e151ec550b882d2e5e1194ac02c6b1e\
                 56063843ff3805a35cf44810091674ef6bf2f080d1bc761770d8b09ee9a01277",
    },
    Model {
        file: "opennlp-pt-ud-gsd-tokens-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-pt-ud-gsd-tokens-1.3-2.5.4.bin",
        sha512: "6ff66aec4835e9d5caeeb50a2e70bd673840ef6219fe1dc4018c8f9999b21b95\
                 9c52fdac68844a0934bbcd77346743627f728fb9c99b7ca468e795fc5291c3bb",
    },
    Model {
        file: "opennlp-en-ud-ewt-pos-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-en-ud-ewt-pos-1.3-2.5.4.bin",
        sha512: "fca85b38d996b7c422401054feedeb0f748cbc03856893cb27670da5d207315e\
                 c9af0c0f818c3107a741a2effca2499b5097ac0b46edf4a5ac10bf77a351ac3e",
    },
    Model {
        file: "opennlp-de-ud-gsd-pos-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-de-ud-gsd-pos-1.3-2.5.4.bin",
        sha512: "9e44d582874464924a8baf8fbfed7500e15112c3dc446bfc5be281bff4dbbf19\
                 5eda0a16ca7e78fde7a64a684c0f780b5987723db800ce88465b9894ef0ba17b",
    },
    Model {
        file: "opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin",
        sha512: "ee81553985908ac25c50fbd8f2ca4222cb42a18845f063c792a1b43639181b70\
                 ce20d57b8736a1604367f5e8c59aef55f8463e04004840cbc88abb7123fd9e7f",
    },
    Model {
        file: "opennlp-de-ud-gsd-lemmas-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-de-ud-gsd-lemmas-1.3-2.5.4.bin",
        sha512: "616c83327e1af12c8952f362fd99be6efcdc4e28526bee1debf27d3446110ded\
                 9beef7e0e2acc9b3094039683c9b4b2b2b0ba32dbf194278d5017e2b21e66371",
    },
    Model {
        file: "langdetect-183.bin",
        url: "https://downloads.apache.org/opennlp/models/langdetect/1.8.3/langdetect-183.bin",
        sha512: "9dc847055e622d9de503a5bf444ccf1428c3578c9698a73b736d8f8e337db018\
                 8ade90ac8cc9196d1d472f58589d578785085b554b1bde856afc5762e9396285",
    },
];

/// The OpenNLP 1.5 English models the sequence crate's beam search, tag
/// dictionary and tag format goldens run, which the generator also loads
/// from the models directory. Their training corpora are LDC corpora or
/// undocumented, so CI does not fetch them: `fetch-models` fetches them
/// only when asked (`--local`), and `generate`, `verify` and `fuzz`
/// always do, since the generator runs them.
pub const LOCAL_MODELS: [Model; 5] = [
    Model {
        file: "en-pos-maxent.bin",
        url: "https://opennlp.sourceforge.net/models-1.5/en-pos-maxent.bin",
        sha512: "7810d7e7c24ed1c6a202dad64ed021310ee62cf324ee866e2e63bf8054199716\
                 2bb5fa2da91fdcfddda39ee275efcdbde738647274d586ba8896940a0467d6cc",
    },
    Model {
        file: "en-pos-perceptron.bin",
        url: "https://opennlp.sourceforge.net/models-1.5/en-pos-perceptron.bin",
        sha512: "01d296d8dc7fdc5e7e93ef8c415e9af752e04d7ce77008b5a864f14cd1a5e53b\
                 2d22fc9b7cbd37ecaa814c6b7a83f2f22fdf531dac2a6784fae723d6ab173e03",
    },
    Model {
        file: "en-chunker.bin",
        url: "https://opennlp.sourceforge.net/models-1.5/en-chunker.bin",
        sha512: "43e83ac9b1b9b0e52d7813e6f4ed2fb3b94dd4146d9b36028795c273ad6ef191\
                 68f1ad5ced3ca11f1b91f81f6fa1324d2070c96342cf538eccadc957a0721730",
    },
    Model {
        file: "en-ner-person.bin",
        url: "https://opennlp.sourceforge.net/models-1.5/en-ner-person.bin",
        sha512: "c62b3460527f103ea1e9b8e0b794822ceb0536b03a843e0ca86f9ae559c1cca0\
                 a4fa03e81ecfec468628be3ee09c17adef08cd460214bd10bb089326427eb04c",
    },
    Model {
        file: "en-ner-location.bin",
        url: "https://opennlp.sourceforge.net/models-1.5/en-ner-location.bin",
        sha512: "f75be6fb88a18ad3bb6f196a72931fa259091cf22e4c05230aace4aa990a6340\
                 d83937c00431165704fe6762a030fe944bae157f4d4a914c4a01deefc52f61b3",
    },
];

/// A MeCab dictionary the generator loads from the models directory: the
/// archive it is published in, the SHA-512 of the archive, the directory
/// the archive unpacks to (where the dictionary's files are) and the
/// encoding of those files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dictionary {
    pub archive: &'static str,
    pub url: &'static str,
    pub sha512: &'static str,
    pub dir: &'static str,
    pub encoding: DictionaryEncoding,
}

/// The encoding of a dictionary's files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictionaryEncoding {
    Utf8,
    /// EUC-JP: `fetch-models` also writes a UTF-8 copy of the files the
    /// dictionary reader reads into `<dir>.utf8`, with `iconv`. The
    /// generator loads the original in EUC-JP and checks that the copy is
    /// the same text; the Rust port reads the copy. Where iconv and Java
    /// decode a code differently (0xA1BD), the copy has what Java decodes.
    EucJp,
}

/// The published MeCab dictionaries, with the SHA-512 of each archive.
/// No project publishes a hash beside its archive; IPADIC and mecab-ko-dic
/// use the hashes OpenNLP's dictionary catalog test resource lists
/// (`dictionary-catalog.properties` at the OpenNLP commit), and UniDic uses
/// the hash of `unidic-mecab-2.1.2.tar.gz` (whose MD5 matches the one
/// `lindera-unidic` pins). IPADIC is under the NAIST and ICOT notice in its
/// `COPYING`, mecab-ko-dic under the Apache License 2.0, and UniDic under
/// the BSD License in its `BSD` file; the archives are fetched, never
/// checked in.
pub const DICTIONARIES: [Dictionary; 3] = [
    Dictionary {
        archive: "mecab-ipadic-2.7.0-20070801.tar.gz",
        url: "https://downloads.sourceforge.net/project/mecab/mecab-ipadic/2.7.0-20070801/mecab-ipadic-2.7.0-20070801.tar.gz",
        sha512: "35ea662cb62f1967849f7ed5781bd6dafef0fe20d63e88d9a0057666e57ed23d\
                 5a0e6fb8d0701a0cc4da43a1050c1b02463bb862decc71c36b7fc2acdc158d86",
        dir: "mecab-ipadic-2.7.0-20070801",
        encoding: DictionaryEncoding::EucJp,
    },
    Dictionary {
        archive: "mecab-ko-dic-2.1.1-20180720.tar.gz",
        url: "https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz",
        sha512: "986f8f9c66c53accd296756bf632c979d2d44b695ada33f36c662f210dba34cd\
                 95d67b61dd8c84a1f7d59f80ee6bc22eb1e9afb5dc6a7f9b6b75b4fbf2f8164f",
        dir: "mecab-ko-dic-2.1.1-20180720",
        encoding: DictionaryEncoding::Utf8,
    },
    Dictionary {
        archive: "unidic-mecab-2.1.2.tar.gz",
        url: "https://lindera.dev/unidic-mecab-2.1.2.tar.gz",
        sha512: "0bb18e1559ae4978f886c2e37d10975fd46ff4c5c4b388e7ff26888a368151ad\
                 d13e1dc9a1450e69d6f5f182b9f6cba67244eb1e65b23099e18eed7942876df2",
        dir: "unidic-mecab-2.1.2",
        encoding: DictionaryEncoding::Utf8,
    },
];

/// A file of a published Hunspell dictionary the generator loads from the
/// models directory: its path there, where it is published, its SHA-512
/// and the licence it is published under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HunspellFile {
    pub file: &'static str,
    pub url: &'static str,
    pub sha512: &'static str,
    pub license: &'static str,
}

/// The commit of LibreOffice's dictionaries repository the Hunspell
/// dictionaries are fetched at.
pub const HUNSPELL_DICTIONARIES_COMMIT: &str = "67032ee957631908098a72edf48dc81cac27e369";

/// The published Hunspell dictionaries the Hunspell goldens read, from
/// LibreOffice's dictionaries repository at
/// [`HUNSPELL_DICTIONARIES_COMMIT`]. The repository publishes no hashes;
/// these are the hashes of the files at that commit. `de_DE_frami` is under
/// the GPL and `hu_HU` under the MPL or the LGPL, so none of them is checked
/// in; `fetch-models` fetches them into `<models>/hunspell`, and CI caches
/// them with the models.
pub const HUNSPELL_FILES: [HunspellFile; 6] = [
    HunspellFile {
        file: "hunspell/en_US.aff",
        url: "https://raw.githubusercontent.com/LibreOffice/dictionaries/67032ee957631908098a72edf48dc81cac27e369/en/en_US.aff",
        sha512: "2b4448dfdff03caf300914415f4642f8d2ba5b650c5f024a12355b420a279ffc\
                 12146649fce092ba591504476634a3d6fd4c079335a27085b396fa76bfd28b74",
        license: "SCOWL (Kevin Atkinson's notice in README_en_US.txt: permissive, MIT-like)",
    },
    HunspellFile {
        file: "hunspell/en_US.dic",
        url: "https://raw.githubusercontent.com/LibreOffice/dictionaries/67032ee957631908098a72edf48dc81cac27e369/en/en_US.dic",
        sha512: "4be737249a8a436d20a02be575dcf6cf2f06f5f2abb840ea5ec0ef0ac73a71fa\
                 0e4669e527c703d5c6b50ef61713a6741b55bc136d559a54bcdeebcd62027988",
        license: "SCOWL (Kevin Atkinson's notice in README_en_US.txt: permissive, MIT-like)",
    },
    HunspellFile {
        file: "hunspell/de_DE_frami.aff",
        url: "https://raw.githubusercontent.com/LibreOffice/dictionaries/67032ee957631908098a72edf48dc81cac27e369/de/de_DE_frami.aff",
        sha512: "6fec308258040c29e6b91bc51381c6e8a631bcdd66e59f84a673705f22a540f6\
                 819e6b4720add394a46ccceff775487cb6f6c240fe5bd357e4d60e21e8ba0ff0",
        license: "GPL-2.0 or GPL-3.0 (igerman98, README_de_DE_frami.txt)",
    },
    HunspellFile {
        file: "hunspell/de_DE_frami.dic",
        url: "https://raw.githubusercontent.com/LibreOffice/dictionaries/67032ee957631908098a72edf48dc81cac27e369/de/de_DE_frami.dic",
        sha512: "ab9e7156f2f31a0fd1d6cad6487f43aec98f1767d46daf7a7a0048b9cbfbd9bc\
                 449f29225a10217ce2d0a722afd8b606f2f6b027dbde2720b093343334f874dd",
        license: "GPL-2.0 or GPL-3.0 (igerman98, README_de_DE_frami.txt)",
    },
    HunspellFile {
        file: "hunspell/hu_HU.aff",
        url: "https://raw.githubusercontent.com/LibreOffice/dictionaries/67032ee957631908098a72edf48dc81cac27e369/hu_HU/hu_HU.aff",
        sha512: "c3e587793dc752aaf9b8dd9262047d8ff24dc7e620ebd4d9023e77bb6eb7c15e\
                 89f39a1851050f413410a298e2b1323cc9abf4627dc96aa15de0d91e26ae1a00",
        license: "MPL-2.0 or LGPL-3.0-or-later (Magyar Ispell 1.9, README_hu_HU.txt)",
    },
    HunspellFile {
        file: "hunspell/hu_HU.dic",
        url: "https://raw.githubusercontent.com/LibreOffice/dictionaries/67032ee957631908098a72edf48dc81cac27e369/hu_HU/hu_HU.dic",
        sha512: "767ac7b9cf190cca8314142dc016ba9b593b6bfbc707eb55314a0a47e8dfdea3\
                 3ceec35880057a6f96897fd289759ff59a2a0feff5cd108d832cb1cfb027d141",
        license: "MPL-2.0 or LGPL-3.0-or-later (Magyar Ispell 1.9, README_hu_HU.txt)",
    },
];

/// A training corpus the generator reads from the models directory: its
/// path there, where it is fetched from, its SHA-512, the licence it is
/// published under and where it is described.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Corpus {
    pub file: &'static str,
    pub url: &'static str,
    pub sha512: &'static str,
    pub license: &'static str,
    pub source: &'static str,
}

/// The training corpora, fetched by `fetch-models` and never checked in.
///
/// The SMS Spam Collection (T. A. Almeida, J. M. Gomez Hidalgo and A.
/// Yamakami, "Contributions to the Study of SMS Spam Filtering: New
/// Collection and Results", DocEng 2011) is published by the UCI Machine
/// Learning Repository under CC BY 4.0, as a zip holding the file
/// `SMSSpamCollection`: 5,574 lines, each `ham` or `spam`, a tab and the
/// message. The file is fetched from a copy in a GitHub repository, pinned
/// by commit, which has those 5,574 lines (4,827 `ham`, 747 `spam`); the
/// hash is of that copy, which has not been compared byte for byte with
/// the file in the UCI archive. The maxent document categorizer goldens
/// (`DoccatGolden.java`) train on it.
pub const CORPORA: [Corpus; 1] = [Corpus {
    file: "corpora/sms-spam-collection.tsv",
    url: "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/5339710562cc46c4a32a369d63f1e8751a18a54e/data/sms.tsv",
    sha512: "2b454257f2d5eff9d829d5be9d0f99af68581367be6bad8e564e1d872112ce6\
             52e7e1d75a5752e4770eb682e7ffd2d0be68b6b015e9588bdf599a30a72307957",
    license: "CC BY 4.0",
    source: "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
}];

/// The suffix of the directory holding the UTF-8 copy of a dictionary
/// that is not UTF-8.
pub const UTF8_COPY_SUFFIX: &str = ".utf8";
