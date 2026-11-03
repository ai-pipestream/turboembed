/// A model file the generator loads, where it is published and its
/// SHA-512: the one published beside it, or for a file published with no
/// hash, the hash of the download.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub file: &'static str,
    pub url: &'static str,
    pub sha512: &'static str,
}

/// The published OpenNLP models the generator loaded: none. Published
/// OpenNLP 2.5 UD and language-detector artifacts are not fetched, pinned
/// or run (#1509, #1653); their goldens go with #1532 and #1644 to #1648.
pub const MODELS: [Model; 0] = [];

/// The upstream all-MiniLM-L6-v2 ONNX export, at the revision the sentence
/// vector bundles are made from, which `fetch` turns into the mean-pooled
/// copy the SentenceVectorsDL goldens run (`crate::pooled`). The generator
/// never opens it. Hugging Face publishes its SHA-256 (6fd5d72f...); this
/// is the SHA-512 of the same bytes. Fetched only when asked (`--local`).
pub const POOLING_SOURCE: Model = Model {
    file: "all-minilm-l6-v2.onnx",
    url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/\
          c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model.onnx",
    sha512: "bcecf6e370eba3d2384b9d98676f79cebadf2462fd583fbd579a87e0149fa369\
             004cd592a77d0acd45767de8f07efe6103e2474a67b7bbdadf0adec8f573bc7f",
};

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
