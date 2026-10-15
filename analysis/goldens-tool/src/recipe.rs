//! What the goldens are made from, as constants: the container's base
//! image, the OpenNLP and OpenNLP addons commits, the models, the checked
//! in fixtures and the input sets. The container recipe
//! (`container/Dockerfile`) repeats the base image and the commits; a test
//! checks that both say the same.

/// The base image, pinned by the digest of its multi-platform index.
pub const BASE_IMAGE: &str = "eclipse-temurin:21-jdk";
pub const BASE_IMAGE_DIGEST: &str = "sha256:3e3c176ffed168beb42c607be9bc1639b466cf00261a0fb04425562c9d0c5c2b";

/// The OpenNLP fork and the commit on its `kristian-3.x-features` branch the
/// generator runs.
pub const OPENNLP_REPO: &str = "https://github.com/ai-pipestream/opennlp";
pub const OPENNLP_COMMIT: &str = "cdf33c653d5d61d5af7d37ffee189a83bd8392ea";

/// The OpenNLP addons repository and the commit on its `ready/numeric`
/// branch whose `numeric` module the generator runs (`NumberScan`). The
/// module is built against the OpenNLP fork above.
pub const ADDONS_REPO: &str = "https://github.com/ai-pipestream/opennlp-addons";
pub const ADDONS_COMMIT: &str = "33e95a4956f1535cdc074056b29246fe88b3c81a";

/// The tag `turbo-goldens image` gives the image it builds. It names the
/// OpenNLP and addons commits, so an image built from other commits has
/// another tag.
pub const IMAGE_TAG: &str = "turbo-goldens-java:opennlp-cdf33c653d5d-addons-33e95a4956f1";

/// A model file the generator loads, where it is published and the SHA-512
/// published beside it.
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
/// Neither project publishes a hash beside the archive; these are the
/// hashes OpenNLP's dictionary catalog test resource lists
/// (`dictionary-catalog.properties` at the OpenNLP commit), which the
/// downloads were checked against. IPADIC is under the NAIST and ICOT
/// notice in its `COPYING`, mecab-ko-dic under the Apache License 2.0; the
/// archives are fetched, never checked in.
pub const DICTIONARIES: [Dictionary; 2] = [
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
];

/// The suffix of the directory holding the UTF-8 copy of a dictionary
/// that is not UTF-8.
pub const UTF8_COPY_SUFFIX: &str = ".utf8";

/// A checked in file the generator loads from the fixtures directory,
/// `analysis/goldens/data/fixtures`: a model trained for the tests or
/// written by the Rust trainer, a
/// dictionary, a WordPiece vocabulary, a frequency lexicon or a file of a
/// MeCab dictionary directory, and the SHA-512 of its bytes. `file` is the
/// path below the fixtures directory, with `/` between directory and file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fixture {
    pub file: &'static str,
    pub sha512: &'static str,
}

/// The fixtures, which are all the files of the fixtures directory. The
/// file names are the ones the generator's `Runs` class opens there; a
/// fixture in a subdirectory is a file of a dictionary directory, and
/// `Runs` opens the directory.
pub const FIXTURES: [Fixture; 39] = [
    Fixture {
        file: "en-abbrev.bin",
        sha512: "759ab1defc61351f53cdd82a26e708789a225feb6e1ec0cfd6e2e54f35f67c14\
                 511f44d0eb73216f9d4c89c0fb26ea28fd6931a7c4f851d09ca487d0a39700fd",
    },
    Fixture {
        file: "en-plain.bin",
        sha512: "8da69d821956999ff898cf23963145cc134beb2bb59aa1ccc5e93d90b0489505\
                 7057d80baf0e8c726bf28dea47021e51eab43689c8dc5fb78c99f19949c93ad8",
    },
    Fixture {
        file: "en-plain-rust.bin",
        sha512: "ef8684db6573d7096b72db1d520fe7181c8ee058d2c9c1abe140556d36da18d1\
                 693e63c2f84faf8f427fdd80b0e0f4bd47ce6a92a7aa53233b0b74c403cff736",
    },
    Fixture {
        file: "en-plain-qn-rust.bin",
        sha512: "23f36eea3d663b3791fd6ee4c5f90e51f7298f4488c815f27b440a2edcac760c\
                 dcb6fbe4da87e1762c09c7f6f5db981cfb4657d0f0a6171a4f6cfaebdb9eda0f",
    },
    Fixture {
        file: "en-plain-perceptron-rust.bin",
        sha512: "5882f58f6cea311ff8b2043509992a31a2870e75fe7fc3e08a96ca7f441e4ef5\
                 8ee8d851ace176942987fa860c7ca37cab6bbcd8c38b5b67abfa57918789b915",
    },
    Fixture {
        file: "tokenizer-abbreviations.bin",
        sha512: "361484f0a67e129fe6f19908704d23e83c8a2bf4551a5f7eec3b40b9082a66b7\
                 026d429233c11639b35bccf042d347d79222cfed62685563f2a926cb8a2460d4",
    },
    Fixture {
        file: "pos-fixture.bin",
        sha512: "70bd00e5c7baba5552e311ad002aa382fd57aa2bc5248ffe6008476546df3799\
                 a4e8578c3e6e90aa51d4a766857d07fa7003d395151d292ede3804fc308d5d32",
    },
    Fixture {
        file: "pos-fixture-rust.bin",
        sha512: "7ce6a0d76f18242966640050d44a94f0f8ff496b2a518a72c0e191a964d32158\
                 8bda11782be6ae3d09f62fec387f01db32bdeb87eb64a741f07bb10359e57328",
    },
    Fixture {
        file: "lemma-fixture-rust.bin",
        sha512: "4fa4c03da55e729af919a2d76a67db49ac0269d073ac05a5af3da9fd8d67ea82\
                 61765d97a3c5e4ca0d22b251e4dc01401f432d6a394d6a6d3fd4b06fecb41402",
    },
    Fixture {
        file: "tags.tagdict",
        sha512: "cbd3dee98987711b48b42967df8e6e3e8400dcba70336fa2c83bfdba03a8c42b\
                 3bd7d5f6f0c80c2f35dc01173795792e117489f11fd23f03f9fcba3fa453417d",
    },
    Fixture {
        file: "lemmatizer-dictionary.txt",
        sha512: "e642dfeb8d0b9f45b1853808878afa8a0a3cbe9f738a0646b9e4cafdaf8bb8c9\
                 15aa548d72833274e877a2b57118906557512d9e6fe22b320c7be87fc964e707",
    },
    Fixture {
        file: "chunker-fixture.bin",
        sha512: "be8e0bfd341644705a652b85b978e4138056922ddb9caea1ac9a2395f0648e55\
                 ad1f1b035d90f6febc0f057f832cecac377f397552fdbdfefadaa27cfa232d6a",
    },
    Fixture {
        file: "chunker-fixture-rust.bin",
        sha512: "71caec6c87065f0872d0ef255ed95443b0c675acf273c2d19a1aefd6c006f1ab\
                 99846bbb28b56d2e78abcb3f78074d00b1135a6dffc4e14ed3957101cf80ee44",
    },
    Fixture {
        file: "english-dictionary.xml",
        sha512: "e824b14f66036b2703cbbd3af6a893f7f07cf282c9e8021ae7be38f10da305a8\
                 7aef48b129923f3973c78f495c2ed073adcad68c90c2f7d94bc6d32870839164",
    },
    Fixture {
        file: "en-head-rules.txt",
        sha512: "4924379ef7ca9376c6953e166512a556006e069e9ba319dc941a74342e3b46aa\
                 64313014d5a801030d5867dc6ea4be9208b75706579840bbcc12fa00aab0cca4",
    },
    Fixture {
        file: "es-head-rules.txt",
        sha512: "b01d1e26e3a298064a86d0222459e1b00fcd5ba4e1bb3859245d86ecd458feb7\
                 93bc208505100e8150f5d1f317762dfe0b01842a7d718245463e77a6a310e900",
    },
    Fixture {
        file: "es-regex-head-rules.txt",
        sha512: "d75363b131a71c1b6fd243e9f339ee97393fe36a9ed89d857c49d89da793dc9b\
                 1e48b92ca0e67fb65c04305fba81d17754b896b2cba90a166348c00feff19a6b",
    },
    Fixture {
        file: "en-detokenizer.xml",
        sha512: "d071b56f77ae5b6aff0a591c1cf7e601a4007b5fdbcc0bd3fc31e8c84e95ec23\
                 2df9765a70bfef786f75dbefcc4cad7d053b34b00ecc805507192a5fddc81260",
    },
    Fixture {
        file: "fr-detokenizer.xml",
        sha512: "9c6c9165060e7ff2801b9cb5e6c778e1e93bc68fc4c3fcd4d44b23dbb3254f04\
                 8e7c9f4d036e1d1bacac6f337f105105990956baf3775f41bc0d97045a25346d",
    },
    Fixture {
        file: "latin-detokenizer.xml",
        sha512: "8de094b90fd1856620ddb7c0d361833055c58d945b13f8263d027797de9510f6\
                 e1cd4b73fffaa6693dab807a11b618ffe8d6ec060d2d4fc40cdc83a91afaa32f",
    },
    Fixture {
        file: "pt-detokenizer.xml",
        sha512: "36be57752d28262efa8fdb0b0d4d997a28f00eb32890419210911303459a1b3a\
                 0ccecc29281f5a415a80b20fb75ac973e52237605960bd7669e2409738f19555",
    },
    Fixture {
        file: "quotes-detokenizer.xml",
        sha512: "d8444302db26d336e30283fa9caf62250b55c671e775629137499b16615e0d2b\
                 6686379cff4d1baa34157deebe61170ebbf9a28d9e0f09622f1aefd7bcf285f0",
    },
    Fixture {
        file: "all-minilm-l6-v2-vocab.txt",
        sha512: "f51f06aae7f580a88998f6f7f24b52495c8d3d289fdbdc21231c05f3a8965783\
                 074d95c17b819186f9a63b622280e8a051105a2161cd0d153fa57db7a0aba9f9",
    },
    Fixture {
        file: "zh-unigram-fixture.txt",
        sha512: "75fe2ae460f0a048e0978dfcf54474752aad6ecde6b64ad86bd59729f3dbce82\
                 9043469e17ad929fed559bccdc1db72a358e706527316e8faf733c63ff144011",
    },
    Fixture {
        file: "jieba-dict-subset.txt",
        sha512: "365d5a67c098b362ec12c166a9b438031fd9ffa5947113bc04e2fad1d623918a\
                 4303cb0f9657e28f4e619fae502f3bf4c412285cfe5d6a128439a614ea60e80c",
    },
    Fixture {
        file: "zh-unigram-modes.txt",
        sha512: "d07069e2b828406873f37f056ca71e2175d8ecbd922d2a34c356f6695c0975e0\
                 3830b7fa8000ba8476312514dca8992b24d848a82abd3c4ce677b0fdf1fe6f9a",
    },
    Fixture {
        file: "mecab-fixture-ja/A-nouns.csv",
        sha512: "3514d1ce9d28d24eafd14406987352f004a85a5980e86971475401e2c87610fb\
                 f383e207498a9067bb37f260dffb3fb03abbedd353dbc08b50d60d24a0aa8847",
    },
    Fixture {
        file: "mecab-fixture-ja/b-others.csv",
        sha512: "1ee8fee5fbed1177dfea712201212a0b6732220f9047a537f8536f31906af2d7\
                 5591b5d4533f21112999142230a44594654eedc85071418dca1df2d1cd777b21",
    },
    Fixture {
        file: "mecab-fixture-ja/char.def",
        sha512: "a1827cdf76d1ad05a1a82d4960a37c5bb2251ac7fadad8b20627fac45de2d758\
                 b4d6ecb2a3cc6313e73df4c929ff9d09a35d1617346a0fcfc438d95d620089c4",
    },
    Fixture {
        file: "mecab-fixture-ja/matrix.def",
        sha512: "5561b2fa77b12841439fa9f81f247749ebe21634e7bc17ed47c1f94698b4b15e\
                 0b9ee433a276edbef3cdf3fd7b19b84f74f478728f8e69551ceb2a93ab4b6286",
    },
    Fixture {
        file: "mecab-fixture-ja/unk.def",
        sha512: "b480bc0b9a28f0876ea756bacecf4f9970d72c33f170fd09279d3f9c92c43b83\
                 1686d01ed98bd541fb84266f752776cba30061607497e59b27d23884efa49832",
    },
    Fixture {
        file: "mecab-fixture-ko/char.def",
        sha512: "771c2403dfb33b08d5f2494463bad93465585d254b532ffadde7515703d01ea4\
                 44deb05a57371dd459d8d4e9eef25b050e2be8a6c7470c052b86e8f1d7b09900",
    },
    Fixture {
        file: "mecab-fixture-ko/matrix.def",
        sha512: "c11d28694008c74e84f9fe2f3c42d25106fb496851bbe92b166af575922ff422\
                 3456cfe2ca3692fee1863a8a3539859ab772e9b9868bea04df41e43cc1ca2575",
    },
    Fixture {
        file: "mecab-fixture-ko/unk.def",
        sha512: "260314c1de3d187ac3fa8ef5a35e354d318af678de099f7a54d3021694cc374b\
                 336e1f223bfbde0dd2969aba9ea2445fc5f549c265c394977aad942cec396374",
    },
    Fixture {
        file: "mecab-fixture-ko/words.csv",
        sha512: "969a6bcd215a61dd8bc7799f69a7b1a6aeb5975d8202bba4c982e39271e208df\
                 8fe1b893dac9c73f5d508901ad8959c66e3c874abc8219db2afa21dc6d0471ce",
    },
    Fixture {
        file: "mecab-fixture-partial/char.def",
        sha512: "45ae8dfcf2398f038d40f0910410dccdb74a2a1ba969bc718c976e675371c72b\
                 d0efb3666b041b201c3a38a27855cbd50a7bfd16685b0764ea87e2104c7b0db4",
    },
    Fixture {
        file: "mecab-fixture-partial/matrix.def",
        sha512: "d7d297def9b3076ff213fba877138bef306126f77e2a3d962ab3be0625e1b4bb\
                 d018207610edde710251e1e86e0d6aae5fd21d1b844edcaa509a809b7c957832",
    },
    Fixture {
        file: "mecab-fixture-partial/unk.def",
        sha512: "677ede432bd5d3d939221a8cc56f36a0dcbad64705b9b02661ef57d10c7ec668\
                 6e44201dc34f5cd028b87e9ae1c1a50ec6e552d0461f13df8ec35adcd609e013",
    },
    Fixture {
        file: "mecab-fixture-partial/words.csv",
        sha512: "7a0d6be9beaf23fec933e7142dc79c20508e9ef7414181996203be9f3ab5b1cd\
                 5cca26be772b523e0ff4d0bf0e895f2c33ec9c615957b9d85cfd271257614d20",
    },
];

/// The input sets under `analysis/goldens/data/inputs`.
pub const INPUT_SETS: [&str; 23] = [
    "en",
    "multi",
    "norm",
    "num",
    "money",
    "quantity",
    "edge",
    "surrogate",
    "abbreviations",
    "alphanumeric",
    "sentence-fixture",
    "detokenizer",
    "subwords",
    "zh",
    "bpe",
    "ja",
    "ko",
    "train",
    "tagged",
    "formats",
    "ner",
    "parse",
    "head-rules",
];

/// The arguments the driver passes to the generator after the fixed ones
/// (the inputs, models and output directories and the commit, which the
/// container's entrypoint supplies). `--dumps` makes it write a decision
/// point dump beside every golden of the [`DUMPED_ANNOTATORS`], so every
/// `generate` makes both.
pub const GENERATOR_ARGS: [&str; 1] = ["--dumps"];

/// The annotator directories the generator writes, which `generate`
/// replaces and `verify` compares. The data directory holds other things
/// (the inputs, the fixtures, the rate snapshot, a README) that neither
/// touches.
pub const ANNOTATORS: [&str; 25] = [
    "sentdetect",
    "tokenize",
    "postag",
    "lemmatize",
    "lemmatize-dictionary",
    "chunk",
    "langdetect",
    "normalize",
    "numbers",
    "money",
    "money-convert",
    "quantity",
    "detokenize",
    "subwords",
    "unigram",
    "bpe",
    "lattice",
    "train",
    "train-events",
    "formats",
    "evaluate",
    "masc",
    "ner",
    "parse",
    "head-rules",
];

/// The annotators with runs that ask a maxent model for their decisions:
/// each golden of theirs whose header names a model has a decision point
/// dump beside it. Their runs without a model (the simple, whitespace and
/// UAX #29 tokenizers and the newline detector) have none, nor do the
/// normalizers, the number scan, the money and quantity annotators,
/// the detokenizer, the WordPiece encoder, the unigram segmenter, the
/// BPE trainer and the lattice tokenizer, which ask no model.
pub const DUMPED_ANNOTATORS: [&str; 6] = ["sentdetect", "tokenize", "postag", "lemmatize", "chunk", "langdetect"];

/// The input set the normalizers run on. They run on no other set, and the
/// other annotators do not run on this one.
pub const NORMALIZER_INPUT_SET: &str = "norm";

/// The input set the number scan runs on. It runs on no other set, and the
/// other annotators do not run on this one.
pub const NUMBER_INPUT_SET: &str = "num";

/// The input set the money annotators run on. They run on no other set, and
/// the other annotators do not run on this one.
pub const MONEY_INPUT_SET: &str = "money";

/// The input set the quantity extractor runs on. It runs on no other set,
/// and the other annotators do not run on this one.
pub const QUANTITY_INPUT_SET: &str = "quantity";

/// The rate snapshot the money conversion reads: a directory beside the
/// inputs directory under `analysis/goldens/data`, which the container gets
/// at `/rates`, and the file in it.
pub const RATES_DIR: &str = "rates";
pub const RATES_FILE: &str = "ecb-snapshot.csv";

#[cfg(test)]
mod tests {
    use super::*;

    fn is_lower_hex(s: &str, len: usize) -> bool {
        s.len() == len && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    #[test]
    fn the_commit_is_forty_hex_digits() {
        assert!(is_lower_hex(OPENNLP_COMMIT, 40), "{OPENNLP_COMMIT}");
        assert!(is_lower_hex(ADDONS_COMMIT, 40), "{ADDONS_COMMIT}");
        assert!(IMAGE_TAG.contains(&format!("opennlp-{}", &OPENNLP_COMMIT[..12])));
        assert!(IMAGE_TAG.ends_with(&format!("addons-{}", &ADDONS_COMMIT[..12])));
    }

    #[test]
    fn the_base_image_is_pinned_by_a_sha256_digest() {
        let hex = BASE_IMAGE_DIGEST.strip_prefix("sha256:").expect("a sha256: digest");
        assert!(is_lower_hex(hex, 64), "{BASE_IMAGE_DIGEST}");
    }

    #[test]
    fn every_model_has_a_file_an_https_url_and_a_sha512() {
        for m in MODELS {
            assert!(is_lower_hex(m.sha512, 128), "{}: {}", m.file, m.sha512);
            assert!(m.url.starts_with("https://"), "{}", m.url);
            assert!(m.url.ends_with(&format!("/{}", m.file)), "{} does not end in {}", m.url, m.file);
            assert!(m.file.ends_with(".bin"));
        }
    }

    #[test]
    fn the_dockerfile_pins_the_same_base_image_and_commit() {
        let dockerfile = include_str!("../container/Dockerfile");
        assert!(dockerfile.contains(&format!("\nFROM {BASE_IMAGE}@{BASE_IMAGE_DIGEST}\n")));
        assert!(dockerfile.contains(&format!("\nARG OPENNLP_COMMIT={OPENNLP_COMMIT}\n")));
        assert!(dockerfile.contains(&format!("\nARG OPENNLP_REPO={OPENNLP_REPO}\n")));
        assert!(dockerfile.contains(&format!("\nARG ADDONS_COMMIT={ADDONS_COMMIT}\n")));
        assert!(dockerfile.contains(&format!("\nARG ADDONS_REPO={ADDONS_REPO}\n")));
    }

    #[test]
    fn every_fixture_has_a_file_and_a_sha512() {
        for f in FIXTURES {
            assert!(is_lower_hex(f.sha512, 128), "{}: {}", f.file, f.sha512);
            match f.file.split_once('/') {
                None => assert!(
                    f.file.ends_with(".bin")
                        || f.file.ends_with(".xml")
                        || f.file.ends_with(".txt")
                        || f.file.ends_with(".tagdict"),
                    "{}",
                    f.file
                ),
                Some((dir, file)) => {
                    assert!(dir.starts_with("mecab-") && !file.contains('/'), "{}", f.file);
                    assert!(file.ends_with(".csv") || file.ends_with(".def"), "{}", f.file);
                }
            }
        }
    }

    #[test]
    fn every_dictionary_has_an_https_url_and_a_sha512() {
        for d in DICTIONARIES {
            assert!(is_lower_hex(d.sha512, 128), "{}: {}", d.archive, d.sha512);
            assert!(d.url.starts_with("https://"), "{}", d.url);
            assert!(d.url.ends_with(&format!("/{}", d.archive)), "{} does not end in {}", d.url, d.archive);
            assert_eq!(d.archive, format!("{}.tar.gz", d.dir));
        }
    }

    /// The name of a fixture as `Runs` opens it: the directory of a
    /// dictionary's file, the file itself otherwise.
    fn opened(file: &str) -> &str {
        file.split_once('/').map_or(file, |(dir, _)| dir)
    }

    #[test]
    fn the_generator_opens_the_same_model_and_fixture_files() {
        let runs = include_str!("../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
        let dictionaries = DICTIONARIES.iter().map(|d| d.dir);
        let copies: Vec<String> = DICTIONARIES
            .iter()
            .filter(|d| d.encoding == DictionaryEncoding::EucJp)
            .map(|d| format!("{}{UTF8_COPY_SUFFIX}", d.dir))
            .collect();
        for file in MODELS
            .iter()
            .map(|m| m.file)
            .chain(FIXTURES.iter().map(|f| opened(f.file)))
            .chain(dictionaries)
            .chain(copies.iter().map(String::as_str))
        {
            assert!(runs.contains(&format!("\"{file}\"")), "Runs.java does not name {file}");
        }
    }

    #[test]
    fn the_generator_names_the_same_input_sets() {
        let runs = include_str!("../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
        for set in INPUT_SETS
            .iter()
            .filter(|s| ![NORMALIZER_INPUT_SET, NUMBER_INPUT_SET, MONEY_INPUT_SET, QUANTITY_INPUT_SET].contains(s))
        {
            assert!(runs.contains(&format!("= \"{set}\";")), "Runs.java does not name the input set {set}");
        }
    }

    #[test]
    fn the_generator_and_the_entrypoint_take_the_generator_args() {
        let main = include_str!("../java/src/main/java/ai/pipestream/turbo/goldens/Main.java");
        for arg in GENERATOR_ARGS {
            assert!(main.contains(&format!("\"{arg}\"")), "Main.java does not take {arg}");
        }
        let dockerfile = include_str!("../container/Dockerfile");
        assert!(
            dockerfile.contains("/inputs /models /fixtures /out \"$(cat /opennlp.commit)\" \"$@\""),
            "the entrypoint does not pass the mounts in the generator's order, or drops its arguments"
        );
    }

    #[test]
    fn the_dumped_annotators_and_the_normalizer_set_are_in_the_recipe() {
        assert!(DUMPED_ANNOTATORS.iter().all(|a| ANNOTATORS.contains(a)));
        assert!(INPUT_SETS.contains(&NORMALIZER_INPUT_SET));
        let normalizer = include_str!("../java/src/main/java/ai/pipestream/turbo/goldens/NormalizerGolden.java");
        assert!(normalizer.contains(&format!("INPUT_SET = \"{NORMALIZER_INPUT_SET}\"")));
    }

    #[test]
    fn the_number_scan_has_its_own_input_set_and_no_dumps() {
        assert!(INPUT_SETS.contains(&NUMBER_INPUT_SET));
        assert!(ANNOTATORS.contains(&"numbers"));
        assert!(!DUMPED_ANNOTATORS.contains(&"numbers"));
        let numbers = include_str!("../java/src/main/java/ai/pipestream/turbo/goldens/NumberGolden.java");
        assert!(numbers.contains(&format!("INPUT_SET = \"{NUMBER_INPUT_SET}\"")));
    }

    #[test]
    fn the_money_annotators_have_their_own_input_set_and_no_dumps() {
        assert!(INPUT_SETS.contains(&MONEY_INPUT_SET));
        assert!(ANNOTATORS.contains(&"money"));
        assert!(ANNOTATORS.contains(&"money-convert"));
        assert!(!DUMPED_ANNOTATORS.contains(&"money"));
        assert!(!DUMPED_ANNOTATORS.contains(&"money-convert"));
        let money = include_str!("../java/src/main/java/ai/pipestream/turbo/goldens/MoneyGolden.java");
        assert!(money.contains(&format!("INPUT_SET = \"{MONEY_INPUT_SET}\"")));
        assert!(money.contains(&format!("SNAPSHOT_DIR = \"{RATES_DIR}\"")));
        assert!(money.contains(&format!("SNAPSHOT_FILE = \"{RATES_FILE}\"")));
    }

    #[test]
    fn the_quantity_extractor_has_its_own_input_set_and_no_dumps() {
        assert!(INPUT_SETS.contains(&QUANTITY_INPUT_SET));
        assert!(ANNOTATORS.contains(&"quantity"));
        assert!(!DUMPED_ANNOTATORS.contains(&"quantity"));
        let quantity = include_str!("../java/src/main/java/ai/pipestream/turbo/goldens/QuantityGolden.java");
        assert!(quantity.contains(&format!("INPUT_SET = \"{QUANTITY_INPUT_SET}\"")));
    }

    #[test]
    fn the_rate_snapshot_is_present() {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data");
        assert!(data.join(RATES_DIR).join(RATES_FILE).is_file());
    }

    #[test]
    fn every_input_set_and_fixture_is_present() {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data");
        for set in INPUT_SETS {
            assert!(data.join("inputs").join(set).is_dir(), "{}", data.join(set).display());
        }
        for f in FIXTURES {
            assert!(data.join("fixtures").join(f.file).is_file(), "{}", f.file);
        }
    }
}
