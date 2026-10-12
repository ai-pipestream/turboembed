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
pub const MODELS: [Model; 9] = [
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
        file: "langdetect-183.bin",
        url: "https://downloads.apache.org/opennlp/models/langdetect/1.8.3/langdetect-183.bin",
        sha512: "9dc847055e622d9de503a5bf444ccf1428c3578c9698a73b736d8f8e337db018\
                 8ade90ac8cc9196d1d472f58589d578785085b554b1bde856afc5762e9396285",
    },
];

/// A checked in file the generator loads from the fixtures directory,
/// `analysis/goldens/data/fixtures`: a model trained for the tests, a
/// dictionary or a WordPiece vocabulary, and the SHA-512 of its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fixture {
    pub file: &'static str,
    pub sha512: &'static str,
}

/// The fixtures, which are all the files of the fixtures directory. The
/// file names are the ones the generator's `Runs` class opens there.
pub const FIXTURES: [Fixture; 10] = [
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
        file: "tokenizer-abbreviations.bin",
        sha512: "361484f0a67e129fe6f19908704d23e83c8a2bf4551a5f7eec3b40b9082a66b7\
                 026d429233c11639b35bccf042d347d79222cfed62685563f2a926cb8a2460d4",
    },
    Fixture {
        file: "english-dictionary.xml",
        sha512: "e824b14f66036b2703cbbd3af6a893f7f07cf282c9e8021ae7be38f10da305a8\
                 7aef48b129923f3973c78f495c2ed073adcad68c90c2f7d94bc6d32870839164",
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
];

/// The input sets under `analysis/goldens/data/inputs`.
pub const INPUT_SETS: [&str; 12] = [
    "en",
    "multi",
    "norm",
    "num",
    "money",
    "edge",
    "surrogate",
    "abbreviations",
    "alphanumeric",
    "sentence-fixture",
    "detokenizer",
    "subwords",
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
pub const ANNOTATORS: [&str; 9] = [
    "sentdetect",
    "tokenize",
    "langdetect",
    "normalize",
    "numbers",
    "money",
    "money-convert",
    "detokenize",
    "subwords",
];

/// The annotators with runs that ask a maxent model for their decisions:
/// each golden of theirs whose header names a model has a decision point
/// dump beside it. Their runs without a model (the simple, whitespace and
/// UAX #29 tokenizers and the newline detector) have none, nor do the
/// normalizers, the number scan, the money annotators, the
/// detokenizer and the WordPiece encoder, which ask no model.
pub const DUMPED_ANNOTATORS: [&str; 3] = ["sentdetect", "tokenize", "langdetect"];

/// The input set the normalizers run on. They run on no other set, and the
/// other annotators do not run on this one.
pub const NORMALIZER_INPUT_SET: &str = "norm";

/// The input set the number scan runs on. It runs on no other set, and the
/// other annotators do not run on this one.
pub const NUMBER_INPUT_SET: &str = "num";

/// The input set the money annotators run on. They run on no other set, and
/// the other annotators do not run on this one.
pub const MONEY_INPUT_SET: &str = "money";

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
            assert!(f.file.ends_with(".bin") || f.file.ends_with(".xml") || f.file.ends_with(".txt"), "{}", f.file);
        }
    }

    #[test]
    fn the_generator_opens_the_same_model_and_fixture_files() {
        let runs = include_str!("../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
        for file in MODELS.iter().map(|m| m.file).chain(FIXTURES.iter().map(|f| f.file)) {
            assert!(runs.contains(&format!("\"{file}\"")), "Runs.java does not name {file}");
        }
    }

    #[test]
    fn the_generator_names_the_same_input_sets() {
        let runs = include_str!("../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
        for set in INPUT_SETS.iter().filter(|s| ![NORMALIZER_INPUT_SET, NUMBER_INPUT_SET, MONEY_INPUT_SET].contains(s))
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
