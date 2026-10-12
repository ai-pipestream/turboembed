//! What the goldens are made from, as constants: the container's base
//! image, the OpenNLP commit, the models and the input sets. The container
//! recipe (`container/Dockerfile`) repeats the base image and the commit;
//! a test checks that both say the same.

/// The base image, pinned by the digest of its multi-platform index.
pub const BASE_IMAGE: &str = "eclipse-temurin:21-jdk";
pub const BASE_IMAGE_DIGEST: &str = "sha256:3e3c176ffed168beb42c607be9bc1639b466cf00261a0fb04425562c9d0c5c2b";

/// The OpenNLP fork and the commit on its `kristian-3.x-features` branch the
/// generator runs.
pub const OPENNLP_REPO: &str = "https://github.com/ai-pipestream/opennlp";
pub const OPENNLP_COMMIT: &str = "cdf33c653d5d61d5af7d37ffee189a83bd8392ea";

/// The tag `turbo-goldens image` gives the image it builds. It names the
/// OpenNLP commit, so an image built from another commit has another tag.
pub const IMAGE_TAG: &str = "turbo-goldens-java:opennlp-cdf33c653d5d";

/// A model file the generator loads, where it is published and the SHA-512
/// published beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub file: &'static str,
    pub url: &'static str,
    pub sha512: &'static str,
}

/// The models, in the order the generator loads them. The file names are
/// the ones the generator's main class opens in the models directory.
pub const MODELS: [Model; 3] = [
    Model {
        file: "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin",
        sha512: "fed5d53ea87bf66afeb08c084f89ca9b95a96de0920dca40487b2ab9972ee6b7\
                 ef211e477db480b7253431de079fd66dfd14329cd7d19ad1f565155eb0cf5191",
    },
    Model {
        file: "opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin",
        url: "https://downloads.apache.org/opennlp/models/ud-models-1.3/opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin",
        sha512: "147d1cdcb89ca5d9bd0ead8fddf2cc14ae543a9151803fe56b3c1fa152b98660\
                 5e34fb6dda4819c68aa22fc9bffa6bfab14f916b7d0bf4892d452686af34778a",
    },
    Model {
        file: "langdetect-183.bin",
        url: "https://downloads.apache.org/opennlp/models/langdetect/1.8.3/langdetect-183.bin",
        sha512: "9dc847055e622d9de503a5bf444ccf1428c3578c9698a73b736d8f8e337db018\
                 8ade90ac8cc9196d1d472f58589d578785085b554b1bde856afc5762e9396285",
    },
];

/// The input sets under `analysis/goldens/data/inputs`.
pub const INPUT_SETS: [&str; 3] = ["en", "multi", "norm"];

/// The annotator directories the generator writes, which `generate`
/// replaces and `verify` compares. The data directory holds other things
/// (the inputs, a README) that neither touches.
pub const ANNOTATORS: [&str; 4] = ["sentdetect", "tokenize", "langdetect", "normalize"];

#[cfg(test)]
mod tests {
    use super::*;

    fn is_lower_hex(s: &str, len: usize) -> bool {
        s.len() == len && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    #[test]
    fn the_commit_is_forty_hex_digits() {
        assert!(is_lower_hex(OPENNLP_COMMIT, 40), "{OPENNLP_COMMIT}");
        assert!(IMAGE_TAG.ends_with(&OPENNLP_COMMIT[..12]));
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
    }

    #[test]
    fn the_generator_opens_the_same_model_files() {
        let main = include_str!("../java/src/main/java/ai/pipestream/turbo/goldens/Main.java");
        for m in MODELS {
            assert!(main.contains(&format!("\"{}\"", m.file)), "Main.java does not name {}", m.file);
        }
    }

    #[test]
    fn every_input_set_is_present() {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/inputs");
        for set in INPUT_SETS {
            assert!(data.join(set).is_dir(), "{}", data.join(set).display());
        }
    }
}
