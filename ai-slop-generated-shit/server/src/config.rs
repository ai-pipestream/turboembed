//! Server configuration: which provider libraries to load and which models
//! to serve, from a JSON file or repeated `--model` flags.

use std::path::PathBuf;

use serde::Deserialize;

use crate::error::{Result, ServeError};

/// One model to serve.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelSpec {
    /// Name clients use (`/v2/models/{name}`, `"model"` in the OpenAI
    /// routes). Defaults to the last path segment of the bundle's
    /// `model_id`.
    #[serde(default)]
    pub name: Option<String>,
    /// Bundle directory.
    pub bundle: PathBuf,
    /// Provider id the model runs on (`cuda`, `openvino`, `metal`, `ggml`,
    /// `hailo`, `static`, `mock`).
    pub provider: String,
    /// Device ordinal within the provider.
    #[serde(default)]
    pub ordinal: u32,
    /// Session buckets as `batch x seq` pairs. A request is served by the
    /// smallest bucket that fits it; a request longer than the longest
    /// bucket is rejected with that limit. Default: batches 1, 8 and the
    /// model's `max_batch` at the model's `max_seq`.
    #[serde(default)]
    pub buckets: Vec<Bucket>,
    /// Sessions per bucket (concurrent requests per bucket).
    #[serde(default = "one")]
    pub sessions: u32,
    /// Concurrent generations (generative models).
    #[serde(default = "one")]
    pub generations: u32,
}

fn one() -> u32 {
    1
}

/// A session shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Bucket {
    /// Rows one run of a session of this shape holds.
    pub batch: u32,
    /// Tokens one row of a session of this shape holds.
    pub seq: u32,
}

/// The server's configuration.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Config {
    /// Provider libraries to load (`libturbo_provider_*.so`).
    #[serde(default)]
    pub provider_libs: Vec<PathBuf>,
    /// Models to serve.
    #[serde(default)]
    pub models: Vec<ModelSpec>,
    /// A directory of static pages served at `/` (a demo front end); none by default.
    #[serde(default)]
    pub pages: Option<PathBuf>,
}

impl Config {
    /// Parse a `--model` flag: `key=value` pairs separated by commas, keys
    /// `name`, `bundle`, `provider`, `ordinal`, `buckets` (`1x128;8x128`),
    /// `sessions`, `generations`.
    pub fn parse_model_flag(text: &str) -> Result<ModelSpec> {
        let mut name = None;
        let mut bundle = None;
        let mut provider = None;
        let mut ordinal = 0;
        let mut buckets = Vec::new();
        let mut sessions = 1;
        let mut generations = 1;
        for pair in text.split(',').filter(|p| !p.trim().is_empty()) {
            let (k, v) = pair
                .split_once('=')
                .ok_or_else(|| ServeError::bad_request(format!("--model: `{pair}` is not key=value")))?;
            let v = v.trim();
            match k.trim() {
                "name" => name = Some(v.to_string()),
                "bundle" => bundle = Some(PathBuf::from(v)),
                "provider" => provider = Some(v.to_string()),
                "ordinal" => ordinal = parse_u32("ordinal", v)?,
                "sessions" => sessions = parse_u32("sessions", v)?,
                "generations" => generations = parse_u32("generations", v)?,
                "buckets" => {
                    for b in v.split(';').filter(|b| !b.trim().is_empty()) {
                        let (batch, seq) = b.trim().split_once('x').ok_or_else(|| {
                            ServeError::bad_request(format!("--model buckets: `{b}` is not BATCHxSEQ"))
                        })?;
                        buckets.push(Bucket {
                            batch: parse_u32("bucket batch", batch)?,
                            seq: parse_u32("bucket seq", seq)?,
                        });
                    }
                }
                other => return Err(ServeError::bad_request(format!("--model: unknown key `{other}`"))),
            }
        }
        let spec = ModelSpec {
            name,
            bundle: bundle.ok_or_else(|| ServeError::bad_request("--model: `bundle=` is required"))?,
            provider: provider.ok_or_else(|| ServeError::bad_request("--model: `provider=` is required"))?,
            ordinal,
            buckets,
            sessions,
            generations,
        };
        spec.validate()?;
        Ok(spec)
    }
}

impl ModelSpec {
    /// A specification from a repository load request; buckets are
    /// `BATCHxSEQ` strings, zero sessions or generations mean the default.
    pub fn from_request(
        name: Option<String>,
        bundle: &str,
        provider: &str,
        ordinal: u32,
        buckets: &[String],
        sessions: u32,
        generations: u32,
    ) -> Result<ModelSpec> {
        if bundle.is_empty() {
            return Err(ServeError::bad_request("load: `bundle` is required"));
        }
        if provider.is_empty() {
            return Err(ServeError::bad_request("load: `provider` is required"));
        }
        let mut parsed = Vec::new();
        for b in buckets {
            let (batch, seq) = b
                .trim()
                .split_once('x')
                .ok_or_else(|| ServeError::bad_request(format!("load: bucket `{b}` is not BATCHxSEQ")))?;
            parsed.push(Bucket { batch: parse_u32("bucket batch", batch)?, seq: parse_u32("bucket seq", seq)? });
        }
        let spec = ModelSpec {
            name,
            bundle: PathBuf::from(bundle),
            provider: provider.to_string(),
            ordinal,
            buckets: parsed,
            sessions: if sessions == 0 { 1 } else { sessions },
            generations: if generations == 0 { 1 } else { generations },
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Reject a specification the server cannot serve: no sessions, no
    /// generations, or a bucket too small to hold a row.
    pub fn validate(&self) -> Result<()> {
        if self.sessions == 0 || self.generations == 0 {
            return Err(ServeError::bad_request(format!(
                "model {}: sessions and generations must be at least 1",
                self.bundle.display()
            )));
        }
        for b in &self.buckets {
            if b.batch == 0 || b.seq < 2 {
                return Err(ServeError::bad_request(format!(
                    "model {}: bucket {}x{} needs batch >= 1 and seq >= 2",
                    self.bundle.display(),
                    b.batch,
                    b.seq
                )));
            }
        }
        Ok(())
    }
}

fn parse_u32(what: &str, v: &str) -> Result<u32> {
    v.parse::<u32>().map_err(|e| ServeError::bad_request(format!("--model {what} `{v}`: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(flag: &str) -> ModelSpec {
        Config::parse_model_flag(flag).unwrap_or_else(|e| panic!("--model {flag}: {e}"))
    }

    fn reject(flag: &str) -> ServeError {
        match Config::parse_model_flag(flag) {
            Ok(spec) => panic!("--model {flag} was accepted as {spec:?}"),
            Err(e) => e,
        }
    }

    #[test]
    fn a_flag_with_only_the_required_keys_takes_the_documented_defaults() {
        let spec = parse("bundle=/bundles/minilm,provider=cuda");
        assert_eq!(spec.bundle, PathBuf::from("/bundles/minilm"), "the bundle path");
        assert_eq!(spec.provider, "cuda", "the provider id");
        assert_eq!(spec.name, None, "no name means the bundle's model_id decides it");
        assert_eq!(spec.ordinal, 0, "the default ordinal");
        assert_eq!(spec.buckets, Vec::new(), "no buckets means the model's own limits decide them");
        assert_eq!(spec.sessions, 1, "the default sessions per bucket");
        assert_eq!(spec.generations, 1, "the default concurrent generations");
    }

    #[test]
    fn every_key_is_parsed() {
        let spec = parse(
            "name=minilm,bundle=/bundles/minilm,provider=cuda,ordinal=2,buckets=1x128;8x256,sessions=3,generations=4",
        );
        assert_eq!(spec.name.as_deref(), Some("minilm"), "the name");
        assert_eq!(spec.bundle, PathBuf::from("/bundles/minilm"), "the bundle path");
        assert_eq!(spec.provider, "cuda", "the provider id");
        assert_eq!(spec.ordinal, 2, "the device ordinal");
        assert_eq!(
            spec.buckets,
            vec![Bucket { batch: 1, seq: 128 }, Bucket { batch: 8, seq: 256 }],
            "the buckets, in the order given"
        );
        assert_eq!(spec.sessions, 3, "sessions per bucket");
        assert_eq!(spec.generations, 4, "concurrent generations");
    }

    #[test]
    fn whitespace_around_keys_and_values_is_ignored() {
        let spec = parse(" name = minilm , bundle = /bundles/minilm , provider = cuda , buckets = 1x128 ; 8x256 ");
        assert_eq!(spec.name.as_deref(), Some("minilm"), "the name");
        assert_eq!(spec.provider, "cuda", "the provider id");
        assert_eq!(spec.buckets, vec![Bucket { batch: 1, seq: 128 }, Bucket { batch: 8, seq: 256 }], "the buckets");
    }

    #[test]
    fn a_missing_required_key_names_the_key() {
        assert!(reject("provider=cuda").message.contains("`bundle=`"), "a flag with no bundle must name `bundle=`");
        assert!(
            reject("bundle=/bundles/minilm").message.contains("`provider=`"),
            "a flag with no provider must name `provider=`"
        );
    }

    #[test]
    fn an_unknown_key_is_rejected_by_name() {
        let e = reject("bundle=/b,provider=cuda,batch=8");
        assert!(e.message.contains("`batch`"), "the error must name the key it did not know: {e}");
    }

    #[test]
    fn a_pair_without_an_equals_sign_is_rejected() {
        let e = reject("bundle=/b,provider=cuda,fast");
        assert!(e.message.contains("`fast`"), "the error must quote what was not a key=value pair: {e}");
    }

    #[test]
    fn a_non_numeric_number_is_rejected_naming_the_key() {
        for (flag, key) in [
            ("bundle=/b,provider=cuda,ordinal=first", "ordinal"),
            ("bundle=/b,provider=cuda,sessions=many", "sessions"),
            ("bundle=/b,provider=cuda,generations=two", "generations"),
            ("bundle=/b,provider=cuda,buckets=onex128", "bucket batch"),
            ("bundle=/b,provider=cuda,buckets=1xlong", "bucket seq"),
        ] {
            let e = reject(flag);
            assert!(e.message.contains(key), "the error for `{flag}` must name `{key}`: {e}");
        }
    }

    #[test]
    fn a_bucket_that_is_not_batch_by_sequence_is_rejected() {
        let e = reject("bundle=/b,provider=cuda,buckets=128");
        assert!(e.message.contains("BATCHxSEQ"), "the error must say the shape a bucket takes: {e}");
    }

    #[test]
    fn a_bucket_that_holds_nothing_is_rejected() {
        for flag in ["bundle=/b,provider=cuda,buckets=0x16", "bundle=/b,provider=cuda,buckets=1x1"] {
            let e = reject(flag);
            assert!(e.message.contains("batch >= 1 and seq >= 2"), "the error for `{flag}` must say the floor: {e}");
        }
    }

    #[test]
    fn zero_sessions_or_generations_are_rejected() {
        for flag in ["bundle=/b,provider=cuda,sessions=0", "bundle=/b,provider=cuda,generations=0"] {
            let e = reject(flag);
            assert!(e.message.contains("at least 1"), "the error for `{flag}` must say the floor: {e}");
        }
    }

    #[test]
    fn empty_pairs_and_an_empty_bucket_list_are_skipped() {
        let spec = parse("bundle=/b,,provider=cuda,buckets=1x16;;");
        assert_eq!(spec.buckets, vec![Bucket { batch: 1, seq: 16 }], "a trailing separator adds no bucket");
    }

    #[test]
    fn a_json_configuration_takes_the_same_defaults_as_the_flag() {
        let config: Config = serde_json::from_str(r#"{"models": [{"bundle": "/bundles/minilm", "provider": "cuda"}]}"#)
            .expect("a minimal JSON configuration");
        assert_eq!(config.provider_libs, Vec::<PathBuf>::new(), "no provider libraries");
        assert_eq!(config.models.len(), 1, "one model");
        assert_eq!(config.models[0].sessions, 1, "sessions defaults to 1 in JSON too");
        assert_eq!(config.models[0].generations, 1, "generations defaults to 1 in JSON too");
        assert_eq!(config.models[0].ordinal, 0, "ordinal defaults to 0 in JSON too");
    }
}
