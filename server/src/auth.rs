//! Constant-time bearer token authentication middleware (`#2346`), separating
//! read/infer principals from admin state-mutating principals.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use http_body::Body;
use tonic::Status;
use tonic::codegen::http::{HeaderMap, Request, Response};

/// Sorted gRPC full paths that mutate server state and require an admin token
/// when bearer authentication is enabled.
const ADMIN_METHODS: &[&str] = &[
    "/turbo.result.v1.AnalysisService/LoadBundle",
    "/turbo.result.v1.AnalysisService/UnloadBundle",
    "/turbo.search.v1.TurboSearchService/CreateIndex",
    "/turbo.search.v1.TurboSearchService/DropIndex",
    "/turbo.search.v1.TurboSearchService/PersistIndex",
    "/turbo.search.v1.TurboSearchService/SetAlias",
    "/turbo.training.v1.TurboCatalogService/InstallModel",
    "/turbo.training.v1.TurboCatalogService/RecordLicenceAcceptance",
    "/turbo.training.v1.TurboCatalogService/RemoveInstalledModel",
    "/turbo.training.v1.TurboTrainingService/DeleteStaticModel",
    "/turbo.training.v1.TurboTrainingService/TrainComponent",
    "/turbo.training.v1.TurboTrainingService/TrainStaticModel",
    "/turbo.vocabulary.v1.TurboVocabularyService/BuildSuggestLexicon",
    "/turbo.vocabulary.v1.TurboVocabularyService/DeleteVocabularyArtifact",
    "/turbo.vocabulary.v1.TurboVocabularyService/ImportCorpusStatistics",
    "/turbo.vocabulary.v1.TurboVocabularyService/ImportDictionary",
    "/turbo.vocabulary.v1.TurboVocabularyService/LearnVocabulary",
    "/turbo.vocabulary.v1.TurboVocabularyService/TrainVoronoiCentroids",
];

/// Hashed bearer token configuration (`#2346`). Raw token values are hashed
/// with SHA-256 at configuration time and never retained or formatted.
#[derive(Clone, PartialEq, Eq)]
pub struct AuthConfig {
    infer_digests: Vec<[u8; 32]>,
    admin_digests: Vec<[u8; 32]>,
}

impl fmt::Debug for AuthConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthConfig")
            .field("infer_tokens", &self.infer_digests.len())
            .field("admin_tokens", &self.admin_digests.len())
            .finish()
    }
}

impl AuthConfig {
    /// Builds an [`AuthConfig`] from read/infer bearer tokens and admin bearer
    /// tokens. Admin tokens are authorized for both read/infer and admin RPCs.
    /// Returns `Err` if both lists are empty or any token is blank.
    pub fn from_tokens<I, A, S1, S2>(infer_tokens: I, admin_tokens: A) -> Result<Self, String>
    where
        I: IntoIterator<Item = S1>,
        A: IntoIterator<Item = S2>,
        S1: AsRef<str>,
        S2: AsRef<str>,
    {
        let mut infer_digests = Vec::new();
        for tok in infer_tokens {
            let trimmed = tok.as_ref().trim();
            if trimmed.is_empty() {
                return Err("bearer token must not be empty".into());
            }
            infer_digests.push(sha256_bytes(trimmed.as_bytes()));
        }
        let mut admin_digests = Vec::new();
        for tok in admin_tokens {
            let trimmed = tok.as_ref().trim();
            if trimmed.is_empty() {
                return Err("admin bearer token must not be empty".into());
            }
            admin_digests.push(sha256_bytes(trimmed.as_bytes()));
        }
        if infer_digests.is_empty() && admin_digests.is_empty() {
            return Err("at least one bearer token or admin bearer token is required".into());
        }
        Ok(Self { infer_digests, admin_digests })
    }

    /// Verifies `headers` for `grpc_path`.
    pub fn check(&self, headers: &HeaderMap, grpc_path: &str) -> Result<(), Status> {
        let Some(token) = parse_bearer(headers) else {
            return Err(Status::unauthenticated("missing or malformed Authorization Bearer token"));
        };
        let candidate = sha256_bytes(token);
        let is_admin = matches_any(&candidate, &self.admin_digests);
        let is_infer = matches_any(&candidate, &self.infer_digests);
        if !is_admin && !is_infer {
            return Err(Status::unauthenticated("invalid bearer token"));
        }
        if is_admin_method(grpc_path) && !is_admin {
            return Err(Status::permission_denied("admin bearer token required for state-mutating RPC"));
        }
        Ok(())
    }
}

fn sha256_bytes(input: &[u8]) -> [u8; 32] {
    *turbo_sha256::sha256(input).as_bytes()
}

fn ct_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    std::hint::black_box(diff) == 0
}

fn matches_any(candidate: &[u8; 32], allowed: &[[u8; 32]]) -> bool {
    let mut hit = 0u8;
    for digest in allowed {
        hit |= u8::from(ct_eq_32(candidate, digest));
    }
    std::hint::black_box(hit) != 0
}

fn parse_bearer(headers: &HeaderMap) -> Option<&[u8]> {
    let raw = headers.get("authorization")?.as_bytes();
    if raw.len() <= 7 {
        return None;
    }
    let (scheme, rest) = raw.split_at(7);
    if !scheme.eq_ignore_ascii_case(b"bearer ") {
        return None;
    }
    let trimmed = trim_ascii(rest);
    (!trimmed.is_empty()).then_some(trimmed)
}

fn trim_ascii(mut b: &[u8]) -> &[u8] {
    while let Some((&first, tail)) = b.split_first() {
        if first.is_ascii_whitespace() {
            b = tail;
        } else {
            break;
        }
    }
    while let Some((&last, head)) = b.split_last() {
        if last.is_ascii_whitespace() {
            b = head;
        } else {
            break;
        }
    }
    b
}

pub(crate) fn is_admin_method(path: &str) -> bool {
    ADMIN_METHODS.binary_search(&path).is_ok()
}

/// Tower layer enforcing [`AuthConfig`] on incoming gRPC requests.
#[derive(Clone, Default)]
pub(crate) struct AuthLayer {
    config: Option<Arc<AuthConfig>>,
}

impl AuthLayer {
    pub(crate) fn new(config: Option<AuthConfig>) -> Self {
        Self { config: config.map(Arc::new) }
    }
}

impl<S> tower::Layer<S> for AuthLayer {
    type Service = AuthService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthService { inner, config: self.config.clone() }
    }
}

#[derive(Clone)]
pub(crate) struct AuthService<S> {
    inner: S,
    config: Option<Arc<AuthConfig>>,
}

impl<S, ReqBody, ResBody> tower::Service<Request<ReqBody>> for AuthService<S>
where
    S: tower::Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ResBody: Body + Default + Send + 'static,
{
    type Response = Response<ResBody>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        if let Some(cfg) = &self.config
            && let Err(status) = cfg.check(req.headers(), req.uri().path())
        {
            let res: Response<ResBody> = status.into_http();
            return Box::pin(async move { Ok(res) });
        }
        let fut = self.inner.call(req);
        Box::pin(fut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admin_methods_are_sorted_for_binary_search() {
        assert!(ADMIN_METHODS.windows(2).all(|w| w[0] < w[1]));
    }
}
