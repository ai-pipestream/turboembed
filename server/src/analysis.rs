//! The AnalysisService rpcs (proto/turbo_analysis.proto,
//! docs/grpc-analysis.md): the capability listing, LoadBundle and
//! UnloadBundle, and the four result round trips. Load and unload are
//! admin calls: the server serves them only when started with the
//! admin flag (default off), and a request's registration is its own
//! Arc identity from registration through loading and rollback.

use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::api::{Failure, TURBO_E_INTERNAL, TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED};
use crate::capabilities::{bundle_capabilities, load_config, result_limits, server_limits};
use crate::kserve::Reply;
use crate::proto::result::analysis_service_server::AnalysisService;
use crate::proto::result::{
    AnalyzeStreamRequest, CapabilitiesRequest, CapabilitiesResponse, ChunkLevelsResult, DocumentGraphRequest,
    DocumentGraphResponse, ExtractionResult, LoadBundleRequest, LoadBundleResponse, RecordsResult, RerankScoresResult,
    SparseTermsResult, TokenVectorsResult, UnloadBundleRequest, UnloadBundleResponse,
};
use crate::results::Wire;
use crate::results::{
    extraction::Extraction, layers::ChunkLevels, records::RecordSet, rerank::RerankScores, tokens::TokenVectors,
};
use crate::state::{Loaded, State};
use crate::status::to_status;
use crate::stream;
use turbo_analysis::View;
use turbo_analysis_layers::SparseTerms;

pub(crate) struct Analysis {
    pub(crate) state: Arc<State>,
    /// Whether the admin calls, LoadBundle and UnloadBundle, are served
    /// (the server's --admin flag, off by default).
    pub(crate) admin: bool,
}

fn plain(f: Failure) -> Status {
    to_status(&f, &[])
}

/// The refusal every LoadBundle and UnloadBundle gets when the server
/// was not started with the admin flag.
fn not_admin(what: &str) -> Status {
    plain(Failure::new(
        TURBO_E_UNSUPPORTED,
        0,
        format!("{what} is admin-only: start the server with --admin to serve it"),
    ))
}

/// Validates a result message the way the layer it mirrors checks
/// itself, and answers with the message unchanged.
fn round_trip<M, W>(req: Request<M>) -> Reply<M>
where
    W: Wire<Msg = M>,
{
    W::from_msg(req.get_ref()).map_err(plain)?;
    Ok(Response::new(req.into_inner()))
}

#[tonic::async_trait]
impl AnalysisService for Analysis {
    type AnalyzeStreamStream = stream::handler::AnswerStream;

    async fn capabilities(&self, _: Request<CapabilitiesRequest>) -> Reply<CapabilitiesResponse> {
        let mut bundles = Vec::new();
        for m in self.state.listing() {
            if let Some(loaded) = m.loaded.get() {
                bundles.push(bundle_capabilities(&m.name, loaded).map_err(plain)?);
            }
        }
        Ok(Response::new(CapabilitiesResponse {
            bundles,
            limits: Some(result_limits()),
            server: Some(server_limits()),
        }))
    }

    async fn load_bundle(&self, req: Request<LoadBundleRequest>) -> Reply<LoadBundleResponse> {
        if !self.admin {
            return Err(not_admin("LoadBundle"));
        }
        let req = req.into_inner();
        let (name, config) = load_config(&req).map_err(plain)?;
        // The registration this request owns, by identity, through
        // loading, the answer, and rollback: a same-named replacement is
        // a different Arc and is never touched.
        let registered = self.state.add(name, config).map_err(plain)?;
        let state = self.state.clone();
        let loading = registered.clone();
        let loaded = tokio::task::spawn_blocking(move || state.load_one(&loading)).await;
        let loaded =
            loaded.map_err(|e| plain(Failure::new(TURBO_E_INTERNAL, 0, format!("the load's task failed: {e}"))))?;
        if let Err(e) = loaded {
            let _ = self.state.remove_if(&registered);
            return Err(Status::internal(e));
        }
        let l: Arc<Loaded> = registered.loaded.get().cloned().ok_or_else(|| {
            plain(Failure::new(TURBO_E_INVALID_STATE, 0, format!("model {} is not loaded yet", registered.name)))
        })?;
        let capabilities = bundle_capabilities(&registered.name, &l).map_err(plain)?;
        Ok(Response::new(LoadBundleResponse { capabilities: Some(capabilities) }))
    }

    async fn unload_bundle(&self, req: Request<UnloadBundleRequest>) -> Reply<UnloadBundleResponse> {
        if !self.admin {
            return Err(not_admin("UnloadBundle"));
        }
        let req = req.into_inner();
        self.state.remove(&req.name).map_err(plain)?;
        Ok(Response::new(UnloadBundleResponse {}))
    }

    async fn round_trip_sparse_terms(&self, req: Request<SparseTermsResult>) -> Reply<SparseTermsResult> {
        round_trip::<_, SparseTerms<View>>(req)
    }

    async fn round_trip_token_vectors(&self, req: Request<TokenVectorsResult>) -> Reply<TokenVectorsResult> {
        round_trip::<_, TokenVectors>(req)
    }

    async fn round_trip_rerank_scores(&self, req: Request<RerankScoresResult>) -> Reply<RerankScoresResult> {
        round_trip::<_, RerankScores>(req)
    }

    async fn round_trip_extraction(&self, req: Request<ExtractionResult>) -> Reply<ExtractionResult> {
        round_trip::<_, Extraction>(req)
    }

    async fn round_trip_records(&self, req: Request<RecordsResult>) -> Reply<RecordsResult> {
        round_trip::<_, RecordSet>(req)
    }

    async fn round_trip_chunk_levels(&self, req: Request<ChunkLevelsResult>) -> Reply<ChunkLevelsResult> {
        round_trip::<_, ChunkLevels>(req)
    }

    async fn round_trip_document_graph(&self, req: Request<DocumentGraphRequest>) -> Reply<DocumentGraphResponse> {
        // Import and re-export can walk a whole document; keep the
        // async runtime free of it.
        let req = req.into_inner();
        let out = tokio::task::spawn_blocking(move || crate::results::document::round_trip_document(&req))
            .await
            .map_err(|e| plain(Failure::new(TURBO_E_INTERNAL, 0, format!("the round trip's task failed: {e}"))))?;
        Ok(Response::new(out.map_err(plain)?))
    }

    async fn analyze_stream(
        &self,
        req: Request<tonic::Streaming<AnalyzeStreamRequest>>,
    ) -> Reply<Self::AnalyzeStreamStream> {
        crate::stream::handler::start(self.state.clone(), req.into_inner()).await
    }
}
