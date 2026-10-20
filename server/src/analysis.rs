//! The AnalysisService rpcs (proto/turbo_analysis.proto,
//! docs/grpc-analysis.md): the capability listing, LoadBundle and
//! UnloadBundle, and the four result round trips.

use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::api::{Failure, TURBO_E_INTERNAL};
use crate::capabilities::{bundle_capabilities, load_config, result_limits};
use crate::kserve::Reply;
use crate::proto::result::analysis_service_server::AnalysisService;
use crate::proto::result::{
    CapabilitiesRequest, CapabilitiesResponse, ExtractionResult, LoadBundleRequest, LoadBundleResponse,
    RerankScoresResult, SparseTermsResult, TokenVectorsResult, UnloadBundleRequest, UnloadBundleResponse,
};
use crate::results::Wire;
use crate::results::{extraction::Extraction, rerank::RerankScores, tokens::TokenVectors};
use crate::state::State;
use crate::status::to_status;
use turbo_analysis::View;
use turbo_analysis_layers::SparseTerms;

pub(crate) struct Analysis {
    pub(crate) state: Arc<State>,
}

fn plain(f: Failure) -> Status {
    to_status(&f, &[])
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
    async fn capabilities(&self, _: Request<CapabilitiesRequest>) -> Reply<CapabilitiesResponse> {
        let mut bundles = Vec::new();
        for m in self.state.listing() {
            if let Some(loaded) = m.loaded.get() {
                bundles.push(bundle_capabilities(&m.name, loaded).map_err(plain)?);
            }
        }
        Ok(Response::new(CapabilitiesResponse { bundles, limits: Some(result_limits()) }))
    }

    async fn load_bundle(&self, req: Request<LoadBundleRequest>) -> Reply<LoadBundleResponse> {
        let req = req.into_inner();
        let (name, config) = load_config(&req).map_err(plain)?;
        self.state.add(name.clone(), config).map_err(plain)?;
        // Only this registration loads, so one LoadBundle's failure
        // cannot fail another's; a failed load leaves nothing served,
        // as a startup refusal does.
        let state = self.state.clone();
        let loading = name.clone();
        let loaded = tokio::task::spawn_blocking(move || state.load_one(&loading)).await;
        let loaded =
            loaded.map_err(|e| plain(Failure::new(TURBO_E_INTERNAL, 0, format!("the load's task failed: {e}"))))?;
        if let Err(e) = loaded {
            // Roll back only this registration, and only if it never
            // loaded: a concurrent unload may have taken the name first.
            if let Ok((_, None)) = self.state.peek(&name) {
                let _ = self.state.remove(&name);
            }
            return Err(Status::internal(e));
        }
        let (_, l) = self.state.ready(&name, "").map_err(plain)?;
        let capabilities = bundle_capabilities(&name, &l).map_err(plain)?;
        Ok(Response::new(LoadBundleResponse { capabilities: Some(capabilities) }))
    }

    async fn unload_bundle(&self, req: Request<UnloadBundleRequest>) -> Reply<UnloadBundleResponse> {
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
}
