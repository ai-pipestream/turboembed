//! The standard gRPC health checking service (`grpc.health.v1.Health`):
//! `Check` and `Watch` map `SERVING` to the same state as `ServerReady`
//! (or `ModelReady` when `service` names a served bundle) and report
//! `NOT_SERVING` before load and during drain.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND};
use crate::kserve::Reply;
use crate::proto::health::health_check_response::ServingStatus;
use crate::proto::health::health_server::Health;
use crate::proto::health::{HealthCheckRequest, HealthCheckResponse};
use crate::state::State;
use crate::status::to_status;

pub(crate) struct HealthService {
    pub(crate) state: Arc<State>,
}

fn service_status(state: &State, service: &str) -> Option<ServingStatus> {
    match service {
        "" | "inference.GRPCInferenceService" | "turbo.result.AnalysisService" | "grpc.health.v1.Health" => {
            if state.is_ready() { Some(ServingStatus::Serving) } else { Some(ServingStatus::NotServing) }
        }
        name => {
            let models = state.models.read().unwrap_or_else(|e| e.into_inner());
            let m = models.iter().find(|m| m.name == name)?;
            if !state.is_draining() && m.loaded.get().is_some() {
                Some(ServingStatus::Serving)
            } else {
                Some(ServingStatus::NotServing)
            }
        }
    }
}

#[tonic::async_trait]
impl Health for HealthService {
    type WatchStream = ReceiverStream<std::result::Result<HealthCheckResponse, Status>>;

    async fn check(&self, req: Request<HealthCheckRequest>) -> Reply<HealthCheckResponse> {
        let service = req.into_inner().service;
        let status = service_status(&self.state, &service).ok_or_else(|| {
            to_status(
                &Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, format!("no service or model {service} is served here")),
                &[],
            )
        })?;
        Ok(Response::new(HealthCheckResponse { status: status as i32 }))
    }

    async fn watch(&self, req: Request<HealthCheckRequest>) -> Reply<Self::WatchStream> {
        let service = req.into_inner().service;
        let state = self.state.clone();
        let mut rx = state.subscribe_health();
        let (tx, out) = mpsc::channel(8);
        tokio::spawn(async move {
            let mut last = None;
            loop {
                let status = service_status(&state, &service).unwrap_or(ServingStatus::ServiceUnknown);
                if last != Some(status) {
                    last = Some(status);
                    if tx.send(Ok(HealthCheckResponse { status: status as i32 })).await.is_err() {
                        break;
                    }
                }
                if state.is_stopping() {
                    break;
                }
                if rx.changed().await.is_err() {
                    break;
                }
            }
        });
        Ok(Response::new(ReceiverStream::new(out)))
    }
}
