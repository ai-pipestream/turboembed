//! Tonic `TurboSearchService` implementation (`#1221`, `#2609`).

use std::collections::{BTreeMap, BTreeSet};
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use tokio::sync::{broadcast, mpsc};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::metadata::MetadataMap;
use tonic::{Code, Request, Response, Status, Streaming};

use crate::api::Failure;
use crate::deadline::{Deadline, deadline_status};
use crate::priority::Priority;
use crate::proto::search::turbo_search_service_server::TurboSearchService;
use crate::proto::search::*;
use crate::search::exec::{run_index_documents, run_reindex, run_search};
use crate::search::ingest::analyze_chunk_terms;
use crate::search::registry::{
    ALREADY_EXISTS_FIELD, CollectionState, SearchRegistry, already_exists, bad_arg, not_admin, not_found,
    providers_list,
};
use crate::search::snapshot::{load_index_from_dir, persist_index_to_dir};
use crate::state::State;
use crate::status::{describe, to_status};

const MAX_STREAM_DOCUMENTS: usize = 4_096;
/// What a watch is told when the server stops under it.
const STOPPING: &str = "the server is stopping";

pub struct SearchService {
    pub(crate) state: Arc<State>,
    pub(crate) registry: Arc<SearchRegistry>,
    pub(crate) admin: bool,
}

fn reply<T>(r: crate::api::Result<T>) -> std::result::Result<Response<T>, Status> {
    r.map(Response::new).map_err(|f| {
        if f.field == ALREADY_EXISTS_FIELD {
            let clean = Failure::new(f.code, 0, f.message);
            let mut md = MetadataMap::new();
            md.insert("turbo-code", clean.code.to_string().parse().expect("ASCII"));
            md.insert("turbo-field", "0".parse().expect("ASCII"));
            Status::with_metadata(Code::AlreadyExists, describe(&clean, &[]), md)
        } else {
            to_status(&f, &[])
        }
    })
}

fn check_rpc_headers<T>(
    state: &State,
    req: &Request<T>,
    default_priority: Priority,
) -> std::result::Result<(Priority, Deadline), Status> {
    let priority = Priority::from_metadata(req.metadata())?.unwrap_or(default_priority);
    let deadline = Deadline::from_request(req)?;
    if deadline.is_expired() {
        state.metrics.record_cancelled_before_exec();
        return Err(deadline_status());
    }
    Ok((priority, deadline))
}

#[tonic::async_trait]
impl TurboSearchService for SearchService {
    async fn list_search_providers(
        &self,
        _req: Request<ListSearchProvidersRequest>,
    ) -> std::result::Result<Response<ListSearchProvidersResponse>, Status> {
        Ok(Response::new(ListSearchProvidersResponse { providers: providers_list() }))
    }

    async fn list_indexes(
        &self,
        _req: Request<ListIndexesRequest>,
    ) -> std::result::Result<Response<ListIndexesResponse>, Status> {
        let indexes = self.registry.indexes.read().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::with_capacity(indexes.len());
        for ix_lock in indexes.values() {
            let ix = ix_lock.read().unwrap_or_else(|e| e.into_inner());
            out.push(ix.summary());
        }
        Ok(Response::new(ListIndexesResponse { indexes: out }))
    }

    async fn create_index(
        &self,
        req: Request<CreateIndexRequest>,
    ) -> std::result::Result<Response<CreateIndexResponse>, Status> {
        let r = req.into_inner();
        if !r.load_from_path.is_empty() && !self.admin {
            return reply(Err(not_admin("CreateIndex.load_from_path")));
        }
        let state = self.state.clone();
        let registry = self.registry.clone();
        let res = tokio::task::spawn_blocking(move || -> crate::api::Result<CreateIndexResponse> {
            if !r.load_from_path.is_empty() {
                let override_name = r.spec.as_ref().map(|s| s.index_name.as_str()).filter(|s| !s.is_empty());
                let snap_dir = registry.snapshot_dir();
                let loaded = load_index_from_dir(&snap_dir, &r.load_from_path, override_name)?;
                let name = loaded.spec.index_name.clone();
                let summary = loaded.summary();
                let mut indexes = registry.indexes.write().unwrap_or_else(|e| e.into_inner());
                if indexes.contains_key(&name) {
                    return Err(already_exists(format!("search index `{name}` already exists")));
                }
                indexes.insert(name, Arc::new(RwLock::new(loaded)));
                return Ok(CreateIndexResponse { summary: Some(summary) });
            }
            let spec = r.spec.ok_or_else(|| bad_arg("CreateIndexRequest.spec is required"))?;
            let summary = registry.create_index(&state, spec)?;
            Ok(CreateIndexResponse { summary: Some(summary) })
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?;
        reply(res)
    }

    async fn delete_index(
        &self,
        req: Request<DeleteIndexRequest>,
    ) -> std::result::Result<Response<DeleteIndexResponse>, Status> {
        let name = req.into_inner().index_name;
        if name.is_empty() {
            return reply(Err(bad_arg("index_name must not be empty")));
        }
        let mut indexes = self.registry.indexes.write().unwrap_or_else(|e| e.into_inner());
        if indexes.remove(&name).is_none() {
            return reply(Err(not_found(format!("no search index `{name}` exists"))));
        }
        let mut aliases = self.registry.aliases.write().unwrap_or_else(|e| e.into_inner());
        aliases.retain(|_, target| target != &name);
        Ok(Response::new(DeleteIndexResponse {}))
    }

    async fn index_documents(
        &self,
        req: Request<Streaming<IndexDocumentRequest>>,
    ) -> std::result::Result<Response<IndexDocumentsResponse>, Status> {
        let (priority, deadline) = check_rpc_headers(&self.state, &req, Priority::Bulk)?;
        let mut stream = req.into_inner();
        let mut items: Vec<IndexDocumentRequest> = Vec::new();
        while let Some(item) = stream.next().await {
            if deadline.is_expired() {
                return Err(deadline_status());
            }
            if items.len() >= MAX_STREAM_DOCUMENTS {
                return reply(Err(bad_arg(format!(
                    "IndexDocuments stream exceeds maximum of {MAX_STREAM_DOCUMENTS} documents"
                ))));
            }
            let doc_req = item?;
            if let Some(first) = items.first()
                && doc_req.index_name != first.index_name
            {
                return reply(Err(bad_arg(format!(
                    "IndexDocuments stream item targets `{}`, expected `{}`",
                    doc_req.index_name, first.index_name
                ))));
            }
            items.push(doc_req);
        }
        if items.is_empty() {
            return reply(Err(bad_arg("IndexDocuments stream must contain at least one document")));
        }
        let res =
            run_index_documents(self.registry.clone(), self.state.clone(), items, priority, deadline.clone()).await;
        if deadline.is_expired() {
            return Err(deadline_status());
        }
        reply(res)
    }

    async fn search(&self, req: Request<SearchRequest>) -> std::result::Result<Response<SearchResponse>, Status> {
        let (priority, deadline) = check_rpc_headers(&self.state, &req, Priority::Interactive)?;
        let r = req.into_inner();
        let res = run_search(self.registry.clone(), self.state.clone(), r, priority, deadline.clone()).await;
        if deadline.is_expired() {
            return Err(deadline_status());
        }
        reply(res)
    }

    async fn persist_index(
        &self,
        req: Request<PersistIndexRequest>,
    ) -> std::result::Result<Response<PersistIndexResponse>, Status> {
        if !self.admin {
            return reply(Err(not_admin("PersistIndex")));
        }
        let r = req.into_inner();
        if r.path.is_empty() {
            return reply(Err(bad_arg("path must not be empty")));
        }
        let registry = self.registry.clone();
        let res = tokio::task::spawn_blocking(move || -> crate::api::Result<PersistIndexResponse> {
            let (resolved_name, ix_lock) = registry.resolve_index(&r.index_name)?;
            let ix = ix_lock.read().unwrap_or_else(|e| e.into_inner());
            let snap_dir = registry.snapshot_dir();
            let written = persist_index_to_dir(&ix, &snap_dir, &r.path)?;
            Ok(PersistIndexResponse { index_name: resolved_name, bytes_written: written })
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?;
        reply(res)
    }

    async fn seal_index(
        &self,
        req: Request<SealIndexRequest>,
    ) -> std::result::Result<Response<SealIndexResponse>, Status> {
        let name = req.into_inner().index_name;
        let (_, ix_lock) = match self.registry.resolve_index(&name) {
            Ok(v) => v,
            Err(f) => return reply(Err(f)),
        };
        let mut ix = ix_lock.write().unwrap_or_else(|e| e.into_inner());
        ix.sealed = true;
        Ok(Response::new(SealIndexResponse { summary: Some(ix.summary()) }))
    }

    async fn reindex(&self, req: Request<ReindexRequest>) -> std::result::Result<Response<ReindexResponse>, Status> {
        let (priority, deadline) = check_rpc_headers(&self.state, &req, Priority::Bulk)?;
        let r = req.into_inner();
        let res = run_reindex(self.registry.clone(), self.state.clone(), r, priority, deadline.clone()).await;
        if deadline.is_expired() {
            return Err(deadline_status());
        }
        reply(res)
    }

    async fn set_alias(
        &self,
        req: Request<SetAliasRequest>,
    ) -> std::result::Result<Response<SetAliasResponse>, Status> {
        let r = req.into_inner();
        if r.alias.is_empty() || r.index_name.is_empty() {
            return reply(Err(bad_arg("alias and index_name must not be empty")));
        }
        let indexes = self.registry.indexes.read().unwrap_or_else(|e| e.into_inner());
        if !indexes.contains_key(&r.index_name) {
            return reply(Err(not_found(format!("target index `{}` does not exist", r.index_name))));
        }
        drop(indexes);
        let mut aliases = self.registry.aliases.write().unwrap_or_else(|e| e.into_inner());
        aliases.insert(r.alias, r.index_name);
        Ok(Response::new(SetAliasResponse {}))
    }

    async fn delete_alias(
        &self,
        req: Request<DeleteAliasRequest>,
    ) -> std::result::Result<Response<DeleteAliasResponse>, Status> {
        let alias = req.into_inner().alias;
        let mut aliases = self.registry.aliases.write().unwrap_or_else(|e| e.into_inner());
        if aliases.remove(&alias).is_none() {
            return reply(Err(not_found(format!("alias `{alias}` does not exist"))));
        }
        Ok(Response::new(DeleteAliasResponse {}))
    }

    async fn list_aliases(
        &self,
        _req: Request<ListAliasesRequest>,
    ) -> std::result::Result<Response<ListAliasesResponse>, Status> {
        let aliases = self.registry.aliases.read().unwrap_or_else(|e| e.into_inner());
        let list = aliases.iter().map(|(a, ix)| AliasEntry { alias: a.clone(), index_name: ix.clone() }).collect();
        Ok(Response::new(ListAliasesResponse { aliases: list }))
    }

    async fn set_collection(
        &self,
        req: Request<SetCollectionRequest>,
    ) -> std::result::Result<Response<SetCollectionResponse>, Status> {
        let r = req.into_inner();
        if r.collection_name.is_empty() {
            return reply(Err(bad_arg("collection_name must not be empty")));
        }
        if !r.oov_alert_threshold.is_finite() || !(0.0..=1.0).contains(&r.oov_alert_threshold) {
            return reply(Err(bad_arg("oov_alert_threshold must be in [0.0, 1.0]")));
        }
        let mut baseline = BTreeSet::new();
        for raw in &r.baseline_terms {
            let analyzed: Vec<String> = analyze_chunk_terms(raw, 0).into_iter().map(|(t, _)| t).collect();
            if analyzed.is_empty() && !raw.is_empty() {
                baseline.insert(raw.to_ascii_lowercase());
            } else {
                baseline.extend(analyzed);
            }
        }
        let mut cols = self.registry.collections.write().unwrap_or_else(|e| e.into_inner());
        let entry = cols.entry(r.collection_name.clone()).or_insert_with(|| CollectionState {
            name: r.collection_name.clone(),
            baseline_terms: BTreeSet::new(),
            known_terms: BTreeSet::new(),
            term_df: BTreeMap::new(),
            total_units: 0,
            total_length: 0,
            oov_alert_threshold: r.oov_alert_threshold,
        });
        entry.oov_alert_threshold = r.oov_alert_threshold;
        if !baseline.is_empty() {
            for t in &baseline {
                entry.known_terms.insert(t.clone());
            }
            entry.baseline_terms = baseline;
        }
        Ok(Response::new(SetCollectionResponse { collection: Some(entry.to_info()) }))
    }

    async fn get_collection(
        &self,
        req: Request<GetCollectionRequest>,
    ) -> std::result::Result<Response<GetCollectionResponse>, Status> {
        let name = req.into_inner().collection_name;
        let cols = self.registry.collections.read().unwrap_or_else(|e| e.into_inner());
        let Some(c) = cols.get(&name) else {
            return reply(Err(not_found(format!("collection `{name}` does not exist"))));
        };
        Ok(Response::new(GetCollectionResponse { collection: Some(c.to_info()) }))
    }

    async fn list_collections(
        &self,
        _req: Request<ListCollectionsRequest>,
    ) -> std::result::Result<Response<ListCollectionsResponse>, Status> {
        let cols = self.registry.collections.read().unwrap_or_else(|e| e.into_inner());
        let list = cols.values().map(CollectionState::to_info).collect();
        Ok(Response::new(ListCollectionsResponse { collections: list }))
    }

    async fn delete_collection(
        &self,
        req: Request<DeleteCollectionRequest>,
    ) -> std::result::Result<Response<DeleteCollectionResponse>, Status> {
        let name = req.into_inner().collection_name;
        let mut cols = self.registry.collections.write().unwrap_or_else(|e| e.into_inner());
        if cols.remove(&name).is_none() {
            return reply(Err(not_found(format!("collection `{name}` does not exist"))));
        }
        Ok(Response::new(DeleteCollectionResponse {}))
    }

    type WatchCollectionStream =
        Pin<Box<dyn tokio_stream::Stream<Item = std::result::Result<CollectionEvent, Status>> + Send>>;

    async fn watch_collection(
        &self,
        req: Request<WatchCollectionRequest>,
    ) -> std::result::Result<Response<Self::WatchCollectionStream>, Status> {
        let r = req.into_inner();
        if r.collection_name.is_empty() {
            return reply(Err(bad_arg("collection_name must not be empty")));
        }
        let mut rx = self.registry.events.subscribe();
        let mut stopping = self.registry.stopping.subscribe();
        if *stopping.borrow() {
            return Err(Status::unavailable(STOPPING));
        }
        let (tx, out_rx) = mpsc::channel(32);
        let col_name = r.collection_name;
        let min_oov = r.min_oov_ratio.max(0.0);
        // A watch has no end of its own, so the task ends it when the
        // server stops (a stop would otherwise wait on it without limit)
        // and ends itself as soon as the caller is gone.
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    ev = rx.recv() => match ev {
                        Ok(ev) => {
                            if ev.collection_name != col_name {
                                continue;
                            }
                            let oov = ev.report.as_ref().map_or(0.0, |rep| rep.oov_ratio);
                            if oov >= min_oov && tx.send(Ok(ev)).await.is_err() {
                                break;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(broadcast::error::RecvError::Closed) => break,
                    },
                    () = async { drop(stopping.wait_for(|s| *s).await) } => {
                        let _ = tx.send(Err(Status::unavailable(STOPPING))).await;
                        break;
                    }
                    () = tx.closed() => break,
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(out_rx))))
    }
}
