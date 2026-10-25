//! Tonic `TurboSearchService` implementation (`#1221`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

use crate::proto::search::turbo_search_service_server::TurboSearchService;
use crate::proto::search::*;
use crate::search::ingest::{analyze_chunk_terms, ingest_document, rebuild_compiled_indexes};
use crate::search::query::execute_search;
use crate::search::registry::{CollectionState, SearchRegistry, bad_arg, providers_list};
use crate::search::snapshot::{load_index_from_path, persist_index_to_path};
use crate::state::State;
use crate::status::to_status;

pub struct SearchService {
    pub(crate) state: Arc<State>,
    pub(crate) registry: Arc<SearchRegistry>,
}

fn reply<T>(r: crate::api::Result<T>) -> std::result::Result<Response<T>, Status> {
    r.map(Response::new).map_err(|f| to_status(&f, &[]))
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
        let registry = self.registry.clone();
        let res = tokio::task::spawn_blocking(move || -> crate::api::Result<CreateIndexResponse> {
            if !r.load_from_path.is_empty() {
                let override_name = r.spec.as_ref().map(|s| s.index_name.as_str()).filter(|s| !s.is_empty());
                let loaded = load_index_from_path(Path::new(&r.load_from_path), override_name)?;
                let name = loaded.spec.index_name.clone();
                let summary = loaded.summary();
                let mut indexes = registry.indexes.write().unwrap_or_else(|e| e.into_inner());
                indexes.insert(name, Arc::new(RwLock::new(loaded)));
                return Ok(CreateIndexResponse { summary: Some(summary) });
            }
            let spec = r.spec.ok_or_else(|| bad_arg("CreateIndexRequest.spec is required"))?;
            let summary = registry.create_index(spec)?;
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
            return reply(Err(bad_arg(format!("no search index `{name}` exists"))));
        }
        let mut aliases = self.registry.aliases.write().unwrap_or_else(|e| e.into_inner());
        aliases.retain(|_, target| target != &name);
        Ok(Response::new(DeleteIndexResponse {}))
    }

    async fn index_documents(
        &self,
        req: Request<Streaming<IndexDocumentRequest>>,
    ) -> std::result::Result<Response<IndexDocumentsResponse>, Status> {
        let mut stream = req.into_inner();
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item?);
        }
        if items.is_empty() {
            return reply(Err(bad_arg("IndexDocuments stream must contain at least one document")));
        }
        let target_arg = items[0].index_name.clone();
        let state = self.state.clone();
        let registry = self.registry.clone();
        let res = tokio::task::spawn_blocking(move || -> crate::api::Result<IndexDocumentsResponse> {
            let (resolved_name, ix_lock) = registry.resolve_index(&target_arg)?;
            let mut ix = ix_lock.write().unwrap_or_else(|e| e.into_inner());
            let mut indexed_docs = 0u32;
            let mut indexed_chunks = 0u32;
            let mut batch_chunk_terms = Vec::new();
            for doc_req in items {
                let added = ingest_document(&mut ix, &state, doc_req, &mut batch_chunk_terms)?;
                indexed_docs += 1;
                indexed_chunks += added;
            }
            rebuild_compiled_indexes(&mut ix)?;
            let col_name = if !ix.spec.collection_name.is_empty() {
                ix.spec.collection_name.clone()
            } else {
                resolved_name.clone()
            };
            let drift =
                registry.record_collection_update(&col_name, &resolved_name, &batch_chunk_terms, ix.bm25.statistics());
            Ok(IndexDocumentsResponse {
                index_name: resolved_name,
                indexed_documents: indexed_docs,
                indexed_chunks,
                drift,
            })
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?;
        reply(res)
    }

    async fn search(&self, req: Request<SearchRequest>) -> std::result::Result<Response<SearchResponse>, Status> {
        let r = req.into_inner();
        let state = self.state.clone();
        let registry = self.registry.clone();
        let res = tokio::task::spawn_blocking(move || -> crate::api::Result<SearchResponse> {
            let (resolved_name, ix_lock) = registry.resolve_index(&r.index_name)?;
            let ix = ix_lock.read().unwrap_or_else(|e| e.into_inner());
            execute_search(resolved_name, &ix, &state, r)
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?;
        reply(res)
    }

    async fn persist_index(
        &self,
        req: Request<PersistIndexRequest>,
    ) -> std::result::Result<Response<PersistIndexResponse>, Status> {
        let r = req.into_inner();
        if r.path.is_empty() {
            return reply(Err(bad_arg("path must not be empty")));
        }
        let registry = self.registry.clone();
        let res = tokio::task::spawn_blocking(move || -> crate::api::Result<PersistIndexResponse> {
            let (resolved_name, ix_lock) = registry.resolve_index(&r.index_name)?;
            let ix = ix_lock.read().unwrap_or_else(|e| e.into_inner());
            let written = persist_index_to_path(&ix, Path::new(&r.path))?;
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
        let r = req.into_inner();
        let state = self.state.clone();
        let registry = self.registry.clone();
        let res = tokio::task::spawn_blocking(move || -> crate::api::Result<ReindexResponse> {
            let target_spec = r.target_spec.ok_or_else(|| bad_arg("ReindexRequest.target_spec is required"))?;
            let target_name = target_spec.index_name.clone();
            let (_, src_lock) = registry.resolve_index(&r.source_index)?;
            let retained_docs = {
                let src = src_lock.read().unwrap_or_else(|e| e.into_inner());
                src.docs.clone()
            };
            registry.create_index(target_spec)?;
            let (_, dst_lock) = registry.resolve_index(&target_name)?;
            {
                let mut dst = dst_lock.write().unwrap_or_else(|e| e.into_inner());
                let mut batch_terms = Vec::new();
                for d in retained_docs {
                    let doc_req = IndexDocumentRequest {
                        index_name: target_name.clone(),
                        doc_id: d.doc_id,
                        text: d.text,
                        graph: d.graph,
                        chunks: d.chunks,
                        attributes: d.attributes.into_iter().collect(),
                    };
                    ingest_document(&mut dst, &state, doc_req, &mut batch_terms)?;
                }
                rebuild_compiled_indexes(&mut dst)?;
            }
            if !r.swap_alias.is_empty() {
                let mut aliases = registry.aliases.write().unwrap_or_else(|e| e.into_inner());
                aliases.insert(r.swap_alias.clone(), target_name.clone());
            }
            let summary = dst_lock.read().unwrap_or_else(|e| e.into_inner()).summary();
            Ok(ReindexResponse { summary: Some(summary), swapped_alias: r.swap_alias })
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?;
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
            return reply(Err(bad_arg(format!("target index `{}` does not exist", r.index_name))));
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
            return reply(Err(bad_arg(format!("alias `{alias}` does not exist"))));
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
            return reply(Err(bad_arg(format!("collection `{name}` does not exist"))));
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
            return reply(Err(bad_arg(format!("collection `{name}` does not exist"))));
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
        let (tx, out_rx) = mpsc::channel(32);
        let col_name = r.collection_name;
        let min_oov = r.min_oov_ratio.max(0.0);
        tokio::spawn(async move {
            while let Ok(ev) = rx.recv().await {
                if ev.collection_name != col_name {
                    continue;
                }
                let oov = ev.report.as_ref().map_or(0.0, |rep| rep.oov_ratio);
                if oov >= min_oov && tx.send(Ok(ev)).await.is_err() {
                    break;
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(out_rx))))
    }
}
