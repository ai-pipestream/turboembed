//! The calls the demo forwards. Every unary and server-streaming method of
//! the server's services is listed with whether it reads or writes, and so
//! is every client-streaming method, as an upload: a browser cannot make a
//! client-streaming call, so each is an HTTP PUT to `/upload` and the
//! method path (upload.rs). The reads are allowed unless the demo is told
//! otherwise, and a write, every upload included, only when named with
//! `--allow`. Bidi methods are not listed.

/// One method: its path, `/<package>.<Service>/<Method>`, whether the
/// server answers it with a stream, and whether it takes a client stream,
/// as an upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Method {
    pub path: &'static str,
    pub streaming: bool,
    pub upload: bool,
    write: bool,
}

impl Method {
    /// Whether the method changes the server.
    pub fn is_write(&self) -> bool {
        self.write
    }
}

const fn read(path: &'static str) -> Method {
    Method { path, streaming: false, upload: false, write: false }
}

const fn write(path: &'static str) -> Method {
    Method { path, streaming: false, upload: false, write: true }
}

/// A client-streaming method, which writes.
const fn upload(path: &'static str) -> Method {
    Method { path, streaming: false, upload: true, write: true }
}

const fn stream(m: Method) -> Method {
    Method { streaming: true, ..m }
}

/// Every method the demo knows.
pub const METHODS: [Method; 56] = [
    read("/inference.GRPCInferenceService/ServerLive"),
    read("/inference.GRPCInferenceService/ServerReady"),
    read("/inference.GRPCInferenceService/ModelReady"),
    read("/inference.GRPCInferenceService/ServerMetadata"),
    read("/inference.GRPCInferenceService/ModelMetadata"),
    read("/inference.GRPCInferenceService/ModelInfer"),
    read("/turbo.result.v1.AnalysisService/Capabilities"),
    write("/turbo.result.v1.AnalysisService/LoadBundle"),
    write("/turbo.result.v1.AnalysisService/UnloadBundle"),
    read("/turbo.result.v1.AnalysisService/RoundTripSparseTerms"),
    read("/turbo.result.v1.AnalysisService/RoundTripTokenVectors"),
    read("/turbo.result.v1.AnalysisService/RoundTripRerankScores"),
    read("/turbo.result.v1.AnalysisService/RoundTripExtraction"),
    read("/turbo.result.v1.AnalysisService/RoundTripRecords"),
    read("/turbo.result.v1.AnalysisService/RoundTripChunkLevels"),
    read("/turbo.result.v1.AnalysisService/RoundTripDocumentGraph"),
    read("/turbo.result.v1.AnalysisService/InspectDocumentGraph"),
    read("/turbo.result.v1.AnalysisService/RoundTripTypedDocument"),
    stream(read("/turbo.result.v1.AnalysisService/AnalyzeDocumentProgressive")),
    read("/turbo.result.v1.AnalysisService/InspectDocumentWindow"),
    read("/turbo.result.v1.AnalysisService/ProjectChunks"),
    read("/turbo.result.v1.AnalysisService/ListOutputFormats"),
    read("/turbo.result.v1.AnalysisService/FormatDocument"),
    read("/turbo.search.v1.TurboSearchService/ListSearchProviders"),
    read("/turbo.search.v1.TurboSearchService/ListIndexes"),
    write("/turbo.search.v1.TurboSearchService/CreateIndex"),
    write("/turbo.search.v1.TurboSearchService/DeleteIndex"),
    read("/turbo.search.v1.TurboSearchService/Search"),
    write("/turbo.search.v1.TurboSearchService/PersistIndex"),
    write("/turbo.search.v1.TurboSearchService/SealIndex"),
    write("/turbo.search.v1.TurboSearchService/Reindex"),
    write("/turbo.search.v1.TurboSearchService/SetAlias"),
    write("/turbo.search.v1.TurboSearchService/DeleteAlias"),
    read("/turbo.search.v1.TurboSearchService/ListAliases"),
    write("/turbo.search.v1.TurboSearchService/SetCollection"),
    read("/turbo.search.v1.TurboSearchService/GetCollection"),
    read("/turbo.search.v1.TurboSearchService/ListCollections"),
    write("/turbo.search.v1.TurboSearchService/DeleteCollection"),
    stream(read("/turbo.search.v1.TurboSearchService/WatchCollection")),
    read("/turbo.training.v1.TurboCatalogService/ListModelCatalog"),
    read("/turbo.training.v1.TurboCatalogService/ListInstalledModels"),
    write("/turbo.training.v1.TurboCatalogService/RecordLicenceAcceptance"),
    stream(write("/turbo.training.v1.TurboCatalogService/InstallModel")),
    read("/turbo.training.v1.TurboTrainingService/ListTeachers"),
    stream(write("/turbo.training.v1.TurboTrainingService/TrainStaticModel")),
    read("/turbo.training.v1.TurboTrainingService/ListStaticModels"),
    write("/turbo.training.v1.TurboTrainingService/DeleteStaticModel"),
    read("/turbo.vocabulary.v1.TurboVocabularyService/ListDictionaryFormats"),
    read("/turbo.vocabulary.v1.TurboVocabularyService/ListDictionaries"),
    read("/turbo.vocabulary.v1.TurboVocabularyService/ListVocabularies"),
    write("/turbo.vocabulary.v1.TurboVocabularyService/DeleteVocabularyArtifact"),
    stream(read("/turbo.vocabulary.v1.TurboVocabularyService/DownloadVocabulary")),
    upload("/turbo.vocabulary.v1.TurboVocabularyService/ImportDictionary"),
    upload("/turbo.vocabulary.v1.TurboVocabularyService/LearnVocabulary"),
    upload("/turbo.search.v1.TurboSearchService/IndexDocuments"),
    upload("/turbo.sink.v1.TurboSinkService/StreamDocuments"),
];

/// Whether `path` is an upload, a client-streaming method: a PUT to
/// `/upload` and the path, never a Connect call.
pub fn is_upload(path: &str) -> bool {
    METHODS.iter().any(|m| m.upload && m.path == path)
}

/// The methods the demo forwards: the reads, and each write named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allow {
    allowed: Vec<Method>,
}

impl Default for Allow {
    fn default() -> Allow {
        Allow { allowed: METHODS.iter().filter(|m| !m.write).copied().collect() }
    }
}

impl Allow {
    /// The reads and the methods named, each `/<package>.<Service>/<Method>`
    /// with or without its leading slash. A name not in the table, such as
    /// a bidi method, is refused.
    pub fn with(names: &[String]) -> Result<Allow, String> {
        let mut allow = Allow::default();
        for name in names {
            let path = name.trim().trim_start_matches('/');
            let Some(m) = METHODS.iter().find(|m| &m.path[1..] == path) else {
                return Err(format!(
                    "--allow {name}: not a unary, server-streaming or upload method of the server (/<package>.<Service>/<Method>)"
                ));
            };
            if !allow.allowed.contains(m) {
                allow.allowed.push(*m);
            }
        }
        Ok(allow)
    }

    /// Whether a method that changes the server is forwarded.
    pub fn writes(&self) -> bool {
        self.allowed.iter().any(|m| m.write)
    }

    /// The method at `path` when it is forwarded; else whether the demo
    /// knows the path at all.
    pub fn get(&self, path: &str) -> Result<Method, bool> {
        match self.allowed.iter().find(|m| m.path == path) {
            Some(m) => Ok(*m),
            None => Err(METHODS.iter().any(|m| m.path == path)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_are_allowed_and_writes_only_when_named() {
        let a = Allow::default();
        assert!(a.get("/turbo.result.v1.AnalysisService/Capabilities").is_ok());
        assert_eq!(a.get("/turbo.training.v1.TurboTrainingService/TrainStaticModel"), Err(true));
        assert_eq!(a.get("/turbo.search.v1.TurboSearchService/IndexDocuments"), Err(true));
        assert_eq!(a.get("/turbo.sink.v1.TurboSinkService/StreamDocuments"), Err(true));
        assert_eq!(a.get("/turbo.result.v1.AnalysisService/AnalyzeStream"), Err(false));
        let a = Allow::with(&["turbo.training.v1.TurboTrainingService/TrainStaticModel".into()]).unwrap();
        assert!(a.get("/turbo.training.v1.TurboTrainingService/TrainStaticModel").unwrap().streaming);
        let a = Allow::with(&["/turbo.search.v1.TurboSearchService/IndexDocuments".into()]).unwrap();
        assert!(a.writes() && a.get("/turbo.search.v1.TurboSearchService/IndexDocuments").unwrap().upload);
        assert!(Allow::with(&["/turbo.result.v1.AnalysisService/AnalyzeStream".into()]).is_err());
    }
}
