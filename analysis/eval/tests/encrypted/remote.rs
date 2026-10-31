//! Explicit real-service test adapter. Reuses the zoo request signer. It is
//! intentionally not a production S3 provider, registry or credential source.
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use turbo_eval::artifact::{
    ArtifactKey, EvaluationArtifactStore, PendingArtifact, Publication, StorageError, StorageOperation,
    TemporaryFileStore,
};
use turbo_zoo::fetch::sigv4::{Credentials, EMPTY_SHA256, amz_date, sign};

#[derive(Clone)]
pub struct Remote {
    endpoint: String,
    host: String,
    bucket: String,
    prefix: String,
    creds: Credentials,
    stage: PathBuf,
    agent: ureq::Agent,
}
fn failed() -> StorageError {
    StorageError::Io { operation: "remote test", source: io::Error::other("remote operation failed") }
}
impl Remote {
    pub fn configured(stage: PathBuf) -> Self {
        let endpoint = std::env::var("ARTIFACT_TEST_ENDPOINT").expect("set test endpoint");
        let host = endpoint_host(&endpoint).expect("TLS endpoint or explicit loopback HTTP test endpoint");
        let bucket = std::env::var("ARTIFACT_TEST_BUCKET").expect("set isolated test bucket");
        assert!(bucket.starts_with("artifact-security-test-"));
        let creds = Credentials {
            key: std::env::var("AWS_ACCESS_KEY_ID").expect("set test access key"),
            secret: std::env::var("AWS_SECRET_ACCESS_KEY").expect("set test secret"),
            token: None,
        };
        Self {
            endpoint,
            host,
            bucket,
            prefix: format!("run-{}", std::process::id()),
            creds,
            stage,
            agent: ureq::Agent::config_builder()
                .http_status_as_error(false)
                .max_redirects(0)
                .timeout_global(Some(std::time::Duration::from_secs(30)))
                .build()
                .into(),
        }
    }
    fn request(
        &self,
        method: &str,
        key: &ArtifactKey,
        payload: &str,
        conditional: bool,
    ) -> Result<ureq::http::request::Builder, StorageError> {
        let path = format!("/{}/{}/{}", self.bucket, self.prefix, key.as_str());
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|_| failed())?.as_secs();
        let extra = if conditional { vec![("if-none-match", "*")] } else { Vec::new() };
        let headers = sign(method, &self.creds, "us-east-1", (&self.host, &path), payload, &extra, &amz_date(now))
            .map_err(|_| failed())?;
        let mut request = ureq::http::Request::builder().method(method).uri(format!("{}{path}", self.endpoint));
        for (name, value) in headers {
            request = request.header(name, value);
        }
        Ok(request)
    }
    pub fn put_file(&self, key: &ArtifactKey, file: &mut File, conditional: bool) -> Result<u16, StorageError> {
        use std::io::{Seek, SeekFrom};
        let mut digest = turbo_sha256::Sha256::new();
        let mut buf = [0; 65536];
        loop {
            let n = file.read(&mut buf).map_err(|_| failed())?;
            if n == 0 {
                break;
            }
            digest.update(&buf[..n]);
        }
        file.seek(SeekFrom::Start(0)).map_err(|_| failed())?;
        let request =
            self.request("PUT", key, &digest.finish().to_hex(), conditional)?.body(&*file).map_err(|_| failed())?;
        self.agent.run(request).map(|r| r.status().as_u16()).map_err(|_| failed())
    }
    pub fn get_status(&self, key: &ArtifactKey, invalid: bool) -> u16 {
        let mut remote = self.clone();
        if invalid {
            remote.creds.secret = "invalid-test-secret".into();
        }
        let request = remote.request("GET", key, EMPTY_SHA256, false).unwrap().body(()).unwrap();
        remote.agent.run(request).map(|r| r.status().as_u16()).unwrap_or(0)
    }
    pub fn erase(&self, key: &ArtifactKey) -> Result<(), StorageError> {
        let request = self.request("DELETE", key, EMPTY_SHA256, false)?.body(()).map_err(|_| failed())?;
        let response = self.agent.run(request).map_err(|_| failed())?;
        if response.status().is_success() { Ok(()) } else { Err(failed()) }
    }
}
impl EvaluationArtifactStore for Remote {
    fn begin(&self, key: &ArtifactKey) -> Result<Box<dyn PendingArtifact>, StorageError> {
        let pending = TemporaryFileStore::new(&self.stage).begin(key)?;
        Ok(Box::new(Pending { pending: Some(pending), remote: self.clone(), key: key.clone(), local_published: false }))
    }
    fn open(&self, key: &ArtifactKey) -> Result<Box<dyn Read>, StorageError> {
        let request = self.request("GET", key, EMPTY_SHA256, false)?.body(()).map_err(|_| failed())?;
        let response = self.agent.run(request).map_err(|_| failed())?;
        if response.status().as_u16() == 404 {
            return Err(StorageError::NotFound(key.clone()));
        }
        if !response.status().is_success() {
            return Err(failed());
        }
        Ok(Box::new(response.into_body().into_reader()))
    }
    fn list(&self) -> Result<Vec<ArtifactKey>, StorageError> {
        // This probe deliberately does not invent a production listing parser.
        Err(StorageError::Unsupported(StorageOperation::List))
    }
    fn delete(&self, key: &ArtifactKey) -> Result<(), StorageError> {
        self.erase(key)
    }
}
struct Pending {
    pending: Option<Box<dyn PendingArtifact>>,
    remote: Remote,
    key: ArtifactKey,
    local_published: bool,
}
impl Write for Pending {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.pending.as_mut().unwrap().write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.pending.as_mut().unwrap().flush()
    }
}
impl PendingArtifact for Pending {
    fn finalize(mut self: Box<Self>) -> Result<Publication, StorageError> {
        self.pending.take().unwrap().finalize()?;
        self.local_published = true;
        let mut file = File::open(self.remote.stage.join(self.key.as_str())).map_err(|_| failed())?;
        let status = self.remote.put_file(&self.key, &mut file, true)?;
        if status == 412 {
            return Err(StorageError::AlreadyExists(self.key.clone()));
        }
        if status != 200 {
            return Err(failed());
        }
        Ok(Publication { key: self.key.clone(), cleanup_error: None })
    }
    fn abort(mut self: Box<Self>) -> Result<(), StorageError> {
        self.pending.take().unwrap().abort()
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        if self.local_published {
            let _ = fs::remove_file(self.remote.stage.join(self.key.as_str()));
        }
    }
}

/// Transport policy for this explicit service probe: remote plaintext HTTP is
/// forbidden. TLS still needs separate live certificate evidence; this parser
/// does not claim to provide it.
fn endpoint_host(endpoint: &str) -> Option<String> {
    let uri: ureq::http::Uri = endpoint.parse().ok()?;
    let host = uri.host()?;
    let scheme = uri.scheme_str()?;
    if scheme != "https" && !(scheme == "http" && matches!(host, "127.0.0.1" | "[::1]")) {
        return None;
    }
    if uri.path() != "/" || uri.query().is_some() {
        return None;
    }
    let authority = uri.authority()?.as_str();
    if authority.contains('@') {
        return None;
    }
    Some(authority.to_owned())
}

#[test]
fn probe_transport_refuses_cleartext_remote_and_credential_urls() {
    assert!(endpoint_host("http://127.0.0.1:9000").is_some());
    assert!(endpoint_host("https://storage.example").is_some());
    assert!(endpoint_host("http://storage.example").is_none());
    assert!(endpoint_host("https://user:password@storage.example").is_none());
    assert!(endpoint_host("https://storage.example/path").is_none());
    assert!(endpoint_host("https://storage.example?secret=value").is_none());
}
