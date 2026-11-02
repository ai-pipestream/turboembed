//! A Google Drive folder as a source (#1736): the Drive API v3 over the
//! zoo's own HTTPS stack, no SDK. The folder is a store root in the
//! `blobs/sha256/<hex>` layout the other stores share; a file not there
//! is looked for at `<artifact-id>/<path>`, the zoo cache's own layout,
//! so a copied cache serves as well.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::Deserialize;

use super::oauth::Renewing;
use super::transport::Transport;
use crate::index::{Artifact, File, Url};

/// The Drive API's own address.
pub const DRIVE_ENDPOINT: &str = "https://www.googleapis.com";

const FOLDER: &str = "application/vnd.google-apps.folder";

/// The most `files.list` pages read for one name.
const PAGES: usize = 20;

/// How a request to Drive is signed in.
#[derive(Clone)]
pub enum DriveAuth {
    /// An OAuth 2.0 access token, sent as a bearer token: one from
    /// `gcloud auth print-access-token`, or any flow the caller runs.
    Token(String),
    /// An API key, enough for a folder shared with anyone who has the
    /// link.
    ApiKey(String),
    /// A user's stored grant: access tokens from its refresh token, renewed
    /// before they run out and once more when Drive turns one down (#1744).
    Refresh(Arc<Renewing>),
}

impl DriveAuth {
    /// `GOOGLE_OAUTH_ACCESS_TOKEN` when `var` has it, else
    /// `GOOGLE_API_KEY`; an empty one counts as unset. `address` is for
    /// the error alone.
    pub fn from_vars(address: &str, var: impl Fn(&str) -> Option<String>) -> Result<DriveAuth, String> {
        let var = |k: &str| var(k).filter(|v| !v.is_empty());
        match (var("GOOGLE_OAUTH_ACCESS_TOKEN"), var("GOOGLE_API_KEY")) {
            (Some(token), _) => Ok(DriveAuth::Token(token)),
            (None, Some(key)) => Ok(DriveAuth::ApiKey(key)),
            (None, None) => Err(format!("{address}: set GOOGLE_OAUTH_ACCESS_TOKEN or GOOGLE_API_KEY")),
        }
    }
}

impl std::fmt::Debug for DriveAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DriveAuth::Token(_) => "Token(..)",
            DriveAuth::ApiKey(_) => "ApiKey(..)",
            DriveAuth::Refresh(_) => "Refresh(..)",
        })
    }
}

/// A Google Drive folder, `gdrive://<folder-id>[/name/...]`: the folder
/// with that id (`root` is the signed-in user's My Drive), or the folder
/// reached from it through the names after it. Shared drives are
/// included. Each file is found with one `files.list` query per folder
/// on the way (folder ids are remembered once found) and read with one
/// `files.get?alt=media`. The token or key travels in a header, never
/// in a URL. A redirect is followed over HTTPS only and carries neither.
pub struct GoogleDrive {
    address: String,
    root: String,
    under: Vec<String>,
    base: String,
    auth: DriveAuth,
    /// Sends the signed request and follows no redirect.
    signed: ureq::Agent,
    /// Follows a redirect, unsigned.
    plain: ureq::Agent,
    /// Folder ids found so far, by their path below the store root.
    folders: Mutex<HashMap<Vec<String>, String>>,
}

impl GoogleDrive {
    /// The folder `address` names, signed in from the environment:
    /// `GOOGLE_OAUTH_ACCESS_TOKEN` when set, else `GOOGLE_API_KEY`.
    pub fn new(address: &str) -> Result<GoogleDrive, String> {
        GoogleDrive::from_vars(address, |k| std::env::var(k).ok())
    }

    /// Whether `address` is one [`GoogleDrive::with`] takes, without
    /// signing in.
    pub fn check_address(address: &str) -> Result<(), String> {
        parse(address).map(|_| ())
    }

    /// [`GoogleDrive::new`], reading the variables through `var`.
    pub fn from_vars(address: &str, var: impl Fn(&str) -> Option<String>) -> Result<GoogleDrive, String> {
        parse(address)?;
        GoogleDrive::with(address, None, DriveAuth::from_vars(address, var)?)
    }

    /// The folder at `endpoint`, or at [`DRIVE_ENDPOINT`] when none is
    /// given. An `https://host` endpoint is HTTPS only, redirects
    /// included. An `http://host:port` endpoint sends the token or key in
    /// clear text, so it is only for a stand-in on loopback in tests.
    pub fn with(address: &str, endpoint: Option<&str>, auth: DriveAuth) -> Result<GoogleDrive, String> {
        let (root, under) = parse(address)?;
        let base = endpoint.unwrap_or(DRIVE_ENDPOINT).trim_end_matches('/');
        let rest = base.strip_prefix("https://").or_else(|| base.strip_prefix("http://"));
        if rest.is_none_or(|r| r.is_empty() || r.contains(['/', '?', '#', '@'])) {
            return Err(format!("{address}: the endpoint {base} is http:// or https:// with a host and port alone"));
        }
        let secret = match &auth {
            DriveAuth::Token(s) | DriveAuth::ApiKey(s) => s,
            // Checked when it was made.
            DriveAuth::Refresh(_) => "-",
        };
        if secret.is_empty() || !secret.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(format!("{address}: the token or key is empty or holds spaces or control characters"));
        }
        let agent = |redirects| {
            ureq::Agent::config_builder()
                .https_only(base.starts_with("https://"))
                .max_redirects(redirects)
                .redirect_auth_headers(ureq::config::RedirectAuthHeaders::Never)
                .accept_encoding("identity")
                .build()
                .into()
        };
        Ok(GoogleDrive {
            address: address.to_owned(),
            root,
            under,
            base: base.to_owned(),
            auth,
            signed: agent(0),
            plain: agent(10),
            folders: Mutex::new(HashMap::new()),
        })
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    /// One GET of `path?query` on the API, signed with a header; at most
    /// `limit` bytes of the body. A redirect is followed unsigned, so the
    /// token or key never leaves the API's own host. With a stored grant,
    /// a 401 renews the token and asks once more. Errors name the path.
    fn call(&self, path: &str, query: &str, limit: u64) -> Result<Vec<u8>, String> {
        match self.send(path, query, limit, false) {
            Err((true, _)) if matches!(self.auth, DriveAuth::Refresh(_)) => self.send(path, query, limit, true),
            r => r,
        }
        .map_err(|(_, e)| e)
    }

    /// One signed GET, as [`GoogleDrive::call`] makes it; an error says
    /// whether Drive answered 401.
    fn send(&self, path: &str, query: &str, limit: u64, renew: bool) -> Result<Vec<u8>, (bool, String)> {
        let url = format!("{}{path}?{query}&supportsAllDrives=true", self.base);
        let bearer = |t: &str| self.signed.get(&url).header("Authorization", format!("Bearer {t}"));
        let req = match &self.auth {
            DriveAuth::Token(t) => bearer(t),
            DriveAuth::Refresh(r) => bearer(&r.bearer(renew).map_err(|e| (false, format!("{}: {e}", self.address)))?),
            DriveAuth::ApiKey(k) => self.signed.get(&url).header("X-Goog-Api-Key", k.as_str()),
        };
        let fail = |e: ureq::Error| (matches!(e, ureq::Error::StatusCode(401)), format!("{}{path}: {e}", self.base));
        let mut resp = req.call().map_err(fail)?;
        if resp.status().is_redirection() {
            let to = resp.headers().get("location").and_then(|v| v.to_str().ok());
            let to = to.ok_or_else(|| (false, format!("{}{path}: a redirect with no Location", self.base)))?;
            let to = resolve(&self.base, path, to);
            let follow = |e: ureq::Error| (false, format!("{}{path}: following its redirect: {e}", self.base));
            resp = self.plain.get(&to).call().map_err(follow)?;
        }
        resp.body_mut().with_config().limit(limit).read_to_vec().map_err(fail)
    }

    /// The id of `parent`'s child called `name` that is (`folder`) or is
    /// not a folder: with `size`, the first whose size it is, else the
    /// first, read page by page (a page may come back short or empty with
    /// a token for the next), and `None` when there is none.
    fn child(&self, parent: &str, name: &str, folder: bool, size: Option<u64>) -> Result<Option<String>, String> {
        let q = format!(
            "name = '{}' and '{parent}' in parents and trashed = false and mimeType {} '{FOLDER}'",
            literal(name),
            if folder { "=" } else { "!=" },
        );
        let fields = encode("nextPageToken,files(id,size)");
        let want = size.map(|s| s.to_string());
        let mut page = String::new();
        // Far more same-named entries than one folder holds in practice;
        // the bound keeps a server that always sends a token from looping.
        for _ in 0..PAGES {
            let query = format!("q={}&fields={fields}&pageSize=1000&includeItemsFromAllDrives=true{page}", encode(&q));
            let body = self.call("/drive/v3/files", &query, 1 << 20)?;
            let list: List =
                serde_json::from_slice(&body).map_err(|e| format!("{}: a files.list reply: {e}", self.address))?;
            let pick = match &want {
                Some(_) => list.files.iter().find(|f| f.size == want),
                None => list.files.first(),
            };
            match pick {
                Some(f) if valid_id(&f.id) => return Ok(Some(f.id.clone())),
                Some(f) => return Err(format!("{}: Drive answered the file id {:?}", self.address, f.id)),
                None => {}
            }
            match list.next_page_token {
                Some(t) if !t.is_empty() => page = format!("&pageToken={}", encode(&t)),
                _ => return Ok(None),
            }
        }
        Err(format!("{}: more than {PAGES} pages of files named {name}", self.address))
    }

    /// The id of the folder at `path` below the store root, if there is one.
    fn folder(&self, path: &[&str]) -> Result<Option<String>, String> {
        let mut at = self.root.clone();
        let mut walked: Vec<String> = Vec::new();
        for name in self.under.iter().map(String::as_str).chain(path.iter().copied()) {
            walked.push(name.to_owned());
            let known = self.folders.lock().map_err(|e| e.to_string())?.get(&walked).cloned();
            at = match known {
                Some(id) => id,
                None => match self.child(&at, name, true, None)? {
                    Some(id) => {
                        self.folders.lock().map_err(|e| e.to_string())?.insert(walked.clone(), id.clone());
                        id
                    }
                    None => return Ok(None),
                },
            };
        }
        Ok(Some(at))
    }

    /// The id of the file at `path` below the store root, if there is one.
    fn find(&self, path: &[&str], size: u64) -> Result<Option<String>, String> {
        let Some((name, dirs)) = path.split_last() else { return Ok(None) };
        match self.folder(dirs)? {
            Some(parent) => self.child(&parent, name, false, Some(size)),
            None => Ok(None),
        }
    }
}

impl Transport for GoogleDrive {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Err(format!("{url}: a Google Drive source serves a file by its SHA-256 or artifact path, not its address"))
    }

    /// The file at `blobs/sha256/<hex>`, else at `<artifact-id>/<path>`,
    /// at most its size and one byte more so the zoo's size check fails
    /// on a longer one.
    fn get_file(&self, artifact: &Artifact, file: &File) -> Result<Vec<u8>, String> {
        let hex = file.sha256.to_string();
        let mut cached = vec![artifact.id.as_str()];
        cached.extend(file.path.as_str().split('/'));
        for path in [vec!["blobs", "sha256", hex.as_str()], cached] {
            if let Some(id) = self.find(&path, file.size)? {
                return self.call(&format!("/drive/v3/files/{id}"), "alt=media", file.size.saturating_add(1));
            }
        }
        Err(format!("{}: no blobs/sha256/{hex} and no {}/{}", self.address, artifact.id.as_str(), file.path.as_str()))
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct List {
    #[serde(default)]
    files: Vec<Listed>,
    #[serde(default)]
    next_page_token: Option<String>,
}

#[derive(Deserialize)]
struct Listed {
    id: String,
    /// Drive sends a size as a decimal string, and none for a folder.
    #[serde(default)]
    size: Option<String>,
}

/// The `Location` of a redirect from `path` on `base` (`scheme://host`),
/// as an absolute URL (RFC 3986 section 5.2): an absolute address as it
/// is, `//host/x` on the request's scheme, `/x` on its host, and any
/// other reference beside the request's last path segment, with `.` and
/// `..` segments removed. Whether the result may be fetched (HTTPS only
/// for an HTTPS endpoint) is the unsigned agent's check.
fn resolve(base: &str, path: &str, location: &str) -> String {
    if location.contains("://") {
        return location.to_owned();
    }
    let scheme = base.split_once("://").map_or("https", |(s, _)| s);
    if location.starts_with("//") {
        return format!("{scheme}:{location}");
    }
    let (reference, query) = match location.find(['?', '#']) {
        Some(at) => location.split_at(at),
        None => (location, ""),
    };
    let joined = if reference.starts_with('/') {
        reference.to_owned()
    } else if reference.is_empty() {
        path.to_owned()
    } else {
        format!("{}/{reference}", path.rsplit_once('/').map_or("", |(dir, _)| dir))
    };
    let mut out: Vec<&str> = Vec::new();
    let segments: Vec<&str> = joined.split('/').skip(1).collect();
    for (i, seg) in segments.iter().enumerate() {
        let last = i + 1 == segments.len();
        match *seg {
            "." if last => out.push(""),
            "." => {}
            ".." => {
                out.pop();
                if last {
                    out.push("");
                }
            }
            s => out.push(s),
        }
    }
    format!("{base}/{}{query}", out.join("/"))
}

/// `gdrive://<id>[/name/...]` as the folder id and the names below it.
fn parse(address: &str) -> Result<(String, Vec<String>), String> {
    let bad = |why: &str| format!("{address}: {why}");
    let rest = address.strip_prefix("gdrive://").ok_or_else(|| bad("a Google Drive source starts gdrive://"))?;
    let rest = rest.strip_suffix('/').unwrap_or(rest);
    let mut parts = rest.split('/');
    let root = parts.next().unwrap_or_default();
    if !valid_id(root) {
        return Err(bad("a folder id is letters, digits, '_' and '-' (or root, for My Drive)"));
    }
    let under: Vec<String> = parts.map(str::to_owned).collect();
    if under.iter().any(|n| n.is_empty() || n == "." || n == ".." || n.chars().any(char::is_control)) {
        return Err(bad("a folder name below the id is not empty, '.' or '..', and holds no control character"));
    }
    Ok((root.to_owned(), under))
}

/// A Drive id: what the API hands out, and all a URL path may carry here.
fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// `name` as a quoted string in a Drive query: `\` and `'` escaped.
fn literal(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c == '\\' || c == '\'' {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Percent-encodes every byte but RFC 3986's unreserved ones.
pub(super) fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{encode, literal, parse, resolve};

    #[test]
    fn a_redirect_location_resolves_against_the_request() {
        let r = |to: &str| resolve("https://api.example", "/drive/v3/files/abc", to);
        assert_eq!(r("media"), "https://api.example/drive/v3/files/media");
        assert_eq!(r("./media?x=1"), "https://api.example/drive/v3/files/media?x=1");
        assert_eq!(r("../other/x"), "https://api.example/drive/v3/other/x");
        assert_eq!(r("../../../../x"), "https://api.example/x");
        assert_eq!(r(".."), "https://api.example/drive/v3/");
        assert_eq!(r("/media/abc"), "https://api.example/media/abc");
        assert_eq!(r("//cdn.example/b"), "https://cdn.example/b");
        assert_eq!(r("https://cdn.example/b?t=1"), "https://cdn.example/b?t=1");
        assert_eq!(r("?alt=media"), "https://api.example/drive/v3/files/abc?alt=media");
        assert_eq!(resolve("http://127.0.0.1:9", "/a/b", "//h/c"), "http://h/c");
    }

    #[test]
    fn an_address_is_a_folder_id_and_names_below_it() {
        assert_eq!(parse("gdrive://root").unwrap(), ("root".into(), vec![]));
        let (id, under) = parse("gdrive://1AbC_d-E/data/my models/").unwrap();
        assert_eq!(id, "1AbC_d-E");
        assert_eq!(under, ["data", "my models"]);
        for bad in ["gdrive://", "gdrive:///x", "gdrive://a b", "gdrive://a/../b", "gdrive://a//b", "s3://a"] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn names_are_quoted_and_queries_encoded() {
        assert_eq!(literal(r"it's a\b"), r"it\'s a\\b");
        assert_eq!(encode("name = 'é'"), "name%20%3D%20%27%C3%A9%27");
    }
}
