//! What the upload tests share: the paths, a PUT sent in parts over one
//! connection, and a relay that counts the bytes the demo sends the server.

#![allow(dead_code)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

pub const IMPORT: &str = "/turbo.vocabulary.v1.TurboVocabularyService/ImportDictionary";
pub const LEARN: &str = "/turbo.vocabulary.v1.TurboVocabularyService/LearnVocabulary";
pub const INDEX: &str = "/turbo.search.v1.TurboSearchService/IndexDocuments";
pub const SINK: &str = "/turbo.sink.v1.TurboSinkService/StreamDocuments";
pub const OCTET: &str = "application/octet-stream";

/// An HTTP/1.1 answer read to its end.
#[derive(Debug)]
pub struct Answer {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Answer {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

/// A PUT open on `addr`: its head sent, its body to follow.
pub struct Put(pub TcpStream);

impl Put {
    /// `extra` are more header lines, each ending `\r\n`; `length` the
    /// Content-Length, or chunked without one.
    pub async fn open(addr: SocketAddr, path: &str, extra: &str, length: Option<u64>) -> Put {
        let mut conn = TcpStream::connect(addr).await.unwrap();
        let framing = match length {
            Some(n) => format!("Content-Length: {n}\r\n"),
            None => "Transfer-Encoding: chunked\r\n".into(),
        };
        let head = format!(
            "PUT /upload{path} HTTP/1.1\r\nHost: demo\r\nConnection: close\r\nContent-Type: {OCTET}\r\n{extra}{framing}\r\n"
        );
        conn.write_all(head.as_bytes()).await.unwrap();
        Put(conn)
    }

    /// Sends `bytes`; false once the demo has stopped reading.
    pub async fn send(&mut self, bytes: &[u8]) -> bool {
        self.0.write_all(bytes).await.is_ok()
    }

    /// Sends `bytes` as one chunk of a chunked body.
    pub async fn chunk(&mut self, bytes: &[u8]) -> bool {
        let mut c = format!("{:x}\r\n", bytes.len()).into_bytes();
        c.extend_from_slice(bytes);
        c.extend_from_slice(b"\r\n");
        self.send(&c).await
    }

    /// The answer, within `secs` seconds.
    pub async fn answer(mut self, secs: u64) -> Answer {
        let mut raw = Vec::new();
        let read = self.0.read_to_end(&mut raw);
        let _ = tokio::time::timeout(Duration::from_secs(secs), read).await.expect("an answer in time");
        parse(&raw)
    }
}

/// The whole of a PUT of `body` with Content-Length.
pub async fn put(addr: SocketAddr, path: &str, extra: &str, body: &[u8]) -> Answer {
    let mut p = Put::open(addr, path, extra, Some(body.len() as u64)).await;
    p.send(body).await;
    p.answer(60).await
}

fn parse(raw: &[u8]) -> Answer {
    let end = raw.windows(4).position(|w| w == b"\r\n\r\n").expect("a head");
    let head = String::from_utf8_lossy(&raw[..end]).into_owned();
    let mut lines = head.split("\r\n");
    let status = lines.next().unwrap().split(' ').nth(1).unwrap().parse().unwrap();
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect::<Vec<_>>();
    let mut body = raw[end + 4..].to_vec();
    if headers.iter().any(|(k, v)| k == "transfer-encoding" && v == "chunked") {
        body = unchunk(&body);
    }
    Answer { status, headers, body }
}

fn unchunk(mut b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(i) = b.windows(2).position(|w| w == b"\r\n") {
        let n = usize::from_str_radix(std::str::from_utf8(&b[..i]).unwrap().trim(), 16).unwrap();
        if n == 0 {
            break;
        }
        out.extend_from_slice(&b[i + 2..i + 2 + n]);
        b = &b[i + 4 + n..];
    }
    out
}

/// A TCP relay to `to` that counts the bytes sent toward it.
pub struct Relay {
    pub addr: SocketAddr,
    pub sent: Arc<AtomicU64>,
}

impl Relay {
    pub async fn start(to: SocketAddr) -> Relay {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let sent = Arc::new(AtomicU64::new(0));
        let count = sent.clone();
        tokio::spawn(async move {
            while let Ok((inbound, _)) = listener.accept().await {
                let count = count.clone();
                tokio::spawn(async move {
                    let outbound = TcpStream::connect(to).await.unwrap();
                    let (mut ir, mut iw) = inbound.into_split();
                    let (mut or, mut ow) = outbound.into_split();
                    tokio::spawn(async move { tokio::io::copy(&mut or, &mut iw).await });
                    let mut buf = vec![0u8; 65536];
                    while let Ok(n) = ir.read(&mut buf).await {
                        if n == 0 || ow.write_all(&buf[..n]).await.is_err() {
                            break;
                        }
                        count.fetch_add(n as u64, Ordering::Relaxed);
                    }
                });
            }
        });
        Relay { addr, sent }
    }

    pub fn sent(&self) -> u64 {
        self.sent.load(Ordering::Relaxed)
    }
}
