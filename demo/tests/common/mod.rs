//! What the demo's tests share: the real server in-process on the CPU, the
//! demo in front of it, and a small HTTP/1.1 client that speaks the
//! Connect protocol by hand: messages are the server's prost types,
//! encoded here, and frames are built and read byte by byte.
#![allow(dead_code)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use prost::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use turbo_kserve::api::Runtime;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES, Device, ModelConfig};
use turbo_kserve::{Serve, Server};
use turbo_kserve_demo::{Allow, Demo, Options};

pub const UNARY: &str = "application/proto";
pub const STREAM: &str = "application/connect+proto";

pub fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata").join(name)
}

/// The runtime index of the CPU.
pub fn cpu() -> u32 {
    let rt = Runtime::create().unwrap();
    (0..16).find(|&i| rt.device_info(i).map(|d| d.kind == turbo::TURBO_DEVICE_CPU).unwrap_or(false)).expect("a CPU")
}

pub fn bundle(name: &str) -> ModelConfig {
    ModelConfig {
        bundle: testdata(name).to_str().unwrap().into(),
        name: None,
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

/// A scratch directory, empty, unique to this process and `name`.
pub fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("demo-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The server serving `bundles`, loaded.
pub async fn server(bundles: &[&str], serve: Serve) -> Server {
    let models = bundles.iter().map(|b| bundle(b)).collect();
    let s = Server::start("127.0.0.1:0".parse().unwrap(), models, DEFAULT_MAX_MESSAGE_BYTES, serve).await.unwrap();
    s.load().await.unwrap();
    s
}

/// The demo in front of `s`, forwarding the reads and the methods `allow`
/// names.
pub async fn demo(s: &Server, allow: &[&str]) -> Demo {
    demo_at(s.local_addr(), allow).await
}

pub async fn demo_at(server: SocketAddr, allow: &[&str]) -> Demo {
    let allow = Allow::with(&allow.iter().map(|a| a.to_string()).collect::<Vec<_>>()).unwrap();
    Demo::start("127.0.0.1:0".parse().unwrap(), &format!("http://{server}"), allow).await.unwrap()
}

/// The demo in front of the server at `server`, started with `options`.
pub async fn demo_with(server: SocketAddr, options: Options) -> Demo {
    Demo::start_with("127.0.0.1:0".parse().unwrap(), &format!("http://{server}"), options).await.unwrap()
}

/// An address nothing listens on: a call the demo forwards there fails
/// `unavailable`, so any other answer was given without the server.
pub async fn nowhere() -> SocketAddr {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap()
}

/// An HTTP/1.1 response being read: its status and headers, then its body
/// as it comes, chunked or not.
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    conn: TcpStream,
    raw: Vec<u8>,
    pub body: Vec<u8>,
    chunked: bool,
    left: usize,
    crlf: bool,
    ended: bool,
}

impl Reply {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    pub fn header_all(&self, name: &str) -> Vec<&str> {
        self.headers.iter().filter(|(k, _)| k == name).map(|(_, v)| v.as_str()).collect()
    }

    async fn read(&mut self) -> bool {
        let mut buf = [0u8; 16384];
        let n = self.conn.read(&mut buf).await.unwrap_or(0);
        self.raw.extend_from_slice(&buf[..n]);
        n > 0
    }

    /// Reads more of the body; false at its end.
    pub async fn more(&mut self) -> bool {
        loop {
            if self.ended {
                return false;
            }
            if !self.chunked {
                if !self.raw.is_empty() {
                    self.body.append(&mut self.raw);
                    return true;
                }
            } else if self.left > 0 {
                if !self.raw.is_empty() {
                    let n = self.left.min(self.raw.len());
                    self.body.extend(self.raw.drain(..n));
                    self.left -= n;
                    self.crlf = self.left == 0;
                    return true;
                }
            } else if self.crlf {
                if self.raw.len() >= 2 {
                    self.raw.drain(..2);
                    self.crlf = false;
                    continue;
                }
            } else if let Some(i) = self.raw.windows(2).position(|w| w == b"\r\n") {
                let line = String::from_utf8(self.raw.drain(..i + 2).collect()).unwrap();
                let size = line.trim_end().split(';').next().unwrap();
                self.left = usize::from_str_radix(size, 16).unwrap();
                if self.left == 0 {
                    self.ended = true;
                }
                continue;
            }
            if !self.read().await {
                self.ended = true;
            }
        }
    }

    pub async fn all(mut self) -> Vec<u8> {
        while self.more().await {}
        self.body
    }

    /// The next envelope: its flags and payload; None at the body's end.
    pub async fn frame(&mut self) -> Option<(u8, Vec<u8>)> {
        loop {
            if self.body.len() >= 5 {
                let len = u32::from_be_bytes(self.body[1..5].try_into().unwrap()) as usize;
                if self.body.len() >= 5 + len {
                    let f: Vec<u8> = self.body.drain(..5 + len).collect();
                    return Some((f[0], f[5..].to_vec()));
                }
            }
            if !self.more().await {
                assert!(self.body.is_empty(), "a partial envelope at the end");
                return None;
            }
        }
    }

    /// Every message of a stream, then its end-of-stream JSON.
    pub async fn frames<M: Message + Default>(mut self) -> (Vec<M>, String) {
        let mut out = Vec::new();
        loop {
            let (flags, payload) = self.frame().await.expect("an end-of-stream envelope");
            match flags {
                0 => out.push(M::decode(payload.as_slice()).unwrap()),
                2 => {
                    assert!(self.frame().await.is_none(), "nothing after the end of the stream");
                    return (out, String::from_utf8(payload).unwrap());
                }
                f => panic!("an envelope with flags {f}"),
            }
        }
    }
}

/// One HTTP/1.1 request; the reply with its head read.
pub async fn http(d: &Demo, method: &str, path: &str, headers: &[(&str, &str)], body: &[u8]) -> Reply {
    let mut conn = TcpStream::connect(d.local_addr()).await.unwrap();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: demo\r\nConnection: close\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    conn.write_all(req.as_bytes()).await.unwrap();
    conn.write_all(body).await.unwrap();
    let mut r = Reply {
        status: 0,
        headers: Vec::new(),
        conn,
        raw: Vec::new(),
        body: Vec::new(),
        chunked: false,
        left: 0,
        crlf: false,
        ended: false,
    };
    let end = loop {
        if let Some(i) = r.raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        assert!(r.read().await, "the connection closed before the head");
    };
    let head = String::from_utf8(r.raw.drain(..end + 4).collect()).unwrap();
    let mut lines = head.trim_end().split("\r\n");
    r.status = lines.next().unwrap().split(' ').nth(1).unwrap().parse().unwrap();
    for l in lines {
        let (k, v) = l.split_once(':').unwrap();
        r.headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
    }
    r.chunked = r.header("transfer-encoding") == Some("chunked");
    r
}

/// A unary Connect call: the reply, its body read whole.
pub async fn unary(d: &Demo, path: &str, msg: &impl Message) -> (Reply, Vec<u8>) {
    let mut r = http(d, "POST", path, &[("Content-Type", UNARY)], &msg.encode_to_vec()).await;
    while r.more().await {}
    let body = std::mem::take(&mut r.body);
    (r, body)
}

/// A unary call the server answers: its reply message.
pub async fn ok<M: Message + Default>(d: &Demo, path: &str, msg: &impl Message) -> M {
    let (r, body) = unary(d, path, msg).await;
    assert_eq!(r.status, 200, "{path}: {}", String::from_utf8_lossy(&body));
    assert_eq!(r.header("content-type"), Some(UNARY));
    M::decode(body.as_slice()).unwrap()
}

/// One message in an envelope with `flags`.
pub fn envelope(flags: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![flags];
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// A server-streaming Connect call; the reply with its head read.
pub async fn stream(d: &Demo, path: &str, msg: &impl Message) -> Reply {
    let r = http(d, "POST", path, &[("Content-Type", STREAM)], &envelope(0, &msg.encode_to_vec())).await;
    assert_eq!(r.status, 200, "{path}");
    assert_eq!(r.header("content-type"), Some(STREAM), "{path}");
    r
}
