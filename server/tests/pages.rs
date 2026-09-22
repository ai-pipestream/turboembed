//! The `--pages` static directory: files under it at `/`, nothing else.

mod common;

use std::sync::Arc;

use turbo_inferstream::http;

fn pages_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("a temporary directory");
    std::fs::write(dir.path().join("index.html"), "<!doctype html><title>t</title>").unwrap();
    std::fs::create_dir(dir.path().join("js")).unwrap();
    std::fs::write(dir.path().join("js/app.js"), "console.log(1)").unwrap();
    std::fs::write(dir.path().join("corpus.json"), "[]").unwrap();
    dir
}

#[tokio::test]
async fn index_files_and_content_types_are_served() {
    let dir = pages_dir();
    let router = http::with_pages(http::router(common::engine()), dir.path()).unwrap();
    for (uri, status, content_type, body) in [
        ("/", 200, "text/html; charset=utf-8", "<!doctype html><title>t</title>"),
        ("/index.html", 200, "text/html; charset=utf-8", "<!doctype html><title>t</title>"),
        ("/js/app.js", 200, "text/javascript; charset=utf-8", "console.log(1)"),
        ("/corpus.json", 200, "application/json", "[]"),
    ] {
        let r = common::get_on(router.clone(), uri).await;
        assert_eq!(r.status.as_u16(), status, "GET {uri}: {}", r.text);
        assert_eq!(r.content_type, content_type, "GET {uri}");
        assert_eq!(r.text, body, "GET {uri}");
    }
}

#[tokio::test]
async fn paths_outside_the_directory_and_missing_files_are_not_found() {
    let dir = pages_dir();
    let router = http::with_pages(http::router(common::engine()), dir.path()).unwrap();
    for uri in ["/nope.html", "/js/", "/js", "/../Cargo.toml", "/js/../../etc/passwd", "/%2e%2e/Cargo.toml"] {
        let r = common::get_on(router.clone(), uri).await;
        assert_eq!(r.status.as_u16(), 404, "GET {uri}: {}", r.text);
    }
}

#[tokio::test]
async fn the_api_routes_take_precedence_over_pages() {
    let dir = pages_dir();
    std::fs::create_dir_all(dir.path().join("v2/health")).unwrap();
    std::fs::write(dir.path().join("v2/health/live"), "page").unwrap();
    let router = http::with_pages(http::router(common::engine()), dir.path()).unwrap();
    let r = common::get_on(router, "/v2/health/live").await;
    assert_eq!(r.status.as_u16(), 200);
    assert_ne!(r.text, "page", "the API answered, not the file");
}

#[test]
fn a_missing_directory_is_refused() {
    let engine = common::engine();
    let missing = std::env::temp_dir().join("inferstream-pages-does-not-exist");
    assert!(http::with_pages(http::router(Arc::clone(&engine)), &missing).is_err());
}
