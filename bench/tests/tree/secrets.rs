//! No private key material in the tree: the TLS certificates and keys of
//! test fixtures are committed as string constants (mid-line, never as a
//! PEM file), and nothing else should ever carry a private key block.

use std::path::Path;

/// A private key PEM starts its first line at the left edge; a key pasted
/// into a Rust string constant does not, so the fixtures in
/// `server/tests/` do not trip this.
fn has_key_block(text: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim_start();
        (line.starts_with("-----BEGIN ") && line.trim_end().ends_with("PRIVATE KEY-----"))
            || line.starts_with("-----BEGIN OPENSSH PRIVATE KEY-----")
    })
}

/// Every committed file carrying a private key block, sorted.
pub fn violations(root: &Path) -> Vec<String> {
    let out = std::process::Command::new("git")
        .current_dir(root)
        .args(["ls-files", "-z"])
        .output()
        .expect("git lists the committed files");
    assert!(out.status.success());
    let mut found = Vec::new();
    for p in String::from_utf8(out.stdout).unwrap().split('\0').filter(|s| !s.is_empty()) {
        let path = root.join(p);
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        if has_key_block(&text) {
            found.push(p.to_owned());
        }
    }
    found.sort();
    found
}

#[test]
fn no_committed_file_holds_a_private_key_block() {
    let found = violations(&crate::common::workspace());
    assert!(found.is_empty(), "private key blocks committed in:\n  {}", found.join("\n  "));
}

#[test]
fn the_key_block_check_fires_on_a_real_pem_and_not_on_a_rust_constant() {
    assert!(has_key_block("-----BEGIN PRIVATE KEY-----\nYm9ndXM=\n-----END PRIVATE KEY-----\n"));
    assert!(has_key_block("  -----BEGIN RSA PRIVATE KEY-----\n"));
    assert!(!has_key_block("const K: &str = \"-----BEGIN PRIVATE KEY-----\\n\";"));
    assert!(!has_key_block("(b\"-----BEGIN PRIVATE KEY-----\", PemKey),"));
    assert!(!has_key_block("no keys here"));
}
