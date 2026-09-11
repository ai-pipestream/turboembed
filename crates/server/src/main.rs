//! `inferstream`: the arch-neutral development binary. Serves only the mock
//! backend — use it to validate clients, auth, and the wire path anywhere.
//! Engine-backed serving lives in the arch binaries: `inferstream-nvidia`,
//! `inferstream-intel`, `inferstream-apple`.

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // No arch: catalog `serve` aliases are rejected with an actionable
    // error; the dev binary serves explicit [[models]] entries only.
    inferstream_server::run_cli("inferstream", None, &inferstream_server::mock_factory()).await
}
