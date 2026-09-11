//! `inferstream`: the arch-neutral development binary. Serves only the mock
//! backend — use it to validate clients, auth, and the wire path anywhere.
//! Engine-backed serving lives in the arch binaries: `inferstream-nvidia`,
//! `inferstream-intel`, `inferstream-apple`.

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    inferstream_server::run_cli("inferstream", &inferstream_server::mock_factory()).await
}
