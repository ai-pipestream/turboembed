# Analyze layer instances (#2240)

The two tests in `demo/web/e2e/layer-instances.test.ts` run Chromium against
real CPU server and demo processes, using binary Connect protobuf. No routes
are intercepted. `server/examples/inspect_layer_fixture.rs` writes the token
graph through the production Rust export codecs. It contains `é 😀` with a
default token on `é` and a named `gold` token on `😀`.

On main `6adf39e59cf5ad6bcec2a9d77532f0a915c0fa75`, both tests fail: the two
boundary controls have the same label, and both token controls read
`turbo:tokens 1`. See `red.log` and `red.png`.

With the fix, the two tests pass, along with the other 11 browser tests.
See `green.log` and `green.png`. Tests select default only, both instances,
named only, and no filter. Returned token marks change accordingly while
all source text remains visible. Boundary controls select independently;
boundaries have no positional annotations in this RPC.

Reproduce with `scripts/e2e-web.sh`: it builds both binaries, generates the
fixture, starts four tiny CPU bundles, and runs all browser tests. Node,
the pinned npm packages, and Playwright Chromium are required. Pass a
screenshot directory to retain screenshots. Browser timings in these logs
are test durations on a shared CPU, not performance benchmarks.

The server regression `inspect_layer_filter` also checks UTF-8, UTF-16 and
code-point offsets, unknown instances, exact pair matching, legacy name
filters, and the intersection of both filter fields. Heatmap, summary and
window outputs are unchanged by these segment filters.
