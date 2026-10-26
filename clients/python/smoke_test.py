#!/usr/bin/env python3
"""Minimal Python client smoke test using only the published turboembed_proto wheel (#1297)."""

from __future__ import annotations

import os
import sys

import grpc
from turboembed_proto import turbo_analysis_pb2, turbo_analysis_pb2_grpc


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TURBO_KSERVE_SMOKE_ADDR", "")
    if not target:
        raise SystemExit("Expected server address as argv[1] or TURBO_KSERVE_SMOKE_ADDR")

    with grpc.insecure_channel(target) as channel:
        stub = turbo_analysis_pb2_grpc.AnalysisServiceStub(channel)

        caps = stub.Capabilities(turbo_analysis_pb2.CapabilitiesRequest(), timeout=15.0)
        assert len(caps.bundles) >= 1, f"expected >= 1 bundle, got {len(caps.bundles)}"
        bundle = caps.bundles[0]
        assert bundle.name == "tiny-bert-bundle", f"unexpected bundle name: {bundle.name!r}"
        assert list(bundle.widths) and bundle.widths[0] == 32, f"unexpected widths: {list(bundle.widths)}"

        requests = [
            turbo_analysis_pb2.AnalyzeStreamRequest(
                header=turbo_analysis_pb2.AnalyzeStreamHeader(
                    order=0,
                    max_inflight=4,
                    batch_linger_ms=0,
                )
            ),
            turbo_analysis_pb2.AnalyzeStreamRequest(
                document=turbo_analysis_pb2.AnalyzeDocument(
                    id="py-doc-1",
                    model="tiny-bert-bundle",
                    text=b"hello from python client stub",
                )
            ),
        ]
        responses = list(stub.AnalyzeStream(iter(requests), timeout=15.0))
        assert len(responses) == 1, f"expected 1 response, got {len(responses)}"
        resp = responses[0]
        assert resp.id == "py-doc-1", f"unexpected response id: {resp.id!r}"
        assert resp.WhichOneof("outcome") == "vectors", f"unexpected outcome: {resp}"
        assert resp.vectors.dim == 32, f"unexpected vector dim: {resp.vectors.dim}"
        assert len(resp.vectors.data) == 32 * 4, f"unexpected data len: {len(resp.vectors.data)}"
        print(f"python stub client ok: bundle={bundle.name} dim={resp.vectors.dim}")


if __name__ == "__main__":
    main()
