#!/usr/bin/env python3
"""Minimal Python client smoke test using only the published turboembed_proto wheel (#1297).

Also sends a heading and a list as a StructureSeed to AnalyzeDocumentProgressive.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys

import grpc
import turboembed_proto
from turboembed_proto import turbo_analysis_pb2, turbo_analysis_pb2_grpc, turbo_typed_annotations_pb2


def import_every_stub() -> None:
    # Each generated module checks its grpcio or protobuf runtime at import.
    for mod in pkgutil.iter_modules(turboembed_proto.__path__):
        importlib.import_module(f"turboembed_proto.{mod.name}")


def block(text: str, kind: int, covers: str, level: int, parent: int | None = None):
    start = text.index(covers)
    b = turbo_analysis_pb2.StructureBlock(kind=kind, start=start, end=start + len(covers), level=level)
    if parent is not None:
        b.parent = parent
    return b


def seed_request(seed):
    return turbo_analysis_pb2.AnalyzePipelineRequest(
        doc_id="py-seed",
        offsets=turbo_typed_annotations_pb2.OFFSET_MODE_CODE_POINT,
        steps=["turbo-sentdetect:rules", "turbo:structure-boundaries"],
        structure_seed=seed,
    )


def structure_seed(stub) -> None:
    """A heading and a two-item list as a StructureSeed, offsets in the code
    points a Python str indexes; then the same list overlapping, refused."""
    pb = turbo_analysis_pb2
    text = "# Caf\u00e9\n\n- \U0001f600 one\n- two\n"
    seed = pb.StructureSeed(
        text=text,
        blocks=[
            block(text, pb.STRUCTURE_BLOCK_KIND_HEADING, "# Caf\u00e9", 1),
            block(text, pb.STRUCTURE_BLOCK_KIND_LIST, "- \U0001f600 one\n- two", 1),
            block(text, pb.STRUCTURE_BLOCK_KIND_LIST_ITEM, "- \U0001f600 one", 1, 1),
            block(text, pb.STRUCTURE_BLOCK_KIND_LIST_ITEM, "- two", 1, 1),
        ],
    )
    complete = None
    for event in stub.AnalyzeDocumentProgressive(seed_request(seed), timeout=15.0):
        kind = event.WhichOneof("event")
        assert kind != "step_failed", f"step failed: {event.step_failed}"
        if kind == "complete":
            complete = event.complete
    assert complete is not None and complete.source_text == text, "the seeded stream ended without its document"
    structure = [layer for layer in complete.layers if layer.layer == "turbo:structure"]
    assert len(structure) == 1 and structure[0].items == 4, f"unexpected structure layer: {structure}"

    seed.blocks[3].start = seed.blocks[2].start + 1
    try:
        list(stub.AnalyzeDocumentProgressive(seed_request(seed), timeout=15.0))
    except grpc.RpcError as e:
        assert e.code() == grpc.StatusCode.INVALID_ARGUMENT, f"unexpected refusal code: {e.code()}"
        assert "block 3 overlaps" in e.details(), f"unexpected refusal: {e.details()}"
    else:
        raise AssertionError("an overlapping seed was accepted")
    print(f"python structure seed ok: blocks={structure[0].items}")


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TURBO_KSERVE_SMOKE_ADDR", "")
    if not target:
        raise SystemExit("Expected server address as argv[1] or TURBO_KSERVE_SMOKE_ADDR")
    import_every_stub()

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
        structure_seed(stub)


if __name__ == "__main__":
    main()
