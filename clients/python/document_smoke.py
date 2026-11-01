#!/usr/bin/env python3
"""One document through AnalyzeDocumentProgressive (default-v1), read back with
InspectDocumentGraph in code points, which is how a Python str is indexed.

Every span must lie within raw_text, every token and sentence slice must equal
the text the annotation carries, and the profile's layers must come back under
the ids Capabilities lists. Uses only the published turboembed_proto wheel.
"""

from __future__ import annotations

import os
import sys

import grpc
from turboembed_proto import turbo_analysis_pb2, turbo_analysis_pb2_grpc, turbo_typed_annotations_pb2

MODEL = "tiny-bert-bundle"
PROFILE = "default-v1"
PLACE = "Zürich"
# Latin with diacritics, an emoji outside the BMP, CJK and a gazetteer place
# name after both, so a span past them is wrong in every unit but the right one.
TEXT = f"Grüße 😀 aus dem Büro.\n\n東京の会議は 🙂 午後です。\n\nThe team meets in {PLACE} next week."


def has_layer(doc, layer_id: str) -> bool:
    # A keyed id such as "turbo:embeddings<turbo:chunks>" names the layer and
    # the layer it hangs on.
    base, _, rest = layer_id.partition("<")
    anchor = rest[:-1] if rest else None
    for layer in doc.layers:
        if layer.id == layer_id:
            return True
        if anchor is not None and layer.id == base and layer.HasField("anchor_layer") and layer.anchor_layer == anchor:
            return True
    return False


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TURBO_KSERVE_SMOKE_ADDR", "")
    if not target:
        raise SystemExit("Expected server address as argv[1] or TURBO_KSERVE_SMOKE_ADDR")
    code_points = turbo_typed_annotations_pb2.OFFSET_MODE_CODE_POINT

    with grpc.insecure_channel(target) as channel:
        stub = turbo_analysis_pb2_grpc.AnalysisServiceStub(channel)
        caps = stub.Capabilities(turbo_analysis_pb2.CapabilitiesRequest(), timeout=15.0)
        profiles = [p for p in caps.profiles if p.name == PROFILE]
        assert profiles, f"Capabilities lists no {PROFILE} profile"

        request = turbo_analysis_pb2.AnalyzePipelineRequest(
            doc_id="py-document", text=TEXT, offsets=code_points, profile=PROFILE, model=MODEL
        )
        graph = None
        for event in stub.AnalyzeDocumentProgressive(request, timeout=60.0):
            kind = event.WhichOneof("event")
            if kind == "step_failed":
                failed = event.step_failed
                raise AssertionError(f"step {failed.step_id} failed: {failed.failure.message}")
            if kind == "complete":
                graph = event.complete.graph
        assert graph is not None, "the stream ended without the document"

        doc = stub.InspectDocumentGraph(
            turbo_typed_annotations_pb2.InspectDocumentGraphRequest(doc_id="py-document", graph=graph, offsets=code_points),
            timeout=60.0,
        )
    text = doc.raw_text
    assert text == TEXT, "raw_text is not the text sent"
    for layer_id in profiles[0].layers:
        assert has_layer(doc, layer_id), f"{PROFILE} lists {layer_id} but the document has no such layer"

    spans = tokens = sentences = 0
    entities = []
    for layer in doc.layers:
        assert layer.status == turbo_typed_annotations_pb2.LAYER_WIRE_STATUS_OK, (
            f"{layer.id}: status {layer.status} {layer.status_detail}"
        )
        assert not layer.id.startswith("opennlp:"), f"a layer id in the opennlp: namespace: {layer.id}"
        for a in layer.annotations:
            if not a.spans:
                continue
            for s in a.spans:
                assert s.start <= s.end <= len(text), f"{layer.id}: span {s.start}..{s.end} outside {len(text)}"
            got = text[a.spans[0].start : a.spans[-1].end]
            spans += 1
            kind = a.WhichOneof("value")
            if kind == "token":
                tokens += 1
                assert got == a.token.surface, f"token slice {got!r} is not its surface {a.token.surface!r}"
            elif kind == "sentence":
                sentences += 1
                assert got == a.sentence.text, f"sentence slice {got!r} is not its text {a.sentence.text!r}"
            elif kind == "entity":
                entities.append(got)
    assert tokens and sentences, f"no tokens or no sentences: {tokens}, {sentences}"
    assert PLACE in entities, f"no entity slices to {PLACE}; entities {entities}"
    print(
        f"python document client ok: layers={len(doc.layers)} spans={spans} "
        f"tokens={tokens} sentences={sentences} entities={entities}"
    )


if __name__ == "__main__":
    main()
