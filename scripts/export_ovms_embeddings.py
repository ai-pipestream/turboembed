#!/usr/bin/env python3
"""Export + SHA-256-pin OVMS embedding pipeline artifacts (Intel arch).

Produces, for each catalog alias, the artifacts an OpenVINO Model Server
DAG pipeline needs on an Intel host (see docs/adding-ovms-embedding-pipelines.md):

    <out>/tokenizer_<name>/1/openvino_tokenizer.{xml,bin}   # openvino_tokenizers
    <out>/embedding_<name>/1/openvino_model.{xml,bin}       # FP16 OpenVINO IR
    <hf-out>/hf_tokenizer_<name>/tokenizer.json             # local Tokenize RPCs

The exported graph bakes pooling (mean or CLS, per model card) and L2
normalization in, exposing exactly the contract the inferstream `ovms`
backend and the existing minilm/mpnet pipelines use:

    inputs:  input_ids, attention_mask          (i64, dynamic [batch, seq])
    outputs: token_embeddings, sentence_embedding

Every export is pinned to an exact Hugging Face revision (commit hash) and
every produced file's SHA-256 is recorded in / verified against
models/manifests/ovms-embeddings.json. Exports are deterministic for a given
(model revision, package versions) pair; the manifest records both so a
mismatch is a signal, not noise.

This script needs a Python environment with: torch (CPU is fine),
transformers, openvino, openvino-tokenizers, sentencepiece. Example setup:

    uv venv --python 3.12 /tmp/ov-export-venv
    uv pip install --python /tmp/ov-export-venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch transformers openvino openvino-tokenizers sentencepiece protobuf

Usage:
    scripts/export_ovms_embeddings.py --list
    scripts/export_ovms_embeddings.py <alias> [...] [--out DIR] [--hf-out DIR]
    scripts/export_ovms_embeddings.py --all --verify-only [--out DIR ...]
    scripts/export_ovms_embeddings.py <alias> ... --update-manifest

Modes:
    (default)          Export any alias whose artifacts are missing or hash-
                       mismatched, then verify SHA-256 against the manifest.
                       Idempotent: matching files are left untouched.
    --verify-only      No export, no network: check every manifest file
                       exists on disk with a matching hash.
    --update-manifest  Maintainer mode: resolve each alias's HF repo to its
                       current main commit, export, hash, and rewrite the
                       manifest entry (including the export environment).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "models" / "manifests" / "ovms-embeddings.json"
DEFAULT_OUT = Path("/work/models/ovms-embedder")
DEFAULT_HF_OUT = Path.home() / "ovms-models"
HF_BASE = "https://huggingface.co"
CHUNK = 1 << 20

# ---------------------------------------------------------------------------
# Alias -> export spec. `name` is the OVMS model-name suffix (tokenizer_<name>,
# embedding_<name>, <name>_pipeline). Pooling follows each model card:
# BGE = CLS, everything else here = mean. All are L2-normalized in-graph.
# ---------------------------------------------------------------------------
SPECS: dict[str, dict] = {
    "bge-small":  {"repo": "BAAI/bge-small-en-v1.5",            "name": "bge_small",  "pooling": "cls",  "dims": 384},
    "bge-base":   {"repo": "BAAI/bge-base-en-v1.5",             "name": "bge_base",   "pooling": "cls",  "dims": 768},
    "bge-large":  {"repo": "BAAI/bge-large-en-v1.5",            "name": "bge_large",  "pooling": "cls",  "dims": 1024},
    "bge-m3":     {"repo": "BAAI/bge-m3",                       "name": "bge_m3",     "pooling": "cls",  "dims": 1024},
    "e5-small":   {"repo": "intfloat/multilingual-e5-small",    "name": "e5_small",   "pooling": "mean", "dims": 384},
    "e5-base":    {"repo": "intfloat/multilingual-e5-base",     "name": "e5_base",    "pooling": "mean", "dims": 768},
    "e5-large":   {"repo": "intfloat/multilingual-e5-large",    "name": "e5_large",   "pooling": "mean", "dims": 1024},
    "minilm-l12": {"repo": "sentence-transformers/all-MiniLM-L12-v2", "name": "minilm_l12", "pooling": "mean", "dims": 384},
    "gte-small":  {"repo": "thenlper/gte-small",                "name": "gte_small",  "pooling": "mean", "dims": 384},
    "gte-base":   {"repo": "thenlper/gte-base",                 "name": "gte_base",   "pooling": "mean", "dims": 768},
    # NomicBERT's remote-code modeling cannot be torch-traced (data-dependent
    # rotary-cache branches fail trace verification on both transformers 5.x
    # and 4.48). Instead we import Nomic's own official ONNX export into
    # OpenVINO and graft mean pooling + L2 normalize onto the graph. The
    # resulting IR keeps the token_type_ids input, which the DAG maps from
    # the tokenizer alongside input_ids/attention_mask.
    "nomic-embed-text": {
        "repo": "nomic-ai/nomic-embed-text-v1.5", "name": "nomic_embed_text",
        "pooling": "mean", "dims": 768, "source": "onnx", "onnx_file": "onnx/model.onnx",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def artifact_paths(name: str, out: Path, hf_out: Path) -> dict[str, Path]:
    return {
        f"tokenizer_{name}/1/openvino_tokenizer.xml": out / f"tokenizer_{name}/1/openvino_tokenizer.xml",
        f"tokenizer_{name}/1/openvino_tokenizer.bin": out / f"tokenizer_{name}/1/openvino_tokenizer.bin",
        f"embedding_{name}/1/openvino_model.xml": out / f"embedding_{name}/1/openvino_model.xml",
        f"embedding_{name}/1/openvino_model.bin": out / f"embedding_{name}/1/openvino_model.bin",
        f"hf_tokenizer_{name}/tokenizer.json": hf_out / f"hf_tokenizer_{name}/tokenizer.json",
    }


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"{path}: unsupported schema_version")
    return manifest


def repo_head(repo: str) -> str:
    import urllib.request
    req = urllib.request.Request(
        f"{HF_BASE}/api/models/{repo}",
        headers={"User-Agent": "inferstream-export-ovms/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        info = json.load(resp)
    sha = info.get("sha")
    if not sha:
        raise RuntimeError(f"HF API returned no commit sha for {repo}")
    return sha


# ---------------------------------------------------------------------------
# Export (heavy imports deferred so --list / --verify-only stay stdlib-only)
# ---------------------------------------------------------------------------

def convert_from_onnx(spec: dict, revision: str):
    """Import an upstream ONNX export into OpenVINO and graft mean pooling +
    L2 normalization onto last_hidden_state (for architectures whose torch
    modeling code cannot be traced, e.g. NomicBERT)."""
    import urllib.request

    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops

    if spec["pooling"] != "mean":
        raise RuntimeError(f"onnx source only implements mean pooling")
    url = f"{HF_BASE}/{spec['repo']}/resolve/{revision}/{spec['onnx_file']}"
    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "model.onnx"
        req = urllib.request.Request(url, headers={"User-Agent": "inferstream-export-ovms/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, onnx_path.open("wb") as f:
            shutil.copyfileobj(resp, f, CHUNK)
        model = ov.Core().read_model(onnx_path)
    lhs = model.outputs[0].get_node().input_value(0)  # last_hidden_state producer
    mask = next(p for p in model.get_parameters()
                if p.get_friendly_name() == "attention_mask")
    maskf = ops.unsqueeze(ops.convert(mask, "f32"), ops.constant(np.int64(2)))
    summed = ops.reduce_sum(ops.multiply(lhs, maskf), ops.constant(np.int64([1])))
    counts = ops.maximum(ops.reduce_sum(maskf, ops.constant(np.int64([1]))),
                         ops.constant(np.float32(1e-9)))
    sent = ops.normalize_l2(ops.divide(summed, counts),
                            ops.constant(np.int64([1])), 1e-12, "max")
    model.add_results([ops.result(sent)])
    return model


def export_alias(alias: str, revision: str, out: Path, hf_out: Path) -> None:
    import openvino as ov
    import torch
    from openvino_tokenizers import convert_tokenizer
    from transformers import AutoModel, AutoTokenizer

    spec = SPECS[alias]
    name, pooling = spec["name"], spec["pooling"]
    repo = spec["repo"]
    print(f"--- exporting {alias} ({repo}@{revision[:12]}, {pooling} pooling) ---")

    hf_tok = AutoTokenizer.from_pretrained(repo, revision=revision)

    if spec.get("source") == "onnx":
        ov_model = convert_from_onnx(spec, revision)
        example = hf_tok(["inferstream export smoke"], return_tensors="pt")
        return finish_export(alias, spec, ov_model, hf_tok, example, out, hf_out)

    # dtype=float32 explicitly: transformers 5.x defaults to the checkpoint
    # dtype, and some repos (e.g. thenlper/gte-*) ship fp16 weights — tracing
    # those yields an fp16-in/out graph, and OVMS then answers Embed with an
    # FP16 tensor the inferstream backend rejects. FP16 compression of the
    # IR happens at save_model time regardless; graph IO must stay FP32.
    model = AutoModel.from_pretrained(repo, revision=revision, dtype=torch.float32)
    model = model.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            out = self.inner(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = out.last_hidden_state
            if pooling == "cls":
                pooled = token_embeddings[:, 0]
            else:  # mean pooling over attention-masked tokens
                mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
                pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            sentence_embedding = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return token_embeddings, sentence_embedding

    example = hf_tok(["inferstream export smoke"], return_tensors="pt")
    example_input = {
        "input_ids": example["input_ids"].to(torch.int64),
        "attention_mask": example["attention_mask"].to(torch.int64),
    }
    with torch.no_grad():
        ov_model = ov.convert_model(
            Wrapper(model),
            example_input=example_input,
            input=[
                ("input_ids", ov.PartialShape([-1, -1]), ov.Type.i64),
                ("attention_mask", ov.PartialShape([-1, -1]), ov.Type.i64),
            ],
        )
    finish_export(alias, spec, ov_model, hf_tok, example, out, hf_out)


def finish_export(alias, spec, ov_model, hf_tok, example, out, hf_out) -> None:
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer

    name = spec["name"]
    ov_model.outputs[0].get_tensor().set_names({"token_embeddings"})
    ov_model.outputs[1].get_tensor().set_names({"sentence_embedding"})

    ov_tok = convert_tokenizer(hf_tok)
    # convert_tokenizer auto-numbers the string input (Parameter_NNNNN); the
    # OVMS DAG template maps the request's "strings" to an input named
    # "Parameter_1" (as the original minilm/mpnet tokenizers expose), so pin it.
    ov_tok.inputs[0].get_node().set_friendly_name("Parameter_1")
    ov_tok.inputs[0].get_tensor().set_names({"Parameter_1"})
    out_names = {n for o in ov_tok.outputs for n in o.get_names()}
    if not {"input_ids", "attention_mask"} <= out_names:
        raise RuntimeError(
            f"{alias}: converted tokenizer outputs {out_names}, need input_ids+attention_mask"
        )

    emb_dir = out / f"embedding_{name}/1"
    tok_dir = out / f"tokenizer_{name}/1"
    hf_dir = hf_out / f"hf_tokenizer_{name}"
    for d in (emb_dir, tok_dir, hf_dir):
        d.mkdir(parents=True, exist_ok=True)
    # FP16 IR (compress_to_fp16 default True) — same as the live minilm/mpnet.
    ov.save_model(ov_model, emb_dir / "openvino_model.xml")
    ov.save_model(ov_tok, tok_dir / "openvino_tokenizer.xml", compress_to_fp16=False)

    with tempfile.TemporaryDirectory() as td:
        hf_tok.save_pretrained(td)
        src = Path(td) / "tokenizer.json"
        if not src.exists():
            raise RuntimeError(f"{alias}: HF tokenizer did not produce tokenizer.json")
        shutil.copy2(src, hf_dir / "tokenizer.json")

    # Sanity: run the exported graph on CPU and check dims + unit norm.
    compiled = ov.compile_model(ov.Core().read_model(emb_dir / "openvino_model.xml"), "CPU")
    feed = {
        "input_ids": example["input_ids"].numpy(),
        "attention_mask": example["attention_mask"].numpy(),
    }
    input_names = {n for i in compiled.inputs for n in i.get_names()}
    if "token_type_ids" in input_names:
        import numpy as np
        feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
    res = compiled(feed)
    vec = res["sentence_embedding"]
    norm = float((vec ** 2).sum() ** 0.5)
    if vec.shape[-1] != spec["dims"]:
        raise RuntimeError(f"{alias}: expected {spec['dims']} dims, got {vec.shape[-1]}")
    if not 0.99 < norm < 1.01:
        raise RuntimeError(f"{alias}: sentence_embedding not L2-normalized (|v|={norm:.4f})")
    print(f"  export OK: dims={vec.shape[-1]} |v|={norm:.4f}")


def export_environment() -> dict:
    import openvino
    import openvino_tokenizers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "openvino": openvino.__version__,
        "openvino_tokenizers": openvino_tokenizers.__version__,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def select_aliases(args, known) -> list[str]:
    if args.all:
        return sorted(known)
    if not args.aliases:
        print("error: no aliases given (use --all or --list)", file=sys.stderr)
        sys.exit(2)
    bad = [a for a in args.aliases if a not in known]
    if bad:
        print(f"error: unknown alias(es): {', '.join(bad)}\n"
              f"known: {', '.join(sorted(known))}", file=sys.stderr)
        sys.exit(2)
    return list(dict.fromkeys(args.aliases))


def verify_alias(alias: str, entry: dict, out: Path, hf_out: Path) -> list[str]:
    problems = []
    paths = artifact_paths(entry["name"], out, hf_out)
    for f in entry["files"]:
        path = paths.get(f["path"])
        if path is None:
            problems.append(f"{alias}: unknown manifest path {f['path']}")
            continue
        if not path.exists():
            problems.append(f"{alias}: {path} — MISSING")
            print(f"  missing   {alias}: {f['path']}")
            continue
        actual = sha256_file(path)
        if actual != f["sha256"]:
            problems.append(
                f"{alias}: {path} — sha256 mismatch (expected {f['sha256']}, got {actual})")
            print(f"  MISMATCH  {alias}: {f['path']}")
        else:
            print(f"  ok        {alias}: {f['path']}")
    return problems


def cmd_verify(aliases, manifest, out, hf_out) -> int:
    failures = []
    for alias in aliases:
        failures += verify_alias(alias, manifest["models"][alias], out, hf_out)
    if failures:
        print(f"\nverify FAILED ({len(failures)} problem(s)):", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        return 1
    print("\nverify OK — all OVMS artifacts present with matching SHA-256.")
    return 0


def cmd_fetch(aliases, manifest, out, hf_out) -> int:
    for alias in aliases:
        entry = manifest["models"][alias]
        paths = artifact_paths(entry["name"], out, hf_out)
        stale = [
            f["path"] for f in entry["files"]
            if not paths[f["path"]].exists() or sha256_file(paths[f["path"]]) != f["sha256"]
        ]
        if not stale:
            print(f"--- {alias}: all artifacts present and verified (skipping export) ---")
            continue
        print(f"--- {alias}: exporting ({len(stale)} file(s) missing/stale) ---")
        export_alias(alias, entry["revision"], out, hf_out)
        problems = verify_alias(alias, entry, out, hf_out)
        if problems:
            print(
                f"error: {alias}: freshly exported artifacts do not match the "
                "manifest. Either the pinned HF revision changed on disk or your "
                "export package versions differ from manifest export_environment. "
                "Inspect, and re-pin with --update-manifest only if intentional.",
                file=sys.stderr,
            )
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 1
    print("\nDone. Register new pipelines in the OVMS config and smoke every alias")
    print("(docs/adding-ovms-embedding-pipelines.md).")
    return 0


def cmd_update_manifest(aliases, manifest_path, out, hf_out) -> int:
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
    else:
        manifest = {
            "schema_version": 1,
            "_comment": (
                "SHA-256 manifest for Intel OVMS embedding pipeline artifacts "
                "(OpenVINO IR + openvino_tokenizers + HF tokenizer.json), as "
                "exported by scripts/export_ovms_embeddings.py. Revisions are "
                "exact HF commit hashes. Hashes are of the FP16 IR produced with "
                "the recorded export_environment; a different toolchain may "
                "produce different (still correct) bytes — verify deliberately."
            ),
            "models": {},
        }
    for alias in aliases:
        spec = SPECS[alias]
        revision = repo_head(spec["repo"])
        print(f"--- pinning {alias} <- {spec['repo']} @ {revision} ---")
        export_alias(alias, revision, out, hf_out)
        paths = artifact_paths(spec["name"], out, hf_out)
        files = []
        for rel, path in paths.items():
            digest = sha256_file(path)
            size = path.stat().st_size
            print(f"  {rel}: sha256={digest[:16]}… size={size}")
            files.append({"path": rel, "sha256": digest, "size": size})
        manifest["models"][alias] = {
            "repo": spec["repo"],
            "revision": revision,
            "name": spec["name"],
            "pipeline": f"{spec['name']}_pipeline",
            "pooling": spec["pooling"],
            "dims": spec["dims"],
            "export_command": (
                f"scripts/export_ovms_embeddings.py {alias} "
                f"--out {out} --hf-out {hf_out}"
            ),
            "export_environment": export_environment(),
            "files": files,
        }
    manifest["models"] = dict(sorted(manifest["models"].items()))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"\nManifest written: {manifest_path} — review and commit it.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export + SHA-256-verify OVMS embedding pipeline artifacts (Intel)."
    )
    parser.add_argument("aliases", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"OVMS model dir (default: {DEFAULT_OUT})")
    parser.add_argument("--hf-out", type=Path, default=DEFAULT_HF_OUT,
                        help=f"dir for hf_tokenizer_<name>/ (default: {DEFAULT_HF_OUT})")
    args = parser.parse_args()

    if args.list:
        print(f"{'alias':<12} {'repo':<42} {'pooling':<8} dims")
        for alias, spec in sorted(SPECS.items()):
            print(f"{alias:<12} {spec['repo']:<42} {spec['pooling']:<8} {spec['dims']}")
        return 0

    if args.update_manifest:
        aliases = select_aliases(args, SPECS)
        return cmd_update_manifest(aliases, args.manifest, args.out, args.hf_out)

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}\n"
              "Generate it with --update-manifest (maintainers) or fetch it from git.",
              file=sys.stderr)
        return 1
    manifest = load_manifest(args.manifest)
    aliases = select_aliases(args, manifest["models"])
    if args.verify_only:
        return cmd_verify(aliases, manifest, args.out, args.hf_out)
    return cmd_fetch(aliases, manifest, args.out, args.hf_out)


if __name__ == "__main__":
    sys.exit(main())
