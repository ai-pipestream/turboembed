#!/usr/bin/env python3
"""Reproducible, hash-verified fetch of inferstream model artifacts.

Downloads the prebuilt ONNX embedding artifacts that the built-in catalog
(config/catalog.toml) resolves for the nvidia arch (backend = "ort") into
models/onnx/<alias>/, verifying a SHA-256 hash for every file against the
committed manifest (models/manifests/embeddings.json). Every URL is pinned
to an exact Hugging Face revision (commit hash), never a floating branch,
so a fetch today and a fetch next year produce byte-identical artifacts —
or fail loudly.

Uses only the Python standard library (no huggingface_hub required).

Usage:
    scripts/fetch_models.py --list
    scripts/fetch_models.py <alias> [<alias> ...]
    scripts/fetch_models.py --all
    scripts/fetch_models.py --all --verify-only
    scripts/fetch_models.py --all --update-manifest [--no-store]

Modes:
    (default)          Download any missing/mismatched files for the given
                       aliases and verify SHA-256. Files already present
                       with a matching hash are skipped (idempotent). A
                       hash mismatch after download is a hard error.
    --verify-only      No network. Check that every manifest file for the
                       given aliases exists on disk with a matching hash;
                       exit non-zero listing anything missing/mismatched.
    --update-manifest  Maintainer mode: resolve each alias's Hugging Face
                       repo to its current main commit, download every
                       file at that pinned revision, compute SHA-256, and
                       rewrite the manifest. Commit the result. With
                       --no-store, files are hashed from the stream and
                       not written to disk.

The manifest also records the Hugging Face repos + pinned revisions the
apple/MLX backend pulls at runtime (mlx_repos); those are informational —
the MLX bridge downloads via huggingface_hub into the HF cache itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "models" / "manifests" / "embeddings.json"
HF_BASE = "https://huggingface.co"
USER_AGENT = "inferstream-fetch-models/1.0"
CHUNK = 1 << 20  # 1 MiB

# ---------------------------------------------------------------------------
# Source of truth for --update-manifest: alias -> ONNX source repo on HF.
# Mirrors the nvidia (backend = "ort") resolutions in config/catalog.toml.
# The sentence-transformers repos ship onnx/model.onnx themselves; the rest
# use Xenova's transformers.js exports; nomic ships its own export.
# ---------------------------------------------------------------------------
ONNX_REPOS: dict[str, str] = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge-small": "Xenova/bge-small-en-v1.5",
    "bge-base": "Xenova/bge-base-en-v1.5",
    "bge-large": "Xenova/bge-large-en-v1.5",
    "bge-m3": "Xenova/bge-m3",
    "e5-small": "Xenova/multilingual-e5-small",
    "e5-base": "Xenova/multilingual-e5-base",
    "e5-large": "Xenova/multilingual-e5-large",
    "gte-small": "Xenova/gte-small",
    "gte-base": "Xenova/gte-base",
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
}

# Files fetched per alias: the ONNX graph (plus its external-data sidecar
# when the repo splits weights out, e.g. Xenova/bge-m3 ships
# onnx/model.onnx + onnx/model.onnx_data), the fast tokenizer, and the model
# config (dims / architecture, kept for reference). --update-manifest lists
# the repo's actual files and picks these, so sidecars are never missed.
ONNX_FILES = ["onnx/model.onnx", "tokenizer.json", "config.json"]
ONNX_SIDECAR_PREFIX = "onnx/model.onnx"  # matches model.onnx_data / .onnx.data

# apple/MLX runtime repos from config/catalog.toml — recorded in the manifest
# (repo + pinned revision) so the expected upstream state is auditable, even
# though the MLX bridge downloads through huggingface_hub at runtime.
MLX_REPOS: dict[str, str] = {
    "minilm": "mlx-community/all-MiniLM-L6-v2-4bit",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
    "bge-small": "mlx-community/bge-small-en-v1.5-4bit",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
    "e5-small": "intfloat/multilingual-e5-small",
    "e5-base": "intfloat/multilingual-e5-base",
    "e5-large": "intfloat/multilingual-e5-large",
    "gte-small": "thenlper/gte-small",
    "gte-base": "thenlper/gte-base",
}


def resolve_url(repo: str, revision: str, path: str) -> str:
    return f"{HF_BASE}/{repo}/resolve/{revision}/{path}"


def http_get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=120)


def repo_info(repo: str) -> tuple[str, list[str]]:
    """(main-branch commit hash, list of files) for a repo via the HF API."""
    with http_get(f"{HF_BASE}/api/models/{repo}") as resp:
        info = json.load(resp)
    sha = info.get("sha")
    if not sha:
        raise RuntimeError(f"HF API returned no commit sha for {repo}")
    files = [s["rfilename"] for s in info.get("siblings", [])]
    return sha, files


def onnx_file_list(repo: str, repo_files: list[str]) -> list[str]:
    """The files we pin for an ONNX alias: ONNX_FILES plus any external-data
    sidecars next to onnx/model.onnx (e.g. onnx/model.onnx_data)."""
    sidecars = sorted(
        f for f in repo_files
        if f.startswith(ONNX_SIDECAR_PREFIX) and f != "onnx/model.onnx"
        # exclude quantized variants like onnx/model.onnx -> model_int8.onnx
        # (those don't share the model.onnx prefix, but be safe about .onnx_data)
        and ("data" in f[len(ONNX_SIDECAR_PREFIX):])
    )
    missing = [f for f in ONNX_FILES if f not in repo_files]
    if missing:
        raise RuntimeError(f"{repo}: required file(s) not in repo: {missing}")
    return [ONNX_FILES[0], *sidecars, *ONNX_FILES[1:]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def stream_download(url: str, dest: Path | None) -> tuple[str, int]:
    """Download url, returning (sha256, size). If dest is given, write the
    file atomically (tmp file in the same directory, then rename)."""
    h = hashlib.sha256()
    size = 0
    tmp = None
    out = None
    try:
        if dest is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd = tempfile.NamedTemporaryFile(
                dir=dest.parent, prefix=dest.name + ".", suffix=".part", delete=False
            )
            tmp = Path(fd.name)
            out = fd
        with http_get(url) as resp:
            while chunk := resp.read(CHUNK):
                h.update(chunk)
                size += len(chunk)
                if out is not None:
                    out.write(chunk)
        if out is not None:
            out.close()
            out = None
            tmp.replace(dest)
            tmp = None
    finally:
        if out is not None:
            out.close()
        if tmp is not None and tmp.exists():
            tmp.unlink()
    return h.hexdigest(), size


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(
            f"{path}: unsupported schema_version {manifest.get('schema_version')!r}"
        )
    return manifest


def human(size: int) -> str:
    if size >= 1 << 30:
        return f"{size / (1 << 30):.2f} GiB"
    if size >= 1 << 20:
        return f"{size / (1 << 20):.1f} MiB"
    return f"{size} B"


def select_aliases(args, known: dict) -> list[str]:
    if args.all:
        return sorted(known)
    if not args.aliases:
        print("error: no aliases given (use --all or --list)", file=sys.stderr)
        sys.exit(2)
    bad = [a for a in args.aliases if a not in known]
    if bad:
        print(
            f"error: unknown alias(es): {', '.join(bad)}\n"
            f"known: {', '.join(sorted(known))}",
            file=sys.stderr,
        )
        sys.exit(2)
    return list(dict.fromkeys(args.aliases))


def cmd_list(manifest_path: Path) -> int:
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        print(f"{'alias':<18} {'repo':<42} revision")
        for alias, entry in sorted(manifest["models"].items()):
            print(f"{alias:<18} {entry['repo']:<42} {entry['revision'][:12]}")
    else:
        print(f"{'alias':<18} repo (manifest not generated yet)")
        for alias, repo in sorted(ONNX_REPOS.items()):
            print(f"{alias:<18} {repo}")
    return 0


def cmd_verify(aliases: list[str], manifest: dict, root: Path) -> int:
    failures = []
    for alias in aliases:
        entry = manifest["models"][alias]
        dest_root = root / entry["dest"]
        for f in entry["files"]:
            path = dest_root / f["path"]
            label = f"{alias}: {path.relative_to(root)}"
            if not path.exists():
                failures.append(f"{label} — MISSING")
                print(f"  missing   {label}")
                continue
            actual = sha256_file(path)
            if actual != f["sha256"]:
                failures.append(
                    f"{label} — sha256 mismatch (expected {f['sha256']}, got {actual})"
                )
                print(f"  MISMATCH  {label}")
            else:
                print(f"  ok        {label}")
    if failures:
        print(f"\nverify FAILED ({len(failures)} problem(s)):", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        return 1
    print(f"\nverify OK — all files present with matching SHA-256.")
    return 0


def cmd_fetch(aliases: list[str], manifest: dict, root: Path) -> int:
    downloaded = skipped = 0
    for alias in aliases:
        entry = manifest["models"][alias]
        dest_root = root / entry["dest"]
        print(f"--- {alias}  <-  {entry['repo']}@{entry['revision'][:12]}  ->  "
              f"{dest_root.relative_to(root)} ---")
        for f in entry["files"]:
            path = dest_root / f["path"]
            if path.exists():
                if sha256_file(path) == f["sha256"]:
                    print(f"  ok (cached)  {f['path']}  [{human(f['size'])}]")
                    skipped += 1
                    continue
                print(f"  stale hash, re-downloading  {f['path']}")
            url = resolve_url(entry["repo"], entry["revision"], f["path"])
            print(f"  downloading  {f['path']}  [{human(f['size'])}] ...", flush=True)
            try:
                actual, size = stream_download(url, path)
            except urllib.error.HTTPError as e:
                print(f"error: {url}: HTTP {e.code} {e.reason}", file=sys.stderr)
                return 1
            if actual != f["sha256"]:
                path.unlink(missing_ok=True)
                print(
                    f"error: SHA-256 mismatch for {alias}/{f['path']}\n"
                    f"  expected {f['sha256']}\n"
                    f"  got      {actual}\n"
                    f"  url      {url}\n"
                    "The file was deleted. If upstream legitimately changed, "
                    "re-pin with --update-manifest and review the diff.",
                    file=sys.stderr,
                )
                return 1
            if size != f["size"]:
                print(
                    f"error: size mismatch for {alias}/{f['path']} "
                    f"(expected {f['size']}, got {size})",
                    file=sys.stderr,
                )
                return 1
            print(f"  verified     {f['path']}  sha256={actual[:16]}…")
            downloaded += 1
    print(f"\nDone: {downloaded} downloaded, {skipped} already present and verified.")
    print("Add the aliases to `serve` in config/nvidia.toml and restart;")
    print("verify with: scripts/smoke-embeddings.sh <host:port> <bearer-token>")
    return 0


def cmd_update_manifest(
    aliases: list[str], manifest_path: Path, root: Path, store: bool
) -> int:
    # Start from the existing manifest so a partial update (subset of
    # aliases) preserves the untouched entries.
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
    else:
        manifest = {
            "schema_version": 1,
            "_comment": (
                "SHA-256 manifest for inferstream model artifacts. Generated by "
                "scripts/fetch_models.py --update-manifest; do not edit hashes by "
                "hand. Revisions are exact HF commit hashes (never floating "
                "branches). mlx_repos records the repos+revisions the apple/MLX "
                "runtime pulls via huggingface_hub at runtime (informational)."
            ),
            "models": {},
            "mlx_repos": {},
        }

    for alias in aliases:
        repo = ONNX_REPOS[alias]
        print(f"--- pinning {alias}  <-  {repo} ---")
        revision, repo_files = repo_info(repo)
        print(f"  revision {revision}")
        dest = f"models/onnx/{alias}"
        files = []
        for rel in onnx_file_list(repo, repo_files):
            url = resolve_url(repo, revision, rel)
            target = (root / dest / rel) if store else None
            print(f"  hashing {rel} ...", flush=True)
            try:
                digest, size = stream_download(url, target)
            except urllib.error.HTTPError as e:
                print(f"error: {url}: HTTP {e.code} {e.reason}", file=sys.stderr)
                return 1
            print(f"    sha256={digest}  size={human(size)}")
            files.append({"path": rel, "sha256": digest, "size": size})
        manifest["models"][alias] = {
            "repo": repo,
            "revision": revision,
            "dest": dest,
            "files": files,
        }
        if alias in MLX_REPOS:
            mlx_repo = MLX_REPOS[alias]
            mlx_rev, _ = repo_info(mlx_repo)
            manifest["mlx_repos"][alias] = {"repo": mlx_repo, "revision": mlx_rev}
            print(f"  mlx runtime repo {mlx_repo}@{mlx_rev[:12]}")

    manifest["models"] = dict(sorted(manifest["models"].items()))
    manifest["mlx_repos"] = dict(sorted(manifest["mlx_repos"].items()))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"\nManifest written: {manifest_path.relative_to(root)} — review and commit it.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hash-verified fetch of inferstream embedding model artifacts.",
    )
    parser.add_argument("aliases", nargs="*", help="catalog aliases to fetch")
    parser.add_argument("--all", action="store_true", help="operate on every alias")
    parser.add_argument("--list", action="store_true", help="list aliases and sources")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="verify existing files against the manifest; no downloads",
    )
    parser.add_argument(
        "--update-manifest", action="store_true",
        help="maintainer mode: re-pin revisions, download, hash, rewrite manifest",
    )
    parser.add_argument(
        "--no-store", action="store_true",
        help="with --update-manifest: hash from the stream without writing files",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help=f"manifest path (default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT,
        help="repo root that dest paths are relative to (default: script's repo)",
    )
    args = parser.parse_args()

    if args.list:
        return cmd_list(args.manifest)

    if args.update_manifest:
        aliases = select_aliases(args, ONNX_REPOS)
        return cmd_update_manifest(aliases, args.manifest, args.root, not args.no_store)

    if not args.manifest.exists():
        print(
            f"error: manifest not found: {args.manifest}\n"
            "Generate it with --update-manifest (maintainers) or fetch it from git.",
            file=sys.stderr,
        )
        return 1
    manifest = load_manifest(args.manifest)
    aliases = select_aliases(args, manifest["models"])

    if args.verify_only:
        return cmd_verify(aliases, manifest, args.root)
    return cmd_fetch(aliases, manifest, args.root)


if __name__ == "__main__":
    sys.exit(main())
