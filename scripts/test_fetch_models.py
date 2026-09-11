"""Offline unit tests for scripts/fetch_models.py and the committed
SHA-256 manifest (models/manifests/embeddings.json).

Run with:  make test-fetch
      or:  python3 -m unittest discover -s scripts -p 'test_*.py'

No network access: verification and idempotence are exercised against tiny
temp-dir fixtures, and the real manifest is checked structurally and against
config/catalog.toml.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_models  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "models" / "manifests" / "embeddings.json"
CATALOG_PATH = REPO_ROOT / "config" / "catalog.toml"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ManifestStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fetch_models.load_manifest(MANIFEST_PATH)

    def test_schema_version(self):
        self.assertEqual(self.manifest["schema_version"], 1)

    def test_covers_every_script_alias(self):
        self.assertEqual(
            set(self.manifest["models"]), set(fetch_models.ONNX_REPOS),
            "manifest aliases must match ONNX_REPOS in fetch_models.py — "
            "run --update-manifest for the missing ones",
        )

    def test_entries_are_pinned_and_hashed(self):
        for alias, entry in self.manifest["models"].items():
            with self.subTest(alias=alias):
                self.assertEqual(entry["repo"], fetch_models.ONNX_REPOS[alias])
                self.assertRegex(
                    entry["revision"], HEX40,
                    "revision must be an exact HF commit hash, not a branch",
                )
                self.assertEqual(entry["dest"], f"models/onnx/{alias}")
                paths = [f["path"] for f in entry["files"]]
                for required in fetch_models.ONNX_FILES:
                    self.assertIn(required, paths)
                for f in entry["files"]:
                    self.assertRegex(f["sha256"], HEX64)
                    self.assertGreater(f["size"], 0)

    def test_external_data_sidecars_present(self):
        # These repos split ONNX weights into an external-data file; a
        # manifest without it would fetch an unloadable graph.
        for alias in ("bge-m3", "e5-large"):
            paths = [f["path"] for f in self.manifest["models"][alias]["files"]]
            self.assertIn("onnx/model.onnx_data", paths, alias)

    def test_mlx_repos_recorded(self):
        for alias, entry in self.manifest["mlx_repos"].items():
            with self.subTest(alias=alias):
                self.assertEqual(entry["repo"], fetch_models.MLX_REPOS[alias])
                self.assertRegex(entry["revision"], HEX40)


class ManifestMatchesCatalog(unittest.TestCase):
    """Every nvidia ORT alias in config/catalog.toml that points into
    models/onnx/ must be fetchable from the manifest, at the same path."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = fetch_models.load_manifest(MANIFEST_PATH)
        with CATALOG_PATH.open("rb") as f:
            cls.catalog = tomllib.load(f)

    def test_catalog_nvidia_ort_aliases_covered(self):
        for alias, tables in self.catalog["models"].items():
            nvidia = tables.get("nvidia")
            if not nvidia or nvidia.get("backend") != "ort":
                continue
            path = nvidia.get("path", "")
            if not path.startswith("models/onnx/"):
                continue  # e.g. minilm's absolute TEI-cache path on krick
            with self.subTest(alias=alias):
                self.assertIn(alias, self.manifest["models"])
                entry = self.manifest["models"][alias]
                fetched = {f"{entry['dest']}/{f['path']}" for f in entry["files"]}
                self.assertIn(path, fetched, "catalog model path not fetched")
                tok_dir = nvidia.get("tokenizer_dir", "")
                self.assertIn(f"{tok_dir}/tokenizer.json", fetched,
                              "catalog tokenizer_dir has no fetched tokenizer.json")


class FixtureVerification(unittest.TestCase):
    """End-to-end verify/fetch logic against a tiny local fixture."""

    def make_fixture(self):
        tmp = Path(tempfile.mkdtemp(prefix="fetch-models-test-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp))
        payload = b"tiny onnx stand-in\n"
        dest = tmp / "models/onnx/tiny"
        dest.mkdir(parents=True)
        (dest / "model.bin").write_bytes(payload)
        manifest = {
            "schema_version": 1,
            "models": {
                "tiny": {
                    "repo": "example/tiny",
                    "revision": "0" * 40,
                    "dest": "models/onnx/tiny",
                    "files": [{
                        "path": "model.bin",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }],
                }
            },
        }
        return tmp, manifest

    def test_verify_ok(self):
        root, manifest = self.make_fixture()
        self.assertEqual(fetch_models.cmd_verify(["tiny"], manifest, root), 0)

    def test_verify_detects_corruption(self):
        root, manifest = self.make_fixture()
        (root / "models/onnx/tiny/model.bin").write_bytes(b"tampered")
        self.assertEqual(fetch_models.cmd_verify(["tiny"], manifest, root), 1)

    def test_verify_detects_missing(self):
        root, manifest = self.make_fixture()
        (root / "models/onnx/tiny/model.bin").unlink()
        self.assertEqual(fetch_models.cmd_verify(["tiny"], manifest, root), 1)

    def test_fetch_is_idempotent_offline(self):
        # A file already present with a matching hash is skipped without any
        # network access (the fixture repo/revision don't exist upstream, so
        # a download attempt would fail loudly).
        root, manifest = self.make_fixture()
        self.assertEqual(fetch_models.cmd_fetch(["tiny"], manifest, root), 0)

    def test_stream_download_atomic_write_and_hash(self):
        root, _ = self.make_fixture()
        src = root / "source.bin"
        payload = b"streamed bytes"
        src.write_bytes(payload)
        dest = root / "out/copy.bin"
        digest, size = fetch_models.stream_download(src.as_uri(), dest)
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
        self.assertEqual(size, len(payload))
        self.assertEqual(dest.read_bytes(), payload)
        self.assertEqual(list(dest.parent.glob("*.part")), [])

    def test_manifest_rejects_unknown_schema(self):
        root, manifest = self.make_fixture()
        manifest["schema_version"] = 99
        path = root / "bad.json"
        path.write_text(json.dumps(manifest))
        with self.assertRaises(RuntimeError):
            fetch_models.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
