"""Fetch and verify pinned GlotLID weights after explicit acceptance.

Usage: fetch_model.py CHECKPOINT REVISION SHA256 LOCAL_DIR

GlotLID is local-only by its training data (docs/licences.md). Naming the
checkpoint in TURBO_ACCEPT_TERMS records acceptance of its model card;
without that exact value this refuses before importing the hub client.
"""

import hashlib
import os
import sys

ACCEPT_ENV = "TURBO_ACCEPT_TERMS"


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    checkpoint, revision, expected, local_dir = sys.argv[1:5]
    if os.environ.get(ACCEPT_ENV, "") != checkpoint:
        raise SystemExit(
            f"refusing to download {checkpoint}: set {ACCEPT_ENV}={checkpoint} "
            "to record acceptance of the model card's terms first (docs/licences.md)"
        )
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(checkpoint, filename="model.bin", revision=revision, local_dir=local_dir)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        os.remove(path)
        raise SystemExit(f"{checkpoint} model.bin: SHA-256 {digest.hexdigest()}, expected {expected}")
    print(f"{checkpoint} model.bin sha256 {expected}", file=sys.stderr)


if __name__ == "__main__":
    main()
