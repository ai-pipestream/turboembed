"""Fetch the pinned GLiNER checkpoint, after explicit acceptance.

Usage: fetch_checkpoint.py CHECKPOINT REVISION LOCAL_DIR

The checkpoint is local-only by its training data and fetched only
after the user accepts the dataset card's terms (docs/licences.md,
urchade/gliner_small-v2.1). Acceptance is recorded by naming the
checkpoint in TURBO_ACCEPT_TERMS; without it this script refuses
before any network or hub code runs, so the Docker build in this
directory cannot fetch the checkpoint for a user who has not
accepted. In Docker:

  docker build --build-arg TURBO_ACCEPT_TERMS=urchade/gliner_small-v2.1 .
"""

import os
import sys

ACCEPT_ENV = "TURBO_ACCEPT_TERMS"


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    checkpoint, revision, local_dir = sys.argv[1:4]
    accepted = os.environ.get(ACCEPT_ENV, "")
    if accepted != checkpoint:
        raise SystemExit(
            f"refusing to download {checkpoint}: set {ACCEPT_ENV}={checkpoint} "
            "to record acceptance of the dataset card's terms first "
            "(docs/licences.md)"
        )
    from huggingface_hub import snapshot_download

    snapshot_download(checkpoint, revision=revision, local_dir=local_dir)


if __name__ == "__main__":
    main()
