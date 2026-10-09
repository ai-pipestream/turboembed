"""Model2Vec's StaticModel and sentence-transformers' StaticEmbedding under
the shared timing protocol: one warm pass, then three runs of at least two
seconds, the best in texts a second.

Arguments: tool (model2vec or st), the model directory, the texts JSON
array, the batch sizes, and optionally a file for the vectors (raw F32).
"""

import json
import sys
import time

import numpy as np


def main(tool, model_dir, texts_path, batches, out=None):
    with open(texts_path, encoding="utf-8") as f:
        texts = json.load(f)
    start = time.perf_counter()
    if tool == "model2vec":
        from model2vec import StaticModel

        m = StaticModel.from_pretrained(model_dir)

        def embed(chunk, b):
            return m.encode(chunk, batch_size=b, use_multiprocessing=False)

        name = "model2vec"
    else:
        try:
            from sentence_transformers.sentence_transformer.modules.static_embedding import StaticEmbedding
            from sentence_transformers.base.modules.normalize import Normalize
        except ImportError:
            from sentence_transformers.models import Normalize, StaticEmbedding
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(modules=[StaticEmbedding.from_model2vec(model_dir), Normalize()], device="cpu")

        def embed(chunk, b):
            return m.encode(chunk, batch_size=b, convert_to_numpy=True, show_progress_bar=False)

        name = "sentence-transformers"
    print(f"{name}: loaded in {(time.perf_counter() - start) * 1e3:.1f} ms", flush=True)
    # "-" times nothing: the run only writes the vectors.
    for b in [] if batches == "-" else [int(x) for x in batches.split(",")]:
        def one_pass():
            for i in range(0, len(texts), b):
                embed(texts[i : i + b], b)

        one_pass()
        best = 0.0
        for _ in range(3):
            start, done = time.perf_counter(), 0
            while time.perf_counter() - start < 2.0:
                one_pass()
                done += len(texts)
            best = max(best, done / (time.perf_counter() - start))
        print(f"{name} batch {b}: {best:.0f} texts/s", flush=True)
    if out:
        v = np.concatenate([np.asarray(embed(texts[i : i + 1024], 1024)) for i in range(0, len(texts), 1024)])
        v.astype("<f4").tofile(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
