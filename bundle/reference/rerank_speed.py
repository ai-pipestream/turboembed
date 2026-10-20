"""Fixed batch matching core/examples/rerank_speed.rs; CPU, one thread."""
import sys
import time
import statistics
import torch
import sentence_transformers
from sentence_transformers import CrossEncoder

torch.set_num_threads(1)
model = CrossEncoder(sys.argv[1], device="cpu", local_files_only=True)
query = "What is the capital of France?"
passages = ["Paris is the capital of France.", "Berlin is the capital of Germany.",
            "Madrid is the capital of Spain.", "Tokyo is the capital of Japan."]
pairs = [(query,p) for p in passages]
for _ in range(3):
    model.predict(pairs, batch_size=4, show_progress_bar=False)
times=[]
for _ in range(15):
    start=time.perf_counter()
    model.predict(pairs, batch_size=4, show_progress_bar=False)
    times.append(time.perf_counter()-start)
median=statistics.median(times)
print(f"CrossEncoder {sentence_transformers.__version__} torch={torch.__version__} batch=4 threads=1 median_seconds={median} pairs_per_second={4/median}")
