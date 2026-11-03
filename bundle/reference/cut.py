"""Which checkpoint tensors a cut or task-headed model may leave unused.

Kept apart from reference.py, which needs torch, so its rules are tested
on their own (test_cut.py).
"""

import re


def cut_layer_key(key, layers):
    """Whether checkpoint tensor `key` belongs to a layer at or past
    `layers`, which a cut model leaves unused. `layer` or `layers` is a
    path component, the first one included: `layers.12.attn...` (a bare
    ModernBERT encoder), `model.layers.12.attn...` (its masked-LM
    checkpoint), `encoder.layer.11.output...` (BERT)."""
    m = re.search(r"(?:^|\.)layers?\.(\d+)\.", key)
    return m is not None and int(m.group(1)) >= layers


def task_heads(keys, prefix):
    """The checkpoint tensors outside the encoder's `prefix`, a task head
    beside it, when the checkpoint names the encoder by that prefix at
    all; none otherwise (a bare encoder checkpoint)."""
    if not any(k.startswith(prefix) for k in keys):
        return set()
    return {k for k in keys if not k.startswith(prefix)}


def unexplained(unexpected, keys, prefix, cut=None):
    """The checkpoint tensors the model did not take that neither a task
    head nor, with `cut`, a layer past the cut accounts for."""
    heads = task_heads(keys, prefix)
    return sorted(k for k in set(unexpected) - heads if cut is None or not cut_layer_key(k, cut))
