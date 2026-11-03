"""The tensors a cut model may leave unused (#1607), on the key shapes of
bare and masked-LM ModernBERT and of BERT checkpoints; no torch needed.

    python3 -m unittest discover -s bundle/reference -p 'test_*.py'
"""

import unittest

from cut import cut_layer_key, unexplained

BARE = ["embeddings.tok_embeddings.weight", "final_norm.weight"] + [
    f"layers.{i}.attn.Wqkv.weight" for i in range(4)
]
MLM = ["model." + k for k in BARE] + ["head.dense.weight", "decoder.bias"]
BERT = ["embeddings.word_embeddings.weight"] + [f"encoder.layer.{i}.output.dense.weight" for i in range(4)]


class CutLayerKey(unittest.TestCase):
    def test_a_layer_is_a_path_component_the_first_one_included(self):
        for key in ("layers.2.attn.Wqkv.weight", "model.layers.2.attn.Wqkv.weight",
                    "encoder.layer.2.output.dense.weight", "model.encoder.layer.2.output.dense.weight"):
            self.assertTrue(cut_layer_key(key, 2), key)
            self.assertFalse(cut_layer_key(key, 3), key)

    def test_a_name_that_only_ends_in_layer_is_not_a_layer(self):
        for key in ("sublayers.5.weight", "player.5.weight", "layers.weight", "layers.5"):
            self.assertFalse(cut_layer_key(key, 0), key)


class Unexplained(unittest.TestCase):
    def test_a_bare_modernbert_cut_at_two_leaves_its_later_layers_unused(self):
        later = ["layers.2.attn.Wqkv.weight", "layers.3.attn.Wqkv.weight"]
        self.assertEqual(unexplained(later, BARE, "model.", cut=2), [])
        # Uncut, the same tensors are a model that did not load.
        self.assertEqual(unexplained(later, BARE, "model."), later)
        # An earlier layer left unused is never the cut's.
        self.assertEqual(unexplained(["layers.1.attn.Wqkv.weight"], BARE, "model.", cut=2),
                         ["layers.1.attn.Wqkv.weight"])

    def test_a_masked_lm_modernbert_cut_leaves_its_head_and_later_layers_unused(self):
        unused = ["head.dense.weight", "decoder.bias", "model.layers.3.attn.Wqkv.weight"]
        self.assertEqual(unexplained(unused, MLM, "model.", cut=3), [])
        self.assertEqual(unexplained(unused, MLM, "model."), ["model.layers.3.attn.Wqkv.weight"])

    def test_a_bert_cut_leaves_its_later_layers_unused(self):
        later = ["encoder.layer.3.output.dense.weight"]
        self.assertEqual(unexplained(later, BERT, "bert.", cut=3), [])
        self.assertEqual(unexplained(later, BERT, "bert.", cut=4), later)

    def test_a_renamed_tensor_is_unexplained_whatever_the_cut(self):
        self.assertEqual(unexplained(["final_norm.scale"], BARE, "model.", cut=1), ["final_norm.scale"])


if __name__ == "__main__":
    unittest.main()
