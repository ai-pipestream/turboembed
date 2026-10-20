# Tiny QA reference

Generated from the existing tiny BERT encoder and a seeded, untrained
BertForQuestionAnswering token head. Cases include a multilingual context,
multiple windows, empty context, and a question longer than the query budget.
This fixture tests inference parity, not question-answering accuracy.

Generator: `bundle/reference/tiny_qa.py`, using `qa_reference.py` for joint
window construction and upstream logits. Versions and seed are in receipt.json.
The reference container used was
`sha256:9bcfa35769f1644b37c257e6664a0febef781852b946aa0be5421d0e90bfff7c`.
The generator accepts the repository root as its sole argument and writes the
intermediate Hugging Face model under target/wave22/qa-upstream.
