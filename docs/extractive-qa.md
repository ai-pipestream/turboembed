# Extractive question answering: implementation in progress

Issue #745 adds a two-logit token head, question/context windows, null-answer
calibration, and answer spans projected to a document view. This branch starts
with the checked decoder in `core/src/qa/mod.rs` and `core/src/qa/decode.rs`.
It does not yet load QA models or advertise a supported runtime QA task.

## Decoder contract

`WindowLogits` carries finite start/end scores, an explicit CLS position,
context byte offsets, and a maximum-context ownership mask. Question, special,
and padding positions have no context offset and cannot start an answer.
All windows are validated before candidate enumeration. Context tokens must
form one ordered segment and every nonempty offset must slice the original
UTF-8 string. Split codepoints, ragged columns, invalid CLS indices, and
non-finite scores or thresholds return errors.

Candidate starts must belong to the window with maximum interior context.
An end token may have another owner, provided it remains in the same window;
otherwise answers spanning an ownership boundary could disappear. Answer
length is bounded in tokens. Duplicate byte spans retain their best score.
Ties prefer earlier byte starts, then shorter spans. Optional non-overlap
selection greedily retains ranked spans.

Null score is the minimum CLS start-plus-end score across windows. The
maximum-context-start rule and minimum-window null aggregation follow the
[Transformers SQuAD-v2 postprocessor](https://github.com/huggingface/transformers/blob/v4.57.1/examples/pytorch/question-answering/utils_qa.py).
This ticket's explicit threshold contract is used: abstain when
`best_span_score - null_score <= null_threshold`, including equality. Scores
are accumulated in F64 from F32 logits. Max-shifted softmax covers the returned
candidate spans plus the null candidate, so probabilities remain finite for
extreme finite logits. Empty window input is invalid; a valid context-free
window produces `Answer::None`.

`core/tests/qa_decode.rs` exercises synthetic boundary cases. These tests prove
decoder behavior, not pretrained QA model parity.

## Remaining integration

- Propose `TASK_EXTRACTIVE_QA` and its C session/result contract in the header.
- Validate a two-output token head and add pinned model conversion/reference
  fixtures with answerable and unanswerable SQuAD-v2 cases.
- Encode question/context pairs with byte offsets, bounded query length,
  context stride, and deterministic maximum-context ownership.
- Run the encoder and token head, then feed checked windows to this decoder.
- Add `ExtractiveQa`, the typed `Answers` layer, `ExtractiveQaAnnotator`, and
  projection through the document's view path to source bytes.
- Reuse the pair/session and contextual-token work from #759 and #741 after
  those changes land; keep model identity and source-view provenance attached.
