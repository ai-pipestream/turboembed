# Training over gRPC

`turbo-kserve` serves `turbo.training.v1.TurboTrainingService` on its port:
static models distilled from an encoder the server already serves
(docs/static.md), sealed into a training directory, and served under
their own name as soon as they are sealed; and classical components
trained from samples the client streams, their model files sent back. `server/proto/turbo_training.proto`
is the crate's own file; generate a client from it in any language with
gRPC.

`--training-dir PATH` names the training directory, made when it does
not exist. Each static model is one child directory, a sealed bundle
named as the model is served. A training works in a `.staging-` child
and moves the sealed bundle into place whole; a server removes any
`.staging-` child it finds when it starts.

## The service

| RPC | Admin | Answer |
|---|---|---|
| `ListTeachers` | no | The served, loaded encoders a static model can be distilled from. |
| `TrainStaticModel` | yes | Distils, seals and serves a static model, streaming its progress. |
| `ListStaticModels` | no | The static models in the training directory, each file checked. |
| `DeleteStaticModel` | yes | Stops serving a static model and removes its directory. |
| `TrainComponent` | yes | Trains a classical component from streamed samples and streams its model file back. |

The admin calls are served only when the server is started with
`--admin`; without it they refuse with `UNIMPLEMENTED` and touch no
file. Without `--training-dir` every call but `ListTeachers` and
`TrainComponent` refuses with `UNIMPLEMENTED`.

## Teachers

A teacher is a served model that is loaded, embeds text, has an
architecture (an encoder the library runs), and is not itself a static
model. `ListTeachers` gives each one's served name, model id and
revision, the SHA-256 of its `manifest.json`, its `hidden` width (the
most `pca_dims` can be) and its vocabulary size.

## Training

`TrainStaticModel` takes the new model's `name`, the `teacher`'s served
name, and the distillation's settings:

| Field | Meaning | 0 or empty |
|---|---|---|
| `pca_dims` | The static vectors' width, at most the teacher's `hidden` and under its vocabulary size. | 256, or `hidden` when that is less. |
| `sif_coefficient` | Model2Vec's SIF coefficient, in (0, 1). | 1e-4. |
| `dtype` | `STATIC_DTYPE_F32` or `STATIC_DTYPE_F16`, how the table is stored. | F32. |
| `max_length` | The tokens a text is cut to, at most 8192. | 512. |
| `vocabulary_artifact_id` | A learned vocabulary in the vocabulary store (docs/grpc-vocabulary.md): its single terms, in its order, are learned terms. | None. |
| `terms` | More learned terms, after the vocabulary's. | None. |
| `revision` | The model's revision. | `1`. |
| `sessions` | Sessions the served model pools. | 1. |

The name is ASCII letters, digits, `.`, `_` and `-`, not starting with
`.`, at most 128 bytes, and no served model or static model may have
it. Every check is made before anything is written, and a refusal is
the call's own status, with `turbo-field` naming the request field. One
training runs at a time; another that passes its checks is refused with `UNAVAILABLE`
(`TURBO_E_BUSY`), which a client may retry. A training adds at most
262,144 learned terms; more is `TURBO_E_CAPACITY`.

The distillation is the bundle tool's (docs/static.md, "Learned terms
and a recipe made from the base"): the teacher's vocabulary entries and each learned term that is
one WordPiece word new to the vocabulary are embedded by the teacher,
reduced by PCA, weighted by Zipf's law, and measured on the quality
texts. The bundle is then sealed: its reference written and every
reference case and quality text checked on the library's CPU device.

The stream is progress, then one `trained` message last. Each progress
message has a `stage`:

| Stage | Meaning |
|---|---|
| `TRAIN_STAGE_ADMITTED` | The checks passed and the training holds the slot. |
| `TRAIN_STAGE_KEPT` | `completed_items` entries of the teacher's `total_items` are kept, with `added_terms` learned terms added and `refused_terms` refused. |
| `TRAIN_STAGE_EMBEDDING` | `completed_items` rows of `total_items` are through the teacher. Repeated. |
| `TRAIN_STAGE_REDUCED` | Reduced to `completed_items` dimensions, keeping `explained_variance`. |
| `TRAIN_STAGE_MEASURED` | The quality texts' `quality`, with the teacher and the static model. |
| `TRAIN_STAGE_SEALED` | The bundle is sealed and verified. |
| `TRAIN_STAGE_LOADED` | The bundle is in the training directory and served under its name on the CPU. |

`trained` describes the model as `ListStaticModels` does, with the
learned terms that became entries (in id order, from the teacher's
vocabulary size up), each refused term with its reason (`EMPTY`,
`UNKNOWN`, `PRESENT`, `WORDS`, `LONG` or `DUPLICATE`, docs/static.md),
and the vocabulary artifact's id and SHA-256 when one was named. From
`TRAIN_STAGE_LOADED` on, `ModelInfer` and `Capabilities` serve the
model.

A training that fails after admission ends the stream with
`TURBO_E_RUNTIME` and leaves no file and no served model. A client that
cancels is seen between steps: the distillation runs to its end, and the
training stops before sealing or before the move, removing its staging.
Publishing is one step under a lock that `DeleteStaticModel` also
takes: the bundle is moved, registered, loaded and checked again, and if
any of it fails the registration and the directory are removed. Once it
is published the model stays; delete it to remove it. A panic in the
training ends the stream with `TURBO_E_INTERNAL`, its staging removed.
The slot is given back after the staging is removed, before the last
message.

## Listing and deleting

`ListStaticModels` lists every child of the training directory in name
order: its model id and revision, the SHA-256 of its manifest, its
width, rows, learned terms and dtype, the teacher it was distilled from
(`TeacherProvenance`: the teacher's model id, revision and manifest
SHA-256), the quality measured, every file its manifest lists with its
size and SHA-256, and whether the server serves this directory under
the name. Every listed file is read again and checked against its size
and SHA-256; `verified` is false when one does not match, and `problem`
names it. A child that is not a static model bundle is listed
unverified, with why. A symbolic link in the directory is not a model:
it is not listed, and `DeleteStaticModel` does not follow it.

`DeleteStaticModel` removes the registration serving the directory, if
any (requests in flight finish), then the directory. `unloaded` says
whether it was served. A model served under the same name from another
directory is not touched. An unknown name is `NOT_FOUND`
(`TURBO_E_BUNDLE_NOT_FOUND`).

## Component training

`TrainComponent` trains one classical component from samples in its
native format and sends the model file back; the server keeps nothing.
The client streams a `TrainComponentStart` first, then the samples as
UTF-8 in any number of `samples` chunks, cut anywhere (inside a
character too), and ends its stream to start the training. It shares
the one training slot with `TrainStaticModel`.

| `component` | Samples, as `turbo-formats` reads them |
|---|---|
| `SENTENCE_DETECTOR` | One sentence a line, a blank line between documents. |
| `TOKENIZER` | One sentence a line, tokens split by whitespace or `<SPLIT>`. |
| `POS_TAGGER` | One sentence a line of `word_TAG` tokens. |
| `LEMMATIZER` | Word, tag and lemma, tab separated, a line a token, a blank line between sentences. |
| `CHUNKER` | Word, tag and chunk tag, space separated, a line a token, a blank line between sentences. |
| `NAME_FINDER` | One sentence a line, `<START:type> ... <END>` around each name. |
| `DOCUMENT_CATEGORIZER` | A category then the document's tokens, whitespace separated, a line each. |
| `LANGUAGE_DETECTOR` | A language code, a tab, then the text, a line each. |

`language` is the code the model records, required for every component
but the language detector, which learns its languages from the
samples. `parameters` are the training parameters a parameters file
names (`Algorithm`, `Iterations`, `Cutoff`, `Tolerance`, ...), each
value a string; unset keys take the trainer's defaults (`MAXENT`, 100
iterations, cutoff 5; the name finder's own). The sentence detector,
tokenizer, POS tagger and lemmatizer train `MAXENT` (GIS) only and
refuse any other `Algorithm`; every trainer but `NAIVEBAYES` needs
`Iterations` of at least 1. `name_type` gives the
name finder's untyped names a type (`default` when empty).

The answer streams `TrainComponentProgress`: `ADMITTED`, `READ` with the
sample count, an `ITERATION` for each finished iteration (with the
events classified correctly, the event count and the trainer's measure,
`Log Likelihood` for GIS and `Training Accuracy` for the perceptron),
then `PACKAGED`. The model file follows as `model` chunks of at most
1 MiB, and `TrainedComponent` ends the stream with the component, the
sample count and the file's size and SHA-256 so the client can check
what it put together. The file loads where a trained model of that
component does: the annotators' `from_trained_archive` (and
`from_trained_archive_with_format`).

Refusals: a first message that is not the start, a second start, an
unnamed component, a missing or longer than 64 byte language,
parameters the trainer refuses, samples that are not UTF-8 (naming the
byte), a refused sample line, or no samples are `INVALID_ARGUMENT`;
more than 256 MiB of samples, or an upload that sends no message for a
minute (`Serve::upload_idle`), is `OUT_OF_RANGE` (`TURBO_E_CAPACITY`);
a call while another training runs is `UNAVAILABLE` (`TURBO_E_BUSY`).
A client that goes away stops the training at its next iteration, the
quasi-Newton trainer's included, and frees the slot; naive Bayes counts
in one pass and is not stopped. A client
that drops the answer but keeps its upload open looks to the server
like a slow one, so the upload idle limit is what frees the slot then.

## Not yet served

`StreamingTraining` (analysis, vocabulary, distillation and indexing as
one transaction) is the rest of issue #1223.
