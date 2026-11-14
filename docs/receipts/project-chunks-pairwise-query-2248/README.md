# ProjectChunks pairwise cosines and query placement receipt (#2248, #2249)

`turbo-kserve` started in process on the CPU reference with
`testdata/tiny-bert-bundle`, every call over gRPC
(`server/tests/inspect_pairwise_query.rs`), and the library's own
placement test (`analysis/chunker/tests/projection.rs`). Machine: a
4-core x86_64 Linux container, no GPU.

The graphs hold six chunk rows over five sentences (one whole-document
row a level up), and their vectors are the served model's own answers to
`ModelInfer`. Every expected number is computed in the test from those
vectors: the pairwise cosines directly, the query point from
`VectorProjection::place` on a query vector the test asks `ModelInfer`
for in the query role.

- `red.log`: the two tests built against the tree before this change.
  They do not compile: the request and response fields, the limits and
  `VectorProjection::place` do not exist.
- `green.log`: the same tests with the change. All pass.

What the server tests hold:

| Test | What it checks |
|---|---|
| `pairwise_cosines_are_the_stored_vectors_cosines` | The upper triangle for level 0 against cosines computed from the stored rows; an empty row and a zero-length row are left out of `items`; the level above lists its one item with no pairs; a level over `max_items` and a `max_items` over the ceiling are refused with `TURBO_E_CAPACITY` naming the cap; a graph without vectors is refused. |
| `a_query_lands_in_the_projections_own_basis` | For L2-normalized and raw stored vectors, in 2D and 3D: the query's coordinates against the same basis computed in the test, and its cosine to every row. |
| `a_query_is_refused_without_vectors_or_their_model` | No vectors, another model's vectors, a width the model does not serve, an empty text, a text over the limit, and all-empty rows. |

Not measured: timing. The pairwise pass is at most 130,816 dot products
of at most 2,048 values, and the query costs one embedding call.
