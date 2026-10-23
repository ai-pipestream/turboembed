# turbo-parser

Constituency parse trees, Penn Treebank bracket I/O, language head rules, and the `CHUNKING` and `TREEINSERT` bottom-up parsers: reading a model and parsing with it (`opennlp.tools.parser`).

Training lives in `turbo-train-components` (`turbo_train_components::parser`): the chunking and tree-insertion event streams, `chunking.Parser.train` and `treeinsert.Parser.train`, writing the model file, and `ParserEvaluator` and `ParserCrossValidator`. `turbo-formats` reads treebanks into `Parse` trees, so this crate cannot depend on `turbo-formats` or on the trainers that do.

It follows OpenNLP at commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the OpenNLP fork, the commit the goldens name (`analysis/goldens-tool/src/recipe/mod.rs`).

## Components

- `parse`:
  - `Parse` (`opennlp.tools.parser.Parse`): constituency tree node with shared `Arc<str>` sentence text, UTF-8 byte `Span` offsets, its `Head`, derivation tracking, tree changes (`insert`, `add`, `adjoin`, `adjoin_root`, `expand_top_node`, `prune_parse`, `fix_possessives`, `add_names`), and node queries (`tag_nodes`, `token_nodes`, `initial_chunks`, `tag_sequence_prob`).
  - `Parse::parse`, `Parse::parse_with_gaps` and `Parse::parse_with_options` (`Parse.parseParse`): the lenient Penn Treebank bracket reader. A type is the leading run of characters other than space, `=` and `-`, or one of `-LRB- -RRB- -LCB- -RCB- -LSB- -RSB- -NONE-` when the label starts with it; with function tags on, only the first tag is kept (`NP-SBJ-TMP` reads as `NP-SBJ`). A bracket without a type is harmless while it covers no token and fails with `ParseError::MissingType` (Java's `NullPointerException`) when it does; an extra `)` fails with `ParseError::UnmatchedClose` (`EmptyStackException`); brackets still open at the end of the line are dropped. A `-NONE-` token is read as a token unless a `GapLabeler` is given; the labeler then labels the extraction chain above it, and the token is left out.
  - `Parse::show` and `Parse::to_string_penn_treebank` (`Parse.show`, `toStringPennTreebank`): the bracket writer, with bracket tokens escaped (`encode_token`, `decode_token`).
  - `space_untokenized_parens` (`ParserTool.spaceUntokenizedParens`): separates round and curly brackets from adjacent characters, with the whitespace of the `WhitespaceMode` it is given.
- `head_rules`:
  - `HeadRules` (`opennlp.tools.parser.HeadRules`) and `GapLabeler` (`opennlp.tools.parser.GapLabeler`).
  - `EnglishHeadRules` (`opennlp.tools.parser.lang.en.HeadRules`): Collins/Magerman English head rules with the fixed `NP`/`NX` passes, subject and object extraction gap labeling, and the rule file reader and writer. OpenNLP's class has no special case for coordination or punctuation; it only reports its punctuation tags.
  - `AncoraSpanishHeadRules` (`opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`): AnCora Spanish head rules, whose tags are Java regular expressions that must match the whole type. `TagPattern` reads the subset the rule files use (literals, `.`, classes, `\` before a character that is not a letter or digit, `*`, `+`, `?`, a leading `^` and a trailing `$`) and refuses the rest with `HeadRulesError::UnsupportedPattern`.
- `stream`:
  - `ParseSampleStream` (`opennlp.tools.parser.ParseSampleStream`), `ParserPosSample` (`opennlp.tools.parser.PosSampleStream`), and `ParserChunkSample` (`opennlp.tools.parser.ChunkSampleStream`).
- `context`:
  - `AbstractContextGenerator` (`opennlp.tools.parser.AbstractContextGenerator`), `Cons` (`opennlp.tools.parser.Cons`), `ChunkContextGenerator` (`opennlp.tools.parser.ChunkContextGenerator`), and `ParserChunkerSequenceValidator` (`opennlp.tools.parser.ParserChunkerSequenceValidator`).
- `model`:
  - `ParserModel`, `ParserType`, `ParserEventType`, `HeadRulesArtifact`, `ParserChunkerModel` and `ParserModelError` (`opennlp.tools.parser.ParserModel`, `ParserType`, `ParserEventTypeEnum`, `ParserChunkerFactory`): reading `.bin` ZIP model packages (`build.model`, `check.model`, `attach.model`, `parsertager.postagger`, `parserchunker.chunker`, `head-rules.headrules`, `manifest.properties`). `turbo_train_components::parser::model` writes them.
- `bottom_up`:
  - `Parser` (`opennlp.tools.parser.Parser`), `ParserEngine` and `ParserFactory` (`opennlp.tools.parser.ParserFactory`), the shared search of `AbstractBottomUpParser` (`parse_bottom_up`), `collapse_punctuation` and `build_dictionary` (`AbstractBottomUpParser.collapsePunctuation`, `buildDictionary`).
- `chunking`:
  - `chunking::Parser`, `chunking::BuildContextGenerator` and `chunking::CheckContextGenerator` (`opennlp.tools.parser.chunking.*`).
- `treeinsert`:
  - `treeinsert::Parser`, `treeinsert::BuildContextGenerator`, `treeinsert::AttachContextGenerator` and `treeinsert::CheckContextGenerator` (`opennlp.tools.parser.treeinsert.*`).
- `annotator`:
  - `ParserTool` (`opennlp.tools.cmdline.parser.ParserTool`) and `ParserAnnotator`, projecting constituent trees onto the `Phrases` layer in `turbo-analysis-layers`.

Both rule readers read the format OpenNLP reads: per line, the field count (`Integer.parseInt`), the type, the direction (`1` is left to right, anything else right to left) and the tags, split as `StringTokenizer` splits them. A count outside 0 to 1,000 tags is an `IOException` with OpenNLP's message, more tags than the count an `ArrayIndexOutOfBoundsException`, fewer a `NullPointerException`, and a line without the first three fields (a blank line too) a `NoSuchElementException`; `HeadRulesError::java_class` names each. A later rule for a type replaces the earlier one. `serialize` writes the rules in the order Java's `HashMap` iterates them.

## Where it differs from OpenNLP

- Spans are UTF-8 byte offsets into the Rust text, where OpenNLP's count UTF-16 units. `Parse::insert` refuses a span that does not fall on character boundaries (`ParseError::NotCharBoundary`, which has no Java counterpart).
- OpenNLP's `Parse` holds a reference to its head node. `Head` keeps the head's span and head index as they were when the head was chosen, which is what OpenNLP's callers read; a later change to that node's span is not seen.
- `Clone` copies the whole subtree, where Java's `clone` shares the children.
- `serialize` does not model a `HashMap` bin that becomes a tree (nine types in one bin of a table of 64 bins or more); such a bin is written in insertion order. No rule file is known to have one.
- OpenNLP's `es-head-rules` doubles its backslashes (`GRUP\\.VERB`, `\\$`). In a Java regular expression that is a backslash followed by any character, so those tags never match a type; `TagPattern` reads them the same way.

## Golden files

`tests/goldens.rs` compares the reader, the writer, both rule classes and the sample conversions with files a JVM running OpenNLP wrote, made by the golden generator (`analysis/goldens-tool`, `ParserGolden.java`). The inputs are the `parse` and `head-rules` sets of `analysis/goldens/data/inputs`; the rule files are the fixtures `en-head-rules.txt`, `es-head-rules.txt` and `es-regex-head-rules.txt` (`analysis/goldens/data/README.md`).

| Run | What it covers |
|---|---|
| `parse/treebank` | `ParseSampleStream` over the `parse` set: 37 parses written back, 8 exceptions, 730 nodes with their spans and heads, and the part-of-speech and chunk samples |
| `parse/treebank+function-tags` | the same with `useFunctionTags` on |
| `parse/en-head-rules`, `parse/en-head-rules+possessives` | `parseParse` with the English rules as gap labeler, then `updateHeads`; the second run calls `pruneParse` and `fixPossesives` first |
| `parse/es-head-rules`, `parse/es-regex-head-rules` | the same with OpenNLP's AnCora rules, and with rules whose tags use the regular expressions `TagPattern` reads |
| `head-rules/en`, `head-rules/es` | the 13 rule files of the `head-rules` set read by each class: the rules written back with `serialize`, or the exception |

The `parse-treebank` runs of `ParserTrainGolden.java` (the n-gram dictionary, the event streams, the context generators, and the chunking and tree-insertion models the JVM trained, with their parses and evaluation) are compared in two places that share `tests/goldens/treebank_runs.rs`: `tests/goldens/treebank.rs` here compares the dictionary and generator runs and the parses of the JVM's models, and `turbo-train-components`' `tests/parser_goldens.rs` compares the event runs and the model runs in full, evaluation included, and checks that a model trained there is the model the JVM wrote. `analysis/goldens-tool/README.md` describes the runs.

`tests/parser_tests.rs` covers what the goldens do not reach: the tree changes the bottom-up parsers make (`add`, `adjoin`, `adjoin_root`, `remove`, `insert`), `add_names`, the punctuation sets, `space_untokenized_parens`, `TagPattern` and the limits of the rule reader.
