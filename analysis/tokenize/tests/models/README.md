# Tokenizer models for the tests

The published OpenNLP UD 1.3 tokenizer models (built with OpenNLP 2.5.4)
the tests once ran are OpenNLP 2.5 artifacts, which are not fetched,
pinned or run (#1509); #1532 removes the JVM goldens that still name them.

The fixture model with an abbreviation dictionary that the tests also
run, `tokenizer-abbreviations.bin`, is trained for this project and checked
in as one of the golden generator's fixtures, in
`analysis/goldens/data/fixtures` (`analysis/goldens/data/README.md` says
where it comes from). The unit tests that do not depend on a model's decisions run
on it, so they need no download.
