/// An input checked in under `analysis/goldens/data/inputs` that holds
/// lines copied from a published corpus: the input's path below the inputs
/// directory and the SHA-512 of its bytes, the corpus file it was cut from
/// with where that file is fetched from and its SHA-512, the line ranges
/// taken from it (1-based, both ends included, in the order they appear in
/// the input), the licence and where the corpus is described.
///
/// The input is the comment lines at its top, an empty line, then the
/// ranges' lines unchanged, so the line count after the first empty line
/// is the ranges' total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Excerpt {
    pub input: &'static str,
    pub sha512: &'static str,
    pub source_file: &'static str,
    pub source_url: &'static str,
    pub source_sha512: &'static str,
    pub lines: &'static [(u32, u32)],
    pub license: &'static str,
    pub source: &'static str,
}

/// The excerpts of published corpora among the inputs.
///
/// `ner/germeval-excerpt.txt` is 105 sentences of the GermEval 2014 named
/// entity training file, `NER-de-train.tsv` as the shared task's data page
/// links it (a Google Drive folder), CC BY 4.0. The ranges are one block
/// of 100 sentences and five single sentences, chosen so that each of the
/// twelve labels (`PER`, `LOC`, `ORG` and `OTH`, each plain, `deriv` and
/// `part`) occurs in both the outer and the inner column; in the full file
/// the inner `OTHpart` is in one sentence and the inner `ORGderiv` and `OTHderiv`
/// in three sentences each. The `ner` runs of `GermEval2014NameSampleStream` read
/// it as they read the other `germeval-` inputs.
pub const EXCERPTS: [Excerpt; 1] = [Excerpt {
    input: "ner/germeval-excerpt.txt",
    sha512: "b8387e481983c27847d0c995d334b298515f554ae12dd195459f3715bd41224e\
             5c1ddffaa5e978ac6790cc0d1d0f5d49bf68d89d60a18794476f9da7f660f3cc",
    source_file: "NER-de-train.tsv",
    source_url: "https://drive.usercontent.google.com/download?id=1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P&export=download",
    source_sha512: "62d786af76254a0682dd28e7b62221a1861979f54529b457148579450947bd79\
                    915f93b69f113b1e999f8fc39b6180c39795a713adf0b1a349b8b1a87c17a606",
    lines: &[(13569, 13606), (156184, 156206), (215761, 215783), (272654, 274778), (356162, 356173), (385626, 385636)],
    license: "CC BY 4.0",
    source: "https://sites.google.com/site/germeval2014ner/data",
}];
