//! Deterministic size fixtures, not trained production checkpoints.
use std::sync::Arc;
use turbo_doccat::{DoccatModel, DocumentCategorizerME, SvmCategorizer, SvmDoccatModel};
use turbo_maxent::ModelFile;
use turbo_train::{JavaString, Manifest, ModelPackage, WritableModel, write_gis_writable};

pub const VOCAB: usize = 16_384;

pub fn word(i: usize) -> &'static str {
    static WORDS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    let words = WORDS.get_or_init(|| {
        let words: Vec<_> = include_str!("../../../goldens/data/fixtures/all-minilm-l6-v2-vocab.txt")
            .lines()
            .filter(|w| w.len() >= 2 && w.bytes().all(|b| b.is_ascii_alphabetic()))
            .collect();
        assert!(words.len() >= VOCAB);
        words
    });
    words[i % VOCAB]
}

struct Gis {
    labels: Vec<JavaString>,
    features: Vec<JavaString>,
    pattern: Vec<u32>,
    weights: Vec<Vec<f64>>,
}
impl WritableModel for Gis {
    fn outcome_labels(&self) -> &[JavaString] {
        &self.labels
    }
    fn pred_labels(&self) -> &[JavaString] {
        &self.features
    }
    fn pattern(&self, _: usize) -> &[u32] {
        &self.pattern
    }
    fn parameters(&self, p: usize) -> &[f64] {
        &self.weights[p]
    }
}

pub fn gis_file(names: bool, width: usize) -> Arc<ModelFile> {
    let labels: Vec<String> = if names {
        ["other", "person-start", "person-cont", "org-start", "org-cont", "place-start", "place-cont"]
            .iter()
            .map(|s| (*s).into())
            .collect()
    } else {
        (0..width).map(|i| format!("class{i}")).collect()
    };
    let width = labels.len();
    let model = Gis {
        labels: labels.iter().map(|s| JavaString::new(s)).collect(),
        features: (0..VOCAB)
            .map(|i| JavaString::new(&format!("{}{}", if names { "w=" } else { "bow=" }, word(i))))
            .collect(),
        pattern: (0..width as u32).collect(),
        weights: (0..VOCAB)
            .map(|i| (0..width).map(|j| (((i * 17 + j * 29) % 101) as f64 - 50.0) * 0.002).collect())
            .collect(),
    };
    let mut package = ModelPackage::new(Manifest::new(
        if names { "TokenNameFinder" } else { "DocumentCategorizerME" },
        "en",
        1700000000000,
    ));
    package.add_entry(if names { "nameFinder.model" } else { "doccat.model" }, write_gis_writable(&model).unwrap());
    Arc::new(ModelFile::open(package.to_bytes()).unwrap())
}

pub fn maxent(width: usize) -> DocumentCategorizerME {
    DocumentCategorizerME::new(DoccatModel::open(Arc::try_unwrap(gis_file(false, width)).unwrap()).unwrap())
}

pub fn svm() -> SvmCategorizer {
    // 64 sparse support vectors, 32 nonzero features per vector, binary probabilities.
    let mut model = String::from(
        "svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv 64\nrho 0.25\nlabel 0 1\nprobA -2.0\nprobB 0.0\nnr_sv 32 32\nSV\n",
    );
    for sv in 0..64 {
        model.push_str(if sv < 32 { "0.125" } else { "-0.125" });
        for j in 0..32 {
            let ix = 1 + sv * 256 + j * 7;
            model.push_str(&format!(" {ix}:{}", (j + 1) as f64 / 32.0));
        }
        model.push('\n');
    }
    let mut side = String::from(
        "svm-doccat 1\nlanguage\teng\nweighting\tBINARY\nselection\tNONE\t-1\nscaling\tnone\ngenerator\tbag-of-words\ncategory\t0\tclass0\ncategory\t1\tclass1\n",
    );
    for i in 0..VOCAB {
        side.push_str(&format!("feature\t{}\t1.0\t0.0\t1.0\tbow={}\n", i + 1, word(i)));
    }
    SvmCategorizer::new(SvmDoccatModel::from_text(&model, &side).unwrap()).unwrap()
}
