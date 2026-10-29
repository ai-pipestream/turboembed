//! An OpenNLP `SvmDoccatModel`, opened from its two exported files: the
//! LIBSVM text model and the sidecar.

use std::path::Path;

use turbo_analysis::ModelRef;

use super::error::{File, ModelError};
use super::libsvm::{self, LibsvmModel};
use super::sidecar::Sidecar;
use super::vector::{Scratch, vectorize};

/// The model a categorizer runs: the LIBSVM classifier and the
/// vocabulary, weighting, scaling and categories around it.
#[derive(Debug, Clone)]
pub struct SvmDoccatModel {
    svm: LibsvmModel,
    sidecar: Sidecar,
    model_ref: ModelRef,
}

/// The one-against-one vote: the category LIBSVM's `svm_predict` picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vote {
    /// The category index (the LIBSVM label).
    pub category: usize,
}

impl SvmDoccatModel {
    /// Opens the model from the text of its LIBSVM file and its sidecar.
    ///
    /// The LIBSVM model must be C-SVC or nu-SVC, and each of its labels a
    /// category index of the sidecar, given once.
    pub fn from_text(libsvm_text: &str, sidecar_text: &str) -> Result<SvmDoccatModel, ModelError> {
        let svm = libsvm::read(libsvm_text)?;
        let sidecar = Sidecar::read(sidecar_text)?;
        if !svm.svm_type().is_classification() {
            return Err(ModelError::NotClassification(svm.svm_type()));
        }
        let n = sidecar.categories.len();
        let mut seen = vec![false; n];
        for &label in svm.labels() {
            let ok = usize::try_from(label).is_ok_and(|l| l < n && !std::mem::replace(&mut seen[l], true));
            if !ok {
                return Err(ModelError::Invalid {
                    file: File::Libsvm,
                    detail: format!("the label {label} is not a category index given once (0 to {})", n - 1),
                });
            }
        }
        let mut bytes = Vec::with_capacity(libsvm_text.len() + sidecar_text.len());
        bytes.extend_from_slice(libsvm_text.as_bytes());
        bytes.extend_from_slice(sidecar_text.as_bytes());
        let model_ref = ModelRef {
            id: format!("opennlp:SvmDoccatModel:{}", sidecar.language.as_deref().unwrap_or("und")),
            revision: "1".into(),
            sha256: turbo_maxent::sha256(&bytes).to_hex(),
        };
        Ok(SvmDoccatModel { svm, sidecar, model_ref })
    }

    /// Opens the model from its two files.
    pub fn open(libsvm_path: &Path, sidecar_path: &Path) -> Result<SvmDoccatModel, ModelError> {
        let read =
            |p: &Path| std::fs::read_to_string(p).map_err(|error| ModelError::Io { path: p.to_path_buf(), error });
        SvmDoccatModel::from_text(&read(libsvm_path)?, &read(sidecar_path)?)
    }

    pub fn libsvm(&self) -> &LibsvmModel {
        &self.svm
    }

    pub fn sidecar(&self) -> &Sidecar {
        &self.sidecar
    }

    /// The categories by index.
    pub fn categories(&self) -> &[String] {
        &self.sidecar.categories
    }

    /// What layers record as their model: `opennlp:SvmDoccatModel:` and
    /// the language (`und` when the sidecar has none), and the SHA-256 of
    /// the LIBSVM file's bytes followed by the sidecar's.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }

    /// The vector of `tokens` (`Scratch::vector`), as `categorize` gives
    /// it to LIBSVM.
    pub fn vectorize(&self, tokens: &[&str], scratch: &mut Scratch) {
        vectorize(&self.sidecar, tokens, scratch);
    }

    /// `svm_predict_values` on the vector of `tokens`: the decision values
    /// into the scratch (`Scratch::decision_values`) and the category the
    /// pairwise votes pick. OpenNLP's `categorize` does not return these;
    /// the goldens record them from the same LIBSVM model.
    pub fn decision_values(&self, tokens: &[&str], scratch: &mut Scratch) -> Vote {
        vectorize(&self.sidecar, tokens, scratch);
        let label = self.svm.predict_values(&scratch.index, &scratch.value, &mut scratch.predict);
        Vote { category: label as usize }
    }
}
