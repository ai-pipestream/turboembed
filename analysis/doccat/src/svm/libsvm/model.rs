//! A LIBSVM model as `svm_model` holds it: the parameters prediction reads,
//! the support vectors in one sparse table, and their coefficients.

/// `svm_parameter.svm_type`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SvmType {
    CSvc,
    NuSvc,
    OneClass,
    EpsilonSvr,
    NuSvr,
}

impl SvmType {
    /// The names LIBSVM writes, in its `svm_type_table` order.
    pub const ALL: [SvmType; 5] =
        [SvmType::CSvc, SvmType::NuSvc, SvmType::OneClass, SvmType::EpsilonSvr, SvmType::NuSvr];

    /// The name in a model file.
    pub const fn name(self) -> &'static str {
        match self {
            SvmType::CSvc => "c_svc",
            SvmType::NuSvc => "nu_svc",
            SvmType::OneClass => "one_class",
            SvmType::EpsilonSvr => "epsilon_svr",
            SvmType::NuSvr => "nu_svr",
        }
    }

    pub fn from_name(name: &str) -> Option<SvmType> {
        SvmType::ALL.into_iter().find(|t| t.name() == name)
    }

    /// C-SVC and nu-SVC: one-against-one classification over labels.
    pub const fn is_classification(self) -> bool {
        matches!(self, SvmType::CSvc | SvmType::NuSvc)
    }
}

/// The kernel and the parameters it reads. A precomputed kernel is not
/// here: it needs the kernel's values as the input, which a document
/// categorizer cannot give, so a model with one is refused when read.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Kernel {
    /// `u'v`.
    Linear,
    /// `(gamma u'v + coef0)^degree`, by repeated squaring.
    Polynomial { degree: i32, gamma: f64, coef0: f64 },
    /// `exp(-gamma |u-v|^2)`.
    Rbf { gamma: f64 },
    /// `tanh(gamma u'v + coef0)`.
    Sigmoid { gamma: f64, coef0: f64 },
}

impl Kernel {
    /// The name in a model file.
    pub const fn name(&self) -> &'static str {
        match self {
            Kernel::Linear => "linear",
            Kernel::Polynomial { .. } => "polynomial",
            Kernel::Rbf { .. } => "rbf",
            Kernel::Sigmoid { .. } => "sigmoid",
        }
    }
}

/// Sparse vectors in one table: vector `i` is
/// `index[start[i]..start[i + 1]]` with its `value`s, indexes strictly
/// ascending.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseRows {
    pub(crate) start: Vec<u32>,
    pub(crate) index: Vec<i32>,
    pub(crate) value: Vec<f64>,
}

impl SparseRows {
    pub(crate) fn new() -> SparseRows {
        SparseRows { start: vec![0], index: Vec::new(), value: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.start.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row `i`'s indexes and values.
    #[inline]
    pub fn row(&self, i: usize) -> (&[i32], &[f64]) {
        let (a, b) = (self.start[i] as usize, self.start[i + 1] as usize);
        (&self.index[a..b], &self.value[a..b])
    }

    pub(crate) fn push(&mut self, index: i32, value: f64) {
        self.index.push(index);
        self.value.push(value);
    }

    pub(crate) fn end_row(&mut self) {
        self.start.push(self.index.len() as u32);
    }
}

/// Platt's sigmoid per class pair, `probA` and `probB`, for C-SVC and
/// nu-SVC probability estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseSigmoid {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}

/// A trained LIBSVM model, as `svm_load_model` reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct LibsvmModel {
    pub(crate) svm_type: SvmType,
    pub(crate) kernel: Kernel,
    pub(crate) nr_class: usize,
    /// One per class pair (`nr_class * (nr_class - 1) / 2`), or one for a
    /// one-class or regression model.
    pub(crate) rho: Vec<f64>,
    /// The class labels, in the order LIBSVM met them; classification only.
    pub(crate) label: Vec<i32>,
    /// The support vectors of each class, in label order; classification only.
    pub(crate) n_sv: Vec<u32>,
    /// `nr_class - 1` coefficient columns of `l` each: column `j` row `i`
    /// at `sv_coef[j * l + i]`.
    pub(crate) sv_coef: Vec<f64>,
    pub(crate) sv: SparseRows,
    pub(crate) probability: Option<PairwiseSigmoid>,
    /// One-class probability marks, `prob_density_marks` (ten).
    pub(crate) density_marks: Option<Vec<f64>>,
    /// The probability parameter of a regression model, `probA[0]`.
    pub(crate) svr_probability: Option<f64>,
}

impl LibsvmModel {
    pub fn svm_type(&self) -> SvmType {
        self.svm_type
    }

    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// `svm_get_nr_class`.
    pub fn nr_class(&self) -> usize {
        self.nr_class
    }

    /// `svm_get_labels`; empty for a one-class or regression model.
    pub fn labels(&self) -> &[i32] {
        &self.label
    }

    /// The number of support vectors, `l`.
    pub fn total_sv(&self) -> usize {
        self.sv.len()
    }

    pub fn support_vectors(&self) -> &SparseRows {
        &self.sv
    }

    /// `rho`, one per class pair.
    pub fn rho(&self) -> &[f64] {
        &self.rho
    }

    /// The coefficients of support vector `i` in column `j`.
    pub fn sv_coef(&self, j: usize, i: usize) -> f64 {
        self.sv_coef[j * self.sv.len() + i]
    }

    /// The sigmoid parameters of every class pair, when the model was
    /// trained with probability estimates.
    pub fn pairwise_sigmoid(&self) -> Option<&PairwiseSigmoid> {
        self.probability.as_ref()
    }

    /// `svm_check_probability_model`.
    pub fn has_probability(&self) -> bool {
        match self.svm_type {
            SvmType::CSvc | SvmType::NuSvc => self.probability.is_some(),
            SvmType::OneClass => self.density_marks.is_some(),
            SvmType::EpsilonSvr | SvmType::NuSvr => self.svr_probability.is_some(),
        }
    }

    /// How many decision values `predict_values` writes: one per class
    /// pair, or one.
    pub fn num_decision_values(&self) -> usize {
        if self.svm_type.is_classification() { self.nr_class * (self.nr_class - 1) / 2 } else { 1 }
    }
}
