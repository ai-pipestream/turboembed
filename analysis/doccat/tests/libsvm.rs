//! The LIBSVM text model reader and the predictor on hand-written models,
//! whose decision values and probabilities follow from LIBSVM's formulas.

use turbo_doccat::svm::libsvm::{self, Kernel, PredictScratch, SvmType};
use turbo_doccat::svm::{File, LineError, ModelError};

/// Two classes, a linear kernel: the decision value is
/// `x1 - x3 - rho`, as `sv_coef` is 1 for the first class's vector and
/// -1 for the second's.
const LINEAR: &str = "svm_type c_svc
kernel_type linear
nr_class 2
total_sv 2
rho 0.25
label 0 1
probA -2.0
probB 0.0
nr_sv 1 1
SV
1.0 1:1.0
-1.0 3:1.0
";

/// The model zlibsvm 3.0.0 wrote for a corpus of one category: no support
/// vectors, no class pairs.
const ONE_CATEGORY: &str = "svm_type c_svc
kernel_type rbf
gamma 0.3333333333333333
nr_class 1
total_sv 0
rho
label 0
probA
probB
nr_sv 0
SV
";

/// Three classes, one support vector each, a polynomial kernel.
const THREE: &str = "svm_type nu_svc
kernel_type polynomial
degree 2
gamma 0.5
coef0 1.0
nr_class 3
total_sv 3
rho 0.1 -0.2 0.3
label 2 0 1
probA -1.5 -1.0 -2.0
probB 0.1 0.2 -0.1
nr_sv 1 1 1
SV
0.5 0.25 1:1.0 2:2.0
-0.5 0.75 2:1.0
-0.25 -0.75 3:4.0
";

fn read(text: &str) -> libsvm::LibsvmModel {
    libsvm::read(text).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn a_two_class_model_reads_its_header_and_vectors() {
    let m = read(LINEAR);
    assert_eq!(m.svm_type(), SvmType::CSvc);
    assert_eq!(m.kernel(), Kernel::Linear);
    assert_eq!(m.nr_class(), 2);
    assert_eq!(m.labels(), [0, 1]);
    assert_eq!(m.total_sv(), 2);
    assert_eq!(m.rho(), [0.25]);
    assert_eq!(m.sv_coef(0, 1), -1.0);
    assert_eq!(m.support_vectors().row(1), (&[3][..], &[1.0][..]));
    assert!(m.has_probability());
    assert_eq!(m.num_decision_values(), 1);
}

#[test]
fn the_linear_decision_value_is_the_weighted_sum_less_rho() {
    let m = read(LINEAR);
    let mut s = PredictScratch::new();
    assert_eq!(m.predict_values(&[1, 2], &[2.0, 7.0], &mut s), 0.0);
    assert_eq!(s.decision_values(), [2.0 - 0.25]);
    assert_eq!(m.predict_values(&[3], &[1.0], &mut s), 1.0);
    assert_eq!(s.decision_values(), [-1.25]);
    // A decision value of exactly 0 votes for the second class.
    assert_eq!(m.predict_values(&[1], &[0.25], &mut s), 1.0);
}

#[test]
fn two_class_probabilities_are_the_pair_sigmoid() {
    let m = read(LINEAR);
    let mut s = PredictScratch::new();
    // f = 1.75 * -2 = -3.5 < 0: 1 / (1 + e^-3.5).
    assert_eq!(m.predict_probability(&[1], &[2.0], &mut s), 0.0);
    let p = 1.0 / (1.0 + (-3.5f64).exp());
    assert_eq!(s.probabilities(), [p, 1.0 - p]);
    // f = -1.25 * -2 = 2.5 >= 0: e^-2.5 / (1 + e^-2.5).
    assert_eq!(m.predict_probability(&[3], &[1.0], &mut s), 1.0);
    let p = (-2.5f64).exp() / (1.0 + (-2.5f64).exp());
    assert_eq!(s.probabilities(), [p, 1.0 - p]);
}

#[test]
fn pair_probabilities_are_clamped_away_from_zero_and_one() {
    let m = read(LINEAR);
    let mut s = PredictScratch::new();
    m.predict_probability(&[1], &[1000.0], &mut s);
    assert_eq!(s.probabilities(), [1.0 - 1e-7, 1.0 - (1.0 - 1e-7)]);
}

#[test]
fn a_one_category_model_predicts_its_category_with_certainty() {
    let m = read(ONE_CATEGORY);
    assert_eq!(m.nr_class(), 1);
    assert!(m.has_probability());
    let mut s = PredictScratch::new();
    assert_eq!(m.predict_values(&[1], &[1.0], &mut s), 0.0);
    assert!(s.decision_values().is_empty());
    assert_eq!(m.predict_probability(&[], &[], &mut s), 0.0);
    assert_eq!(s.probabilities(), [1.0]);
}

#[test]
fn three_classes_vote_pair_by_pair_and_couple_their_probabilities() {
    let m = read(THREE);
    assert_eq!(m.kernel(), Kernel::Polynomial { degree: 2, gamma: 0.5, coef0: 1.0 });
    let x = (&[1, 2][..], &[1.0, 1.0][..]);
    // Kernel values (0.5 x.sv + 1)^2: sv0 = 3 -> 6.25, sv1 = 1 -> 2.25, sv2 = 0 -> 1.
    let k = [6.25, 2.25, 1.0];
    let mut s = PredictScratch::new();
    let label = m.predict_values(x.0, x.1, &mut s);
    // Pair (0,1): coef column 0 of sv0, column 0 of sv1; (0,2): column 1 of
    // sv0, column 0 of sv2; (1,2): column 1 of sv1 and of sv2.
    let d01 = 0.5 * k[0] + -0.5 * k[1] - 0.1;
    let d02 = 0.25 * k[0] + -0.25 * k[2] - -0.2;
    let d12 = 0.75 * k[1] + -0.75 * k[2] - 0.3;
    assert_eq!(s.decision_values(), [d01, d02, d12]);
    // Class 0 wins both of its pairs; its label is 2.
    assert_eq!(label, 2.0);
    assert_eq!(m.predict_probability(x.0, x.1, &mut s), 2.0);
    let p = s.probabilities();
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{p:?}");
    assert!(p[0] > p[1] && p[0] > p[2], "{p:?}");
}

#[test]
fn rbf_and_sigmoid_kernels_read_their_parameters() {
    let rbf = LINEAR.replace("kernel_type linear\n", "kernel_type rbf\ngamma 0.5\n");
    assert_eq!(read(&rbf).kernel(), Kernel::Rbf { gamma: 0.5 });
    let mut s = PredictScratch::new();
    read(&rbf).predict_values(&[1], &[1.0], &mut s);
    // |x - sv0|^2 = 0, |x - sv1|^2 = 2.
    assert_eq!(s.decision_values(), [1.0 - (-0.5f64 * 2.0).exp() - 0.25]);
    let sigmoid = LINEAR.replace("kernel_type linear\n", "kernel_type sigmoid\ngamma 0.5\ncoef0 -1.0\n");
    assert_eq!(read(&sigmoid).kernel(), Kernel::Sigmoid { gamma: 0.5, coef0: -1.0 });
}

#[test]
fn numbers_read_in_the_c_and_java_forms() {
    let c = LINEAR
        .replace("rho 0.25", "rho 2.5000000000000000e-01")
        .replace("1.0 1:1.0 ", "1.0E0 1:1.00000000000000000e+00");
    assert_eq!(read(&c), read(LINEAR));
}

#[test]
fn one_class_and_regression_models_have_one_decision_value() {
    let one = "svm_type one_class\nkernel_type linear\nnr_class 2\ntotal_sv 1\nrho 0.5\nSV\n2.0 1:1.0\n";
    let m = read(one);
    assert!(!m.has_probability());
    let mut s = PredictScratch::new();
    assert_eq!(m.predict_values(&[1], &[1.0], &mut s), 1.0);
    assert_eq!(s.decision_values(), [1.5]);
    assert_eq!(m.predict_values(&[1], &[0.0], &mut s), -1.0);
    let svr = one.replace("one_class", "epsilon_svr");
    assert_eq!(read(&svr).predict_values(&[1], &[1.0], &mut s), 1.5);
    let marks = one.replace("SV\n", "prob_density_marks 0 1 2 3 4 5 6 7 8 9\nSV\n");
    let m = read(&marks);
    assert!(m.has_probability());
    assert_eq!(m.predict_probability(&[1], &[2.0], &mut s), 1.0);
    // Decision value 3.5 lies below the fifth mark.
    assert_eq!(s.probabilities(), [0.4, 0.6]);
}

fn line_error(text: &str) -> (usize, LineError) {
    match libsvm::read(text) {
        Err(ModelError::Line { file: File::Libsvm, line, error }) => (line, error),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_malformed_model_names_the_line_and_the_problem() {
    assert_eq!(
        line_error(&LINEAR.replace("c_svc", "c_svm")).1,
        LineError::BadValue { field: "svm_type", text: "c_svm".into() }
    );
    assert_eq!(
        line_error(&LINEAR.replace("kernel_type linear", "kernel_type precomputed")),
        (2, LineError::PrecomputedKernel)
    );
    assert_eq!(
        line_error(&LINEAR.replace("rho 0.25", "rho 0.25 0.5")).1,
        LineError::Count { field: "rho", expected: 1, found: 2 }
    );
    assert_eq!(
        line_error(&LINEAR.replace("nr_class 2\n", "").replace("label", "nr_class 2\nlabel")).1,
        LineError::Before { field: "rho", needs: "nr_class" }
    );
    assert_eq!(
        line_error(&LINEAR.replace("total_sv 2\n", "total_sv 2\ntotal_sv 2\n")),
        (5, LineError::Repeated("total_sv"))
    );
    assert_eq!(
        line_error(&LINEAR.replace("probB 0.0", "probB zero")).1,
        LineError::BadValue { field: "probB", text: "zero".into() }
    );
    assert_eq!(
        line_error(&LINEAR.replace("-1.0 3:1.0", "-1.0 3:1.0 2:1.0")),
        (12, LineError::Unordered { field: "SV", index: 2 })
    );
    assert_eq!(
        line_error(&LINEAR.replace("-1.0 3:1.0", "-1.0 3=1.0")).1,
        LineError::BadValue { field: "SV", text: "3=1.0".into() }
    );
    assert_eq!(line_error(&LINEAR.replace("nr_sv", "bias 1\nnr_sv")).1, LineError::UnknownKeyword("bias".into()));
    assert_eq!(line_error(&format!("{LINEAR}1.0 1:1.0\n")), (13, LineError::Trailing));
}

#[test]
fn a_model_missing_records_says_which() {
    let missing = |text: &str| libsvm::read(text).unwrap_err().to_string();
    assert_eq!(missing(&LINEAR[..LINEAR.find("SV\n").unwrap()]), "LIBSVM model: no SV record");
    assert_eq!(missing(&LINEAR.replace("label 0 1\n", "")), "LIBSVM model: no label record");
    assert_eq!(missing(&LINEAR.replace("probB 0.0\n", "")), "LIBSVM model: no probB record");
    assert_eq!(missing(&LINEAR.replace("kernel_type linear", "kernel_type rbf")), "LIBSVM model: no gamma record");
    assert_eq!(
        missing(&LINEAR.replace("nr_sv 1 1", "nr_sv 1 2")),
        "LIBSVM model: nr_sv [1, 2] does not add up to total_sv 2"
    );
    assert_eq!(
        missing(&LINEAR.replace("total_sv 2", "total_sv 3").replace("nr_sv 1 1", "nr_sv 1 2")),
        "LIBSVM model: 3 support vectors expected, 2 found"
    );
    assert_eq!(missing(&LINEAR.replace("probA -2.0\n", "")), "LIBSVM model: no probA record");
    // A regression model keeps the Laplace parameter in probA.
    let svr = read(&LINEAR.replace("c_svc", "nu_svr"));
    assert!(svr.has_probability());
    assert!(svr.labels().is_empty());
    assert!(svr.pairwise_sigmoid().is_none());
}

#[test]
fn a_model_without_probabilities_says_so() {
    let m = read(&LINEAR.replace("probA -2.0\nprobB 0.0\n", ""));
    assert!(!m.has_probability());
    let mut s = PredictScratch::new();
    assert_eq!(m.predict_probability(&[1], &[2.0], &mut s), 0.0);
    assert!(s.probabilities().is_empty());
    assert_eq!(s.decision_values(), [1.75]);
}
