//! The LIBSVM text model format, as `svm_save_model` writes it and
//! `svm_load_model` reads it: header records (`svm_type`, `kernel_type`,
//! `degree`, `gamma`, `coef0`, `nr_class`, `total_sv`, `rho`, `label`,
//! `probA`, `probB`, `prob_density_marks`, `nr_sv`), then `SV` and one
//! line per support vector: its `nr_class - 1` coefficients and its
//! `index:value` pairs.
//!
//! Numbers are read with Java's grammar (`Double.parseDouble`,
//! `Integer.parseInt`), so the shortest decimals the Java port writes and
//! the `%.17g` decimals of the C library both read back to the bits that
//! were written. The reader is stricter than LIBSVM's: a keyword must be
//! exact, a count must match, each support vector's indexes must ascend
//! (the kernels merge two sorted vectors), and the parameters a kernel
//! reads must be given rather than left at zero.

use turbo_jvm::text::{parse_double, parse_int};

use super::model::{Kernel, LibsvmModel, PairwiseSigmoid, SparseRows, SvmType};
use crate::svm::error::{File, LineError, ModelError};
use crate::svm::text::lines;

/// The number of `prob_density_marks` LIBSVM writes.
const DENSITY_MARKS: usize = 10;

fn line_error(line: usize, error: LineError) -> ModelError {
    ModelError::Line { file: File::Libsvm, line, error }
}

fn invalid(detail: String) -> ModelError {
    ModelError::Invalid { file: File::Libsvm, detail }
}

fn double(field: &'static str, text: &str) -> Result<f64, LineError> {
    parse_double(text).map_err(|_| LineError::BadValue { field, text: text.into() })
}

fn int(field: &'static str, text: &str) -> Result<i32, LineError> {
    parse_int(text).map_err(|_| LineError::BadValue { field, text: text.into() })
}

/// Exactly `n` doubles from the whitespace separated `args`.
fn doubles(field: &'static str, args: &str, n: usize) -> Result<Vec<f64>, LineError> {
    let out = args.split_ascii_whitespace().map(|t| double(field, t)).collect::<Result<Vec<_>, _>>()?;
    if out.len() != n {
        return Err(LineError::Count { field, expected: n, found: out.len() });
    }
    Ok(out)
}

fn ints(field: &'static str, args: &str, n: usize) -> Result<Vec<i32>, LineError> {
    let out = args.split_ascii_whitespace().map(|t| int(field, t)).collect::<Result<Vec<_>, _>>()?;
    if out.len() != n {
        return Err(LineError::Count { field, expected: n, found: out.len() });
    }
    Ok(out)
}

/// The header records as read, before they are checked against each other.
#[derive(Default)]
struct Header {
    svm_type: Option<SvmType>,
    kernel: Option<&'static str>,
    degree: Option<i32>,
    gamma: Option<f64>,
    coef0: Option<f64>,
    nr_class: Option<usize>,
    total_sv: Option<usize>,
    rho: Option<Vec<f64>>,
    label: Option<Vec<i32>>,
    prob_a: Option<Vec<f64>>,
    prob_b: Option<Vec<f64>>,
    density_marks: Option<Vec<f64>>,
    n_sv: Option<Vec<i32>>,
}

/// Sets `slot` once.
fn once<T>(slot: &mut Option<T>, field: &'static str, value: T) -> Result<(), LineError> {
    if slot.is_some() {
        return Err(LineError::Repeated(field));
    }
    *slot = Some(value);
    Ok(())
}

impl Header {
    /// The number of class pairs, which the per-pair records need first.
    fn pairs(&self, field: &'static str) -> Result<usize, LineError> {
        let k = self.nr_class.ok_or(LineError::Before { field, needs: "nr_class" })?;
        Ok(k * (k - 1) / 2)
    }

    fn classes(&self, field: &'static str) -> Result<usize, LineError> {
        self.nr_class.ok_or(LineError::Before { field, needs: "nr_class" })
    }

    fn record(&mut self, keyword: &str, args: &str) -> Result<(), LineError> {
        match keyword {
            "svm_type" => {
                let t = SvmType::from_name(args).ok_or(LineError::BadValue { field: "svm_type", text: args.into() })?;
                once(&mut self.svm_type, "svm_type", t)
            }
            "kernel_type" => {
                let k = match args {
                    "linear" => "linear",
                    "polynomial" => "polynomial",
                    "rbf" => "rbf",
                    "sigmoid" => "sigmoid",
                    "precomputed" => return Err(LineError::PrecomputedKernel),
                    _ => return Err(LineError::BadValue { field: "kernel_type", text: args.into() }),
                };
                once(&mut self.kernel, "kernel_type", k)
            }
            "degree" => once(&mut self.degree, "degree", int("degree", args)?),
            "gamma" => once(&mut self.gamma, "gamma", double("gamma", args)?),
            "coef0" => once(&mut self.coef0, "coef0", double("coef0", args)?),
            "nr_class" => {
                let k = int("nr_class", args)?;
                if k < 1 {
                    return Err(LineError::BadValue { field: "nr_class", text: args.into() });
                }
                once(&mut self.nr_class, "nr_class", k as usize)
            }
            "total_sv" => {
                let l = int("total_sv", args)?;
                if l < 0 {
                    return Err(LineError::BadValue { field: "total_sv", text: args.into() });
                }
                once(&mut self.total_sv, "total_sv", l as usize)
            }
            "rho" => {
                let v = doubles("rho", args, self.pairs("rho")?)?;
                once(&mut self.rho, "rho", v)
            }
            "label" => {
                let v = ints("label", args, self.classes("label")?)?;
                once(&mut self.label, "label", v)
            }
            "probA" => {
                let v = doubles("probA", args, self.pairs("probA")?)?;
                once(&mut self.prob_a, "probA", v)
            }
            "probB" => {
                let v = doubles("probB", args, self.pairs("probB")?)?;
                once(&mut self.prob_b, "probB", v)
            }
            "prob_density_marks" => {
                let v = doubles("prob_density_marks", args, DENSITY_MARKS)?;
                once(&mut self.density_marks, "prob_density_marks", v)
            }
            "nr_sv" => {
                let v = ints("nr_sv", args, self.classes("nr_sv")?)?;
                once(&mut self.n_sv, "nr_sv", v)
            }
            _ => Err(LineError::UnknownKeyword(keyword.into())),
        }
    }

    fn kernel(&self) -> Result<Kernel, ModelError> {
        let missing = |field| ModelError::Missing { file: File::Libsvm, field };
        let gamma = || self.gamma.ok_or(missing("gamma"));
        let coef0 = || self.coef0.ok_or(missing("coef0"));
        Ok(match self.kernel.ok_or(missing("kernel_type"))? {
            "linear" => Kernel::Linear,
            "polynomial" => {
                Kernel::Polynomial { degree: self.degree.ok_or(missing("degree"))?, gamma: gamma()?, coef0: coef0()? }
            }
            "rbf" => Kernel::Rbf { gamma: gamma()? },
            _ => Kernel::Sigmoid { gamma: gamma()?, coef0: coef0()? },
        })
    }
}

/// Reads a model from the text of a LIBSVM model file.
pub fn read(text: &str) -> Result<LibsvmModel, ModelError> {
    let mut lines = lines(text);
    let mut header = Header::default();
    loop {
        let Some((n, line)) = lines.next() else {
            return Err(ModelError::Missing { file: File::Libsvm, field: "SV" });
        };
        if line == "SV" {
            break;
        }
        let (keyword, args) = line.split_once(' ').unwrap_or((line, ""));
        header.record(keyword, args).map_err(|e| line_error(n, e))?;
    }
    let missing = |field| ModelError::Missing { file: File::Libsvm, field };
    let svm_type = header.svm_type.ok_or(missing("svm_type"))?;
    let kernel = header.kernel()?;
    let nr_class = header.nr_class.ok_or(missing("nr_class"))?;
    let l = header.total_sv.ok_or(missing("total_sv"))?;
    let rho = header.rho.take().ok_or(missing("rho"))?;
    let (label, n_sv) = if svm_type.is_classification() {
        let label = header.label.take().ok_or(missing("label"))?;
        let n_sv = header.n_sv.take().ok_or(missing("nr_sv"))?;
        if n_sv.iter().any(|&c| c < 0) || n_sv.iter().map(|&c| c as usize).sum::<usize>() != l {
            return Err(invalid(format!("nr_sv {n_sv:?} does not add up to total_sv {l}")));
        }
        (label, n_sv.into_iter().map(|c| c as u32).collect())
    } else {
        if nr_class != 2 {
            return Err(invalid(format!("a {} model has nr_class 2, not {nr_class}", svm_type.name())));
        }
        (Vec::new(), Vec::new())
    };
    // A regression model keeps its Laplace parameter in probA.
    let svr_probability = match svm_type {
        SvmType::EpsilonSvr | SvmType::NuSvr => header.prob_a.as_ref().and_then(|a| a.first().copied()),
        _ => None,
    };
    let probability = match (header.prob_a.take(), header.prob_b.take()) {
        (Some(a), Some(b)) if svm_type.is_classification() => Some(PairwiseSigmoid { a, b }),
        (Some(_), None) if svm_type.is_classification() => return Err(missing("probB")),
        (None, Some(_)) => return Err(missing("probA")),
        _ => None,
    };
    let density_marks = header.density_marks.take().filter(|_| svm_type == SvmType::OneClass);

    let columns = nr_class - 1;
    let mut sv_coef = vec![0.0; columns * l];
    let mut sv = SparseRows::new();
    for i in 0..l {
        let (n, line) = lines.next().ok_or_else(|| invalid(format!("{l} support vectors expected, {i} found")))?;
        read_vector(line, i, l, columns, &mut sv_coef, &mut sv).map_err(|e| line_error(n, e))?;
    }
    if let Some((n, _)) = lines.find(|(_, line)| !line.is_empty()) {
        return Err(line_error(n, LineError::Trailing));
    }
    Ok(LibsvmModel {
        svm_type,
        kernel,
        nr_class,
        rho,
        label,
        n_sv,
        sv_coef,
        sv,
        probability,
        density_marks,
        svr_probability,
    })
}

/// One support vector line: `columns` coefficients, then `index:value`
/// pairs with ascending indexes.
fn read_vector(
    line: &str,
    i: usize,
    l: usize,
    columns: usize,
    sv_coef: &mut [f64],
    sv: &mut SparseRows,
) -> Result<(), LineError> {
    let mut fields = line.split_ascii_whitespace();
    for j in 0..columns {
        let text = fields.next().ok_or(LineError::Count { field: "sv_coef", expected: columns, found: j })?;
        sv_coef[j * l + i] = double("sv_coef", text)?;
    }
    let mut last: Option<i32> = None;
    for pair in fields {
        let (index, value) = pair.split_once(':').ok_or(LineError::BadValue { field: "SV", text: pair.into() })?;
        let index = int("SV index", index)?;
        if last.is_some_and(|p| index <= p) {
            return Err(LineError::Unordered { field: "SV", index: i64::from(index) });
        }
        last = Some(index);
        sv.push(index, double("SV value", value)?);
    }
    sv.end_row();
    Ok(())
}
