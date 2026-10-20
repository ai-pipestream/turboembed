//! Adapt sequence objectives to the existing QN minimizer. Event training
//! retains its arithmetic, scratch layout and Java-compatible history.
use super::{QnParams, QnStop, minimizer::Minimizer, objective::NegLogLikelihood};
use std::cell::RefCell;
pub(super) trait Objective {
    fn dimension(&self) -> usize;
    fn scratch(&self) -> usize {
        0
    }
    fn chunks(&self) -> usize {
        1
    }
    fn value_at(&self, x: &[f64], scratch: &mut [f64]) -> f64;
    fn gradient_at(&self, x: &[f64], scratch: &mut [f64], gradient: &mut [f64], chunk: &mut [f64]);
}
impl Objective for NegLogLikelihood<'_> {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn scratch(&self) -> usize {
        self.num_outcomes
    }
    fn chunks(&self) -> usize {
        self.chunks.len()
    }
    fn value_at(&self, x: &[f64], scratch: &mut [f64]) -> f64 {
        self.value_at(x, scratch)
    }
    fn gradient_at(&self, x: &[f64], scratch: &mut [f64], gradient: &mut [f64], chunk: &mut [f64]) {
        self.gradient_at(x, scratch, gradient, chunk);
    }
}
type Function<'a> = dyn Fn(&[f64], &mut [f64]) -> Result<f64, String> + 'a;
struct Cached<'a> {
    dimension: usize,
    function: &'a Function<'a>,
    cache: RefCell<(Vec<f64>, f64, Vec<f64>)>,
    error: RefCell<Option<String>>,
}
impl Cached<'_> {
    fn evaluate(&self, x: &[f64]) {
        let mut c = self.cache.borrow_mut();
        if c.0 == x {
            return;
        }
        c.2.fill(0.0);
        match (self.function)(x, &mut c.2) {
            Ok(v) if v.is_finite() && c.2.iter().all(|g| g.is_finite()) => c.1 = v,
            result => {
                *self.error.borrow_mut() =
                    Some(result.err().unwrap_or_else(|| "non-finite objective or gradient".into()));
                c.1 = f64::NAN;
            }
        }
        c.0 = x.to_vec();
    }
}
impl Objective for Cached<'_> {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn value_at(&self, x: &[f64], _: &mut [f64]) -> f64 {
        self.evaluate(x);
        self.cache.borrow().1
    }
    fn gradient_at(&self, x: &[f64], _: &mut [f64], out: &mut [f64], _: &mut [f64]) {
        self.evaluate(x);
        out.copy_from_slice(&self.cache.borrow().2);
    }
}
/// Minimize a finite differentiable objective from zero with existing
/// L-BFGS/Armijo machinery. The callback includes any L2 penalty. L1 is
/// unsupported here, since this entry point returns unscaled parameters.
pub fn minimize_objective(
    dimension: usize,
    params: &QnParams,
    function: &Function<'_>,
) -> Result<(Vec<f64>, QnStop), String> {
    params.validate().map_err(|e| e.to_string())?;
    if dimension == 0
        || dimension > 100_000_000
        || params.iterations <= 0
        || params.l1_cost != 0.0
        || !params.l2_cost.is_finite()
    {
        return Err("invalid generic QN parameters".into());
    }
    let cached = Cached {
        dimension,
        function,
        cache: RefCell::new((Vec::new(), 0.0, vec![0.0; dimension])),
        error: RefCell::new(None),
    };
    let origin = vec![0.0; dimension];
    cached.evaluate(&origin);
    if let Some(e) = cached.error.borrow_mut().take() {
        return Err(e);
    }
    if cached.cache.borrow().2.iter().all(|&g| g == 0.0) {
        return Ok((origin, QnStop::GradientNorm));
    }
    let mut minimizer = Minimizer::new(&cached, params);
    let result = minimizer.minimize(&mut |_, _| {});
    if let Some(e) = cached.error.borrow_mut().take() {
        return Err(e);
    }
    Ok((minimizer.search.next_point, result.map_err(|e| e.to_string())?))
}
