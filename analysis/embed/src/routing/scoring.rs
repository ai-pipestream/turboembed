use super::{Role, Route, RoutingError};
use turbo_vector_index::{Hit, Space, VectorIndex};

/// Output plus the producer's space. Always retain the actual output identity;
/// do not relabel existing vectors with a newly selected model's identity.
#[derive(Clone, Copy, Debug)]
pub struct QueryVector<'a> {
    pub space: &'a Space,
    pub values: &'a [f32],
}
impl<H> Route<'_, H> {
    /// Score only when the selected configuration, the query producer and the
    /// destination index all agree. Even empty indexes and k=0 are checked.
    /// Numeric and length checks are also performed by the index.
    pub fn top_k<M>(&self, index: &VectorIndex<M>, query: QueryVector<'_>, k: usize) -> Result<Vec<Hit>, RoutingError> {
        if !matches!(self.capability.role, Role::DenseQuery | Role::StaticVectors) {
            return Err(RoutingError::NotDenseQuery(self.capability.role));
        }
        self.check_space(query.space)?;
        self.check_space(index.space())?;
        Ok(index.top_k(query.space, query.values, k)?)
    }
    fn check_space(&self, actual: &Space) -> Result<(), RoutingError> {
        if self.space != actual {
            return Err(RoutingError::CrossSpace {
                expected: Box::new(self.space.clone()),
                actual: Box::new(actual.clone()),
            });
        }
        Ok(())
    }
}
