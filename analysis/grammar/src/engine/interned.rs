//! Time values a parse builds, kept beside the chart so a node carries a
//! small index instead of the value itself.
//!
//! An equal value always gets the same index, so two nodes with equal
//! time values compare equal and the chart keeps the better one.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::time::TimeValue;
use crate::value::{TimeId, Value};

#[derive(Debug, Default)]
struct Table {
    values: Vec<TimeValue>,
    index: HashMap<TimeValue, u32>,
}

/// The time values of one parse. Rules add to it through a shared
/// reference while they build.
#[derive(Debug, Default)]
pub struct Interned(RefCell<Table>);

impl Interned {
    /// The node value for `v`.
    pub fn intern(&self, v: TimeValue) -> Value {
        let mut t = self.0.borrow_mut();
        let next = t.values.len() as u32;
        let id = *t.index.entry(v).or_insert(next);
        if id == next {
            t.values.push(v);
        }
        Value::Time(TimeId(id))
    }

    /// The time value a node value holds.
    pub fn get(&self, v: &Value) -> Option<TimeValue> {
        match *v {
            Value::Time(TimeId(id)) => self.0.borrow().values.get(id as usize).copied(),
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        self.0.borrow().values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        let mut t = self.0.borrow_mut();
        t.values.clear();
        t.index.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::spec::{DateSpec, Point};
    use turbo_analysis_layers::TimeGrain;

    #[test]
    fn equal_values_share_an_index() {
        let table = Interned::default();
        let today = TimeValue::point(Point::date(DateSpec::Rel { grain: TimeGrain::Day, n: 0 }));
        let tomorrow = TimeValue::point(Point::date(DateSpec::Rel { grain: TimeGrain::Day, n: 1 }));
        let a = table.intern(today);
        let b = table.intern(tomorrow);
        assert_ne!(a, b);
        assert_eq!(table.intern(today), a);
        assert_eq!(table.get(&b), Some(tomorrow));
        assert_eq!(table.get(&Value::Empty), None);
        assert_eq!(table.len(), 2);
        table.clear();
        assert!(table.is_empty());
    }
}
