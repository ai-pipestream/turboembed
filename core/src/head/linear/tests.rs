use super::*;

fn layer() -> Linear {
    Linear::new(2, 1, vec![1.0, 2.0], vec![0.0]).unwrap()
}

#[test]
fn mutations_invalidate_the_lazy_packed_copy() {
    let mut l = layer();
    assert!(l.matrix.get().is_none(), "construction does not pack unused weights");
    let mut out = [0.0];
    l.apply(&[1.0, 1.0], &mut out);
    assert_eq!(out, [3.0]);
    assert!(l.matrix.get().is_some());
    l.weight_mut()[0] = 3.0;
    assert!(l.matrix.get().is_none());
    l.apply(&[1.0, 1.0], &mut out);
    assert_eq!(out, [5.0]);
    l.bias_mut()[0] = 1.0;
    assert!(l.matrix.get().is_none());
    l.apply(&[1.0, 1.0], &mut out);
    assert_eq!(out, [6.0]);
    let clone = l.clone();
    assert_eq!(l, clone);
    assert!(clone.matrix.get().is_none(), "cloning does not pack an unused decoder");
    clone.apply(&[1.0, 1.0], &mut out);
    assert_eq!(out, [6.0]);
}

#[test]
fn products_never_hold_the_initialization_lock() {
    let l = layer();
    for _ in 0..2 {
        assert!(l.with_matrix(|_| {
            assert!(l.initialize.try_lock().is_ok(), "first and warm products run without the lock");
        }));
    }
}

/// Mutating through the accessors re-arms the finiteness scan, so a
/// layer made non-finite after a passing validate is refused again.
#[test]
fn mutation_rearms_the_finiteness_scan() {
    let mut l = layer();
    assert_eq!(l.validate("test"), Ok(()));
    assert_eq!(l.validate("test"), Ok(()));
    l.weight_mut()[0] = f32::NAN;
    assert_eq!(l.validate("test").unwrap_err().code, crate::status::INVALID_ARGUMENT);
    l.weight_mut()[0] = 1.0;
    assert_eq!(l.validate("test"), Ok(()));
    l.bias_mut()[0] = f32::INFINITY;
    assert_eq!(l.validate("test").unwrap_err().code, crate::status::INVALID_ARGUMENT);
}

#[test]
fn shared_first_use_and_warm_products_agree() {
    let l = layer();
    let gate = std::sync::Barrier::new(8);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                gate.wait();
                for _ in 0..20 {
                    let mut out = [0.0];
                    l.apply(&[1.0, 1.0], &mut out);
                    assert_eq!(out, [3.0]);
                }
            });
        }
    });
}
