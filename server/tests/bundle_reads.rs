//! The load path's own reads of a bundle's manifest fail the load with
//! the bundle's error (issue #2517, F14). Both helpers used to swallow a
//! failed `Bundle::open`: the task peek picked TASK_EMBED, and the head
//! listing answered no heads, so a bundle served without its head steps
//! and its requests were refused for a head it does provide.

mod common;

use common::*;
use turbo::TURBO_TASK_EMBED;
use turbo_kserve::progressive::heads;
use turbo_kserve::tasks::peek_task;

fn missing() -> String {
    tiny().join("no-such-bundle").to_str().unwrap().to_string()
}

#[test]
fn the_task_peek_fails_with_the_bundle_error() {
    assert_eq!(peek_task(tiny().to_str().unwrap()).unwrap(), TURBO_TASK_EMBED);
    let e = peek_task(&missing()).unwrap_err();
    assert_ne!(e.code, 0, "{e:?}");
    assert!(!e.message.is_empty(), "{e:?}");
}

#[test]
fn the_head_listing_fails_with_the_bundle_error() {
    assert!(heads::available(tiny().to_str().unwrap()).unwrap().is_empty(), "the tiny bundle has no heads");
    let e = heads::available(&missing()).unwrap_err();
    assert_ne!(e.code, 0, "{e:?}");
    assert!(!e.message.is_empty(), "{e:?}");
}
