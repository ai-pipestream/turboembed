//! The capability matrix: what the cpu offers for embed is what its
//! sessions run.

use crate::common::*;
use crate::support::*;
use turbo::*;

#[test]
fn the_cpu_offers_embed_as_its_sessions_run_it() {
    let l = tiny();
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let mut cap: turbo_capability = unsafe { std::mem::zeroed() };
        cap.struct_size = size_of::<turbo_capability>() as u32;
        assert_eq!(unsafe { turbo_runtime_capability(l.rt, cpu(l.rt), TURBO_TASK_EMBED, p, &mut cap, null_err()) }, 0);
        let info = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap().info();
        assert_eq!(cap.dtype, info.compute_dtype, "precision {p}");
        assert_eq!(cap.options_honored, 0b1111_1111, "every field of turbo_embed_options");
    }
}
