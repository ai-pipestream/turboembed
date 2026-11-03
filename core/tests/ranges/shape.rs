//! The shape of a range run's result.

use std::ptr;

use crate::common::*;
use crate::support::*;
use turbo::*;

#[test]
fn a_range_run_reports_its_shape() {
    let l = tiny();
    let s = session_of(&l, 6);
    let (t, lens) = tokens_of(&tiny_bundle(), &RANGES_TEXTS);
    let ranges = [range(0, 0, 4), range(0, 4, lens[0] as u32), range(1, 0, 2), range(2, 1, 3)];
    let count = ranges.len();
    for (normalize, pool_then_normalize) in
        [(TURBO_NORMALIZE_L2, TURBO_STAGE_HOST), (TURBO_NORMALIZE_NONE, TURBO_STAGE_UNUSED)]
    {
        s.write_ranges(&t.batch(), &ranges, Some(&normalized(normalize))).unwrap();
        let r = s.run().unwrap();
        let i = r.info();
        assert_eq!((i.batch as usize, i.dim), (count, 32));
        assert_eq!(i.bytes, (count * 32 * 4) as u64);
        assert_eq!((i.dtype, i.placement), (TURBO_DTYPE_F32, TURBO_PLACE_HOST));
        assert_eq!(i.stage[TURBO_EMBED_STAGE_POOL], TURBO_STAGE_HOST);
        assert_eq!(i.stage[TURBO_EMBED_STAGE_NORMALIZE], pool_then_normalize, "normalize {normalize}");

        let mut flat = vec![f32::NAN; count * 32];
        let mut written = 0u64;
        let rc = unsafe {
            turbo_result_read(r.0, flat.as_mut_ptr().cast(), (count * 32 * 4) as u64, &mut written, ptr::null_mut())
        };
        assert_eq!((rc, written), (0, (count * 32 * 4) as u64));
        assert!(flat.iter().all(|v| v.is_finite()));

        let mut buf = ptr::null_mut();
        assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buf, ptr::null_mut()) }, 0);
        let mut d: turbo_buffer_desc = unsafe { std::mem::zeroed() };
        d.struct_size = size_of::<turbo_buffer_desc>() as u32;
        let mut host = ptr::null_mut();
        unsafe {
            assert_eq!(turbo_buffer_get_desc(buf, &mut d, ptr::null_mut()), 0);
            assert_eq!(turbo_buffer_host_ptr(buf, &mut host, ptr::null_mut()), 0);
        }
        assert_eq!((d.ndim, d.shape, d.bytes), (2, [count as u64, 32], (count * 32 * 4) as u64));
        let in_place = unsafe { std::slice::from_raw_parts(host as *const f32, count * 32) };
        assert_eq!(bits(in_place), bits(&flat), "the buffer is the vectors where they are");
        unsafe { turbo_buffer_release(buf) };
    }
}
