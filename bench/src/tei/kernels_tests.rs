use super::*;

const TRACE: &str = "Start (ns),Duration (ns),CorrId,GrdX,Name
100,10,1,1,\"void gemm<float, (int)128>(params)\"
120,5,2,,[CUDA memcpy Host-to-Device]
140,20,3,4,ucopy_f32
300000200,30,4,1,\"void gemm<float, (int)128>(params)\"
300000250,10,5,1,ucopy_f32
300000300,3,6,,[CUDA memset]
";

#[test]
fn a_trace_reads_with_commas_in_its_names() {
    let a = parse_trace(TRACE).unwrap();
    assert_eq!(a.len(), 6);
    assert_eq!(a[0], Activity { start: 100, duration: 10, copy: false });
    assert!(a[1].copy && !a[2].copy && a[5].copy);
    assert!(parse_trace("Name,Start (ns),Duration (ns)\n").is_err(), "Name must be last");
    assert!(parse_trace("Start (ns),Duration (ns),Name\nx,1,k\n").is_err());
}

#[test]
fn requests_are_the_runs_between_idle_gaps() {
    let a = parse_trace(TRACE).unwrap();
    let runs = split(a.clone(), 2, SPLIT_NS).unwrap();
    assert_eq!(runs.iter().map(Vec::len).collect::<Vec<_>>(), [3, 3]);
    let r = requests(a.clone(), 2).unwrap();
    assert_eq!(r[0], Request { kernel_ns: 30, copy_ns: 5, kernels: 2 });
    assert_eq!(r[1], Request { kernel_ns: 40, copy_ns: 3, kernels: 2 });
    let k = kernel_time("nsys 2023.4\n", &r).unwrap();
    assert!(kernel_time("nsys", &[]).is_err(), "no requests is no kernel time");
    assert_eq!((k.profiler.as_str(), k.requests, k.kernels_p50), ("nsys 2023.4", 2, 2));
    assert_eq!((k.p50_ms, k.p99_ms), (30.0 / 1e6, 40.0 / 1e6));
    assert_eq!(k.copy_p50_ms, 3.0 / 1e6);
    // Three requests need two idle gaps; the trace has one.
    let wrong = requests(a, 3).unwrap_err();
    assert!(wrong.contains("fewer than the 2 idle gaps"), "{wrong}");
}

#[test]
fn a_long_gap_inside_a_request_is_cut_at_the_longest_gaps() {
    // Two requests of two kernels, the first with a pause of 2 SPLIT
    // inside it, the gap between them 3 SPLIT: the longest gap cuts.
    let act = |s| Activity { start: s, duration: 10, copy: false };
    let at = |k: u64| k * SPLIT_NS;
    let a = [act(0), act(at(2)), act(at(5)), act(at(5) + 100)];
    let r = requests(a.to_vec(), 2).unwrap();
    assert_eq!((r[0].kernels, r[1].kernels), (2, 2));
    // The same pause with the requests closer than it: the cut falls
    // inside the first, and the counts differ.
    let a = [act(0), act(at(3)), act(at(4) + 10), act(at(4) + 100)];
    let e = requests(a.to_vec(), 2).unwrap_err();
    assert!(e.contains("[1, 3]") && e.contains("median of 1"), "{e}");
    // Counts that vary a little are requests; a merged pair is not.
    let run = |at: u64, k: u64| (0..k).map(move |i| act(at + i * 100));
    let a: Vec<Activity> = run(0, 20).chain(run(at(2), 22)).chain(run(at(4), 19)).collect();
    assert_eq!(requests(a, 3).unwrap().iter().map(|r| r.kernels).collect::<Vec<_>>(), [20, 22, 19]);
    let a: Vec<Activity> = run(0, 20).chain(run(at(2), 40)).chain(run(at(4), 20)).collect();
    assert!(requests(a, 3).unwrap_err().contains("median of 20"));
    // Gaps all shorter than SPLIT are no requests.
    let a = [act(0), act(SPLIT_NS / 2), act(SPLIT_NS)];
    assert!(split(a.to_vec(), 2, SPLIT_NS).is_none());
    assert!(split(a.to_vec(), 1, SPLIT_NS).is_some_and(|r| r.len() == 1));
}

#[test]
fn a_kernel_time_longer_than_the_request_is_refused() {
    let k = |p50_ms| KernelTime {
        profiler: "nsys".into(),
        requests: 25,
        p50_ms,
        p99_ms: p50_ms,
        copy_p50_ms: 0.0,
        kernels_p50: 1,
    };
    assert!(check(&k(5.0), 9.9).is_ok());
    assert!(check(&k(9.9), 9.9).is_ok(), "equal is not longer");
    // The #1399 run: kernels traced in the end-to-end server read
    // 77 ms against an unprofiled TEI's 35 ms.
    let e = check(&k(77.14), 35.2).unwrap_err();
    assert!(e.contains("77.140 ms") && e.contains("35.200 ms") && e.contains("nothing is recorded"), "{e}");
}

#[test]
fn the_procedure_names_the_clocks() {
    let p = procedure(200, 30, &[2610, 795, 2595]);
    assert!(p.contains("min 795 MHz, median 2595 MHz") && p.contains("idle 50 ms"), "{p}");
    assert!(p.contains("30 untimed") && p.contains("another container"), "{p}");
    assert!(procedure(1, 0, &[]).contains("no SM clock read"));
}

#[test]
fn a_gap_is_measured_from_the_last_end() {
    // A long activity covers the next start: one run.
    let act = |(s, d)| Activity { start: s, duration: d, copy: false };
    let a = [(0, 2 * SPLIT_NS), (SPLIT_NS + 10, 10)].map(act);
    assert!(split(a.to_vec(), 2, SPLIT_NS).is_none());
    // A short one in between: the gap is from the long one's end, not
    // from the short one's, so the third is still in the run.
    let a = [(0, 2 * SPLIT_NS), (10, 10), (SPLIT_NS + 100, 10)].map(act);
    assert!(split(a.to_vec(), 2, SPLIT_NS).is_none());
}

#[test]
fn the_launch_script_keeps_the_images_library_path_for_the_router() {
    let l = launch();
    assert_eq!((l[0].as_str(), l[2].as_str()), ("-c", "sh"));
    assert!(l[1].contains("--env-var=LD_LIBRARY_PATH=\"$own\"") && l[1].ends_with("text-embeddings-router \"$@\""));
    let n = Nsys { target: "/x/t".into(), libs: Some("/x/l".into()), importer: None };
    let shown = mounts(&n, Path::new("/w"), true).join(" ");
    assert!(
        shown.contains("src=<tei-nsys>,") && shown.contains("src=<tei-nsys-libs>,") && shown.contains("src=<work>,")
    );
    assert!(!shown.contains("/x/"));
}
