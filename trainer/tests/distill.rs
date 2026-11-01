//! Gradient verification and dark-knowledge transfer tests for
//! teacher-to-student distillation (#1520).

use turbo::manifest::{Adapter, Bilstm, Decode, HashEmbed, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::rng::Rng;
use turbo_trainer::{
    Config, DistillExample, DistillOptions, DistillTask, DistillWork, Example, StudentGate, Tagger, distill_loss,
    distill_multi_head, distill_tagger, feature_align_loss, fit, make_distill_examples, measure_throughput,
    score_tagger, tagger_distill_loss,
};

const EPS: f64 = 1e-6;

fn random(rng: &mut Rng, n: usize, scale: f64) -> Vec<f64> {
    (0..n).map(|_| rng.uniform(scale)).collect()
}

fn check_grad(what: &str, x: &mut [f64], analytic: &[f64], f: &mut dyn FnMut(&[f64]) -> f64) {
    assert_eq!(x.len(), analytic.len(), "{what}");
    for i in 0..x.len() {
        let keep = x[i];
        x[i] = keep + EPS;
        let up = f(x);
        x[i] = keep - EPS;
        let down = f(x);
        x[i] = keep;
        let numeric = (up - down) / (2.0 * EPS);
        let tol = 1e-6 * (1.0 + numeric.abs().max(analytic[i].abs()));
        assert!((numeric - analytic[i]).abs() <= tol, "{what}[{i}]: numeric {numeric}, analytic {}", analytic[i]);
    }
}

fn spec(labels: &[&str], adapter: Option<(u32, u32)>, hash: bool) -> HeadSpec {
    HeadSpec {
        name: "distil_test".into(),
        layer: "turboembed:pos".into(),
        kind: HeadKind::TokenTagger,
        labels: labels.iter().map(|s| (*s).to_owned()).collect(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: adapter.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode: Decode::Argmax,
        biaffine: None,
        pos_input: None,
        hash_embed: hash.then_some(HashEmbed { dim: 2, rows: [8, 4, 4, 4] }),
        layer_mix: None,
        span: None,
        relation: None,
        coref: None,
        license_class: None,
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.pos".into(),
    }
}

#[test]
fn distill_loss_gradients_match_finite_differences_across_temperatures_and_alphas() {
    let mut rng = Rng::new(42);
    let (n, labels) = (4, 5);
    let gold = [2u32, 0, 4, 1];
    let teacher = random(&mut rng, n * labels, 2.5);
    for &temperature in &[1.0, 2.0, 4.0] {
        for &alpha in &[0.0, 0.5, 1.0] {
            let mut student = random(&mut rng, n * labels, 2.0);
            let mut dscores = vec![0.0; n * labels];
            let loss = distill_loss(&student, &teacher, &gold, labels, temperature, alpha, &mut dscores).unwrap();
            assert!(loss >= 0.0 && loss.is_finite());
            let what = format!("T={temperature},alpha={alpha}");
            check_grad(&what, &mut student, &dscores, &mut |s| {
                distill_loss(s, &teacher, &gold, labels, temperature, alpha, &mut vec![0.0; n * labels]).unwrap()
            });
        }
    }
}

#[test]
fn whole_student_tagger_distill_gradients_match_finite_differences() {
    let forms = ["The", "quick", "fox", "jumps"];
    let gold = [1u32, 2, 0, 3];
    let hidden = 4;
    let labels = 4;
    for (k, (adapter, hash)) in [(None, false), (Some((3, 1)), false), (Some((2, 1)), true)].into_iter().enumerate() {
        let s = spec(&["N", "V", "J", "D"], adapter, hash);
        let mut rng = Rng::new(100 + k as u64);
        let mut student = Tagger::<f64>::new(&s, hidden, &mut rng).unwrap();
        for i in 0..student.params().len() {
            student.param_mut(i).iter_mut().for_each(|v| *v += rng.uniform(0.25));
        }
        let features = random(&mut rng, gold.len() * hidden, 1.0);
        let teacher_scores = random(&mut rng, gold.len() * labels, 2.0);
        let f = hash.then_some(&forms[..]);
        for &(temperature, alpha) in &[(1.0, 0.0), (2.0, 0.5), (4.0, 1.0)] {
            let mut grads = student.zero_grads();
            let mut work = DistillWork::default();
            tagger_distill_loss(
                &student,
                &features,
                f,
                &teacher_scores,
                &gold,
                temperature,
                alpha,
                &mut grads,
                &mut work,
            )
            .unwrap();
            for (ti, g) in grads.iter().enumerate() {
                let mut p = student.params()[ti].clone();
                let name = format!("case {k} T={temperature} alpha={alpha} {}", student.names()[ti]);
                check_grad(&name, &mut p, g, &mut |v| {
                    let mut m = student.clone();
                    m.param_mut(ti).copy_from_slice(v);
                    tagger_distill_loss(
                        &m,
                        &features,
                        f,
                        &teacher_scores,
                        &gold,
                        temperature,
                        alpha,
                        &mut m.zero_grads(),
                        &mut DistillWork::default(),
                    )
                    .unwrap()
                });
            }
        }
    }
}

#[test]
fn feature_alignment_loss_gradients_match_finite_differences() {
    let mut rng = Rng::new(77);
    let (n, d_s, d_t) = (3, 4, 6);
    let mut proj_w = random(&mut rng, d_t * d_s, 0.6);
    let mut proj_b = random(&mut rng, d_t, 0.4);
    let mut h_s = random(&mut rng, n * d_s, 1.0);
    let h_t = random(&mut rng, n * d_t, 1.0);
    let weight = 0.75;
    let (mut dw, mut db, mut dx) = (vec![0.0; proj_w.len()], vec![0.0; proj_b.len()], vec![0.0; h_s.len()]);
    let loss =
        feature_align_loss(&proj_w, &proj_b, &h_s, &h_t, d_s, d_t, weight, &mut dw, &mut db, Some(&mut dx)).unwrap();
    assert!(loss > 0.0);
    let (b0, x0) = (proj_b.clone(), h_s.clone());
    check_grad("proj_w", &mut proj_w, &dw, &mut |w| {
        feature_align_loss(w, &b0, &x0, &h_t, d_s, d_t, weight, &mut vec![0.0; w.len()], &mut vec![0.0; d_t], None)
            .unwrap()
    });
    let w0 = proj_w.clone();
    check_grad("proj_b", &mut proj_b, &db, &mut |b| {
        feature_align_loss(&w0, b, &x0, &h_t, d_s, d_t, weight, &mut vec![0.0; w0.len()], &mut vec![0.0; d_t], None)
            .unwrap()
    });
    check_grad("student_features", &mut h_s, &dx, &mut |x| {
        feature_align_loss(&w0, &b0, x, &h_t, d_s, d_t, weight, &mut vec![0.0; w0.len()], &mut vec![0.0; d_t], None)
            .unwrap()
    });
}

#[test]
fn invalid_options_and_inputs_return_typed_errors() {
    let d = DistillOptions::default();
    assert!(d.validate().is_ok());
    for bad in [
        DistillOptions { temperature: 0.0, ..d },
        DistillOptions { temperature: -1.0, ..d },
        DistillOptions { temperature: f32::NAN, ..d },
        DistillOptions { alpha: -0.01, ..d },
        DistillOptions { alpha: 1.01, ..d },
        DistillOptions { alpha: f32::NAN, ..d },
        DistillOptions { max_accuracy_drop: -0.001, ..d },
        DistillOptions { epochs: 0, ..d },
        DistillOptions { batch_size: 0, ..d },
        DistillOptions { lr: 0.0, ..d },
        DistillOptions { warmup: 1.0, ..d },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }

    let mut dscores = vec![1.0f64; 6];
    assert!(distill_loss(&[0.0; 6], &[0.0; 6], &[0, 1], 3, 0.0, 0.5, &mut dscores).is_err());
    assert!(distill_loss(&[0.0; 6], &[0.0; 6], &[0, 1], 3, 2.0, 1.5, &mut dscores).is_err());
    assert!(distill_loss(&[0.0; 6], &[0.0; 6], &[0, 3], 3, 2.0, 0.5, &mut dscores).is_err());
    assert!(distill_loss(&[0.0; 5], &[0.0; 6], &[0, 1], 3, 2.0, 0.5, &mut dscores).is_err());
    assert_eq!(dscores, vec![1.0; 6], "refused calls leave dscores untouched");
}

fn synthetic_pair(seed: u64, count: usize, d_teacher: usize, d_student: usize) -> (Vec<Example>, Vec<Example>) {
    let mut rng = Rng::new(seed);
    let mut teacher_out = Vec::with_capacity(count);
    let mut student_out = Vec::with_capacity(count);
    for _ in 0..count {
        let len = 5;
        let mut t_features = Vec::with_capacity(len * d_teacher);
        let mut s_features = Vec::with_capacity(len * d_student);
        let mut labels = Vec::with_capacity(len);
        for _ in 0..len {
            let c = rng.below(4);
            labels.push(c as u32);
            for j in 0..d_teacher {
                let signal = if j % 4 == c { 1.2 } else { 0.0 };
                t_features.push((signal + rng.uniform(0.25)) as f32);
            }
            for j in 0..d_student {
                let signal = if j % 4 == c { 1.2 } else { 0.0 };
                s_features.push((signal + rng.uniform(0.25)) as f32);
            }
        }
        teacher_out.push(Example { features: t_features, labels: labels.clone(), forms: Vec::new() });
        student_out.push(Example { features: s_features, labels, forms: Vec::new() });
    }
    (teacher_out, student_out)
}

#[test]
fn soft_target_distillation_transfers_teacher_predictions_and_passes_gate() {
    let d_teacher = 12;
    let d_student = 4;
    let mut teacher_spec = spec(&["A", "B", "C", "D"], Some((8, 1)), false);
    teacher_spec.decode = Decode::Crf;
    let student_spec = spec(&["A", "B", "C", "D"], None, false);

    let (t_train, s_train) = synthetic_pair(1, 120, d_teacher, d_student);
    let (t_val, s_val) = synthetic_pair(2, 40, d_teacher, d_student);
    let (t_test, s_test) = synthetic_pair(3, 40, d_teacher, d_student);

    let cfg = Config { epochs: 20, batch: 16, lr: 1e-2, patience: 0, threads: 1, ..Config::default() };
    let (teacher, _) = fit(&teacher_spec, d_teacher, &t_train, &t_val, &cfg).unwrap();
    let teacher_viterbi_acc = score_tagger(&teacher, &t_test).unwrap();
    assert!(teacher_viterbi_acc >= 0.95);

    let d_train = make_distill_examples(&teacher, &t_train, &s_train).unwrap();
    let d_val = make_distill_examples(&teacher, &t_val, &s_val).unwrap();
    let d_test = make_distill_examples(&teacher, &t_test, &s_test).unwrap();

    // Pure soft-target distillation (alpha = 1.0) with corrupted training gold labels
    // still recovers high test accuracy solely from the teacher's dark-knowledge logits.
    let corrupted_train: Vec<DistillExample> = d_train
        .iter()
        .map(|e| DistillExample {
            features: e.features.clone(),
            teacher_scores: e.teacher_scores.clone(),
            teacher_tags: e.teacher_tags.clone(),
            labels: vec![0; e.labels.len()],
            forms: e.forms.clone(),
        })
        .collect();
    let opts = DistillOptions {
        epochs: 25,
        batch_size: 16,
        lr: 1.5e-2,
        temperature: 2.0,
        alpha: 1.0,
        max_accuracy_drop: 0.03,
        ..DistillOptions::default()
    };
    let (student, report) = distill_tagger(&student_spec, d_student, &corrupted_train, &d_val, &d_test, &opts).unwrap();
    assert_eq!(report.teacher_test_accuracy, teacher_viterbi_acc);
    assert!(report.teacher_test_accuracy >= 0.95, "{report:?}");
    assert!(report.student_test_accuracy >= 0.95, "{report:?}");
    assert!(report.within_margin, "{report:?}");

    // Student with Decode::Crf is refused until CRF parameters are distilled.
    let mut crf_student = student_spec.clone();
    crf_student.decode = Decode::Crf;
    assert!(distill_tagger(&crf_student, d_student, &corrupted_train, &d_val, &d_test, &opts).is_err());

    let throughput = measure_throughput(&teacher, &t_test, &student, &s_test, 5).unwrap();
    assert!(throughput.student_tokens_per_sec > 0.0 && throughput.teacher_tokens_per_sec > 0.0);
    let gate = StudentGate::evaluate(&report, throughput, opts.max_accuracy_drop, 0.1);
    assert!(gate.passed);
    gate.verify().unwrap();

    let strict_fail = StudentGate::evaluate(&report, throughput, -0.5, 0.1);
    assert!(strict_fail.verify().is_err());

    // Multi-head distillation across two tasks.
    let tasks = vec![
        DistillTask { spec: student_spec.clone(), train: d_train.clone(), val: d_val.clone(), test: d_test.clone() },
        DistillTask { spec: student_spec, train: d_train, val: d_val, test: d_test },
    ];
    let multi = distill_multi_head(d_student, &tasks, &DistillOptions { alpha: 0.7, ..opts }).unwrap();
    assert_eq!(multi.len(), 2);
    assert!(multi.iter().all(|(_, r)| r.within_margin));
}

#[test]
fn crf_teachers_with_identical_emissions_and_manual_examples_score_accurately() {
    let mut teacher_spec = spec(&["A", "B"], None, false);
    teacher_spec.decode = Decode::Crf;
    let student_spec = spec(&["A", "B"], None, false);

    // Build two CRF teachers with identical emission weights (all zero logits)
    // and different CRF start/transition/end parameters.
    let mut teacher_a = Tagger::<f32>::new(&teacher_spec, 2, &mut Rng::new(7)).unwrap();
    teacher_a.param_mut(0).fill(0.0);
    teacher_a.param_mut(1).fill(0.0);
    let mut teacher_b = teacher_a.clone();

    // teacher_a favors path [0, 0]; teacher_b favors path [1, 1] under identical emissions.
    teacher_a.param_mut(2).copy_from_slice(&[2.0, -2.0, -2.0, 2.0]);
    teacher_a.param_mut(3).copy_from_slice(&[2.0, -2.0]);
    teacher_a.param_mut(4).copy_from_slice(&[2.0, -2.0]);

    teacher_b.param_mut(2).copy_from_slice(&[2.0, -2.0, -2.0, 2.0]);
    teacher_b.param_mut(3).copy_from_slice(&[-2.0, 2.0]);
    teacher_b.param_mut(4).copy_from_slice(&[-2.0, 2.0]);

    let ex = Example { features: vec![1.0, 0.0, 0.0, 1.0], labels: vec![1, 1], forms: Vec::new() };
    let d_a = make_distill_examples(&teacher_a, std::slice::from_ref(&ex), std::slice::from_ref(&ex)).unwrap();
    let d_b = make_distill_examples(&teacher_b, std::slice::from_ref(&ex), std::slice::from_ref(&ex)).unwrap();
    assert_eq!(d_a[0].teacher_scores, d_b[0].teacher_scores, "identical emission scores");
    assert_eq!(d_a[0].teacher_tags, vec![0, 0]);
    assert_eq!(d_b[0].teacher_tags, vec![1, 1]);

    let opts = DistillOptions { epochs: 2, batch_size: 1, ..DistillOptions::default() };
    let (_, rep_a) = distill_tagger(&student_spec, 2, &d_a, &d_a, &d_a, &opts).unwrap();
    let (_, rep_b) = distill_tagger(&student_spec, 2, &d_b, &d_b, &d_b, &opts).unwrap();
    assert_eq!(rep_a.teacher_test_accuracy, score_tagger(&teacher_a, std::slice::from_ref(&ex)).unwrap());
    assert_eq!(rep_b.teacher_test_accuracy, score_tagger(&teacher_b, std::slice::from_ref(&ex)).unwrap());
    assert_eq!(rep_a.teacher_test_accuracy, 0.0);
    assert_eq!(rep_b.teacher_test_accuracy, 1.0);

    // Manually constructed DistillExample whose runtime-decoded teacher_tags ([1, 1])
    // differ from argmax(teacher_scores) ([0, 0]) is scored using teacher_tags.
    let manual = DistillExample {
        features: vec![1.0, 0.0, 0.0, 1.0],
        teacher_scores: vec![3.0, -1.0, 3.0, -1.0],
        teacher_tags: vec![1, 1],
        labels: vec![1, 1],
        forms: Vec::new(),
    };
    let (_, rep_manual) =
        distill_tagger(&student_spec, 2, std::slice::from_ref(&manual), &[], std::slice::from_ref(&manual), &opts)
            .unwrap();
    assert_eq!(rep_manual.teacher_val_accuracy, 1.0);
    assert_eq!(rep_manual.teacher_test_accuracy, 1.0);

    // Mismatched or out-of-range teacher_tags are refused.
    let bad_len = DistillExample { teacher_tags: vec![1], ..manual.clone() };
    let bad_tag = DistillExample { teacher_tags: vec![1, 2], ..manual };
    assert!(distill_tagger(&student_spec, 2, &[bad_len], &[], &[], &opts).is_err());
    assert!(distill_tagger(&student_spec, 2, &[bad_tag], &[], &[], &opts).is_err());
}
