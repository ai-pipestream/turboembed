use super::*;

#[test]
fn lora_options_require_lora() {
    for option in ["--lora-alpha", "--lora-layers", "--lora-lr"] {
        let args = Args {
            rest: vec!["--package", "missing", "--hold-out", "location", option, "2"]
                .into_iter()
                .map(String::from)
                .collect(),
        };
        let error = train(&[], "wnut17", Path::new("missing"), false, args).unwrap_err();
        assert_eq!(error.to_string(), format!("{option} requires --lora"));
    }
}

#[test]
#[ignore = "loads the real small-v2.5 package and MultiCoNER corpus"]
fn long_guidelines_train_in_batches_with_exclusive_gold_spans() {
    let root = std::env::var_os("GLINER_V2_5_PACKAGE").expect("GLINER_V2_5_PACKAGE");
    let dir = std::env::var_os("TURBO_MULTICONER_DIR").expect("TURBO_MULTICONER_DIR");
    let corpus = turbo_trainer_recipes::multiconer::read(Path::new(&dir), true).unwrap();
    let (labels, _) = type_schema(std::slice::from_ref(&corpus)).unwrap();
    let held = ["location", "VisualWork", "Scientist", "Clothing"];
    let labels: Vec<String> = labels.into_iter().filter(|x| !held.contains(&x.as_str())).collect();
    let sentence = corpus
        .splits
        .train
        .iter()
        .find(|s| s.labels.iter().any(|tag| tag.strip_prefix("B-").is_some_and(|x| labels.iter().any(|y| y == x))))
        .expect("a sentence with a seen entity");
    let model = Manifest::load(&turbo_pins::require(&root)).unwrap();
    let guidelines = read_guidelines(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("records/open-span-guidelines-v1.json").display().to_string(),
    )
    .unwrap();
    let words: Vec<&str> = sentence.words.iter().map(String::as_str).collect();
    assert!(gliner_eval::batches(&model, &words, &labels, 0.8, Some(&guidelines)).unwrap().len() > 1);
    let trainer = OutputTrainer::load(
        model.config(),
        &std::fs::read(turbo_pins::require(&root).join("heads.safetensors")).unwrap(),
    )
    .unwrap();
    let (loss, grads, _) =
        crate::train_gliner_lora::sentence(&model, &trainer, sentence, &labels, 0.8, Some(&guidelines)).unwrap();
    assert!(loss > 0.0);
    assert!(grads[0].iter().any(|&x| x != 0.0));
}

#[test]
#[ignore = "loads the real NuNER package and WNUT corpus"]
fn nuner_linear_output_step_serializes_and_reloads() {
    let root = std::env::var_os("GLINER_NUNER_PACKAGE").expect("GLINER_NUNER_PACKAGE");
    let dir = std::env::var_os("TURBO_WNUT_DIR").expect("TURBO_WNUT_DIR");
    let corpus = turbo_trainer_recipes::wnut17::read(Path::new(&dir), true).unwrap();
    let labels: Vec<String> =
        turbo_trainer_recipes::wnut17::TYPES.iter().filter(|&&x| x != "location").map(|&x| x.to_owned()).collect();
    let sentence = corpus
        .splits
        .train
        .iter()
        .find(|s| s.labels.iter().any(|tag| tag.strip_prefix("B-").is_some_and(|x| labels.iter().any(|y| y == x))))
        .expect("a sentence with a seen entity");
    let original = std::fs::read(turbo_pins::require(&root).join("heads.safetensors")).unwrap();
    let mut model = Manifest::load(&turbo_pins::require(&root)).unwrap();
    let mut trainer = OutputTrainer::load(model.config(), &original).unwrap();
    assert_eq!(trainer.params().len(), 2);
    let (_, mut grads, _) = crate::train_gliner_lora::sentence(&model, &trainer, sentence, &labels, 0.8, None).unwrap();
    let mut optimizer = output_optimizer(Settings { lr: 1e-6, total: 1, ..Settings::default() }, &trainer).unwrap();
    trainer.step(&mut optimizer, &mut grads).unwrap();
    model.apply_output(&trainer).unwrap();
    let heads = trainer.write_heads(&original).unwrap();
    let out = turbo_scratch::Scratch::at(
        std::env::temp_dir().join(format!("turbo-nuner-linear-test-{}", std::process::id())),
    );
    Manifest::write_tuned(&turbo_pins::require(&root), &out, &heads).unwrap();
    let reloaded = Manifest::load(&out).unwrap();
    let words: Vec<&str> = sentence.words.iter().map(String::as_str).collect();
    let schema = gliner_eval::schema(&labels, 0.8, None).unwrap();
    assert_eq!(model.training_input(&words, &schema).unwrap().1, reloaded.training_input(&words, &schema).unwrap().1);
}
