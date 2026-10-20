//! turbo_model_load, turbo_model_get_info and turbo_model_release on the
//! CPU backend this build links, over a small BERT bundle each test writes:
//! docs/bundle.md loader rules 6 to 8 on top of 1 to 5, and where the
//! weights live once loaded.

mod common;

use std::ffi::c_void;
use std::fs;
use std::ptr;

use common::*;
use serde_json::{Value, json};
use turbo::status::*;
use turbo::*;

fn refused(f: &Fixture) -> Failure {
    match f.load() {
        Ok(_) => panic!("the model loaded"),
        Err(e) => e,
    }
}

fn with(name: &str, edit: impl FnOnce(&mut Value)) -> Failure {
    let mut f = Fixture::model(name);
    edit(&mut f.manifest);
    refused(&f)
}

/// The small BERT's tensors with `edit` applied to the one named `name`.
fn edited(name: &str, edit: impl FnOnce(&mut Tensor)) -> Vec<Tensor> {
    let mut t = tiny_weights(0);
    edit(t.iter_mut().find(|t| t.name == name).expect("the small BERT has it"));
    t
}

fn with_weights(name: &str, tensors: &[Tensor]) -> Failure {
    let mut f = Fixture::model(name);
    f.weights("weights/model.safetensors", tensors);
    refused(&f)
}

#[path = "models/artifacts.rs"]
mod artifacts;
#[path = "models/hef.rs"]
mod hef;
#[path = "models/info.rs"]
mod info;
#[path = "models/weights.rs"]
mod weights;
