//! Helpers the session tests share.

use std::ptr;

use crate::common::*;
use turbo::*;

pub fn session(l: &Loaded) -> Session {
    Session::create(l.m, None).unwrap_or_else(|e| panic!("{e:?}"))
}

pub fn null_err() -> *mut turbo_error {
    ptr::null_mut()
}
