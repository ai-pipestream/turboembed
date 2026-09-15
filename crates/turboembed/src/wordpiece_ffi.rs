//! FFI for `include/wordpiece.h`. Write-through into arena i32/i64 rows.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, CString};
use std::os::raw::c_void;
use std::path::Path;
use std::ptr;

pub const WORDPIECE_OK: i32 = 0;

pub type wordpiece_vocab = c_void;

unsafe extern "C" {
    fn wordpiece_vocab_load(path: *const c_char, out: *mut *mut wordpiece_vocab) -> i32;
    fn wordpiece_vocab_destroy(v: *mut wordpiece_vocab);
    fn wordpiece_encode_sentence(
        v: *const wordpiece_vocab,
        utf8: *const c_char,
        utf8_len: usize,
        ids: *mut c_void,
        mask: *mut c_void,
        types: *mut c_void,
        pos: *mut c_void,
        seq: u32,
        stride: u32,
        elem_width: u32,
    ) -> i32;
    fn wordpiece_hot_alloc_counter() -> u64;
    fn wordpiece_hot_alloc_counter_reset();
}

/// SOLIDIFY 5: process-wide heap counter on the WordPiece hot path.
/// rustc 1.97+ requires `unsafe` at the call site for `unsafe extern "C"`.
pub fn hot_alloc_counter() -> u64 {
    unsafe { wordpiece_hot_alloc_counter() }
}

pub fn hot_alloc_counter_reset() {
    unsafe { wordpiece_hot_alloc_counter_reset() }
}

pub struct WordPiece {
    raw: *mut wordpiece_vocab,
}

unsafe impl Send for WordPiece {}
unsafe impl Sync for WordPiece {}

impl Drop for WordPiece {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { wordpiece_vocab_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

impl WordPiece {
    pub fn load(path: &Path) -> Result<Self, String> {
        let c = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| "wordpiece path is not a C string".to_string())?;
        let mut raw = ptr::null_mut();
        let st = unsafe { wordpiece_vocab_load(c.as_ptr(), &mut raw) };
        if st != WORDPIECE_OK || raw.is_null() {
            return Err(format!(
                "wordpiece_vocab_load failed for {}",
                path.display()
            ));
        }
        Ok(Self { raw })
    }

    /// Write [CLS] tokens [SEP] [PAD…] into caller i64 (ORT) or i32 rows.
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep the private wrapper aligned with the native write-through ABI"
    )]
    pub fn encode_sentence(
        &self,
        text: &str,
        ids: *mut c_void,
        mask: *mut c_void,
        types: *mut c_void,
        seq: u32,
        stride: u32,
        elem_width: u32,
    ) -> Result<(), String> {
        let st = unsafe {
            wordpiece_encode_sentence(
                self.raw,
                text.as_ptr() as *const c_char,
                text.len(),
                ids,
                mask,
                types,
                ptr::null_mut(),
                seq,
                stride,
                elem_width,
            )
        };
        if st != WORDPIECE_OK {
            return Err("wordpiece_encode_sentence write-through failed".into());
        }
        Ok(())
    }
}

/// Select native tokenization only from the model's validated tokenizer configuration.
/// An explicit tokenizer path is authoritative, including an unsupported result.
pub fn load_beside_model(model_path: &str, tokenizer_dir: Option<&str>) -> Option<WordPiece> {
    if let Some(path) = tokenizer_dir {
        let p = Path::new(path);
        let json = if p.is_dir() {
            p.join("tokenizer.json")
        } else {
            p.to_owned()
        };
        return (json.extension().is_some_and(|ext| ext == "json"))
            .then(|| WordPiece::load(&json).ok())
            .flatten();
    }
    let dir = Path::new(model_path).parent()?;
    let adjacent = dir.join("tokenizer.json");
    if adjacent.is_file() {
        return WordPiece::load(&adjacent).ok();
    }
    WordPiece::load(&dir.parent()?.join("tokenizer.json")).ok()
}
