#![forbid(unsafe_code)]

extern crate regex_syntax;
extern crate smol_str;

use crate::algo::str::{SafeStr};
use crate::env::*;
use crate::parsing::*;

use std::fs::{File};
use std::io::{Read, BufReader};
use std::path::{Path};
use std::str::{from_utf8};

pub mod algo;
pub mod env;
pub mod parsing;
pub mod re;

pub fn mangle_js<S: AsRef<str>>(src: &str, globls: &[S]) -> Result<SafeStr, (CharSpan, ParseErr)> {
  let mut env = Env::default();
  for globl in globls.iter() {
    let _ = env.fresh_global(globl.as_ref());
  }
  let trie = tokenizer_trie();
  let tokens = Tokenizer_::new(trie, src.chars());
  let mut parser = Parser_::new(tokens);
  loop {
    match parser.next_stmt(&mut env)? {
      None => {
        break;
      }
      Some(_) => {}
    }
  }
  env.mangle_idents();
  Ok(env.render_min().into())
}

pub fn mangle_js_file<P: AsRef<Path>, S: AsRef<str>>(src_path: P, globls: &[S]) -> Result<SafeStr, (CharSpan, ParseErr)> {
  let src_file = File::open(src_path).unwrap();
  let mut reader = BufReader::new(src_file);
  let mut buf = Vec::new();
  reader.read_to_end(&mut buf).unwrap();
  let src = from_utf8(&buf).unwrap();
  mangle_js(src, globls)
}
