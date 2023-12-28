#![forbid(unsafe_code)]

extern crate regex_syntax;
extern crate smol_str;

use crate::algo::str::{SafeStr};
use crate::env::*;
use crate::parsing::*;

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
