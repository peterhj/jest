extern crate jest;

use std::env;
use std::fs::{File};
use std::io::{Read, BufReader};
use std::str::{from_utf8};

fn main() {
  let args: Vec<_> = env::args().collect();
  if args.len() <= 1 {
    println!("usage: jest <input.js> [<global> ...]");
    return;
  }
  let file = File::open(&args[1]).unwrap();
  let mut reader = BufReader::new(file);
  let mut buf = Vec::new();
  reader.read_to_end(&mut buf).unwrap();
  let src = from_utf8(&buf).unwrap();
  let globls = &args[2 .. ];
  let res = jest::mangle_js(src, globls);
  match res {
    Err(e) => {
      println!("DEBUG:  mangle_js: err={:?}", e);
    }
    Ok(s) => {
      print!("{}", s.as_raw_str());
    }
  }
}
