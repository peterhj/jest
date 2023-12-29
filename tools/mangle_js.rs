extern crate jest;

use std::env;

fn main() {
  let args: Vec<_> = env::args().collect();
  if args.len() <= 1 {
    println!("usage: jest <input.js> [<global> ...]");
    return;
  }
  let src_path = &args[1];
  let globls = &args[2 .. ];
  let res = jest::mangle_js_file(src_path, globls);
  match res {
    Err(e) => {
      eprintln!("DEBUG:  mangle_js: err={:?}", e);
    }
    Ok(s) => {
      print!("{}", s.as_raw_str());
    }
  }
}
