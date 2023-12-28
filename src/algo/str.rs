use crate::algo::*;

use std::convert::{TryInto};
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};

pub fn safe_ascii(s: &[u8]) -> SmolStr {
  let mut buf = String::new();
  for &x in s.iter() {
    if x <= 0x20 {
      buf.push(' ');
    } else if x < 0x7f {
      buf.push(x.try_into().unwrap());
    } else {
      buf.push('?');
    }
  }
  buf.into()
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SafeStr {
  raw:  SmolStr,
}

impl From<SmolStr> for SafeStr {
  fn from(raw: SmolStr) -> SafeStr {
    SafeStr{raw}
  }
}

impl From<String> for SafeStr {
  fn from(s: String) -> SafeStr {
    SafeStr{raw: s.into()}
  }
}

impl<'a> From<&'a str> for SafeStr {
  fn from(s: &'a str) -> SafeStr {
    SafeStr{raw: s.into()}
  }
}

impl SafeStr {
  pub fn into_raw(self) -> SmolStr {
    self.raw
  }

  pub fn as_raw_str(&self) -> &str {
    self.raw.as_str()
  }
}

impl Debug for SafeStr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{:?}", safe_ascii(self.raw.as_bytes()))
  }
}

impl Display for SafeStr {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}", safe_ascii(self.raw.as_bytes()))
  }
}

/* json-like string (un-)escape via rustc_serialize:

Copyright (c) 2014 The Rust Project Developers

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE. */

fn decode_hex_escape<I: Iterator<Item=char>>(src: &mut I) -> Result<u16, ()> {
  let mut i = 0;
  let mut n = 0;
  while i < 4 {
    let c = match src.next() {
      Some(c) => c,
      None => return Err(())
    };
    n = match c {
      c @ '0' ..= '9' => n * 16 + ((c as u16) - ('0' as u16)),
      c @ 'a' ..= 'f' => n * 16 + (10 + (c as u16) - ('a' as u16)),
      c @ 'A' ..= 'F' => n * 16 + (10 + (c as u16) - ('A' as u16)),
      _ => return Err(())
    };
    i += 1;
  }
  Ok(n)
}

pub fn unescape_str<I: Iterator<Item=char>>(src: &mut I, delim: char) -> Result<String, ()> {
  let c = match src.next() {
    None => {
      return Err(());
    }
    Some(c) => c
  };
  if c != delim {
    return Err(());
  }

  let mut dst = String::new();
  let mut escape = false;

  loop {
    let c = match src.next() {
      None => {
        return Err(());
      }
      Some(c) => c
    };

    if escape {
      match c {
        '"' => dst.push('"'),
        '\\' => dst.push('\\'),
        '/' => dst.push('/'),
        'b' => dst.push('\x08'),
        'f' => dst.push('\x0c'),
        'n' => dst.push('\n'),
        'r' => dst.push('\r'),
        't' => dst.push('\t'),
        'u' => match decode_hex_escape(src)? {
          0xDC00 ..= 0xDFFF => {
            //return self.error(LoneLeadingSurrogateInHexEscape)
            return Err(());
          }

          // Non-BMP characters are encoded as a sequence of
          // two hex escapes, repdstenting UTF-16 surrogates.
          n1 @ 0xD800 ..= 0xDBFF => {
            match (src.next(), src.next()) {
              (Some('\\'), Some('u')) => (),
              //_ => return self.error(UnexpectedEndOfHexEscape),
              _ => return Err(())
            }

            let n2 = decode_hex_escape(src)?;
            if n2 < 0xDC00 || n2 > 0xDFFF {
              //return self.error(LoneLeadingSurrogateInHexEscape)
              return Err(());
            }
            let c = (((n1 - 0xD800) as u32) << 10 |
                 (n2 - 0xDC00) as u32) + 0x1_0000;
            dst.push(char::from_u32(c).unwrap());
          }

          n => match char::from_u32(n as u32) {
            Some(c) => dst.push(c),
            //None => return self.error(InvalidUnicodeCodePoint),
            None => return Err(())
          },
        },
        //_ => return self.error(InvalidEscape),
        _ => return Err(())
      }
      escape = false;
    } else if c == '\\' {
      escape = true;
    } else {
      if c == delim {
        return Ok(dst);
      } else if c <= '\u{1F}' {
        //return self.error(ControlCharacterInString),
        return Err(());
      } else {
        dst.push(c);
      }
    }
  }
}

pub fn escape_str(src: &str) -> String {
  let mut dst = String::new();
  dst.push_str("\"");

  let mut start = 0;

  for (i, byte) in src.bytes().enumerate() {
    let escaped = match byte {
      b'"' => "\\\"",
      b'\\' => "\\\\",
      b'\x00' => "\\u0000",
      b'\x01' => "\\u0001",
      b'\x02' => "\\u0002",
      b'\x03' => "\\u0003",
      b'\x04' => "\\u0004",
      b'\x05' => "\\u0005",
      b'\x06' => "\\u0006",
      b'\x07' => "\\u0007",
      b'\x08' => "\\b",
      b'\t' => "\\t",
      b'\n' => "\\n",
      b'\x0b' => "\\u000b",
      b'\x0c' => "\\f",
      b'\r' => "\\r",
      b'\x0e' => "\\u000e",
      b'\x0f' => "\\u000f",
      b'\x10' => "\\u0010",
      b'\x11' => "\\u0011",
      b'\x12' => "\\u0012",
      b'\x13' => "\\u0013",
      b'\x14' => "\\u0014",
      b'\x15' => "\\u0015",
      b'\x16' => "\\u0016",
      b'\x17' => "\\u0017",
      b'\x18' => "\\u0018",
      b'\x19' => "\\u0019",
      b'\x1a' => "\\u001a",
      b'\x1b' => "\\u001b",
      b'\x1c' => "\\u001c",
      b'\x1d' => "\\u001d",
      b'\x1e' => "\\u001e",
      b'\x1f' => "\\u001f",
      b'\x7f' => "\\u007f",
      _ => { continue; }
    };

    if start < i {
      dst.push_str(&src[start..i]);
    }

    dst.push_str(escaped);

    start = i + 1;
  }

  if start != src.len() {
    dst.push_str(&src[start..]);
  }

  dst.push_str("\"");
  dst
}
