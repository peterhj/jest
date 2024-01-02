use crate::algo::*;
use crate::algo::str::*;
use crate::env::*;
use crate::re::{ReTrie, len_utf8};

use std::cmp::{min};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::iter::{Peekable, once};
use std::mem::{replace};
use std::panic::{Location};
use std::rc::{Rc};

#[derive(Clone)]
pub struct CharCursor<I> {
  pos:  usize,
  buf:  String,
  gen:  I,
}

impl<I: Iterator<Item=char>> CharCursor<I> {
  pub fn new(gen: I) -> CharCursor<I> {
    CharCursor{
      pos:  0,
      buf:  String::new(),
      gen,
    }
  }

  pub fn lookahead<'a>(&'a mut self) -> CharCursorLookahead<'a, I> {
    CharCursorLookahead{
      pos:  self.pos,
      buf:  &mut self.buf,
      gen:  &mut self.gen,
    }
  }

  pub fn buffer_ahead(&mut self, capacity: usize) {
    if self.pos + capacity <= self.buf.len() {
      return;
    }
    while !(self.pos + capacity <= self.buf.len()) {
      let c = match self.gen.next() {
        None => return,
        Some(c) => c
      };
      self.buf.push(c);
    }
  }

  pub fn to_str(&self) -> &str {
    self.buf.get(self.pos .. ).unwrap()
  }

  pub fn advance(&mut self, offset: usize) {
    // FIXME: utf8 char boundary.
    //self.pos += offset;
    self.pos = min(self.pos + offset, self.buf.len());
    assert!(self.pos <= self.buf.len());
    match self.buf.get(self.pos .. ) {
      None => panic!("bug"),
      Some(_) => {}
    }
  }

  pub fn peek(&mut self) -> Option<char> {
    if self.pos > self.buf.len() {
      panic!("bug");
    } else if self.pos < self.buf.len() {
      let c = self.buf.get(self.pos .. ).unwrap().chars().next().unwrap();
      return Some(c);
    } else if self.pos > 0 /*&& self.pos == self.buf.borrow().len() */{
      self.buf.clear();
      self.pos = 0;
    }
    let c = self.gen.next()?;
    self.buf.push(c);
    Some(c)
  }
}

impl<I: Iterator<Item=char>> Iterator for CharCursor<I> {
  type Item = char;

  fn next(&mut self) -> Option<char> {
    if self.pos > self.buf.len() {
      panic!("bug");
    } else if self.pos < self.buf.len() {
      let c = self.buf.get(self.pos .. ).unwrap().chars().next().unwrap();
      self.pos += len_utf8(c as _);
      return Some(c);
    } else if self.pos > 0 /*&& self.pos == self.buf.borrow().len() */{
      self.buf.clear();
      self.pos = 0;
    }
    self.gen.next()
  }
}

pub struct CharCursorLookahead<'a, I> {
  pos:  usize,
  buf:  &'a mut String,
  gen:  &'a mut I,
}

impl<'a, I: Iterator<Item=char>> CharCursorLookahead<'a, I> {
  pub fn position(&self) -> usize {
    self.pos
  }

  pub fn buffer(&self) -> &str {
    self.buf
  }

  pub fn to_str(&self) -> &str {
    self.buf.get(self.pos .. ).unwrap()
  }
}

impl<'a, I: Iterator<Item=char>> Iterator for CharCursorLookahead<'a, I> {
  type Item = char;

  fn next(&mut self) -> Option<char> {
    if self.pos > self.buf.len() {
      panic!("bug");
    } else if self.pos == self.buf.len() {
      match self.gen.next() {
        None => {
          return None;
        }
        Some(next_c) => {
          self.buf.push(next_c);
        }
      }
    }
    let c = self.buf.get(self.pos .. ).unwrap().chars().next().unwrap();
    self.pos += len_utf8(c as _);
    Some(c)
  }
}

#[derive(Clone, Copy)]
pub struct CharSpan {
  pub start: u32,
  pub end: u32,
  //pub fin: u32,
}

impl Debug for CharSpan {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    if self.is_noloc() {
      write!(f, "CharSpan(.)")
    } else {
      write!(f, "CharSpan({}:{})", self.start, self.end)
    }
  }
}

impl Default for CharSpan {
  fn default() -> CharSpan {
    CharSpan{start: u32::max_value(), end: u32::max_value()}
  }
}

impl CharSpan {
  pub fn is_noloc(&self) -> bool {
    self.start == u32::max_value() && self.end == u32::max_value()
  }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Token {
  Space,
  Newline,
  CommentNewline(SafeStr),
  Comma,
  Dot,
  Query,
  Bang,
  Tilde,
  Slash,
  //Backslash,
  Star,
  Plus,
  PlusPlus,
  PlusEqual,
  Dash,
  DashDash,
  DashEqual,
  Semi,
  Colon,
  Equal,
  EqualEqual,
  EqualEqualEqual,
  BangEqual,
  BangEqualEqual,
  Geq,
  Gt,
  Leq,
  Lt,
  Neq,
  Not,
  Amp,
  AmpAmp,
  Bar,
  BarBar,
  LParen,
  RParen,
  LBrack,
  RBrack,
  LCurly,
  RCurly,
  Null,
  Undefined,
  False,
  True,
  New,
  Function,
  For,
  If,
  Else,
  Return,
  Var,
  Int(i64),
  Float(i64, i64),
  Atom(SafeStr),
  Ident(SafeStr),
  _Eof,
  _Bot,
}

impl Token {
  pub fn try_int_from(s: &str) -> Result<Token, ()> {
    Ok(Token::Int(i64::from_str_radix(s, 10).map_err(|_| ())?))
  }

  pub fn try_float_from(s: &str) -> Result<Token, ()> {
    let parts: Vec<_> = s.split(".").collect();
    if parts.len() != 2 {
      return Err(());
    }
    let x = i64::from_str_radix(parts[0], 10).map_err(|_| ())?;
    let y = i64::from_str_radix(parts[1], 10).map_err(|_| ())?;
    Ok(Token::Float(x, y))
  }
}

pub fn tokenizer_trie() -> Rc<ReTrie<Token>> {
  let mut tr = ReTrie::default();
  tr.push(r"[ ]+",  |_| Token::Space);
  tr.push(r"//",    |_| Token::CommentNewline(SafeStr::default()));
  tr.push(r"/",     |_| Token::Slash);
  //tr.push(r"\\",    |_| Token::Backslash);
  tr.push(r"\n",    |_| Token::Newline);
  //tr.push(r"=>",    |_| Token::REqualArrow);
  tr.push(r"===",   |_| Token::EqualEqualEqual);
  tr.push(r"==",    |_| Token::EqualEqual);
  tr.push(r"=",     |_| Token::Equal);
  tr.push(r">=",    |_| Token::Geq);
  tr.push(r">",     |_| Token::Gt);
  tr.push(r"<=",    |_| Token::Leq);
  tr.push(r"<",     |_| Token::Lt);
  tr.push(r";",     |_| Token::Semi);
  tr.push(r":",     |_| Token::Colon);
  tr.push(r",",     |_| Token::Comma);
  tr.push(r"\.",    |_| Token::Dot);
  tr.push(r"\?",    |_| Token::Query);
  tr.push(r"!==",   |_| Token::BangEqualEqual);
  tr.push(r"!=",    |_| Token::BangEqual);
  tr.push(r"!",     |_| Token::Bang);
  tr.push(r"\&\&",  |_| Token::AmpAmp);
  tr.push(r"\&",    |_| Token::Amp);
  tr.push(r"\|\|",  |_| Token::BarBar);
  tr.push(r"\|",    |_| Token::Bar);
  tr.push(r"\+=",   |_| Token::PlusEqual);
  tr.push(r"\+\+",  |_| Token::PlusPlus);
  tr.push(r"\+",    |_| Token::Plus);
  tr.push(r"\-=",   |_| Token::DashEqual);
  tr.push(r"\-\-",  |_| Token::DashDash);
  tr.push(r"\-",    |_| Token::Dash);
  tr.push(r"\*",    |_| Token::Star);
  tr.push(r"\(",    |_| Token::LParen);
  tr.push(r"\)",    |_| Token::RParen);
  tr.push(r"\[",    |_| Token::LBrack);
  tr.push(r"\]",    |_| Token::RBrack);
  tr.push(r"\{",    |_| Token::LCurly);
  tr.push(r"\}",    |_| Token::RCurly);
  tr.push(r"null",  |_| Token::Null);
  tr.push(r"undefined", |_| Token::Undefined);
  tr.push(r"false", |_| Token::False);
  tr.push(r"true",  |_| Token::True);
  tr.push(r"new ",  |_| Token::New);
  tr.push(r"function", |_| Token::Function);
  tr.push(r"for",   |_| Token::For);
  tr.push(r"if",    |_| Token::If);
  tr.push(r"else",  |_| Token::Else);
  tr.push(r"return", |_| Token::Return);
  tr.push(r"var",   |_| Token::Var);
  tr.push(r"[A-Za-z_\$][A-Za-z0-9_\$]*", |s| Token::Ident(s.into()));
  tr.push(r"[0-9]+\.[0-9]+", |s| Token::try_float_from(s).unwrap());
  tr.push(r"[0-9]+", |s| Token::try_int_from(s).unwrap());
  tr.into()
}

#[derive(Clone)]
pub struct Tokenizer_<I> {
  trie: Rc<ReTrie<Token>>,
  eof:  Option<CharSpan>,
  peek: Option<(CharSpan, Token, usize)>,
  off:  usize,
  src:  CharCursor<I>,
}

impl<I: Iterator<Item=char>> Tokenizer_<I> {
  pub fn new(trie: Rc<ReTrie<Token>>, src: I) -> Tokenizer_<I> {
    Tokenizer_{
      trie,
      eof:  None,
      peek: None,
      off:  0,
      src:  CharCursor::new(src),
    }
  }
}

impl<I: Iterator<Item=char>> Iterator for Tokenizer_<I> {
  type Item = (CharSpan, Token);

  fn next(&mut self) -> Option<(CharSpan, Token)> {
    //eprintln!("DEBUG:  Tokenizer::next: off={}", self.off);
    if let Some((next_span, next_tok, next_off)) = self.peek.take() {
      //eprintln!("DEBUG:  Tokenizer::next: next tok={:?} off={}", next_tok, next_off);
      self.off = next_off;
      return Some((next_span, next_tok));
    }
    if let Some(end_span) = self.eof {
      //eprintln!("DEBUG:  Tokenizer::next: eof");
      return Some((end_span, Token::_Eof));
    }
    let (mut span, mut tok) = {
      let c = self.src.peek()?;
      //eprintln!("DEBUG:  Tokenizer::next: c={:?}", c);
      //eprintln!("DEBUG:  Tokenizer::next: c={:?} ({})", c, c as u32);
      if c == '\"' {
        let mut it = self.src.lookahead();
        let pos0 = it.position();
        // NB: Correct position accounting requires `unescape_str`
        // to consume exactly the chars of the string and no more.
        let s = match unescape_str(&mut it, '\"') {
          Err(_) => {
            //eprintln!("DEBUG:  Tokenizer::next: eof 2");
            let start = self.off as _;
            let end = self.off as _;
            self.eof = Some(CharSpan{start, end});
            return Some((CharSpan{start, end}, Token::_Eof));
          }
          Ok(s) => s
        };
        //eprintln!("DEBUG:  Tokenizer::next: s (unescaped): {}", safe_ascii(s.as_bytes()));
        let pos = it.position();
        let o = pos - pos0;
        drop(it);
        self.src.advance(o);
        let start = self.off as _;
        self.off += o;
        let end = self.off as _;
        (CharSpan{start, end}, Token::Atom(s.into()))
      } else {
        // FIXME FIXME: tokenizer lookahead completely breaks long idents.
        self.src.buffer_ahead(100);
        let mut it = self.src.lookahead();
        let (next_tok, o) = match self.trie.match_at(&it.to_str(), 0) {
          None => {
            //eprintln!("DEBUG:  Tokenizer::next: eof 3");
            let start = self.off as _;
            let end = self.off as _;
            self.eof = Some(CharSpan{start, end});
            return Some((CharSpan{start, end}, Token::_Eof));
          }
          Some((next_tok, next_off)) => {
            //eprintln!("DEBUG:  Tokenizer::next: next tok={:?} off={}", next_tok, next_off);
            (next_tok, next_off)
          }
        };
        drop(it);
        self.src.advance(o);
        let start = self.off as _;
        self.off += o;
        let end = self.off as _;
        (CharSpan{start, end}, next_tok)
      }
    };
    match &mut tok {
      &mut Token::CommentNewline(ref mut s_) => {
        let mut s = String::new();
        loop {
          // NB: comment could end on eof.
          let c = match self.src.peek() {
            None => {
              break;
            }
            Some(c) => c
          };
          if c == '\n' {
            break;
          }
          let o = len_utf8(c as _);
          s.push(c);
          assert_eq!(Some(c), self.src.next());
          self.off += o;
        }
        span.end = self.off as _;
        let _ = replace(s_, s.into());
      }
      _ => {}
    }
    //eprintln!("DEBUG:  Tokenizer::next: tok={:?} off={}", tok, self.off);
    Some((span, tok))
  }
}

pub type Span = CharSpan;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Ident {
  Global(SafeStr),
  Local(SafeStr),
  Fresh(SafeStr),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Atom {
  pub val:  SafeStr,
}

pub type IdentNum = Num;
pub type AtomNum = Num;
pub type SemitermNum = Num;
pub type SemistmtNum = Num;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Semiterm {
  Null,
  Undefined,
  Bool(bool),
  Int(i64),
  Float(i64, i64),
  Atom(AtomNum),
  Ident(IdentNum),
  New(SemitermNum),
  Index(SemitermNum, SemitermNum),
  Field(SemitermNum, IdentNum),
  Group(SemitermNum),
  Apply(SemitermNum, Vec<SemitermNum>),
  Function(Vec<IdentNum>, Vec<SemistmtNum>),
  Array(Vec<SemitermNum>),
  Object(Vec<(AtomNum, SemitermNum)>),
  PrefixOp(Token, SemitermNum),
  SuffixOp(SemitermNum, Token),
  BinaryOp(SemitermNum, Token, SemitermNum),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Semistmt {
  Just(SemitermNum),
  Return(SemitermNum),
  AssignFresh(IdentNum, SemitermNum),
  Assign(SemitermNum, SemitermNum),
  AddAssign(SemitermNum, SemitermNum),
  SubAssign(SemitermNum, SemitermNum),
  //MulAssign(SemitermNum, SemitermNum),
  Conditions(Vec<(SemitermNum, Vec<SemistmtNum>)>),
  Induction(IdentNum, SemitermNum, SemitermNum, SemitermNum, Vec<SemistmtNum>),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct StaticLoc(&'static Location<'static>);

impl Debug for StaticLoc {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, " line={} col={} ", self.0.line(), self.0.column())
  }
}

#[derive(Clone, Debug)]
pub enum ParseErr {
  Eof,
  NotImpl(StaticLoc, Token),
  Unexpected(Token),
  ExpectedSpace(Token),
  Expected(StaticLoc, Token, Token),
  Bot(StaticLoc),
  _Bot,
}

#[derive(Clone)]
pub struct Parser_<Tokens> where Tokens: Iterator<Item=(CharSpan, Token)> {
  tokens: Peekable<Tokens>,
  cur: Option<(CharSpan, Token)>,
  lastspan: CharSpan,
  trace: bool,
}

impl<Tokens: Iterator<Item=(CharSpan, Token)>> Parser_<Tokens> {
  pub fn new(tokens: Tokens) -> Parser_<Tokens> {
    let tokens = tokens.peekable();
    let mut this = Parser_{tokens, cur: None, lastspan: CharSpan{start: 0, end: 0}, trace: false};
    this.next();
    this
  }

  pub fn next(&mut self) {
    self.cur = match self.tokens.next() {
      None => {
        Some((CharSpan{start: self.lastspan.end, end: self.lastspan.end}, Token::_Eof))
      }
      Some(t) => {
        self.lastspan = t.0;
        Some(t)
      }
    };
  }

  pub fn peek_(&mut self) -> (CharSpan, Token) {
    match self.tokens.peek() {
      None => {
        (CharSpan{start: self.lastspan.end, end: self.lastspan.end}, Token::_Eof)
      }
      Some(&(span, ref tok)) => (span, tok.clone())
    }
  }

  pub fn current_span(&self) -> CharSpan {
    self.cur.as_ref().map(|&(span, _)| span).unwrap()
  }

  pub fn current_tok(&self) -> Token {
    self.cur.as_ref().map(|&(_, ref t)| t.clone()).unwrap()
  }

  #[inline]
  pub fn current(&self) -> Token {
    self.current_tok()
  }

  pub fn current_ref(&self) -> &Token {
    self.cur.as_ref().map(|&(_, ref t)| t).unwrap()
  }

  pub fn current_(&self) -> (CharSpan, Token) {
    self.cur.as_ref().map(|&(span, ref t)| (span, t.clone())).unwrap()
  }

  pub fn maybe_skip_space(&mut self) -> Result<bool, (CharSpan, ParseErr)> {
    let mut comment = false;
    match self.current_ref() {
      &Token::Space |
      &Token::Newline => {
        self.next();
      }
      &Token::CommentNewline(_) => {
        self.next();
        comment = true;
      }
      _ => return Ok(false)
    }
    loop {
      match self.current_ref() {
        &Token::Space => {
          self.next();
        }
        &Token::Newline => {
          self.next();
          comment = false;
        }
        &Token::CommentNewline(_) => {
          self.next();
          comment = true;
        }
        _ => {
          if comment {
            self.next();
          } else {
            return Ok(true);
          }
        }
      }
    }
  }

  pub fn skip_space(&mut self) -> Result<(), (CharSpan, ParseErr)> {
    let mut comment = false;
    match self.current() {
      Token::Space |
      Token::Newline => {
        self.next();
      }
      Token::CommentNewline(_) => {
        comment = true;
        self.next();
      }
      t => {
        return Err((self.current_span(), ParseErr::ExpectedSpace(t)));
      }
    }
    loop {
      match self.current_ref() {
        &Token::Space => {
          self.next();
        }
        &Token::Newline => {
          self.next();
          comment = false;
        }
        &Token::CommentNewline(_) => {
          self.next();
          comment = true;
        }
        _ => {
          if comment {
            self.next();
          } else {
            return Ok(());
          }
        }
      }
    }
  }
}

#[track_caller]
pub fn e_bot<T>(span: CharSpan) -> Result<T, (CharSpan, ParseErr)> {
  let loc = StaticLoc(Location::caller()).into();
  Err((span, ParseErr::Bot(loc)))
}

#[track_caller]
pub fn e_exp<T>(span: CharSpan, expect: Token, actual: Token) -> Result<T, (CharSpan, ParseErr)> {
  let loc = StaticLoc(Location::caller()).into();
  Err((span, ParseErr::Expected(loc, expect, actual)))
}

#[track_caller]
pub fn e_notimpl<T>(span: CharSpan, tok: Token) -> Result<T, (CharSpan, ParseErr)> {
  let loc = StaticLoc(Location::caller()).into();
  Err((span, ParseErr::NotImpl(loc, tok)))
}

impl<Tokens: Iterator<Item=(CharSpan, Token)> + Clone> Parser_<Tokens> {
  pub fn term_nud_(&mut self, span: CharSpan, tok: Token, env: &mut Env) -> Result<Num, (CharSpan, ParseErr)> {
    if self.trace { eprintln!("DEBUG:  Parser::term_nud: tok={:?} span={:?}", tok, span); }
    match tok {
      Token::Null => {
        if self.trace { eprintln!("DEBUG:  Parser::term_nud: Null"); }
        let term = env.put_term(Semiterm::Null);
        Ok(term)
      }
      Token::Undefined => {
        if self.trace { eprintln!("DEBUG:  Parser::term_nud: Undefined"); }
        let term = env.put_term(Semiterm::Undefined);
        Ok(term)
      }
      Token::False => {
        if self.trace { eprintln!("DEBUG:  Parser::term_nud: False"); }
        let term = env.put_term(Semiterm::Bool(false));
        Ok(term)
      }
      Token::True => {
        if self.trace { eprintln!("DEBUG:  Parser::term_nud: True"); }
        let term = env.put_term(Semiterm::Bool(true));
        Ok(term)
      }
      Token::Int(x) => {
        if self.trace { eprintln!("DEBUG:  Parser::term_nud: Int"); }
        let term = env.put_term(Semiterm::Int(x));
        Ok(term)
      }
      Token::Float(x, y) => {
        if self.trace { eprintln!("DEBUG:  Parser::term_nud: Int"); }
        let term = env.put_term(Semiterm::Float(x, y));
        Ok(term)
      }
      Token::Atom(val) => {
        if self.trace { eprintln!("DEBUG:  Parser::term_nud: Atom"); }
        let atom = env.put_atom(Atom{val});
        let term = env.put_term(Semiterm::Atom(atom));
        Ok(term)
      }
      Token::Ident(val) => {
        if self.trace { eprintln!("DEBUG:  Parser::term_nud: Ident"); }
        // FIXME: global.
        let ident = match env.get_global(val.as_raw_str()) {
          None => {
            env.put_ident(Ident::Local(val))
          }
          Some(id) => id
        };
        let term = env.put_term(Semiterm::Ident(ident));
        Ok(term)
      }
      Token::Dash => {
        let op = Token::Dash;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let rterm = self.term_(span, tok, env)?;
        let term = env.put_term(Semiterm::PrefixOp(op, rterm));
        return Ok(term);
      }
      Token::Bang => {
        let op = Token::Bang;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let rterm = self.term_(span, tok, env)?;
        let term = env.put_term(Semiterm::PrefixOp(op, rterm));
        return Ok(term);
      }
      Token::LParen => {
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let inner = self.term_(span, tok, env)?;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::RParen => {
            let term = env.put_term(Semiterm::Group(inner));
            return Ok(term);
          }
          t => {
            return e_exp(span, Token::RParen, t);
          }
        }
      }
      Token::LBrack => {
        let mut first = true;
        let mut inner = Vec::new();
        loop {
          if !first {
            self.maybe_skip_space()?;
            let (span, tok) = self.current_();
            self.next();
            match tok {
              Token::RBrack => {
                let term = env.put_term(Semiterm::Array(inner));
                return Ok(term);
              }
              Token::Comma => {}
              t => {
                if self.trace { eprintln!("DEBUG:  Parser::term_nud:   LBrack: not impl: tok={:?} span={:?}", t, span); }
                return e_bot(span);
              }
            }
          }
          self.maybe_skip_space()?;
          let (span, tok) = self.current_();
          self.next();
          if first {
            match tok {
              Token::RBrack => {
                let term = env.put_term(Semiterm::Array(inner));
                return Ok(term);
              }
              _ => {}
            }
          }
          let item = self.term_(span, tok, env)?;
          inner.push(item);
          first = false;
        }
      }
      Token::LCurly => {
        let mut first = true;
        let mut inner = Vec::new();
        loop {
          if !first {
            self.maybe_skip_space()?;
            let (span, tok) = self.current_();
            self.next();
            match tok {
              Token::RCurly => {
                let term = env.put_term(Semiterm::Object(inner));
                return Ok(term);
              }
              Token::Comma => {}
              _ => {
                return e_bot(span);
              }
            }
          }
          self.maybe_skip_space()?;
          let (span, tok) = self.current_();
          self.next();
          if first {
            match tok {
              Token::RCurly => {
                let term = env.put_term(Semiterm::Object(inner));
                return Ok(term);
              }
              _ => {}
            }
          }
          let key = match tok {
            Token::Atom(key_s) => {
              env.put_atom(Atom{val: key_s})
            }
            _ => {
              return e_bot(span);
            }
          };
          self.maybe_skip_space()?;
          let (span, tok) = self.current_();
          self.next();
          match tok {
            Token::Colon => {}
            t => {
              return e_exp(span, Token::Colon, t);
            }
          }
          self.maybe_skip_space()?;
          let (span, tok) = self.current_();
          self.next();
          let val = self.term_(span, tok, env)?;
          inner.push((key, val));
          first = false;
        }
      }
      Token::New => {
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let rterm = self.term_(span, tok, env)?;
        let term = env.put_term(Semiterm::New(rterm));
        return Ok(term);
      }
      Token::Function => {
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::LParen => {}
          t => {
            return e_exp(span, Token::LParen, t);
          }
        }
        let mut first = true;
        let mut args = Vec::new();
        loop {
          if !first {
            self.maybe_skip_space()?;
            let (span, tok) = self.current_();
            self.next();
            match tok {
              Token::RParen => {
                break;
              }
              Token::Comma => {}
              _ => {
                return e_bot(span);
              }
            }
          }
          self.maybe_skip_space()?;
          let (span, tok) = self.current_();
          self.next();
          if first {
            match tok {
              Token::RParen => {
                break;
              }
              _ => {}
            }
          }
          match tok {
            Token::Ident(ident_s) => {
              let arg = env.put_ident(Ident::Fresh(ident_s));
              args.push(arg);
              first = false;
            }
            _ => {
              return e_bot(span);
            }
          }
        }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::LCurly => {}
          t => {
            return e_exp(span, Token::LCurly, t);
          }
        }
        let mut body = Vec::new();
        loop {
          self.maybe_skip_space()?;
          let (span, tok) = self.current_();
          self.next();
          match tok {
            Token::RCurly => {
              break;
            }
            _ => {}
          }
          let stmt = self.stmt_nud_(span, tok, false, env)?;
          body.push(stmt);
        }
        let term = env.put_term(Semiterm::Function(args, body));
        return Ok(term);
      }
      t => {
        if self.trace { eprintln!("DEBUG:  Parser::term_nud: not impl: tok={:?} span={:?}", t, span); }
        return e_notimpl(span, t.clone());
      }
    }
  }

  pub fn term_led_(&mut self, lterm: Num, span: CharSpan, tok: Token, env: &mut Env) -> Result<Num, (CharSpan, ParseErr)> {
    if self.trace { eprintln!("DEBUG:  Parser::term_led: lterm={:?} tok={:?} span={:?}", lterm, tok, span); }
    match &tok {
      &Token::PlusPlus |
      &Token::DashDash => {
        let op = tok;
        let term = env.put_term(Semiterm::SuffixOp(lterm, op));
        return Ok(term);
      }
      &Token::Plus |
      &Token::Dash |
      &Token::Gt |
      &Token::Geq |
      &Token::Lt |
      &Token::Leq |
      &Token::EqualEqual |
      &Token::EqualEqualEqual |
      &Token::BangEqual |
      &Token::AmpAmp |
      &Token::BarBar => {
        let op = tok;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let rterm = self.term_(span, tok, env)?;
        let term = env.put_term(Semiterm::BinaryOp(lterm, op, rterm));
        return Ok(term);
      }
      &Token::Dot => {
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::Ident(ident_s) => {
            let ident = env.put_ident(Ident::Local(ident_s));
            let term = env.put_term(Semiterm::Field(lterm, ident));
            return Ok(term);
          }
          _ => {
            return e_bot(span);
          }
        }
      }
      &Token::LBrack => {
        if self.trace { eprintln!("DEBUG:  Parser::term_led: LBrack"); }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let arg = self.term_(span, tok, env)?;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        if self.trace { eprintln!("DEBUG:  Parser::term_led:   LBrack: expect RBrack: tok={:?} span={:?}", tok, span); }
        match tok {
          Token::RBrack => {}
          t => {
            return e_exp(span, Token::RBrack, t);
          }
        }
        let term = env.put_term(Semiterm::Index(lterm, arg));
        return Ok(term);
      }
      &Token::LParen => {
        if self.trace { eprintln!("DEBUG:  Parser::term_led: LParen"); }
        let mut first = true;
        let mut args = Vec::new();
        loop {
          if !first {
            self.maybe_skip_space()?;
            let (span, tok) = self.current_();
            self.next();
            if self.trace { eprintln!("DEBUG:  Parser::term_led:   LParen: expect RParen or Comma: tok={:?} span={:?}", tok, span); }
            match tok {
              Token::RParen => {
                if self.trace { eprintln!("DEBUG:  Parser::term_led:   LParen: RParen"); }
                let term = env.put_term(Semiterm::Apply(lterm, args));
                return Ok(term);
              }
              Token::Comma => {
                if self.trace { eprintln!("DEBUG:  Parser::term_led:   LParen: Comma"); }
              }
              t => {
                return e_bot(span);
              }
            }
          }
          self.maybe_skip_space()?;
          let (span, tok) = self.current_();
          self.next();
          if first {
            if self.trace { eprintln!("DEBUG:  Parser::term_led:   LParen: maybe RParen: tok={:?} span={:?}", tok, span); }
            match tok {
              Token::RParen => {
                if self.trace { eprintln!("DEBUG:  Parser::term_led:   LParen: RParen"); }
                let term = env.put_term(Semiterm::Apply(lterm, args));
                return Ok(term);
              }
              _ => {}
            }
          }
          if self.trace { eprintln!("DEBUG:  Parser::term_led:   LParen: term: tok={:?} span={:?}", tok, span); }
          let arg = self.term_(span, tok, env)?;
          args.push(arg);
          first = false;
        }
      }
      t => {
        if self.trace { eprintln!("DEBUG:  Parser::term_led: not impl: lterm={:?} tok={:?} span={:?}", lterm, t, span); }
        return e_notimpl(span, t.clone());
      }
    }
  }

  pub fn term_(&mut self, span: CharSpan, tok: Token, env: &mut Env) -> Result<Num, (CharSpan, ParseErr)> {
    if self.trace { eprintln!("DEBUG:  Parser::term: try term nud... tok={:?} span={:?}", tok, span); }
    let mut term = match self.term_nud_(span, tok, env) {
      Err((e_span, e)) => {
        if self.trace { eprintln!("DEBUG:  Parser::term:   nud: err={:?} span={:?}", e, e_span); }
        return Err((e_span, e));
      }
      Ok(t) => t
    };
    if self.trace { eprintln!("DEBUG:  Parser::term:   nud: ok: {:?}", term); }
    loop {
      // NB: should peek in case of Space.
      self.maybe_skip_space()?;
      let (span, tok) = self.current_();
      match &tok {
        &Token::Semi |
        &Token::Equal |
        &Token::PlusEqual |
        &Token::DashEqual |
        &Token::RCurly |
        &Token::RBrack |
        &Token::RParen |
        // FIXME: also stop at reserved words.
        &Token::Bang |
        &Token::Colon |
        &Token::Comma => {
          if self.trace { eprintln!("DEBUG:  Parser::term:   led: delim: tok={:?} span={:?}", tok, span); }
          if self.trace { eprintln!("DEBUG:  Parser::term:   led: ok: {:?}", term); }
          return Ok(term);
        }
        _ => {}
      }
      self.next();
      term = match self.term_led_(term, span, tok, env) {
        Err((e_span, e)) => {
          if self.trace { eprintln!("DEBUG:  Parser::term:   led: err={:?} span={:?}", e, e_span); }
          return Err((e_span, e));
        }
        Ok(t) => t
      };
      if self.trace { eprintln!("DEBUG:  Parser::term:   led: ok: {:?}", term); }
    }
  }

  pub fn stmt_rterm_(&mut self, term: Num, /*span: CharSpan, tok: Token,*/ top_level: bool, env: &mut Env) -> Result<Num, (CharSpan, ParseErr)> {
    self.maybe_skip_space()?;
    let (span, tok) = self.current_();
    self.next();
    if self.trace { eprintln!("DEBUG:  Parser::stmt_rterm: tok={:?} span={:?}", tok, span); }
    match tok {
      Token::Semi => {
        if self.trace { eprintln!("DEBUG:  Parser::stmt_rterm: Semi"); }
        let stmt = env.put_stmt_(top_level, Semistmt::Just(term));
        return Ok(stmt);
      }
      Token::Equal => {
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let rterm = self.term_(span, tok, env)?;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::Semi => {}
          t => {
            return e_exp(span, Token::Semi, t);
          }
        }
        let stmt = env.put_stmt_(top_level, Semistmt::Assign(term, rterm));
        return Ok(stmt);
      }
      Token::PlusEqual => {
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let rterm = self.term_(span, tok, env)?;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::Semi => {}
          t => {
            return e_exp(span, Token::Semi, t);
          }
        }
        let stmt = env.put_stmt_(top_level, Semistmt::AddAssign(term, rterm));
        return Ok(stmt);
      }
      Token::DashEqual => {
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let rterm = self.term_(span, tok, env)?;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::Semi => {}
          t => {
            return e_exp(span, Token::Semi, t);
          }
        }
        let stmt = env.put_stmt_(top_level, Semistmt::SubAssign(term, rterm));
        return Ok(stmt);
      }
      t => {
        if self.trace { eprintln!("DEBUG:  Parser::stmt_rterm: not impl: tok={:?} span={:?}", t, span); }
        return e_notimpl(span, t.clone());
      }
    }
  }

  pub fn stmt_nud_(&mut self, span: CharSpan, tok: Token, top_level: bool, env: &mut Env) -> Result<Num, (CharSpan, ParseErr)> {
    {
      if self.trace { eprintln!("DEBUG:  Parser::stmt_nud: try just term... tok={:?} span={:?}", tok, span); }
      let mut parser = self.clone();
      let ckpt = env.checkpoint();
      match parser.term_(span, tok.clone(), env) {
        Err(_) => {
          env.restore(ckpt);
        }
        Ok(term) => {
          let _ = replace(self, parser);
          env.commit(ckpt);
          return self.stmt_rterm_(term, top_level, env);
        }
      }
    }
    if self.trace { eprintln!("DEBUG:  Parser::stmt_nud: try proper stmt... tok={:?} span={:?}", tok, span); }
    match tok {
      // TODO TODO
      Token::Return => {
        if self.trace { eprintln!("DEBUG:  Parser::stmt_nud: Return"); }
        let (span, tok) = self.current_();
        match tok {
          Token::Semi => {
            self.next();
            let stmt = env.put_stmt_(top_level, Semistmt::Return(Num::nil()));
            return Ok(stmt);
          }
          // FIXME: could have paren group w/out space.
          _ => {}
        }
        self.skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let rterm = self.term_(span, tok, env)?;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::Semi => {}
          t => {
            return e_exp(span, Token::Semi, t);
          }
        }
        let stmt = env.put_stmt_(top_level, Semistmt::Return(rterm));
        return Ok(stmt);
      }
      Token::Var => {
        if self.trace { eprintln!("DEBUG:  Parser::stmt_nud: Var"); }
        self.skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let ident = match tok {
          Token::Ident(ident_s) => {
            env.put_ident(Ident::Fresh(ident_s))
          }
          _ => {
            return e_bot(span);
          }
        };
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::Equal => {}
          t => {
            //eprintln!("DEBUG:  Parser::stmt_nud:   Var: expected Equal, got tok={:?}", t);
            return e_exp(span, Token::Equal, t);
          }
        }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let rterm = self.term_(span, tok, env)?;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::Semi => {}
          t => {
            //eprintln!("DEBUG:  Parser::stmt_nud:   Var: expected Semi, got tok={:?} span={:?}", t, span);
            return e_exp(span, Token::Semi, t);
          }
        }
        let stmt = env.put_stmt_(top_level, Semistmt::AssignFresh(ident, rterm));
        return Ok(stmt);
      }
      Token::If => {
        if self.trace { eprintln!("DEBUG:  Parser::stmt_nud: If"); }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::LParen => {}
          t => {
            return e_exp(span, Token::LParen, t);
          }
        }
        let (span, tok) = self.current_();
        self.next();
        let mut cond = self.term_(span, tok, env)?;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::RParen => {}
          t => {
            return e_exp(span, Token::RParen, t);
          }
        }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::LCurly => {}
          t => {
            return e_exp(span, Token::LCurly, t);
          }
        }
        let mut last_case = false;
        let mut cases = Vec::new();
        loop {
          let mut body = Vec::new();
          loop {
            self.maybe_skip_space()?;
            let (span, tok) = self.current_();
            self.next();
            match tok {
              Token::RCurly => {
                if last_case {
                  cases.push((cond, body));
                  let stmt = env.put_stmt_(top_level, Semistmt::Conditions(cases));
                  return Ok(stmt);
                }
                break;
              }
              _ => {}
            }
            let stmt = self.stmt_nud_(span, tok, false, env)?;
            body.push(stmt);
          }
          cases.push((cond, replace(&mut body, Vec::new())));
          self.maybe_skip_space()?;
          let (span, tok) = self.current_();
          match tok {
            Token::Else => {
              if last_case {
                return e_bot(span);
              }
              self.next();
              self.maybe_skip_space()?;
              let (span, tok) = self.current_();
              self.next();
              match tok {
                Token::LCurly => {
                  cond = Num::nil();
                  last_case = true;
                  continue;
                }
                Token::If => {}
                _ => {
                  return e_bot(span);
                }
              }
              self.maybe_skip_space()?;
              let (span, tok) = self.current_();
              self.next();
              match tok {
                Token::LParen => {}
                t => {
                  return e_exp(span, Token::LParen, t);
                }
              }
              let (span, tok) = self.current_();
              self.next();
              cond = self.term_(span, tok, env)?;
              self.maybe_skip_space()?;
              let (span, tok) = self.current_();
              self.next();
              match tok {
                Token::RParen => {}
                t => {
                  return e_exp(span, Token::RParen, t);
                }
              }
              self.maybe_skip_space()?;
              let (span, tok) = self.current_();
              self.next();
              match tok {
                Token::LCurly => {}
                t => {
                  return e_exp(span, Token::LCurly, t);
                }
              }
              /*continue;*/
            }
            _ => {
              let stmt = env.put_stmt_(top_level, Semistmt::Conditions(cases));
              return Ok(stmt);
            }
          }
        }
      }
      Token::For => {
        if self.trace { eprintln!("DEBUG:  Parser::stmt_nud: For"); }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::LParen => {}
          t => {
            return e_exp(span, Token::LParen, t);
          }
        }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::Var => {}
          t => {
            return e_exp(span, Token::Var, t);
          }
        }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let ident = match tok {
          Token::Ident(ident_s) => {
            env.put_ident(Ident::Fresh(ident_s))
          }
          _ => {
            return e_bot(span);
          }
        };
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::Equal => {}
          t => {
            return e_exp(span, Token::Equal, t);
          }
        }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let iv = self.term_(span, tok, env)?;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::Semi => {}
          t => {
            return e_exp(span, Token::Semi, t);
          }
        }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let cond = self.term_(span, tok, env)?;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::Semi => {}
          t => {
            return e_exp(span, Token::Semi, t);
          }
        }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        let acc = self.term_(span, tok, env)?;
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::RParen => {}
          t => {
            return e_exp(span, Token::RParen, t);
          }
        }
        self.maybe_skip_space()?;
        let (span, tok) = self.current_();
        self.next();
        match tok {
          Token::LCurly => {}
          t => {
            return e_exp(span, Token::LCurly, t);
          }
        }
        let mut body = Vec::new();
        loop {
          self.maybe_skip_space()?;
          let (span, tok) = self.current_();
          self.next();
          match tok {
            Token::RCurly => {
              break;
            }
            _ => {}
          }
          let stmt = self.stmt_nud_(span, tok, false, env)?;
          body.push(stmt);
        }
        let stmt = env.put_stmt_(top_level, Semistmt::Induction(ident, iv, cond, acc, body));
        return Ok(stmt);
      }
      t => {
        if self.trace { eprintln!("DEBUG:  Parser::stmt_nud: not impl: tok={:?} span={:?}", t, span); }
        return e_notimpl(span, t.clone());
      }
    }
  }

  pub fn next_stmt(&mut self, env: &mut Env) -> Result<Option<Num>, (CharSpan, ParseErr)> {
    if self.trace { eprintln!("DEBUG:  Parser::next_stmt: start"); }
    self.maybe_skip_space()?;
    let (span, tok) = self.current_();
    match tok {
      Token::_Eof => {
        return Ok(None);
      }
      _ => {}
    }
    self.next();
    if self.trace { eprintln!("DEBUG:  Parser::next_stmt:   stmt_nud: tok={:?} span={:?}", tok, span); }
    let stmt = self.stmt_nud_(span, tok, true, env)?;
    if self.trace { eprintln!("DEBUG:  Parser::next_stmt:   done"); }
    return Ok(Some(stmt));
  }
}
