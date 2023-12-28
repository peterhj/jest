use crate::algo::*;
use crate::algo::str::*;
use crate::parsing::{Token, Ident, Atom, Semiterm, Semistmt};

use std::convert::{TryInto};
use std::fmt::{Debug, Formatter, Result as FmtResult};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Num {
  raw_: i32,
}

impl Debug for Num {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "Num({})", self.raw_)
  }
}

impl Num {
  #[inline]
  pub fn nil() -> Num {
    Num{raw_: 0}
  }

  #[inline]
  pub fn is_nil(self) -> bool {
    self.raw_ == 0
  }

  #[inline]
  pub fn increment(self) -> Num {
    Num{raw_: self.raw_ + 1}
  }
}

pub struct Checkpoint {
  idx_: u32,
}

impl Checkpoint {
  pub fn index(&self) -> usize {
    self.idx_ as usize
  }
}

pub enum StackEntry {
  Ckpt(i32),
  PutIdent(Num),
  PutAtom(Num),
  PutTerm(Num),
  PutStmt(Num),
  PutToplvl(Num),
  //_Bot,
}

pub struct Env {
  stack: Vec<StackEntry>,
  ctr: i32,
  globl: BTreeMap<SmolStr, Num>,
  //sub: BTreeMap<Num, Num>,
  sub: BTreeMap<SmolStr, Num>,
  skip: usize,
  ident: BTreeMap<Num, Ident>,
  atom: BTreeMap<Num, Atom>,
  term: BTreeMap<Num, Semiterm>,
  stmt: BTreeMap<Num, Semistmt>,
  toplvl: Vec<Num>,
}

impl Default for Env {
  fn default() -> Env {
    let mut env = Env{
      stack: Vec::new(),
      ctr: 0,
      globl: BTreeMap::new(),
      sub: BTreeMap::new(),
      skip: 0,
      ident: BTreeMap::new(),
      atom: BTreeMap::new(),
      term: BTreeMap::new(),
      stmt: BTreeMap::new(),
      toplvl: Vec::new(),
    };
    let _ = env.fresh_global("console");
    let _ = env.fresh_global("document");
    //let _ = env.fresh_global("window");
    let _ = env.fresh_global("Event");
    let _ = env.fresh_global("JSON");
    let _ = env.fresh_global("XMLHttpRequest");
    env
  }
}

impl Env {
  pub fn checkpoint(&mut self) -> Checkpoint {
    let idx_ = self.stack.len() as u32;
    self.stack.push(StackEntry::Ckpt(self.ctr));
    Checkpoint{idx_}
  }

  pub fn restore(&mut self, ckpt: Checkpoint) {
    let start = ckpt.index();
    let end = self.stack.len();
    for idx in (start .. end).rev() {
      match &self.stack[idx] {
        &StackEntry::Ckpt(ctr) => {
          assert!(ctr <= self.ctr);
          self.ctr = ctr;
        }
        &StackEntry::PutIdent(x) => {
          let item = self.ident.remove(&x);
          assert!(item.is_some());
        }
        &StackEntry::PutAtom(x) => {
          let item = self.atom.remove(&x);
          assert!(item.is_some());
        }
        &StackEntry::PutTerm(x) => {
          let item = self.term.remove(&x);
          assert!(item.is_some());
        }
        &StackEntry::PutStmt(x) => {
          let item = self.stmt.remove(&x);
          assert!(item.is_some());
        }
        &StackEntry::PutToplvl(x) => {
          match self.toplvl.pop() {
            None => panic!("bug"),
            Some(y) => {
              assert_eq!(x, y);
            }
          }
        }
        _ => panic!("bug")
      }
    }
    self.stack.truncate(start);
  }

  pub fn commit(&mut self, ckpt: Checkpoint) {
    self.stack.truncate(ckpt.index());
  }

  pub fn _fresh(&mut self) -> Num {
    let x = self.ctr + 1;
    assert!(x > 0);
    self.ctr = x;
    Num{raw_: x}
  }

  pub fn fresh_global(&mut self, val: &str) -> Num {
    let val: SafeStr = val.into();
    let ident = Ident::Global(val.clone());
    let x = self._fresh();
    self.ident.insert(x, ident);
    self.globl.insert(val.into_raw(), x);
    x
  }

  pub fn get_global(&mut self, val: &str) -> Option<Num> {
    match self.globl.get(val) {
      Some(&x) => Some(x),
      None => None
    }
  }

  pub fn put_ident(&mut self, ident: Ident) -> Num {
    let x = self._fresh();
    self.ident.insert(x, ident);
    if self.stack.len() > 0 {
      self.stack.push(StackEntry::PutIdent(x));
    }
    x
  }

  pub fn put_atom(&mut self, atom: Atom) -> Num {
    let x = self._fresh();
    self.atom.insert(x, atom);
    if self.stack.len() > 0 {
      self.stack.push(StackEntry::PutAtom(x));
    }
    x
  }

  pub fn put_term(&mut self, term: Semiterm) -> Num {
    let x = self._fresh();
    self.term.insert(x, term);
    if self.stack.len() > 0 {
      self.stack.push(StackEntry::PutTerm(x));
    }
    x
  }

  pub fn put_stmt_(&mut self, top_level: bool, stmt: Semistmt) -> Num {
    if top_level {
      self.put_stmt_top_level(stmt)
    } else {
      self.put_stmt(stmt)
    }
  }

  pub fn put_stmt(&mut self, stmt: Semistmt) -> Num {
    let x = self._fresh();
    self.stmt.insert(x, stmt);
    if self.stack.len() > 0 {
      self.stack.push(StackEntry::PutStmt(x));
    }
    x
  }

  pub fn put_stmt_top_level(&mut self, stmt: Semistmt) -> Num {
    let x = self.put_stmt(stmt);
    self.toplvl.push(x);
    if self.stack.len() > 0 {
      self.stack.push(StackEntry::PutToplvl(x));
    }
    x
  }

  pub fn _mangle(&mut self, item: Num) {
    match self.ident.get(&item).map(|v| v.clone()) {
      Some(Ident::Global(_)) => {}
      Some(Ident::Local(s)) |
      Some(Ident::Fresh(s)) => {
        //match self.sub.get(&item) {}
        match self.sub.get(s.as_raw_str()) {
          None => {
            let x = self._fresh();
            loop {
              let mut nval = String::new();
              let mut idx = self.sub.len() + self.skip;
              loop {
                let m = idx % 27;
                let c = if m == 0 {
                  '_'
                } else if m < 27 {
                  (0x60 + m as u8).try_into().unwrap()
                } else {
                  unreachable!();
                };
                nval.push(c);
                if idx < 27 {
                  break;
                }
                idx /= 27;
              }
              if self.globl.contains_key(nval.as_str()) {
                self.skip += 1;
                continue;
              }
              // FIXME: local, fresh.
              self.ident.insert(x, Ident::Local(nval.into()));
              break;
            }
            //self.sub.insert(item, x);
            self.sub.insert(s.clone().into_raw(), x);
          }
          Some(_) => {}
        }
      }
      _ => {}
    }
    match self.term.get(&item).map(|v| v.clone()) {
      Some(Semiterm::Ident(ident)) => {
        self._mangle(ident);
      }
      Some(Semiterm::New(t)) => {
        self._mangle(t);
      }
      Some(Semiterm::Index(t, idx)) => {
        self._mangle(t);
        self._mangle(idx);
      }
      Some(Semiterm::Field(t, key)) => {
        self._mangle(t);
        // NB: do not mangle the field key.
      }
      Some(Semiterm::Group(t)) => {
        self._mangle(t);
      }
      Some(Semiterm::Apply(t, args)) => {
        self._mangle(t);
        for (k, &arg) in args.iter().enumerate() {
          self._mangle(arg);
        }
      }
      Some(Semiterm::Function(args, body)) => {
        for (_, &arg) in args.iter().enumerate() {
          self._mangle(arg);
        }
        for &stmt in body.iter() {
          self._mangle(stmt);
        }
      }
      Some(Semiterm::Array(items)) => {
        for (_, &item) in items.iter().enumerate() {
          self._mangle(item);
        }
      }
      Some(Semiterm::Object(items)) => {
        for (_, &(key, val)) in items.iter().enumerate() {
          self._mangle(key);
          self._mangle(val);
        }
      }
      Some(Semiterm::PrefixOp(op, t)) => {
        self._mangle(t);
      }
      Some(Semiterm::SuffixOp(t, op)) => {
        self._mangle(t);
      }
      Some(Semiterm::BinaryOp(lt, op, rt)) => {
        self._mangle(lt);
        self._mangle(rt);
      }
      _ => {}
    }
    match self.stmt.get(&item).map(|v| v.clone()) {
      Some(Semistmt::Just(t)) => {
        self._mangle(t);
      }
      Some(Semistmt::Return(t)) => {
        self._mangle(t);
      }
      Some(Semistmt::AssignFresh(ident, t)) => {
        self._mangle(ident);
        self._mangle(t);
      }
      Some(Semistmt::Assign(lt, rt)) => {
        self._mangle(lt);
        self._mangle(rt);
      }
      Some(Semistmt::AddAssign(lt, rt)) => {
        self._mangle(lt);
        self._mangle(rt);
      }
      Some(Semistmt::SubAssign(lt, rt)) => {
        self._mangle(lt);
        self._mangle(rt);
      }
      Some(Semistmt::Conditions(cases)) => {
        for (_, &(cond, ref body)) in cases.iter().enumerate() {
          self._mangle(cond);
          for &stmt in body.iter() {
            self._mangle(stmt);
          }
        }
      }
      Some(Semistmt::Induction(ident, iv, cond, acc, body)) => {
        self._mangle(ident);
        self._mangle(iv);
        self._mangle(cond);
        self._mangle(acc);
        for &stmt in body.iter() {
          self._mangle(stmt);
        }
      }
      _ => {}
    }
  }

  pub fn mangle_idents(&mut self) {
    // TODO TODO
    /*eprintln!("DEBUG:  Env::mangle_idents: ident ={:?}", self.ident);
    eprintln!("DEBUG:  Env::mangle_idents: atom  ={:?}", self.atom);
    eprintln!("DEBUG:  Env::mangle_idents: term  ={:?}", self.term);
    eprintln!("DEBUG:  Env::mangle_idents: stmt  ={:?}", self.stmt);
    eprintln!("DEBUG:  Env::mangle_idents: toplvl={:?}", self.toplvl);*/
    for &stmt in self.toplvl.clone().iter() {
      self._mangle(stmt);
    }
  }

  pub fn _render_nomangle(&self, item: Num) -> SafeStr {
    match self.ident.get(&item) {
      Some(&Ident::Global(ref s)) |
      Some(&Ident::Local(ref s)) |
      Some(&Ident::Fresh(ref s)) => {
        return s.clone();
      }
      _ => {}
    }
    panic!("bug");
  }

  pub fn _render(&self, item: Num) -> SafeStr {
    match self.ident.get(&item) {
      Some(&Ident::Global(ref s)) => {
        return s.clone();
      }
      Some(&Ident::Local(ref s)) |
      Some(&Ident::Fresh(ref s)) => {
        match self.sub.get(s.as_raw_str()) {
          Some(&x) => {
            match self.ident.get(&x) {
              Some(&Ident::Local(ref s)) |
              Some(&Ident::Fresh(ref s)) => {
                return s.clone();
              }
              _ => panic!("bug")
            }
          }
          None => {
            return s.clone();
          }
        }
      }
      _ => {}
    }
    match self.atom.get(&item) {
      Some(&Atom{ref val}) => {
        return escape_str(val.as_raw_str()).into();
      }
      _ => {}
    }
    match self.term.get(&item) {
      Some(&Semiterm::Null) => {
        return "null".into();
      }
      Some(&Semiterm::Bool(x)) => {
        return match x {
          false => "false",
          true  => "true"
        }.into();
      }
      Some(&Semiterm::Int(x)) => {
        return format!("{}", x).into();
      }
      Some(&Semiterm::Float(x, y)) => {
        return format!("{}.{}", x, y).into();
      }
      Some(&Semiterm::Atom(atom)) => {
        return self._render(atom);
      }
      Some(&Semiterm::Ident(ident)) => {
        return self._render(ident);
      }
      Some(&Semiterm::New(t)) => {
        return format!("new {}",
            self._render(t).as_raw_str(),
        ).into();
      }
      Some(&Semiterm::Index(t, idx)) => {
        return format!("{}[{}]",
            self._render(t).as_raw_str(),
            self._render(idx).as_raw_str(),
        ).into();
      }
      Some(&Semiterm::Field(t, key)) => {
        return format!("{}.{}",
            self._render(t).as_raw_str(),
            self._render_nomangle(key).as_raw_str(),
        ).into();
      }
      Some(&Semiterm::Group(t)) => {
        return format!("({})",
            self._render(t).as_raw_str(),
        ).into();
      }
      Some(&Semiterm::Apply(t, ref args)) => {
        let mut dst = String::new();
        dst.push_str(self._render(t).as_raw_str());
        dst.push_str("(");
        for (k, &arg) in args.iter().enumerate() {
          if k > 0 {
            dst.push_str(",");
          }
          dst.push_str(self._render(arg).as_raw_str());
        }
        dst.push_str(")");
        return dst.into();
      }
      Some(&Semiterm::Function(ref args, ref body)) => {
        let mut dst = String::new();
        // FIXME: min render should be optional.
        dst.push_str("function(");
        //dst.push_str("function (");
        for (k, &arg) in args.iter().enumerate() {
          if k > 0 {
            dst.push_str(",");
          }
          dst.push_str(self._render(arg).as_raw_str());
        }
        dst.push_str("){");
        //dst.push_str(") {\n");
        for &stmt in body.iter() {
          dst.push_str(self._render_min(stmt).as_raw_str());
          //dst.push_str("\n");
        }
        dst.push_str("}");
        return dst.into();
      }
      Some(&Semiterm::Array(ref items)) => {
        let mut dst = String::new();
        dst.push_str("[");
        for (k, &item) in items.iter().enumerate() {
          if k > 0 {
            dst.push_str(",");
          }
          dst.push_str(self._render(item).as_raw_str());
        }
        dst.push_str("]");
        return dst.into();
      }
      Some(&Semiterm::Object(ref items)) => {
        let mut dst = String::new();
        dst.push_str("{");
        for (k, &(key, val)) in items.iter().enumerate() {
          if k > 0 {
            dst.push_str(",");
          }
          dst.push_str(self._render(key).as_raw_str());
          dst.push_str(":");
          dst.push_str(self._render(val).as_raw_str());
        }
        dst.push_str("}");
        return dst.into();
      }
      Some(&Semiterm::PrefixOp(ref op, t)) => {
        let mut dst = String::new();
        dst.push_str(match op {
          &Token::Dash => "-",
          &Token::Bang => "!",
          _ => panic!("bug")
        });
        dst.push_str(self._render(t).as_raw_str());
        return dst.into();
      }
      Some(&Semiterm::SuffixOp(t, ref op)) => {
        let mut dst = String::new();
        dst.push_str(self._render(t).as_raw_str());
        dst.push_str(match op {
          &Token::PlusPlus => "++",
          &Token::DashDash => "--",
          _ => panic!("bug")
        });
        return dst.into();
      }
      Some(&Semiterm::BinaryOp(lt, ref op, rt)) => {
        let mut dst = String::new();
        dst.push_str(self._render(lt).as_raw_str());
        dst.push_str(match op {
          &Token::Plus => "+",
          &Token::Dash => "-",
          &Token::AmpAmp => "&&",
          &Token::BarBar => "||",
          &Token::Gt => ">",
          &Token::Geq => ">=",
          &Token::Lt => "<",
          &Token::Leq => "<=",
          &Token::EqualEqual => "==",
          &Token::EqualEqualEqual => "===",
          &Token::BangEqual => "!=",
          &Token::BangEqualEqual => "!==",
          _ => panic!("bug")
        });
        dst.push_str(self._render(rt).as_raw_str());
        return dst.into();
      }
      _ => {}
    }
    match self.stmt.get(&item) {
      Some(&Semistmt::Just(t)) => {
        return format!("{};",
            self._render(t).as_raw_str(),
        ).into();
      }
      Some(&Semistmt::Return(t)) => {
        if t.is_nil() {
          return "return;".into();
        } else {
          return format!("return {};",
              self._render(t).as_raw_str(),
          ).into();
        }
      }
      Some(&Semistmt::AssignFresh(ident, t)) => {
        return format!("var {} = {};",
            self._render(ident).as_raw_str(),
            self._render(t).as_raw_str(),
        ).into();
      }
      Some(&Semistmt::Assign(lt, rt)) => {
        return format!("{} = {};",
            self._render(lt).as_raw_str(),
            self._render(rt).as_raw_str(),
        ).into();
      }
      Some(&Semistmt::AddAssign(lt, rt)) => {
        return format!("{} += {};",
            self._render(lt).as_raw_str(),
            self._render(rt).as_raw_str(),
        ).into();
      }
      Some(&Semistmt::SubAssign(lt, rt)) => {
        return format!("{} -= {};",
            self._render(lt).as_raw_str(),
            self._render(rt).as_raw_str(),
        ).into();
      }
      Some(&Semistmt::Conditions(ref cases)) => {
        let mut dst = String::new();
        for (k, &(cond, ref body)) in cases.iter().enumerate() {
          if k > 0 {
            dst.push_str(" else ");
          }
          if cond.is_nil() {
            // FIXME
            /*assert!(k > 0);*/
            if k == 0 {
              dst.push_str("/* WARNING: Conditions: k=0 */");
            }
            dst.push_str("{\n");
          } else {
            dst.push_str("if (");
            dst.push_str(self._render(cond).as_raw_str());
            dst.push_str(") {\n");
          }
          for &stmt in body.iter() {
            dst.push_str(self._render(stmt).as_raw_str());
            dst.push_str("\n");
          }
          dst.push_str("}");
        }
        return dst.into();
      }
      Some(&Semistmt::Induction(ident, iv, cond, acc, ref body)) => {
        let mut dst = String::new();
        dst.push_str("for (var ");
        dst.push_str(self._render(ident).as_raw_str());
        dst.push_str(" = ");
        dst.push_str(self._render(iv).as_raw_str());
        dst.push_str("; ");
        dst.push_str(self._render(cond).as_raw_str());
        dst.push_str("; ");
        dst.push_str(self._render(acc).as_raw_str());
        dst.push_str(") {\n");
        for &stmt in body.iter() {
          dst.push_str(self._render(stmt).as_raw_str());
          dst.push_str("\n");
        }
        dst.push_str("}");
        return dst.into();
      }
      _ => {}
    }
    unimplemented!();
  }

  pub fn _render_min(&self, item: Num) -> SafeStr {
    match self.stmt.get(&item) {
      Some(&Semistmt::Just(t)) => {
        return format!("{};",
            self._render(t).as_raw_str(),
        ).into();
      }
      Some(&Semistmt::Return(t)) => {
        if t.is_nil() {
          return "return;".into();
        } else {
          return format!("return {};",
              self._render(t).as_raw_str(),
          ).into();
        }
      }
      Some(&Semistmt::AssignFresh(ident, t)) => {
        return format!("var {}={};",
            self._render(ident).as_raw_str(),
            self._render(t).as_raw_str(),
        ).into();
      }
      Some(&Semistmt::Assign(lt, rt)) => {
        return format!("{}={};",
            self._render(lt).as_raw_str(),
            self._render(rt).as_raw_str(),
        ).into();
      }
      Some(&Semistmt::AddAssign(lt, rt)) => {
        return format!("{}+={};",
            self._render(lt).as_raw_str(),
            self._render(rt).as_raw_str(),
        ).into();
      }
      Some(&Semistmt::SubAssign(lt, rt)) => {
        return format!("{}-={};",
            self._render(lt).as_raw_str(),
            self._render(rt).as_raw_str(),
        ).into();
      }
      Some(&Semistmt::Conditions(ref cases)) => {
        let mut dst = String::new();
        for (k, &(cond, ref body)) in cases.iter().enumerate() {
          if k > 0 {
            dst.push_str("else ");
          }
          if cond.is_nil() {
            /*assert!(k > 0);*/
            dst.push_str("{");
          } else {
            dst.push_str("if(");
            dst.push_str(self._render(cond).as_raw_str());
            dst.push_str("){");
          }
          for &stmt in body.iter() {
            dst.push_str(self._render_min(stmt).as_raw_str());
          }
          dst.push_str("}");
        }
        return dst.into();
      }
      Some(&Semistmt::Induction(ident, iv, cond, acc, ref body)) => {
        let mut dst = String::new();
        dst.push_str("for(var ");
        dst.push_str(self._render(ident).as_raw_str());
        dst.push_str("=");
        dst.push_str(self._render(iv).as_raw_str());
        dst.push_str(";");
        dst.push_str(self._render(cond).as_raw_str());
        dst.push_str(";");
        dst.push_str(self._render(acc).as_raw_str());
        dst.push_str("){");
        for &stmt in body.iter() {
          dst.push_str(self._render_min(stmt).as_raw_str());
        }
        dst.push_str("}");
        return dst.into();
      }
      _ => {}
    }
    unimplemented!();
  }

  pub fn render(&self) -> SafeStr {
    let mut dst = String::new();
    for &stmt in self.toplvl.iter() {
      dst.push_str(self._render(stmt).as_raw_str());
      dst.push_str("\n");
    }
    dst.into()
  }

  pub fn render_min(&self) -> SafeStr {
    let mut dst = String::new();
    for &stmt in self.toplvl.iter() {
      dst.push_str(self._render_min(stmt).as_raw_str());
    }
    dst.into()
  }
}
