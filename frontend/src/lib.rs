mod ast;
mod ctx;
mod lex;
mod macros;
mod parse;
mod mir;

use crate::ctx::Ctx;
use std::fs::File;
use std::io::Read;

pub fn do_frontend(file: &str) {
    let mut file = File::open(file).unwrap();

    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let mut ctx = Ctx::new();
    // TODO: error handling and avoid the extra map allocation here by passing the direct iterator
    // to the parser
    // possibly use a RWLock on the ctx to share it mutably
    let tokens = lex::tokenize(&mut ctx, &contents)
        .map(|t| t.unwrap().0)
        .collect();

    let _module = parse::parse(&mut ctx, tokens);
}
