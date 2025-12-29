mod ast;
mod ast_visitor;
mod ctx;
mod lex;
mod macros;
mod mir;
mod parse;
mod span;
mod typecheck;

pub use mir::{Function, visualize};
pub use span::{DUMMY_SPAN, SourceMap, Span, SpanExt, Spanned};

use crate::ctx::Ctx;
use ast_visitor::{AstToMIR, AstVisitor};
use std::fs::File;
use std::io::Read;

pub fn do_frontend(file: &str) -> mir::Module {
    let mut file = File::open(file).unwrap();

    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    contents = contents.trim().to_string();

    let mut ctx = Ctx::new();
    // TODO: error handling and avoid the extra map allocation here by passing the direct iterator
    // to the parser
    // possibly use a RWLock on the ctx to share it mutably
    let tokens = lex::tokenize(&mut ctx, &contents)
        .map(|t| t.unwrap())
        .collect();

    let mut module = parse::parse(&mut ctx, tokens).unwrap();

    typecheck::typecheck(&mut module).unwrap();

    let mut ast_visitor = AstToMIR::new(ctx);
    ast_visitor.visit_module(module);

    ast_visitor.produce_module()
}
