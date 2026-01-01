mod ast;
mod ast_visitor;
mod ctx;
mod lex;
mod macros;
pub mod mir;
mod parse;
mod span;
mod typecheck;

pub use ctx::{Ctx, Symbol};
pub use mir::{Function, visualize};
pub use span::{Diagnostic, DUMMY_SPAN, SourceMap, Span, SpanExt, Spanned};
pub use lex::BinOp;

use ast_visitor::{AstToMIR, AstVisitor};
use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;

pub fn do_frontend(file: &str) -> Result<(mir::Module, Ctx), String> {
    let mut file = File::open(file).map_err(|e| format!("failed to open file: {}", e))?;

    let mut contents = String::new();
    file.read_to_string(&mut contents).map_err(|e| format!("failed to read file: {}", e))?;
    contents = contents.trim().to_string();

    let source_map = SourceMap::new(contents.clone());

    let mut ctx = Ctx::default();
    // TODO: error handling and avoid the extra map allocation here by passing the direct iterator
    // to the parser
    // possibly use a RWLock on the ctx to share it mutably
    let tokens_result: Result<Vec<_>, _> = lex::tokenize(&mut ctx, &contents).collect();
    let tokens = tokens_result.map_err(|e| e.format(&source_map))?;
    let tokens = VecDeque::from(tokens);

    let mut module = parse::parse(&mut ctx, tokens).map_err(|e| e.format(&source_map))?;

    typecheck::typecheck(&mut module).map_err(|e| e.format(&source_map))?;

    let mut ast_visitor = AstToMIR::new(&ctx);
    ast_visitor.visit_module(module);

    Ok((ast_visitor.produce_module(), ctx))
}
