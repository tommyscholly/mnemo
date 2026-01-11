use crate::ast::*;
use crate::mir;

pub trait AstVisitor {
    fn visit_expr(&mut self, expr: Expr) -> mir::RValue;
    fn visit_stmt(&mut self, stmt: Stmt);
    fn visit_decl(&mut self, decl: Decl);
    fn visit_module(&mut self, module: Module);
    fn visit_block(&mut self, block: Block);
}
