use crate::ast::*;
use crate::ctx::{Ctx, Symbol};
use crate::mir::{self, Function};

use std::collections::{BTreeMap, HashMap};

fn ast_type_to_mir_type(ty: &TypeKind) -> mir::Ty {
    match ty {
        TypeKind::Int => mir::Ty::Int,
        TypeKind::Unit => mir::Ty::Unit,
        TypeKind::Alloc(kind, _) => match kind {
            // TODO: resolve tuple tys
            AllocKind::Tuple(tys) => {
                let tys = tys.iter().map(ast_type_to_mir_type).collect();

                mir::Ty::Tuple(tys)
            }

            AllocKind::DynArray(ty) => mir::Ty::DynArray(Box::new(ast_type_to_mir_type(ty))),
            AllocKind::Array(ty, len) => mir::Ty::Array(Box::new(ast_type_to_mir_type(ty)), *len),
        },
        TypeKind::Ptr(ty) => mir::Ty::Ptr(Box::new(ast_type_to_mir_type(ty))),
        TypeKind::Char => mir::Ty::Char,
        _ => todo!(),
    }
}

pub trait AstVisitor {
    fn visit_expr(&mut self, expr: Expr) -> mir::RValue;
    fn visit_stmt(&mut self, stmt: Stmt);
    fn visit_decl(&mut self, decl: Decl);
    fn visit_module(&mut self, module: Module);
    fn visit_block(&mut self, block: Block);
}

#[derive(Debug)]
pub struct AstToMIR<'a> {
    // TODO: we will use this later for symbol lookups and other things
    #[allow(unused)]
    ctx: &'a Ctx,
    current_function: Option<Symbol>,
    current_block: mir::BlockId,
    symbol_table: HashMap<Symbol, mir::LocalId>,
    function_table: HashMap<Symbol, Function>,
    constants: HashMap<Symbol, mir::RValue>,
    phi_functions_to_generate: BTreeMap<mir::LocalId, Vec<mir::LocalId>>,
    externs: Vec<mir::Extern>,
}

impl<'a> AstToMIR<'a> {
    pub fn new(ctx: &'a Ctx) -> Self {
        let function_table = HashMap::new();
        // function_table.insert(
        //     Symbol(-1),
        //     Function {
        //         function_id: 0,
        //         blocks: Vec::new(),
        //         parameters: 1,
        //         return_ty: mir::Ty::Unit,
        //         locals: Vec::new(),
        //     },
        // );

        Self {
            ctx,
            current_function: None,
            current_block: 0,
            symbol_table: HashMap::new(),
            function_table,
            constants: HashMap::new(),
            phi_functions_to_generate: BTreeMap::new(),
            externs: Vec::new(),
        }
    }

    fn new_local(&mut self, ty: &TypeKind) -> mir::LocalId {
        let ty = ast_type_to_mir_type(ty);
        let current_function = self.get_current_function();
        let local_id = current_function.locals.len();
        let local = mir::Local::new(local_id, ty);
        current_function.locals.push(local);
        local_id
    }

    fn new_local_with_ty(&mut self, ty: mir::Ty) -> mir::LocalId {
        let current_function = self.get_current_function();
        let local_id = current_function.locals.len();
        let local = mir::Local::new(local_id, ty);
        current_function.locals.push(local);
        local_id
    }

    // BLOCKS USE 1 BASED INDEXING
    fn get_current_block(&mut self) -> &mut mir::BasicBlock {
        &mut self
            .function_table
            .get_mut(&self.current_function.unwrap())
            .unwrap()
            .blocks[self.current_block - 1]
    }

    fn get_block(&mut self, block_id: mir::BlockId) -> &mut mir::BasicBlock {
        &mut self
            .function_table
            .get_mut(&self.current_function.unwrap())
            .unwrap()
            .blocks[block_id - 1]
    }

    fn get_current_function(&mut self) -> &mut Function {
        self.function_table
            .get_mut(&self.current_function.unwrap())
            .unwrap()
    }

    pub fn produce_module(mut self) -> mir::Module {
        let function_table = std::mem::take(&mut self.function_table);
        let functions = function_table.into_values().collect();

        let constants = std::mem::take(&mut self.constants);
        let constants = constants.into_values().collect();

        mir::Module {
            functions,
            constants,
            externs: self.externs,
        }
    }
}

impl AstVisitor for AstToMIR<'_> {
    fn visit_expr(&mut self, expr: Expr) -> mir::RValue {
        match expr.node {
            ExprKind::Value(v) => match v {
                ValueKind::Int(i) => mir::RValue::Use(mir::Operand::Constant(i)),
                ValueKind::Ident(i) => mir::RValue::Use(mir::Operand::Local(self.symbol_table[&i])),
            },
            ExprKind::BinOp { op, lhs, rhs } => {
                let lhs = self.visit_expr(*lhs);
                let lhs_op = match lhs {
                    mir::RValue::Use(op) => op,
                    _ => {
                        let lhs_local = self.new_local(&TypeKind::Int);
                        self.get_current_block()
                            .stmts
                            .push(mir::Statement::Assign(lhs_local, lhs));
                        mir::Operand::Local(lhs_local)
                    }
                };

                let rhs = self.visit_expr(*rhs);
                let rhs_op = match rhs {
                    mir::RValue::Use(op) => op,
                    _ => {
                        let rhs_local = self.new_local(&TypeKind::Int);
                        self.get_current_block()
                            .stmts
                            .push(mir::Statement::Assign(rhs_local, rhs));
                        mir::Operand::Local(rhs_local)
                    }
                };

                mir::RValue::BinOp(op, lhs_op, rhs_op)
            }
            ExprKind::Call(Call { callee, args }) => {
                let callee_sym = callee.node;
                let args = args.into_iter().map(|a| self.visit_expr(a)).collect();

                let next_block_idx = self.current_block + 1;
                let next_block = mir::BasicBlock::new(next_block_idx);

                let dest = self.get_current_function().locals.len() - 1;
                let call_transfer = mir::Terminator::Call {
                    function_id: self.function_table[&callee_sym].function_id,
                    args,
                    destination: Some(dest),
                    target: next_block_idx,
                };
                self.get_current_block().terminator = call_transfer;
                self.get_current_function().blocks.push(next_block);
                self.current_block = next_block_idx;

                mir::RValue::Use(mir::Operand::Local(dest))
            }
            ExprKind::Allocation {
                kind,
                elements,
                region: _,
            } => {
                let mut ops = Vec::new();
                for elem in elements {
                    let elem = self.visit_expr(elem);
                    let op = match elem {
                        mir::RValue::Use(op) => op,
                        _ => {
                            let elem_local = self.new_local(&TypeKind::Int);
                            self.get_current_block()
                                .stmts
                                .push(mir::Statement::Assign(elem_local, elem));
                            mir::Operand::Local(elem_local)
                        }
                    };
                    ops.push(op);
                }

                match kind {
                    AllocKind::Array(ty, _) => {
                        let ty = ast_type_to_mir_type(&ty);
                        mir::RValue::Alloc(mir::AllocKind::Array(ty), ops)
                    }
                    AllocKind::Tuple(tys) => {
                        let tys = tys.into_iter().map(|t| ast_type_to_mir_type(&t)).collect();

                        mir::RValue::Alloc(mir::AllocKind::Tuple(tys), ops)
                    }

                    AllocKind::DynArray(ty) => {
                        let ty = ast_type_to_mir_type(&ty);
                        mir::RValue::Alloc(mir::AllocKind::DynArray(ty), ops)
                    }
                }
            }
        }
    }

    fn visit_stmt(&mut self, stmt: Stmt) {
        match stmt.node {
            StmtKind::ValDec { name, ty, expr } => {
                // all types should be resolved at this point
                let ty = ty.unwrap();
                let local_id = self.new_local(&ty.node);

                self.symbol_table.insert(name.node, local_id);

                let rvalue = self.visit_expr(expr);
                let stmt = mir::Statement::Assign(local_id, rvalue);
                let block = self.get_current_block();
                block.stmts.push(stmt);
            }
            StmtKind::Assign { name, expr } => {
                let name_sym = name.node;
                let local_id = self.symbol_table[&name_sym];
                let rvalue = self.visit_expr(expr);
                let old_local = &self.get_current_function().locals[local_id];
                let ty = old_local.ty.clone();
                let new_local_id = self.new_local_with_ty(ty);

                self.phi_functions_to_generate
                    .entry(local_id)
                    .or_default()
                    .push(new_local_id);

                let stmt = mir::Statement::Assign(new_local_id, rvalue);
                self.get_current_block().stmts.push(stmt);
            }
            StmtKind::Call(Call { callee, args }) => {
                let callee_sym = callee.node;
                let args = args.into_iter().map(|a| self.visit_expr(a)).collect();

                let next_block_idx = self.current_block + 1;
                let next_block = mir::BasicBlock::new(next_block_idx);

                let call_transfer = mir::Terminator::Call {
                    function_id: self.function_table[&callee_sym].function_id,
                    args,
                    destination: None,
                    target: next_block_idx,
                };
                self.get_current_block().terminator = call_transfer;
                self.get_current_function().blocks.push(next_block);
                self.current_block = next_block_idx;
            }
            StmtKind::IfElse(IfElse { cond, then, else_ }) => {
                // store the current block id of where the if starts so we can refer to it later
                let current_block_id = self.current_block;

                let cond_local = self.new_local(&TypeKind::Int);
                let cond_rvalue = self.visit_expr(cond);
                let cond_stmt = mir::Statement::Assign(cond_local, cond_rvalue);
                self.get_current_block().stmts.push(cond_stmt);

                // TODO: refactor the block creation here
                let then_block_entrance_id = self.current_block + 1;
                let mut then_block = mir::BasicBlock::new(then_block_entrance_id);
                // SAFETY: safe to do because we always know there will be another block
                // TODO: figure out basic block params
                then_block.terminator = mir::Terminator::Br(then_block_entrance_id + 1);
                self.get_current_function().blocks.push(then_block);
                self.current_block = then_block_entrance_id;
                self.visit_block(then);

                let mut else_block_id = if let Some(else_) = else_ {
                    let else_block_entrance_id = self.current_block + 1;
                    let mut else_block = mir::BasicBlock::new(else_block_entrance_id);
                    // SAFETY: same as above
                    // TODO: figure out basic block params
                    else_block.terminator = mir::Terminator::Br(else_block_entrance_id + 1);
                    self.get_current_function().blocks.push(else_block);
                    self.current_block = else_block_entrance_id;
                    self.visit_block(else_);
                    let newest_block = self.get_current_block().block_id;
                    Some(newest_block)
                } else {
                    None
                };

                let join_block_entrance_id = self.current_block + 1;
                let join_block = mir::BasicBlock::new(join_block_entrance_id);
                self.get_current_function().blocks.push(join_block);
                self.current_block = join_block_entrance_id;

                let phi_functions_to_generate = std::mem::take(&mut self.phi_functions_to_generate);
                for (local_id, phi_ids) in phi_functions_to_generate {
                    let phi = mir::Statement::Phi(local_id, phi_ids);
                    self.get_current_block().stmts.push(phi);
                }

                if let Some(id) = else_block_id {
                    self.get_block(id).terminator = mir::Terminator::Br(join_block_entrance_id);
                }

                else_block_id.get_or_insert(join_block_entrance_id);

                let if_transfer = mir::Terminator::BrIf(
                    cond_local,
                    then_block_entrance_id,
                    else_block_id.unwrap(),
                );
                self.get_block(current_block_id).terminator = if_transfer;
            }
        }
    }

    fn visit_block(&mut self, block: Block) {
        let block_inner = block.node;

        let new_block = mir::BasicBlock::new(self.current_block + 1);
        self.get_current_function().blocks.push(new_block);

        self.current_block += 1;
        for stmt in block_inner.stmts {
            self.visit_stmt(stmt);
        }
    }

    fn visit_decl(&mut self, decl: Decl) {
        match decl.node {
            DeclKind::Extern { name, sig } => {
                let sig_inner = sig.node;
                let param_types = sig_inner
                    .params
                    .types
                    .into_iter()
                    .map(|t| ast_type_to_mir_type(&t.node))
                    .collect();

                let return_ty = sig_inner
                    .return_ty
                    .map(|t| ast_type_to_mir_type(&t.node))
                    .unwrap_or(mir::Ty::Unit);

                let name = self.ctx.resolve(name.node).to_string();
                let extern_ = mir::Extern {
                    name,
                    params: param_types,
                    return_ty,
                };
                self.externs.push(extern_);
            }
            DeclKind::Constant { name, ty: _, expr } => {
                let name_sym = name.node;
                let rvalue = self.visit_expr(expr);
                self.constants.insert(name_sym, rvalue);
            }
            DeclKind::TypeDef { name: _, def: _ } => todo!(),
            DeclKind::Procedure {
                name,
                fn_ty,
                sig,
                block,
            } => {
                let name_sym = name.node;
                // TODO: we do not use function types yet
                let _fn_ty = fn_ty;
                let sig_inner = sig.node;
                let return_ty = sig_inner
                    .return_ty
                    .map(|t| ast_type_to_mir_type(&t.node))
                    .unwrap_or(mir::Ty::Unit);

                let function = mir::Function {
                    name: self.ctx.resolve(name_sym).to_string(),
                    function_id: self.function_table.len(),
                    blocks: Vec::new(),
                    parameters: sig_inner.params.patterns.len(),
                    return_ty,
                    locals: Vec::new(),
                };

                self.function_table.insert(name_sym, function);
                self.current_function = Some(name_sym);

                for (pat, type_) in sig_inner
                    .params
                    .patterns
                    .into_iter()
                    .zip(sig_inner.params.types.into_iter())
                {
                    let local_id = self.new_local(&type_.node);
                    // TODO: we will have more patterns than just symbols eventually
                    let PatKind::Symbol(pat_name) = pat.node;
                    self.symbol_table.insert(pat_name, local_id);
                }

                self.visit_block(block);
                self.current_function = None;
            }
        }
    }

    fn visit_module(&mut self, module: Module) {
        for decl in module.declarations {
            self.visit_decl(decl);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::{lex, parse, span::Spanned, typecheck};

    use super::*;

    fn tokenify(s: &str) -> (Ctx, VecDeque<Spanned<crate::lex::Token>>) {
        let mut ctx = Ctx::new();
        let tokens = lex::tokenize(&mut ctx, s).map(|t| t.unwrap()).collect();
        (ctx, tokens)
    }

    fn parseify(ctx: &mut Ctx, tokens: VecDeque<Spanned<crate::lex::Token>>) -> Module {
        let mut module = parse::parse(ctx, tokens).unwrap();
        typecheck::typecheck(&mut module).unwrap();
        module
    }

    const BASIC_MODULE_SRC: &str = "foo :: (x: int, y: int): int { \
                if x {\
                    x = x + 1\
                    println(x)\
                } else {\
                    y = y + 1\
                    println(y)\
                }\
            }";

    fn build_basic_module_function() -> mir::Function {
        let (mut ctx, tokens) = tokenify(BASIC_MODULE_SRC);
        let module = parseify(&mut ctx, tokens);

        let mut ast_to_mir = AstToMIR::new(&ctx);
        ast_to_mir.visit_module(module);

        let mut function_table = std::mem::take(&mut ast_to_mir.function_table);
        function_table.remove(&Symbol(0)).unwrap()
    }

    #[test]
    fn test_basic_module_locals() {
        let func = build_basic_module_function();

        assert_eq!(
            func.locals,
            vec![
                mir::Local {
                    id: 0,
                    ty: mir::Ty::Int
                },
                mir::Local {
                    id: 1,
                    ty: mir::Ty::Int
                },
                mir::Local {
                    id: 2,
                    ty: mir::Ty::Int
                },
                mir::Local {
                    id: 3,
                    ty: mir::Ty::Int
                },
                mir::Local {
                    id: 4,
                    ty: mir::Ty::Int
                },
            ]
        );
        assert_eq!(func.function_id, 1);
        assert_eq!(func.parameters, 2);
    }

    #[test]
    fn test_basic_module_entry_block() {
        let func = build_basic_module_function();

        let expected_block = mir::BasicBlock {
            block_id: 1,
            stmts: vec![mir::Statement::Assign(
                2,
                mir::RValue::Use(mir::Operand::Local(0)),
            )],
            terminator: mir::Terminator::BrIf(2, 2, 7),
        };

        assert_eq!(func.blocks[0], expected_block);
    }

    #[test]
    fn test_basic_module_then_blocks() {
        let func = build_basic_module_function();

        let expected_br_block = mir::BasicBlock {
            block_id: 2,
            stmts: vec![],
            terminator: mir::Terminator::Br(3),
        };
        let expected_body_block = mir::BasicBlock {
            block_id: 3,
            stmts: vec![mir::Statement::Assign(
                3,
                mir::RValue::BinOp(
                    lex::BinOp::Add,
                    mir::Operand::Local(0),
                    mir::Operand::Constant(1),
                ),
            )],
            terminator: mir::Terminator::Call {
                function_id: 0,
                args: vec![mir::RValue::Use(mir::Operand::Local(0))],
                destination: None,
                target: 4,
            },
        };
        let expected_return_block = mir::BasicBlock {
            block_id: 4,
            stmts: vec![],
            terminator: mir::Terminator::Return,
        };

        assert_eq!(func.blocks[1], expected_br_block);
        assert_eq!(func.blocks[2], expected_body_block);
        assert_eq!(func.blocks[3], expected_return_block);
    }

    #[test]
    fn test_basic_module_else_blocks() {
        let func = build_basic_module_function();

        let expected_br_block = mir::BasicBlock {
            block_id: 5,
            stmts: vec![],
            terminator: mir::Terminator::Br(6),
        };
        let expected_body_block = mir::BasicBlock {
            block_id: 6,
            stmts: vec![mir::Statement::Assign(
                4,
                mir::RValue::BinOp(
                    lex::BinOp::Add,
                    mir::Operand::Local(1),
                    mir::Operand::Constant(1),
                ),
            )],
            terminator: mir::Terminator::Call {
                function_id: 0,
                args: vec![mir::RValue::Use(mir::Operand::Local(1))],
                destination: None,
                target: 7,
            },
        };
        let expected_exit_block = mir::BasicBlock {
            block_id: 7,
            stmts: vec![],
            terminator: mir::Terminator::Br(8),
        };

        assert_eq!(func.blocks[4], expected_br_block);
        assert_eq!(func.blocks[5], expected_body_block);
        assert_eq!(func.blocks[6], expected_exit_block);
    }

    #[test]
    fn test_basic_module_join_block() {
        let func = build_basic_module_function();

        let expected_block = mir::BasicBlock {
            block_id: 8,
            stmts: vec![
                mir::Statement::Phi(0, vec![3]),
                mir::Statement::Phi(1, vec![4]),
            ],
            terminator: mir::Terminator::Return,
        };

        assert_eq!(func.blocks[7], expected_block);
    }
}
