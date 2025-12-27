use crate::ast::*;
use crate::ctx::{Ctx, Symbol};
use crate::mir::{self, Function};

use std::collections::{BTreeMap, HashMap};

fn ast_type_to_mir_type(ty: &Type) -> mir::Ty {
    match ty {
        Type::Int => mir::Ty::Int,
        // Type::Bool => mir::Ty::Bool,
        // Type::Unit => mir::Ty::Unit,
        // will have more types later
        #[allow(unreachable_patterns)]
        _ => todo!(),
    }
}

pub struct MIRCtx {
    pub function_table: HashMap<Symbol, mir::Function>,
}

pub trait AstVisitor {
    fn visit_expr(&mut self, expr: Expr) -> mir::RValue;
    fn visit_stmt(&mut self, stmt: Stmt);
    fn visit_decl(&mut self, decl: Decl);
    fn visit_module(&mut self, module: Module);
    fn visit_block(&mut self, block: Block);
}

#[derive(Debug)]
pub struct AstToMIR {
    // TODO: we will use this later for symbol lookups and other things
    #[allow(unused)]
    ctx: Ctx,
    current_function: Option<Symbol>,
    current_block: mir::BlockId,
    symbol_table: HashMap<Symbol, mir::LocalId>,
    function_table: HashMap<Symbol, Function>,
    constants: HashMap<Symbol, mir::RValue>,
    phi_functions_to_generate: BTreeMap<mir::LocalId, Vec<mir::LocalId>>,
}

impl AstToMIR {
    pub fn new(ctx: Ctx) -> Self {
        let mut function_table = HashMap::new();
        function_table.insert(
            Symbol(-1),
            Function {
                function_id: 0,
                blocks: Vec::new(),
                parameters: 1,
                locals: Vec::new(),
            },
        );

        Self {
            ctx,
            current_function: None,
            current_block: 0,
            symbol_table: HashMap::new(),
            function_table,
            constants: HashMap::new(),
            phi_functions_to_generate: BTreeMap::new(),
        }
    }

    fn new_local(&mut self, ty: Type) -> mir::LocalId {
        let ty = ast_type_to_mir_type(&ty);
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
}

impl AstVisitor for AstToMIR {
    fn visit_expr(&mut self, expr: Expr) -> mir::RValue {
        match expr {
            Expr::Value(v) => match v {
                Value::Int(i) => mir::RValue::Use(mir::Operand::Constant(i)),
                Value::Ident(i) => mir::RValue::Use(mir::Operand::Local(self.symbol_table[&i])),
            },
            Expr::BinOp { op, lhs, rhs } => {
                let lhs = self.visit_expr(*lhs);
                let rhs = self.visit_expr(*rhs);
                mir::RValue::BinOp(op, Box::new(lhs), Box::new(rhs))
            }
            Expr::Call(Call { callee, args }) => {
                let args = args.into_iter().map(|a| self.visit_expr(a)).collect();

                let next_block_idx = self.current_block + 1;
                let next_block = mir::BasicBlock::new(next_block_idx);

                let dest = self.get_current_function().locals.len() - 1;
                let call_transfer = mir::Terminator::Call {
                    function_id: self.function_table[&callee].function_id,
                    args,
                    destination: Some(dest),
                    target: next_block_idx,
                };
                self.get_current_block().terminator = call_transfer;
                self.get_current_function().blocks.push(next_block);
                self.current_block = next_block_idx;

                mir::RValue::Use(mir::Operand::Local(dest))
            }
        }
    }

    fn visit_stmt(&mut self, stmt: Stmt) {
        match stmt {
            Stmt::ValDec { name, ty, expr } => {
                // all types should be resolved at this point
                let ty = ty.unwrap();
                let local_id = self.new_local(ty);

                self.symbol_table.insert(name, local_id);

                let rvalue = self.visit_expr(expr);
                let stmt = mir::Statement::Assign(local_id, rvalue);
                let block = self.get_current_block();
                block.stmts.push(stmt);
            }
            Stmt::Assign { name, expr } => {
                let local_id = self.symbol_table[&name];
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
            Stmt::Call(Call { callee, args }) => {
                let args = args.into_iter().map(|a| self.visit_expr(a)).collect();

                let next_block_idx = self.current_block + 1;
                let next_block = mir::BasicBlock::new(next_block_idx);

                let call_transfer = mir::Terminator::Call {
                    function_id: self.function_table[&callee].function_id,
                    args,
                    destination: None,
                    target: next_block_idx,
                };
                self.get_current_block().terminator = call_transfer;
                self.get_current_function().blocks.push(next_block);
                self.current_block = next_block_idx;
            }
            Stmt::IfElse(IfElse { cond, then, else_ }) => {
                // store the current block id of where the if starts so we can refer to it later
                let current_block_id = self.current_block;

                let cond_local = self.new_local(Type::Int);
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
        let new_block = mir::BasicBlock::new(self.current_block + 1);
        self.get_current_function().blocks.push(new_block);

        self.current_block += 1;
        for stmt in block.stmts {
            self.visit_stmt(stmt);
        }
    }

    fn visit_decl(&mut self, decl: Decl) {
        match decl {
            Decl::Constant { name, ty: _, expr } => {
                let rvalue = self.visit_expr(expr);
                self.constants.insert(name, rvalue);
            }
            Decl::TypeDef { name, def } => todo!(),
            Decl::Procedure {
                name,
                fn_ty,
                sig,
                block,
            } => {
                // TODO: we do not use function types yet
                let _fn_ty = fn_ty;
                let function = mir::Function {
                    function_id: self.function_table.len(),
                    blocks: Vec::new(),
                    parameters: sig.params.patterns.len() as u32,
                    locals: Vec::new(),
                };

                self.function_table.insert(name, function);
                self.current_function = Some(name);

                for (pat, type_) in sig
                    .params
                    .patterns
                    .into_iter()
                    .zip(sig.params.types.into_iter())
                {
                    let local_id = self.new_local(type_);
                    // TODO: we will have more patterns than just symbols eventually
                    #[allow(irrefutable_let_patterns)]
                    let Pat::Symbol(name) = pat else {
                        panic!("expected symbol pattern")
                    };
                    self.symbol_table.insert(name, local_id);
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

    use crate::{
        lex::{self, Token},
        parse,
    };

    use super::*;

    fn tokenify(s: &str) -> (Ctx, VecDeque<Token>) {
        let mut ctx = Ctx::new();
        let tokens = lex::tokenize(&mut ctx, s).map(|t| t.unwrap().0).collect();
        (ctx, tokens)
    }

    fn parseify(ctx: &mut Ctx, tokens: VecDeque<Token>) -> Module {
        parse::parse(ctx, tokens).unwrap()
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

        let mut ast_to_mir = AstToMIR::new(ctx);
        ast_to_mir.visit_module(module);

        let mut mir_ctx = MIRCtx {
            function_table: std::mem::take(&mut ast_to_mir.function_table),
        };

        mir_ctx.function_table.remove(&Symbol(0)).unwrap()
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
                    mir::RValue::Use(mir::Operand::Local(0)).into(),
                    mir::RValue::Use(mir::Operand::Constant(1)).into(),
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
                    mir::RValue::Use(mir::Operand::Local(1)).into(),
                    mir::RValue::Use(mir::Operand::Constant(1)).into(),
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
