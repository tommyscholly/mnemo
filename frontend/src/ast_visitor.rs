use crate::Spanned;
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
            AllocKind::Record(_) => unreachable!(),
            AllocKind::Variant(_) => unreachable!(),
            AllocKind::Str(_) => mir::Ty::Str,
        },
        TypeKind::Ptr(ty) => mir::Ty::Ptr(Box::new(ast_type_to_mir_type(ty))),
        TypeKind::Char => mir::Ty::Char,
        TypeKind::Variant(variants) => {
            let mir_variant_tys = variants
                .iter()
                .map(|variant| {
                    let adts = &variant.adts;
                    let variant_ty = if adts.len() == 1 {
                        ast_type_to_mir_type(&adts[0].node)
                    } else if adts.is_empty() {
                        mir::Ty::Unit
                    } else {
                        let tys = adts
                            .iter()
                            .map(|tk| ast_type_to_mir_type(&tk.node))
                            .collect();

                        mir::Ty::Tuple(tys)
                    };

                    let union_tag = variant.name.node.0;

                    (union_tag as u8, variant_ty)
                })
                .collect();

            mir::Ty::TaggedUnion(mir_variant_tys)
        }
        TypeKind::Record(fields) => {
            let field_tys = fields
                .iter()
                .map(|f| {
                    ast_type_to_mir_type(
                        &f.ty
                            .as_ref()
                            .expect("all field types should be resolved by now")
                            .node,
                    )
                })
                .collect();
            mir::Ty::Record(field_tys)
        }
        TypeKind::Bool => mir::Ty::Bool,
        tk => panic!("unimplemented type kind {:?}", tk),
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
    variants: HashMap<u8, mir::Ty>,
    // store local types so we can resolve field accesses
    local_types: HashMap<mir::LocalId, TypeKind>,
    externs: Vec<mir::Extern>,
    in_assign_expr: bool,
}

impl<'a> AstToMIR<'a> {
    pub fn new(ctx: &'a Ctx) -> Self {
        Self {
            ctx,
            current_function: None,
            current_block: 0,
            symbol_table: HashMap::new(),
            function_table: HashMap::new(),
            constants: HashMap::new(),
            phi_functions_to_generate: BTreeMap::new(),
            variants: HashMap::new(),
            local_types: HashMap::new(),
            externs: Vec::new(),
            in_assign_expr: false,
        }
    }

    fn update_type_information(&mut self, ty: &mir::Ty) {
        #[allow(clippy::single_match)]
        match ty {
            mir::Ty::TaggedUnion(tags_tys) => {
                for (tag, ty) in tags_tys {
                    self.variants.insert(*tag, ty.clone());
                }
            }
            _ => {}
        }
    }

    fn new_local(&mut self, ty: &TypeKind) -> mir::LocalId {
        let current_function = self.get_current_function();
        let local_id = current_function.locals.len();

        self.local_types.insert(local_id, ty.clone());

        let ty = ast_type_to_mir_type(ty);
        self.update_type_information(&ty);

        let local = mir::Local::new(local_id, ty);
        self.get_current_function().locals.push(local);
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

    fn resolve_callee_function_name(&mut self, callee: &Expr) -> Option<Symbol> {
        let callee_expr = &callee.node;

        match callee_expr {
            ExprKind::Value(ValueKind::Ident(name)) => {
                if self.function_table.contains_key(name)
                    || self
                        .externs
                        .iter()
                        .any(|e| e.name == self.ctx.resolve(*name))
                {
                    Some(*name)
                } else {
                    None
                }
            }
            ExprKind::FieldAccess(_, method_name) => {
                // TOOD: we are doing UFCS: x.method() = method(x)
                // same as type checking, we will need to switch this for structural impls
                let name = method_name;

                if self.function_table.contains_key(name)
                    || self
                        .externs
                        .iter()
                        .any(|e| e.name == self.ctx.resolve(*name))
                {
                    Some(*name)
                } else {
                    None
                }
            }
            _ => {
                // for first-class functions, we would need to check that callee's type is TypeKind::Fn
                panic!("unimplemented callee type {:?}", callee_expr);
            }
        }
    }

    fn call(&mut self, callee: Expr, args: Vec<Expr>, dest: Option<mir::LocalId>) {
        let args = args.into_iter().map(|a| self.visit_expr(a)).collect();

        let name = self.resolve_callee_function_name(&callee);
        assert!(name.is_some());

        let name = self.ctx.resolve(name.unwrap()).to_string();
        let call = mir::Statement::Call {
            function_name: name,
            args,
            destination: dest,
        };
        self.get_current_block().stmts.push(call);
    }

    fn rvalue_to_local(&mut self, rvalue_ty: mir::Ty, rvalue: mir::RValue) -> mir::LocalId {
        match rvalue {
            mir::RValue::Use(mir::Operand::Copy(mir::Place { local, .. })) => local,
            mir::RValue::Use(mir::Operand::Constant(_)) => {
                let local_id = self.new_local_with_ty(rvalue_ty);
                self.get_current_block()
                    .stmts
                    .push(mir::Statement::Assign(local_id, rvalue));
                local_id
            }
            _ => todo!(),
        }
    }

    // very similar to rvalue_to_local, but we need to ensure that the type of the local is correct
    // TOOD: can we unify these two functions?
    fn ensure_local(&mut self, rvalue: mir::RValue, expected_ty: &TypeKind) -> mir::LocalId {
        match rvalue {
            mir::RValue::Use(mir::Operand::Copy(place)) => place.local,
            mir::RValue::Use(mir::Operand::Constant(_)) => {
                let ty = ast_type_to_mir_type(expected_ty);
                let local_id = self.new_local_with_ty(ty);
                self.get_current_block()
                    .stmts
                    .push(mir::Statement::Assign(local_id, rvalue));
                local_id
            }
            _ => todo!(),
        }
    }

    fn get_scrutinee_type_and_local(&mut self, scrutinee: Expr) -> (TypeKind, mir::LocalId) {
        let scru_rvalue = self.visit_expr(scrutinee);
        match &scru_rvalue.place() {
            Some(place) => {
                let scru_ty = self.local_types.get(&place.local).unwrap().clone();
                let scru_local = self.ensure_local(scru_rvalue, &scru_ty);
                (scru_ty, scru_local)
            }
            None => {
                // need to load this into memory now
                match &scru_rvalue {
                    mir::RValue::Use(mir::Operand::Constant(cnst)) => {
                        let scru_ty = match cnst {
                            mir::Constant::Int(_) => TypeKind::Int,
                            mir::Constant::Bool(_) => TypeKind::Bool,
                        };

                        let scru_local = self.new_local(&scru_ty);
                        self.get_current_block()
                            .stmts
                            .push(mir::Statement::Assign(scru_local, scru_rvalue));
                        (scru_ty, scru_local)
                    }
                    _ => todo!(),
                }
            }
        }
    }

    fn get_variant_tag(&self, scrutinee_ty: &TypeKind, variant_name: Symbol) -> Option<u8> {
        match scrutinee_ty {
            TypeKind::Variant(variants) => {
                let variant = variants.iter().find(|v| v.name.node == variant_name);
                variant.map(|v| v.name.node.0 as u8)
            }
            _ => None,
        }
    }

    fn get_variant_index(&self, scrutinee_ty: &TypeKind, variant_name: Symbol) -> Option<usize> {
        match scrutinee_ty {
            TypeKind::Variant(variants) => {
                variants.iter().position(|v| v.name.node == variant_name)
            }
            _ => None,
        }
    }

    fn bind_pattern(&mut self, pat: &Pat, scrutinee_local: mir::LocalId, _scrutinee_ty: &TypeKind) {
        match &pat.node {
            PatKind::Symbol(name) => {
                self.symbol_table.insert(*name, scrutinee_local);
            }
            PatKind::Wildcard => {}
            _ => {}
        }
    }

    fn extract_variant_payload(
        &mut self,
        scrutinee_local: mir::LocalId,
        _variant_tag: u8,
        _binding_pat: &Pat,
        scrutinee_ty: &TypeKind,
        tag: usize,
    ) -> mir::LocalId {
        let payload_local = self.new_local(&TypeKind::Int);
        let field_ty = match scrutinee_ty {
            TypeKind::Variant(variants) => variants[tag].adts.clone(),
            _ => panic!("expected variant type, got {:?}", scrutinee_ty),
        };
        let field_mir_ty = ast_type_to_mir_type(&field_ty[0].node);
        let place = mir::Place::new(scrutinee_local, mir::PlaceKind::Field(1, field_mir_ty));
        let rvalue = mir::RValue::Use(mir::Operand::Copy(place));
        self.get_current_block()
            .stmts
            .push(mir::Statement::Assign(payload_local, rvalue));
        payload_local
    }

    #[allow(unused)]
    fn get_record_field_index(&self, scrutinee_ty: &TypeKind, field_name: Symbol) -> Option<usize> {
        match scrutinee_ty {
            TypeKind::Record(fields) => fields
                .iter()
                .enumerate()
                .find(|(_, f)| f.name == field_name)
                .map(|(idx, _)| idx),
            _ => None,
        }
    }

    #[allow(unused)]
    fn extract_record_field(
        &mut self,
        scrutinee_local: mir::LocalId,
        field_idx: usize,
    ) -> mir::LocalId {
        let field_ty = self
            .local_types
            .get(&scrutinee_local)
            .and_then(|ty| match ty {
                TypeKind::Record(fields) => fields.get(field_idx).and_then(|f| f.ty.clone()),
                _ => None,
            })
            .unwrap_or_else(|| Spanned::new(TypeKind::Int, Default::default()));

        let field_mir_ty = ast_type_to_mir_type(&field_ty.node);
        let payload_local = self.new_local_with_ty(field_mir_ty.clone());
        let place = mir::Place::new(
            scrutinee_local,
            mir::PlaceKind::Field(field_idx, field_mir_ty),
        );
        let rvalue = mir::RValue::Use(mir::Operand::Copy(place));
        self.get_current_block()
            .stmts
            .push(mir::Statement::Assign(payload_local, rvalue));
        payload_local
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
                ValueKind::Int(i) => {
                    mir::RValue::Use(mir::Operand::Constant(mir::Constant::Int(i)))
                }
                ValueKind::Bool(b) => {
                    mir::RValue::Use(mir::Operand::Constant(mir::Constant::Bool(b)))
                }
                ValueKind::Ident(i) => {
                    let local_id = self.symbol_table[&i];
                    let place = mir::Place::new(local_id, mir::PlaceKind::Deref);
                    let op = mir::Operand::Copy(place);
                    mir::RValue::Use(op)
                }
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

                        let place = mir::Place::new(lhs_local, mir::PlaceKind::Deref);
                        mir::Operand::Copy(place)
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

                        let place = mir::Place::new(rhs_local, mir::PlaceKind::Deref);
                        mir::Operand::Copy(place)
                    }
                };

                mir::RValue::BinOp(op, lhs_op, rhs_op)
            }
            ExprKind::Call(Call {
                callee,
                args,
                returned_ty,
            }) => {
                assert!(returned_ty.is_some());
                let returned_ty = returned_ty.unwrap();

                let dest = if self.in_assign_expr {
                    self.get_current_function().locals.len() - 1
                } else {
                    self.new_local(&returned_ty.node)
                };

                self.call(*callee, args, Some(dest));

                let place = mir::Place::new(dest, mir::PlaceKind::Deref);
                let op = mir::Operand::Copy(place);
                // this makes an irrelevant copy, from %dest to %dest
                mir::RValue::Use(op)
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

                            let place = mir::Place::new(elem_local, mir::PlaceKind::Deref);
                            mir::Operand::Copy(place)
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
                    AllocKind::Record(fields) => {
                        let tys = fields
                            .iter()
                            .map(|f| {
                                ast_type_to_mir_type(
                                    &f.ty
                                        .as_ref()
                                        .expect("all field types should be resolved by now")
                                        .node,
                                )
                            })
                            .collect();

                        mir::RValue::Alloc(mir::AllocKind::Record(tys), ops)
                    }
                    AllocKind::Variant(variant_name) => {
                        let variant_id = variant_name.0 as u8;
                        let ty = self.variants.get(&variant_id).unwrap();

                        mir::RValue::Alloc(mir::AllocKind::Variant(variant_id, ty.clone()), ops)
                    }
                    AllocKind::Str(s) => mir::RValue::Alloc(mir::AllocKind::Str(s), ops),
                }
            }
            ExprKind::FieldAccess(expr, field_name) => {
                let mir_expr = self.visit_expr(*expr);
                // SAFETY: we should have type checked this, which means that the expr should be a place
                let place = mir_expr.place().unwrap();
                let expr_ty = self.local_types.get(&place.local).unwrap();

                let TypeKind::Record(fields) = expr_ty else {
                    panic!("expected record type, got {:?}", expr_ty);
                };

                let (field_idx, field_ty) = fields
                    .iter()
                    .enumerate()
                    .find(|(_, f)| f.name == field_name)
                    .map(|(idx, f)| (idx, ast_type_to_mir_type(&f.ty.as_ref().unwrap().node)))
                    .expect("field not found");

                mir::RValue::Use(mir::Operand::Copy(mir::Place::new(
                    place.local,
                    mir::PlaceKind::Field(field_idx, field_ty),
                )))
            }
            ExprKind::Index(expr, index) => {
                let mir_expr = self.visit_expr(*expr);
                // SAFETY: we should have type checked this, which means that the expr should be a place
                let place = mir_expr.place().unwrap();

                let mir_index = self.visit_expr(*index);
                // locals have to be ints, and should have been typechecked by now
                let mir_local = self.rvalue_to_local(mir::Ty::Int, mir_index);

                mir::RValue::Use(mir::Operand::Copy(mir::Place::new(
                    place.local,
                    mir::PlaceKind::Index(mir_local),
                )))
            }
            ExprKind::TupleAccess(expr, index) => {
                let mir_expr = self.visit_expr(*expr);
                // SAFETY: we should have type checked this, which means that the expr should be a place
                let place = mir_expr.place().unwrap();

                let TypeKind::Alloc(AllocKind::Tuple(tys), _) =
                    self.local_types.get(&place.local).unwrap()
                else {
                    panic!("expected tuple type, got {:?}", place.kind);
                };

                let field_ty = ast_type_to_mir_type(&tys[index]);

                mir::RValue::Use(mir::Operand::Copy(mir::Place::new(
                    place.local,
                    mir::PlaceKind::Field(index, field_ty),
                )))
            }
        }
    }

    fn visit_stmt(&mut self, stmt: Stmt) {
        match stmt.node {
            StmtKind::ValDec {
                name,
                ty,
                expr,
                is_comptime: _,
            } => {
                // all types should be resolved at this point
                let ty = ty.unwrap();
                let local_id = self.new_local(&ty.node);

                self.symbol_table.insert(name.node, local_id);

                self.in_assign_expr = true;
                let rvalue = self.visit_expr(expr);
                self.in_assign_expr = false;
                let stmt = mir::Statement::Assign(local_id, rvalue);
                let block = self.get_current_block();
                block.stmts.push(stmt);
            }
            StmtKind::Assign { name, expr } => {
                let name_sym = name.node;
                let local_id = self.symbol_table[&name_sym];
                self.in_assign_expr = true;
                let rvalue = self.visit_expr(expr);
                self.in_assign_expr = false;
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
            StmtKind::Call(Call {
                callee,
                args,
                returned_ty: _,
            }) => {
                self.call(*callee, args, None);
            }
            StmtKind::IfElse(if_else) => {
                let IfElse { cond, then, else_ } = *if_else;
                // store the current block id of where the if starts so we can refer to it later
                let current_block_id = self.current_block;

                let cond_local = self.new_local(&TypeKind::Bool);
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
                let last_then_block_id = self.current_block;

                let (mut else_block_id, last_else_block_id) = if let Some(else_) = else_ {
                    let else_block_entrance_id = self.current_block + 1;
                    let mut else_block = mir::BasicBlock::new(else_block_entrance_id);
                    // SAFETY: same as above
                    // TODO: figure out basic block params
                    else_block.terminator = mir::Terminator::Br(else_block_entrance_id + 1);
                    self.get_current_function().blocks.push(else_block);
                    self.current_block = else_block_entrance_id;
                    self.visit_block(else_);
                    let newest_block = self.get_current_block().block_id;
                    (Some(else_block_entrance_id), Some(newest_block))
                } else {
                    (None, None)
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

                if let Some(id) = last_else_block_id {
                    self.get_block(id).terminator = mir::Terminator::Br(join_block_entrance_id);
                }

                self.get_block(last_then_block_id).terminator =
                    mir::Terminator::Br(join_block_entrance_id);

                else_block_id.get_or_insert(join_block_entrance_id);

                let if_transfer = mir::Terminator::BrIf(
                    cond_local,
                    then_block_entrance_id,
                    else_block_id.unwrap(),
                );
                self.get_block(current_block_id).terminator = if_transfer;
            }
            StmtKind::Match(Match { scrutinee, arms }) => {
                let (scru_ty, scru_local) = self.get_scrutinee_type_and_local(scrutinee);

                let current_block_id = self.current_block;

                let mut arm_infos = Vec::new();
                let mut cases = Vec::new();
                let mut default_arm_block_id = None;

                for arm in arms {
                    let arm_block_id = self.current_block + 1;
                    let mut arm_block = mir::BasicBlock::new(arm_block_id);
                    arm_block.terminator = mir::Terminator::Br(arm_block_id + 1);
                    self.get_current_function().blocks.push(arm_block);

                    match &arm.pat.node {
                        PatKind::Variant { name, bindings } => {
                            if let Some(tag) = self.get_variant_tag(&scru_ty, *name) {
                                cases.push((tag as i32, arm_block_id));

                                if !bindings.is_empty() {
                                    let idx = self.get_variant_index(&scru_ty, *name).unwrap();
                                    let payload_local = self.extract_variant_payload(
                                        scru_local,
                                        tag,
                                        &bindings[0],
                                        &scru_ty,
                                        idx,
                                    );
                                    self.bind_pattern(&bindings[0], payload_local, &scru_ty);
                                }
                            }
                        }
                        PatKind::Wildcard => {
                            default_arm_block_id = Some(arm_block_id);
                        }
                        PatKind::Symbol(_) => {
                            self.bind_pattern(&arm.pat, scru_local, &scru_ty);
                            default_arm_block_id = Some(arm_block_id);
                        }
                        PatKind::Literal(lit) => match lit {
                            ValueKind::Int(i) => {
                                cases.push((*i, arm_block_id));
                            }
                            _ => todo!(),
                        },
                        // PatKind::Record(fields) => {
                        //     for field in fields {
                        //         let field_idx = self.get_record_field_index(&scru_ty, field.name);
                        //         if let Some(idx) = field_idx {
                        //             let field_local = self.extract_record_field(scru_local, idx);
                        //             self.symbol_table.insert(field.name, field_local);
                        //         }
                        //     }
                        // }
                        _ => {}
                    }

                    self.current_block = arm_block_id;
                    self.visit_block(arm.body);
                    arm_infos.push((arm_block_id, self.current_block));
                }

                let join_block_id = self.current_block + 1;
                let join_block = mir::BasicBlock::new(join_block_id);
                self.get_current_function().blocks.push(join_block);

                let phi_functions_to_generate = std::mem::take(&mut self.phi_functions_to_generate);
                for (local_id, phi_ids) in phi_functions_to_generate {
                    let phi = mir::Statement::Phi(local_id, phi_ids);
                    self.get_current_block().stmts.push(phi);
                }

                for (_, arm_end_id) in &arm_infos {
                    self.get_block(*arm_end_id).terminator = mir::Terminator::Br(join_block_id);
                }

                let jump_table = mir::JumpTable {
                    default: default_arm_block_id.unwrap_or(join_block_id),
                    cases,
                };

                let br_table = mir::Terminator::BrTable(scru_local, jump_table);
                self.get_block(current_block_id).terminator = br_table;

                self.current_block = join_block_id;
            }
            StmtKind::Return(expr) => {
                let local = if let Some(expr) = expr {
                    let expr = self.visit_expr(expr);

                    // SAFETY: this was typed checked
                    let return_ty = self.get_current_function().return_ty.clone();
                    Some(self.rvalue_to_local(return_ty, expr))
                } else {
                    None
                };

                let ret = mir::Terminator::Return(local);
                self.get_block(self.current_block).terminator = ret;
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
            DeclKind::Extern {
                name,
                sig,
                generic_params: _,
            } => {
                let sig_inner = sig.node;
                let param_types = sig_inner
                    .params
                    .params
                    .iter()
                    .map(|p| ast_type_to_mir_type(&p.ty.node))
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
            DeclKind::Constant {
                name,
                ty: _,
                expr,
                is_comptime: _,
            } => {
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
                constraints: _,
                monomorph_of: _,
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
                    blocks: Vec::new(),
                    parameters: sig_inner.params.params.len(),
                    return_ty,
                    locals: Vec::new(),
                };

                self.local_types.clear();
                self.function_table.insert(name_sym, function);
                self.current_function = Some(name_sym);

                for param in sig_inner.params.params.into_iter() {
                    let local_id = self.new_local(&param.ty.node);

                    match param.pattern.node {
                        PatKind::Symbol(pat_name) => {
                            self.symbol_table.insert(pat_name, local_id);
                        }
                        _ => todo!(),
                    }
                }

                self.visit_block(block);
                self.current_function = None;
                self.current_block = 0;
            }
        }
    }

    fn visit_module(&mut self, module: Module) {
        for decl in module.declarations {
            self.visit_decl(decl);
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use std::collections::VecDeque;
//
//     use crate::{lex, parse, span::Spanned, typecheck};
//
//     use super::*;
//
//     fn tokenify(s: &str) -> (Ctx, VecDeque<Spanned<crate::lex::Token>>) {
//         let mut ctx = Ctx::default();
//         let tokens = lex::tokenize(&mut ctx, s).map(|t| t.unwrap()).collect();
//         (ctx, tokens)
//     }
//
//     fn parseify(ctx: &mut Ctx, tokens: VecDeque<Spanned<crate::lex::Token>>) -> Module {
//         let mut module = parse::parse(ctx, tokens).unwrap();
//         typecheck::typecheck(ctx, &mut module).unwrap();
//         module
//     }
//
//     const BASIC_MODULE_SRC: &str = "
//     bar :: extern (x: int)
//
//     foo :: (x: int, y: int, z: bool): int { \
//                 if z {\
//                     x = x + 1\
//                     bar(x)\
//                 } else {\
//                     y = y + 1\
//                     bar(y)\
//                 }\
//             }";
//
//     fn build_basic_module_function() -> mir::Function {
//         let (mut ctx, tokens) = tokenify(BASIC_MODULE_SRC);
//         let module = parseify(&mut ctx, tokens);
//
//         let mut ast_to_mir = AstToMIR::new(&ctx);
//         ast_to_mir.visit_module(module);
//
//         let mut function_table = std::mem::take(&mut ast_to_mir.function_table);
//         function_table.remove(&Symbol(2)).unwrap()
//     }
//
//     #[test]
//     fn test_basic_module_locals() {
//         let func = build_basic_module_function();
//
//         assert_eq!(
//             func.locals,
//             vec![
//                 mir::Local {
//                     id: 0,
//                     ty: mir::Ty::Int
//                 },
//                 mir::Local {
//                     id: 1,
//                     ty: mir::Ty::Int
//                 },
//                 mir::Local {
//                     id: 2,
//                     ty: mir::Ty::Int
//                 },
//                 mir::Local {
//                     id: 3,
//                     ty: mir::Ty::Bool
//                 },
//                 mir::Local {
//                     id: 4,
//                     ty: mir::Ty::Int
//                 },
//                 mir::Local {
//                     id: 5,
//                     ty: mir::Ty::Int
//                 },
//             ]
//         );
//         assert_eq!(func.name, "foo".to_string());
//         assert_eq!(func.parameters, 3);
//     }
//
//     #[test]
//     fn test_basic_module_entry_block() {
//         let func = build_basic_module_function();
//
//         let expected_block = mir::BasicBlock {
//             block_id: 1,
//             stmts: vec![mir::Statement::Assign(
//                 2,
//                 mir::RValue::Use(mir::Operand::Copy(mir::Place::new(
//                     0,
//                     mir::PlaceKind::Deref,
//                 ))),
//             )],
//             terminator: mir::Terminator::BrIf(2, 2, 7),
//         };
//
//         assert_eq!(func.blocks[0], expected_block);
//     }
//
//     #[test]
//     fn test_basic_module_then_blocks() {
//         let func = build_basic_module_function();
//
//         let expected_br_block = mir::BasicBlock {
//             block_id: 2,
//             stmts: vec![],
//             terminator: mir::Terminator::Br(3),
//         };
//         let expected_body_block = mir::BasicBlock {
//             block_id: 3,
//             stmts: vec![
//                 mir::Statement::Assign(
//                     3,
//                     mir::RValue::BinOp(
//                         lex::BinOp::Add,
//                         mir::Operand::Copy(mir::Place::new(0, mir::PlaceKind::Deref)),
//                         mir::Operand::Constant(mir::Constant::Int(1)),
//                     ),
//                 ),
//                 mir::Statement::Call {
//                     function_name: "bar".to_string(),
//                     args: vec![mir::RValue::Use(mir::Operand::Copy(mir::Place::new(
//                         0,
//                         mir::PlaceKind::Deref,
//                     )))],
//                     destination: None,
//                 },
//             ],
//             terminator: mir::Terminator::Br(4),
//         };
//         let expected_return_block = mir::BasicBlock {
//             block_id: 4,
//             stmts: vec![],
//             terminator: mir::Terminator::Return,
//         };
//
//         assert_eq!(func.blocks[1], expected_br_block);
//         assert_eq!(func.blocks[2], expected_body_block);
//         assert_eq!(func.blocks[3], expected_return_block);
//     }
//
//     #[test]
//     fn test_basic_module_else_blocks() {
//         let func = build_basic_module_function();
//
//         let expected_br_block = mir::BasicBlock {
//             block_id: 5,
//             stmts: vec![],
//             terminator: mir::Terminator::Br(6),
//         };
//         let expected_body_block = mir::BasicBlock {
//             block_id: 6,
//             stmts: vec![
//                 mir::Statement::Assign(
//                     4,
//                     mir::RValue::BinOp(
//                         lex::BinOp::Add,
//                         mir::Operand::Copy(mir::Place::new(1, mir::PlaceKind::Deref)),
//                         mir::Operand::Constant(mir::Constant::Int(1)),
//                     ),
//                 ),
//                 mir::Statement::Call {
//                     function_name: "bar".to_string(),
//                     args: vec![mir::RValue::Use(mir::Operand::Copy(mir::Place::new(
//                         1,
//                         mir::PlaceKind::Deref,
//                     )))],
//                     destination: None,
//                 },
//             ],
//             terminator: mir::Terminator::Br(7),
//         };
//         let expected_exit_block = mir::BasicBlock {
//             block_id: 7,
//             stmts: vec![],
//             terminator: mir::Terminator::Br(8),
//         };
//
//         assert_eq!(func.blocks[4], expected_br_block);
//         assert_eq!(func.blocks[5], expected_body_block);
//         assert_eq!(func.blocks[6], expected_exit_block);
//     }
//
//     #[test]
//     fn test_basic_module_join_block() {
//         let func = build_basic_module_function();
//
//         let expected_block = mir::BasicBlock {
//             block_id: 8,
//             stmts: vec![
//                 mir::Statement::Phi(0, vec![3]),
//                 mir::Statement::Phi(1, vec![4]),
//             ],
//             terminator: mir::Terminator::Return,
//         };
//
//         assert_eq!(func.blocks[7], expected_block);
//     }
// }
