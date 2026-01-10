#![allow(unused)]

use std::collections::{HashMap, HashSet};

use crate::{
    Ctx,
    ast::{
        self, AllocKind, Block, BlockInner, Call, ComptimeValue, Decl, DeclKind, Expr, ExprKind,
        IfElse, Match, MatchArm, Module, Param, Params, Pat, PatKind, RecordField, Region,
        Signature, SignatureInner, Stmt, StmtKind, Type, TypeKind, TypedValue, ValueKind,
        VariantField,
    },
    ctx::Symbol,
    lex::BinOp,
    span::{DUMMY_SPAN, Diagnostic, Span, Spanned},
};

#[derive(Debug)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub span: Span,
}

impl TypeError {
    fn new(kind: TypeErrorKind, span: Span) -> Self {
        Self { kind, span }
    }
}

impl Diagnostic for TypeError {
    fn span(&self) -> &Span {
        &self.span
    }

    fn message(&self) -> String {
        format!("type error {:?}", self.kind)
    }

    fn label(&self) -> Option<String> {
        Some(format!("{:?}", self.kind))
    }
}

#[derive(Debug)]
pub enum TypeErrorKind {
    MissingMainFunction,
    ExpectedArrayIndex,
    ExpectedVariant,
    ExpectedType {
        expected: TypeKind,
        found: TypeKind,
    },
    VariantMismatch {
        expected: Vec<VariantField>,
        found: VariantField,
    },
    FnTypeExpected,
    SignatureMismatch {
        expected: Box<SignatureInner>,
        found: Box<SignatureInner>,
    },
    RecordFieldMissing {
        field_name: Symbol,
        declared_fields: Vec<Symbol>,
    },
    RecordFieldMismatch {
        field_name: Symbol,
        expected: TypeKind,
        found: TypeKind,
    },
    ReturnTypeMismatch {
        expected: TypeKind,
        found: TypeKind,
    },
    UnknownSymbol(Symbol),
    UnknownField(Symbol),
    ArgCountMismatch {
        expected: usize,
        found: usize,
    },
    UnknownMethod(Symbol),
    NotCallable,
    ExpectedRecord,
    // Comptime-related errors
    NotComptime,
    ComptimeArgRequired,
    GenericInstantiation(String),
    TypeNotResolved(Symbol),
    InvalidComptimeOperation,
}

pub type TypecheckResult<T> = Result<T, TypeError>;

pub struct Analyzer<'a> {
    pub front_ctx: &'a mut Ctx,
    pub type_map: HashMap<Symbol, Type>,
    pub function_sigs: HashMap<Symbol, Signature>,
    pub comptime_env: HashMap<Symbol, ComptimeValue>,
    pub monomorph_cache: HashMap<ast::MonomorphKey, DeclKind>,
    pub module_decls: HashMap<Symbol, DeclKind>,
    pub pending_monomorphs: Vec<Decl>,
}

impl<'a> Analyzer<'a> {
    pub fn new(front_ctx: &'a mut Ctx) -> Self {
        Self {
            front_ctx,
            type_map: HashMap::new(),
            function_sigs: HashMap::new(),
            comptime_env: HashMap::new(),
            monomorph_cache: HashMap::new(),
            module_decls: HashMap::new(),
            pending_monomorphs: Vec::new(),
        }
    }

    fn type_of_comptime_value(cv: &ComptimeValue) -> TypeKind {
        match cv {
            ComptimeValue::Int(_) => TypeKind::Int,
            ComptimeValue::Bool(_) => TypeKind::Bool,
            ComptimeValue::Type(tk) => TypeKind::Type,
            ComptimeValue::Array(_) => {
                TypeKind::Alloc(AllocKind::DynArray(Box::new(TypeKind::Int)), Region::Stack)
            }
            ComptimeValue::Unit => TypeKind::Unit,
        }
    }

    fn eval_comptime_binop(
        &self,
        op: &BinOp,
        lhs: &ComptimeValue,
        rhs: &ComptimeValue,
    ) -> Result<ComptimeValue, TypeError> {
        match (op, lhs, rhs) {
            (BinOp::Add, ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Int(*a + *b))
            }
            (BinOp::Sub, ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Int(*a - *b))
            }
            (BinOp::Mul, ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Int(*a * *b))
            }
            (BinOp::Div, ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                if *b == 0 {
                    Err(TypeError::new(
                        TypeErrorKind::InvalidComptimeOperation,
                        DUMMY_SPAN,
                    ))
                } else {
                    Ok(ComptimeValue::Int(*a / *b))
                }
            }
            (BinOp::Mod, ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                if *b == 0 {
                    Err(TypeError::new(
                        TypeErrorKind::InvalidComptimeOperation,
                        DUMMY_SPAN,
                    ))
                } else {
                    Ok(ComptimeValue::Int(*a % *b))
                }
            }
            (BinOp::EqEq, ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Bool(*a == *b))
            }
            (BinOp::NEq, ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Bool(*a != *b))
            }
            (BinOp::Gt, ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Bool(*a > *b))
            }
            (BinOp::GtEq, ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Bool(*a >= *b))
            }
            (BinOp::Lt, ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Bool(*a < *b))
            }
            (BinOp::LtEq, ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                Ok(ComptimeValue::Bool(*a <= *b))
            }
            (BinOp::And, ComptimeValue::Bool(a), ComptimeValue::Bool(b)) => {
                Ok(ComptimeValue::Bool(*a && *b))
            }
            (BinOp::Or, ComptimeValue::Bool(a), ComptimeValue::Bool(b)) => {
                Ok(ComptimeValue::Bool(*a || *b))
            }
            _ => Err(TypeError::new(
                TypeErrorKind::InvalidComptimeOperation,
                DUMMY_SPAN,
            )),
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn substitute_type(&self, ty: &TypeKind, subs: &HashMap<Symbol, TypeKind>) -> TypeKind {
        match ty {
            TypeKind::TypeAlias(sym) => subs.get(sym).cloned().unwrap_or_else(|| ty.clone()),
            TypeKind::Ptr(inner) => TypeKind::Ptr(Box::new(self.substitute_type(inner, subs))),
            TypeKind::Alloc(kind, region) => TypeKind::Alloc(kind.clone(), *region),
            TypeKind::Fn(sig) => {
                let new_params: Vec<_> = sig
                    .params
                    .params
                    .iter()
                    .map(|p| Param {
                        pattern: p.pattern.clone(),
                        ty: Type::synthetic(self.substitute_type(&p.ty.node, subs)),
                        is_comptime: p.is_comptime,
                    })
                    .collect();
                TypeKind::Fn(Box::new(SignatureInner {
                    params: Params { params: new_params },
                    return_ty: sig
                        .return_ty
                        .as_ref()
                        .map(|t| Type::synthetic(self.substitute_type(&t.node, subs))),
                }))
            }
            TypeKind::Resolved(inner) => self.substitute_type(inner, subs),
            _ => ty.clone(),
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn substitute_expr(&self, expr: &Expr, subs: &HashMap<Symbol, TypeKind>) -> Expr {
        match &expr.node {
            ExprKind::Value(vk) => Spanned::new(ExprKind::Value(vk.clone()), expr.span.clone()),
            ExprKind::Call(call) => Spanned::new(
                ExprKind::Call(Call {
                    callee: Box::new(self.substitute_expr(&call.callee, subs)),
                    args: call
                        .args
                        .iter()
                        .map(|a| self.substitute_expr(a, subs))
                        .collect(),
                    returned_ty: call.returned_ty.clone(),
                }),
                expr.span.clone(),
            ),
            ExprKind::BinOp { op, lhs, rhs } => Spanned::new(
                ExprKind::BinOp {
                    op: *op,
                    lhs: Box::new(self.substitute_expr(lhs, subs)),
                    rhs: Box::new(self.substitute_expr(rhs, subs)),
                },
                expr.span.clone(),
            ),
            ExprKind::FieldAccess(inner, field) => Spanned::new(
                ExprKind::FieldAccess(Box::new(self.substitute_expr(inner, subs)), *field),
                expr.span.clone(),
            ),
            ExprKind::TupleAccess(inner, idx) => Spanned::new(
                ExprKind::TupleAccess(Box::new(self.substitute_expr(inner, subs)), *idx),
                expr.span.clone(),
            ),
            ExprKind::Index(inner, idx) => Spanned::new(
                ExprKind::Index(
                    Box::new(self.substitute_expr(inner, subs)),
                    Box::new(self.substitute_expr(idx, subs)),
                ),
                expr.span.clone(),
            ),
            ExprKind::Allocation {
                kind,
                elements,
                region,
            } => Spanned::new(
                ExprKind::Allocation {
                    kind: kind.clone(),
                    elements: elements
                        .iter()
                        .map(|e| self.substitute_expr(e, subs))
                        .collect(),
                    region: *region,
                },
                expr.span.clone(),
            ),
        }
    }

    fn substitute_stmt(&self, stmt: &Stmt, subs: &HashMap<Symbol, TypeKind>) -> Stmt {
        match &stmt.node {
            StmtKind::ValDec {
                name,
                ty,
                expr,
                is_comptime,
            } => Spanned::new(
                StmtKind::ValDec {
                    name: name.clone(),
                    ty: ty
                        .as_ref()
                        .map(|t| Type::synthetic(self.substitute_type(&t.node, subs))),
                    expr: self.substitute_expr(expr, subs),
                    is_comptime: *is_comptime,
                },
                stmt.span.clone(),
            ),
            StmtKind::Assign { name, expr } => Spanned::new(
                StmtKind::Assign {
                    name: name.clone(),
                    expr: self.substitute_expr(expr, subs),
                },
                stmt.span.clone(),
            ),
            StmtKind::Call(call) => Spanned::new(
                StmtKind::Call(Call {
                    callee: Box::new(self.substitute_expr(&call.callee, subs)),
                    args: call
                        .args
                        .iter()
                        .map(|a| self.substitute_expr(a, subs))
                        .collect(),
                    returned_ty: call.returned_ty.clone(),
                }),
                stmt.span.clone(),
            ),
            StmtKind::IfElse(if_else) => Spanned::new(
                StmtKind::IfElse(Box::new(IfElse {
                    cond: self.substitute_expr(&if_else.cond, subs),
                    then: self.substitute_block(&if_else.then, subs),
                    else_: if_else
                        .else_
                        .as_ref()
                        .map(|b| self.substitute_block(b, subs)),
                })),
                stmt.span.clone(),
            ),
            StmtKind::Return(expr) => Spanned::new(
                StmtKind::Return(expr.as_ref().map(|e| self.substitute_expr(e, subs))),
                stmt.span.clone(),
            ),
            StmtKind::Match(m) => Spanned::new(
                StmtKind::Match(Match {
                    scrutinee: self.substitute_expr(&m.scrutinee, subs),
                    arms: m
                        .arms
                        .iter()
                        .map(|arm| MatchArm {
                            pat: arm.pat.clone(),
                            body: self.substitute_block(&arm.body, subs),
                        })
                        .collect(),
                }),
                stmt.span.clone(),
            ),
        }
    }

    fn substitute_block(&self, block: &Block, subs: &HashMap<Symbol, TypeKind>) -> Block {
        Spanned::new(
            BlockInner {
                stmts: block
                    .node
                    .stmts
                    .iter()
                    .map(|s| self.substitute_stmt(s, subs))
                    .collect(),
                expr: block
                    .node
                    .expr
                    .as_ref()
                    .map(|e| self.substitute_expr(e, subs)),
            },
            block.span.clone(),
        )
    }

    #[allow(clippy::only_used_in_recursion)]
    fn resolve_generic_type(&self, ty: &TypeKind, subs: &HashMap<Symbol, TypeKind>) -> TypeKind {
        match ty {
            TypeKind::TypeAlias(sym) => subs.get(sym).cloned().unwrap_or_else(|| ty.clone()),
            TypeKind::Ptr(inner) => TypeKind::Ptr(Box::new(self.resolve_generic_type(inner, subs))),
            TypeKind::Alloc(kind, region) => TypeKind::Alloc(kind.clone(), *region),
            TypeKind::Fn(sig) => {
                let new_params: Vec<_> = sig
                    .params
                    .params
                    .iter()
                    .map(|p| Param {
                        pattern: p.pattern.clone(),
                        ty: Type::synthetic(self.resolve_generic_type(&p.ty.node, subs)),
                        is_comptime: p.is_comptime,
                    })
                    .collect();
                TypeKind::Fn(Box::new(SignatureInner {
                    params: Params { params: new_params },
                    return_ty: sig
                        .return_ty
                        .as_ref()
                        .map(|t| Type::synthetic(self.resolve_generic_type(&t.node, subs))),
                }))
            }
            TypeKind::Resolved(inner) => self.resolve_generic_type(inner, subs),
            _ => ty.clone(),
        }
    }

    fn monomorphize(
        &mut self,
        base_fn: Symbol,
        comptime_args: Vec<ComptimeValue>,
    ) -> Result<Decl, TypeError> {
        let key = ast::MonomorphKey {
            base_fn,
            comptime_args: comptime_args.clone(),
        };

        if let Some(cached) = self.monomorph_cache.get(&key).cloned() {
            return Ok(Spanned::new(cached, DUMMY_SPAN));
        }

        let Some(sig) = self.function_sigs.get(&base_fn).cloned() else {
            return Err(TypeError::new(
                TypeErrorKind::UnknownSymbol(base_fn),
                DUMMY_SPAN,
            ));
        };

        let sig_inner = sig.node;
        let mut subs = HashMap::new();
        for (param, cv) in sig_inner.params.params.iter().zip(comptime_args.iter()) {
            if let PatKind::Symbol(sym) = param.pattern.node {
                let ty = match cv {
                    ComptimeValue::Int(_) => TypeKind::Int,
                    ComptimeValue::Bool(_) => TypeKind::Bool,
                    ComptimeValue::Type(tk) => tk.clone(),
                    ComptimeValue::Array(_) => {
                        TypeKind::Alloc(AllocKind::DynArray(Box::new(TypeKind::Int)), Region::Stack)
                    }
                    ComptimeValue::Unit => TypeKind::Unit,
                };
                subs.insert(sym, ty);
            }
        }

        let monomorphized_sig = SignatureInner {
            params: Params {
                params: sig_inner
                    .params
                    .params
                    .iter()
                    .filter_map(|p| {
                        if p.is_comptime {
                            None
                        } else {
                            Some(Param {
                                pattern: p.pattern.clone(),
                                ty: Type::synthetic(self.resolve_generic_type(&p.ty.node, &subs)),
                                is_comptime: false,
                            })
                        }
                    })
                    .collect(),
            },
            return_ty: sig_inner
                .return_ty
                .as_ref()
                .map(|t| Type::synthetic(self.resolve_generic_type(&t.node, &subs))),
        };

        let new_name = format!("{}_{}", self.front_ctx.resolve(base_fn), {
            let mut parts = Vec::new();
            for cv in &comptime_args {
                match cv {
                    ComptimeValue::Int(n) => parts.push(format!("I{}", n)),
                    ComptimeValue::Bool(b) => parts.push(format!("B{}", b)),
                    ComptimeValue::Type(tk) => parts.push(format!("T{:?}", tk)),
                    _ => parts.push("X".to_string()),
                }
            }
            parts.join("_")
        });

        let Some(original_decl) = self.module_decls.get_mut(&base_fn) else {
            return Err(TypeError::new(
                TypeErrorKind::UnknownSymbol(base_fn),
                DUMMY_SPAN,
            ));
        };

        let DeclKind::Procedure {
            block, is_comptime, ..
        } = original_decl
        else {
            return Err(TypeError::new(
                TypeErrorKind::GenericInstantiation("not a procedure".to_string()),
                DUMMY_SPAN,
            ));
        };

        *is_comptime = true;
        // TOOD: remove this clone, its so we drob the mut borrow
        let block = block.clone();

        let monomorphized_block = self.substitute_block(&block, &subs);

        let monomorphized_decl = DeclKind::Procedure {
            name: Spanned::new(self.front_ctx.intern(&new_name), DUMMY_SPAN),
            fn_ty: Some(Type::synthetic(TypeKind::Fn(Box::new(
                monomorphized_sig.clone(),
            )))),
            sig: Spanned::new(monomorphized_sig, DUMMY_SPAN),
            constraints: vec![],
            block: monomorphized_block,
            monomorph_of: Some(key.clone()),
            is_comptime: false,
        };

        self.monomorph_cache
            .insert(key.clone(), monomorphized_decl.clone());
        let decl = Spanned::new(monomorphized_decl, DUMMY_SPAN);
        self.pending_monomorphs.push(decl.clone());
        Ok(decl)
    }

    pub fn analyze_expr(&mut self, expr: &mut Expr) -> Result<TypedValue, TypeError> {
        match &mut expr.node {
            ExprKind::Value(v) => match v {
                ValueKind::Int(n) => Ok(TypedValue::comptime(
                    TypeKind::Int,
                    ComptimeValue::Int(*n as i128),
                )),
                ValueKind::Ident(sym) => {
                    if let Some(cv) = self.comptime_env.get(sym) {
                        let ty = Self::type_of_comptime_value(cv);
                        return Ok(TypedValue::comptime(ty, cv.clone()));
                    }
                    let ty = self
                        .type_map
                        .get(sym)
                        .ok_or_else(|| {
                            TypeError::new(TypeErrorKind::UnknownSymbol(*sym), expr.span.clone())
                        })?
                        .node
                        .clone();
                    Ok(TypedValue::runtime(ty))
                }
                ValueKind::Bool(b) => Ok(TypedValue::comptime(
                    TypeKind::Bool,
                    ComptimeValue::Bool(*b),
                )),
                ValueKind::Type(tk) => Ok(TypedValue::comptime(
                    TypeKind::Type,
                    ComptimeValue::Type(tk.clone()),
                )),
            },
            ExprKind::BinOp { op, lhs, rhs } => {
                let lhs_val = self.analyze_expr(lhs)?;
                let rhs_val = self.analyze_expr(rhs)?;

                if lhs_val.ty != rhs_val.ty {
                    return Err(TypeError::new(
                        TypeErrorKind::ExpectedType {
                            expected: lhs_val.ty.clone(),
                            found: rhs_val.ty.clone(),
                        },
                        expr.span.clone(),
                    ));
                }

                if lhs_val.is_comptime() && rhs_val.is_comptime() {
                    let result = self.eval_comptime_binop(
                        op,
                        lhs_val.comptime_value.as_ref().unwrap(),
                        rhs_val.comptime_value.as_ref().unwrap(),
                    )?;
                    Ok(TypedValue::comptime(lhs_val.ty, result))
                } else {
                    Ok(TypedValue::runtime(lhs_val.ty))
                }
            }
            ExprKind::Call(call) => self.analyze_call(call),
            ExprKind::Allocation {
                kind,
                region,
                elements,
            } => {
                let mut elem_vals = Vec::new();
                for elem in elements {
                    elem_vals.push(self.analyze_expr(elem)?);
                }

                let region_handle = region.unwrap_or(Region::Stack);
                let resolved_kind = match kind {
                    AllocKind::Tuple(tys) => {
                        let mut types = Vec::new();
                        for elem in elem_vals.into_iter() {
                            types.push(elem.unwrap_type());
                        }

                        if !tys.is_empty() && tys.len() != types.len() && *tys != types {
                            panic!("expected tuple types to be equal");
                        }
                        AllocKind::Tuple(types)
                    }
                    AllocKind::Variant(variant_name) => {
                        let mut adts = Vec::new();
                        for val in elem_vals.into_iter() {
                            adts.push(Type::synthetic(val.unwrap_type()));
                        }
                        return Ok(TypedValue::runtime(TypeKind::Variant(vec![VariantField {
                            name: Spanned::default(*variant_name),
                            adts,
                        }])));
                    }
                    AllocKind::Record(fields) => {
                        let mut filled_fields = Vec::new();
                        for (field, elem_val) in fields.iter_mut().zip(elem_vals.into_iter()) {
                            let elem_ty = Type::synthetic(elem_val.ty);
                            filled_fields.push(RecordField {
                                name: field.name,
                                ty: Some(elem_ty),
                            });
                        }
                        return Ok(TypedValue::runtime(TypeKind::Record(filled_fields)));
                    }
                    _ => kind.clone(),
                };
                Ok(TypedValue::runtime(TypeKind::Alloc(
                    resolved_kind,
                    region_handle,
                )))
            }
            ExprKind::FieldAccess(expr, field) => {
                let expr_val = self.analyze_expr(expr)?;
                let field_ty = Self::get_field_type(&expr_val.ty, *field, expr.span.clone())?;
                Ok(TypedValue::runtime(field_ty))
            }
            ExprKind::TupleAccess(expr, index) => {
                let expr_val = self.analyze_expr(expr)?;
                let TypeKind::Alloc(AllocKind::Tuple(tys), _) = &expr_val.ty else {
                    return Err(TypeError::new(
                        TypeErrorKind::ExpectedType {
                            expected: TypeKind::Alloc(AllocKind::Tuple(vec![]), Region::Stack),
                            found: expr_val.ty.clone(),
                        },
                        expr.span.clone(),
                    ));
                };
                if *index >= tys.len() {
                    return Err(TypeError::new(
                        TypeErrorKind::ExpectedArrayIndex,
                        expr.span.clone(),
                    ));
                }
                let ty = tys[*index].clone();
                Ok(TypedValue::runtime(ty))
            }
            ExprKind::Index(expr, index) => {
                let expr_val = self.analyze_expr(expr)?;
                self.analyze_expr(index)?;
                let TypeKind::Alloc(AllocKind::Array(ty, _), _) = &expr_val.ty else {
                    return Err(TypeError::new(
                        TypeErrorKind::ExpectedType {
                            expected: TypeKind::Alloc(
                                AllocKind::Array(Box::new(TypeKind::Int), 0),
                                Region::Stack,
                            ),
                            found: expr_val.ty.clone(),
                        },
                        expr.span.clone(),
                    ));
                };
                Ok(TypedValue::runtime((**ty).clone()))
            }
        }
    }

    fn get_field_type(
        expr_ty: &TypeKind,
        field: Symbol,
        span: Span,
    ) -> Result<TypeKind, TypeError> {
        match expr_ty {
            TypeKind::Record(fields) => fields
                .iter()
                .find(|f| f.name == field)
                .and_then(|f| f.ty.as_ref())
                .map(|t| t.node.clone())
                .ok_or_else(|| TypeError::new(TypeErrorKind::UnknownField(field), span)),
            _ => Err(TypeError::new(TypeErrorKind::ExpectedRecord, span)),
        }
    }

    fn analyze_call(&mut self, call: &mut Call) -> Result<TypedValue, TypeError> {
        let arg_vals: Vec<_> = call
            .args
            .iter_mut()
            .map(|arg| self.analyze_expr(arg))
            .collect::<Result<_, _>>()?;

        let (callee_sig, span) = self.resolve_callee_signature(&call.callee)?;

        let params = &callee_sig.node.params.params;
        let expected_count = params.len();
        let found_count = call.args.len();

        let is_variadic = params
            .last()
            .map(|p| p.ty.node == TypeKind::Variadic)
            .unwrap_or(false);

        if !is_variadic && expected_count != found_count {
            return Err(TypeError::new(
                TypeErrorKind::ArgCountMismatch {
                    expected: expected_count,
                    found: found_count,
                },
                call.callee.span.clone(),
            ));
        }

        let min_expected = if is_variadic {
            expected_count - 1
        } else {
            expected_count
        };
        if found_count < min_expected {
            return Err(TypeError::new(
                TypeErrorKind::ArgCountMismatch {
                    expected: min_expected,
                    found: found_count,
                },
                call.callee.span.clone(),
            ));
        }

        let mut comptime_args = Vec::new();
        let mut non_comptime_args = Vec::new();
        for (i, (arg_val, param)) in arg_vals.iter().zip(params.iter()).enumerate() {
            if param.is_comptime {
                if let Some(cv) = &arg_val.comptime_value {
                    comptime_args.push(cv.clone());
                } else {
                    return Err(TypeError::new(
                        TypeErrorKind::ComptimeArgRequired,
                        call.callee.span.clone(),
                    ));
                }
            } else {
                non_comptime_args.push(call.args[i].clone());
            };
        }

        let mut subs = HashMap::new();
        for (param, cv) in params.iter().zip(comptime_args.iter()) {
            if let PatKind::Symbol(sym) = param.pattern.node {
                let ty = match cv {
                    ComptimeValue::Int(_) => TypeKind::Int,
                    ComptimeValue::Bool(_) => TypeKind::Bool,
                    ComptimeValue::Type(tk) => tk.clone(),
                    ComptimeValue::Array(_) => {
                        todo!()
                    }
                    ComptimeValue::Unit => TypeKind::Unit,
                };
                subs.insert(sym, ty);
            }
        }

        let mut return_ty = callee_sig
            .node
            .return_ty
            .as_ref()
            .map(|t| t.node.clone())
            .unwrap_or(TypeKind::Unit);

        if !comptime_args.is_empty()
            && let ExprKind::Value(ValueKind::Ident(fn_name)) = &call.callee.node
        {
            let monomorphized_decl = self.monomorphize(*fn_name, comptime_args)?;
            if let DeclKind::Procedure { name, sig, .. } = monomorphized_decl.node {
                return_ty = sig
                    .node
                    .return_ty
                    .as_ref()
                    .map(|t| t.node.clone())
                    .unwrap_or(TypeKind::Unit);

                call.callee.node = ExprKind::Value(ValueKind::Ident(name.node));
                call.args = non_comptime_args;
            }
        }

        for (i, (arg, param)) in call.args.iter().zip(params.iter()).enumerate() {
            let arg_val = &arg_vals[i];
            if param.ty.node != TypeKind::Variadic {
                let resolved_param_ty = self.resolve_type_with_subs(&param.ty.node, &subs);
                self.structural_typecheck(&arg_val.ty, &resolved_param_ty, arg.span.clone())?;
            }
        }

        Ok(TypedValue::runtime(return_ty))
    }

    #[allow(clippy::only_used_in_recursion)]
    fn resolve_type_with_subs(&self, ty: &TypeKind, subs: &HashMap<Symbol, TypeKind>) -> TypeKind {
        match ty {
            TypeKind::TypeAlias(sym) => subs.get(sym).cloned().unwrap_or_else(|| ty.clone()),
            TypeKind::Ptr(inner) => {
                TypeKind::Ptr(Box::new(self.resolve_type_with_subs(inner, subs)))
            }
            TypeKind::Alloc(kind, region) => TypeKind::Alloc(kind.clone(), *region),
            TypeKind::Fn(sig) => {
                let new_params: Vec<_> = sig
                    .params
                    .params
                    .iter()
                    .map(|p| Param {
                        pattern: p.pattern.clone(),
                        ty: Type::synthetic(self.resolve_type_with_subs(&p.ty.node, subs)),
                        is_comptime: p.is_comptime,
                    })
                    .collect();
                TypeKind::Fn(Box::new(SignatureInner {
                    params: Params { params: new_params },
                    return_ty: sig
                        .return_ty
                        .as_ref()
                        .map(|t| Type::synthetic(self.resolve_type_with_subs(&t.node, subs))),
                }))
            }
            TypeKind::Resolved(inner) => self.resolve_type_with_subs(inner, subs),
            _ => ty.clone(),
        }
    }

    fn resolve_callee_signature(&self, callee: &Expr) -> Result<(Signature, Span), TypeError> {
        match &callee.node {
            ExprKind::Value(ValueKind::Ident(name)) => {
                let Some(sig) = self.function_sigs.get(name) else {
                    return Err(TypeError::new(
                        TypeErrorKind::UnknownSymbol(*name),
                        callee.span.clone(),
                    ));
                };
                Ok((sig.clone(), callee.span.clone()))
            }
            ExprKind::FieldAccess(receiver, method_name) => {
                let Some(sig) = self.function_sigs.get(method_name) else {
                    return Err(TypeError::new(
                        TypeErrorKind::UnknownMethod(*method_name),
                        callee.span.clone(),
                    ));
                };
                Ok((sig.clone(), callee.span.clone()))
            }
            _ => Err(TypeError::new(
                TypeErrorKind::NotCallable,
                callee.span.clone(),
            )),
        }
    }

    fn structural_typecheck(
        &self,
        expr_ty: &TypeKind,
        declared_ty: &TypeKind,
        declared_ty_span: Span,
    ) -> Result<(), TypeError> {
        match (expr_ty, declared_ty) {
            (TypeKind::Variant(expr_variant), TypeKind::Variant(declared_variants)) => {
                let expr_variant = expr_variant.first().unwrap();
                if !declared_variants.contains(expr_variant) {
                    return Err(TypeError::new(
                        TypeErrorKind::VariantMismatch {
                            expected: declared_variants.clone(),
                            found: expr_variant.clone(),
                        },
                        declared_ty_span,
                    ));
                }
            }
            (TypeKind::Record(expr_fields), TypeKind::Record(declared_fields)) => {
                let expr_field_set: HashMap<u32, Option<TypeKind>> = expr_fields
                    .iter()
                    .map(|f| (f.name.0, f.ty.as_ref().map(|t| t.node.clone())))
                    .collect();

                let declared_field_set: HashMap<u32, TypeKind> = declared_fields
                    .iter()
                    .map(|f| (f.name.0, f.ty.as_ref().unwrap().node.clone()))
                    .collect();

                for (field_name, field_ty) in &declared_field_set {
                    if !expr_field_set.contains_key(field_name) {
                        return Err(TypeError::new(
                            TypeErrorKind::RecordFieldMissing {
                                field_name: Symbol(*field_name),
                                declared_fields: declared_field_set
                                    .keys()
                                    .cloned()
                                    .map(Into::into)
                                    .collect(),
                            },
                            declared_ty_span,
                        ));
                    }

                    let expr_field_ty = &expr_field_set[field_name];
                    if expr_field_ty.as_ref() != Some(field_ty) {
                        return Err(TypeError::new(
                            TypeErrorKind::RecordFieldMismatch {
                                field_name: Symbol(*field_name),
                                expected: field_ty.clone(),
                                found: expr_field_ty.clone().unwrap_or(TypeKind::Int),
                            },
                            declared_ty_span,
                        ));
                    }
                }
            }
            (TypeKind::Alloc(AllocKind::Str(_), _), TypeKind::Ptr(c))
                if (**c) == TypeKind::Char => {}
            _ => {
                if declared_ty != expr_ty {
                    return Err(TypeError::new(
                        TypeErrorKind::ExpectedType {
                            expected: declared_ty.clone(),
                            found: expr_ty.clone(),
                        },
                        declared_ty_span,
                    ));
                }
            }
        }
        Ok(())
    }

    fn bind_pattern(&mut self, pat: &Pat, ty: &Type) {
        match &pat.node {
            PatKind::Symbol(name) => {
                self.type_map.insert(*name, ty.clone());
            }
            _ => todo!(),
        }
    }

    pub fn typecheck_module(&mut self, module: &mut Module) -> Result<(), TypeError> {
        for decl in &module.declarations {
            if let DeclKind::Procedure { name, .. } = &decl.node {
                self.module_decls.insert(name.node, decl.node.clone());
            }
        }
        for decl in &mut module.declarations {
            self.analyze_decl(decl)?;
        }

        for decl in &mut module.declarations {
            if let DeclKind::Procedure {
                name, is_comptime, ..
            } = &mut decl.node
                && let Some(proc) = self.module_decls.get(&name.node)
                && let DeclKind::Procedure {
                    is_comptime: comptime,
                    ..
                } = &proc
            {
                *is_comptime = *comptime;
            }
        }

        module.declarations.extend(self.pending_monomorphs.clone());
        self.check_entry_point()
    }

    fn check_entry_point(&mut self) -> Result<(), TypeError> {
        let mut found_main = false;
        for function in self.function_sigs.keys() {
            let function_name = self.front_ctx.resolve(*function);
            if function_name == "main" {
                self.front_ctx.update(*function, "__entry");
                found_main = true;
            }
        }
        #[cfg(not(test))]
        if !found_main {
            return Err(TypeError::new(
                TypeErrorKind::MissingMainFunction,
                DUMMY_SPAN,
            ));
        }
        Ok(())
    }

    fn analyze_decl(&mut self, decl: &mut Decl) -> Result<(), TypeError> {
        match &mut decl.node {
            DeclKind::Extern {
                name,
                sig,
                generic_params: _,
            } => {
                self.function_sigs.insert(name.node, sig.clone());
                Ok(())
            }
            DeclKind::Constant {
                name,
                ty,
                expr,
                is_comptime,
            } => {
                let val = self.analyze_expr(expr)?;
                if let Some(declared_ty) = ty {
                    self.structural_typecheck(
                        &val.ty,
                        &declared_ty.node,
                        declared_ty.span.clone(),
                    )?;
                }
                let inferred_ty = ty
                    .clone()
                    .unwrap_or_else(|| Type::synthetic(val.ty.clone()));
                if *is_comptime {
                    if let Some(cv) = val.comptime_value {
                        self.comptime_env.insert(name.node, cv);
                    } else {
                        return Err(TypeError::new(
                            TypeErrorKind::NotComptime,
                            expr.span.clone(),
                        ));
                    }
                }
                *ty = Some(inferred_ty.clone());
                self.type_map.insert(name.node, inferred_ty);
                Ok(())
            }
            DeclKind::TypeDef { name: _, def: _ } => Ok(()),
            DeclKind::Procedure {
                name,
                fn_ty,
                sig,
                block,
                constraints: _,
                monomorph_of: _,
                is_comptime: _,
            } => {
                if let Some(declared_fn_ty) = fn_ty {
                    let TypeKind::Fn(fn_sig) = &declared_fn_ty.node else {
                        return Err(TypeError::new(
                            TypeErrorKind::FnTypeExpected,
                            declared_fn_ty.span.clone(),
                        ));
                    };
                    if sig.node != **fn_sig {
                        return Err(TypeError::new(
                            TypeErrorKind::SignatureMismatch {
                                expected: fn_sig.clone(),
                                found: Box::new(sig.node.clone()),
                            },
                            sig.span.clone(),
                        ));
                    }
                } else {
                    *fn_ty = Some(Type::synthetic(TypeKind::Fn(Box::new(sig.node.clone()))));
                }
                let previous_comptime = std::mem::take(&mut self.comptime_env);
                for param in &sig.node.params.params {
                    if let PatKind::Symbol(sym) = param.pattern.node {
                        self.type_map.insert(sym, param.ty.clone());
                    }
                }
                self.function_sigs.insert(name.node, sig.clone());
                self.analyze_block(block)?;
                self.comptime_env = previous_comptime;
                if let Some(expr) = &mut block.node.expr
                    && let Some(ret_ty) = &sig.node.return_ty
                {
                    let expr_val = self.analyze_expr(expr)?;
                    if expr_val.ty != ret_ty.node {
                        return Err(TypeError::new(
                            TypeErrorKind::ReturnTypeMismatch {
                                expected: ret_ty.node.clone(),
                                found: expr_val.ty,
                            },
                            expr.span.clone(),
                        ));
                    }
                }
                Ok(())
            }
        }
    }

    fn analyze_block(&mut self, block: &mut Block) -> Result<(), TypeError> {
        for stmt in &mut block.node.stmts {
            self.analyze_stmt(stmt)?;
        }
        if let Some(expr) = &mut block.node.expr {
            self.analyze_expr(expr)?;
        }
        Ok(())
    }

    fn analyze_stmt(&mut self, stmt: &mut Stmt) -> Result<(), TypeError> {
        match &mut stmt.node {
            StmtKind::ValDec {
                name,
                ty,
                expr,
                is_comptime,
            } => {
                let val = self.analyze_expr(expr)?;
                if let Some(declared_ty) = ty {
                    self.structural_typecheck(
                        &val.ty,
                        &declared_ty.node,
                        declared_ty.span.clone(),
                    )?;
                }
                let inferred_ty = ty
                    .clone()
                    .unwrap_or_else(|| Type::synthetic(val.ty.clone()));
                if *is_comptime {
                    if let Some(cv) = val.comptime_value {
                        self.comptime_env.insert(name.node, cv);
                    } else {
                        return Err(TypeError::new(
                            TypeErrorKind::NotComptime,
                            expr.span.clone(),
                        ));
                    }
                }
                *ty = Some(inferred_ty.clone());
                self.type_map.insert(name.node, inferred_ty);
                Ok(())
            }
            StmtKind::Assign { name, expr } => {
                let val = self.analyze_expr(expr)?;
                let Some(expected_type) = self.type_map.get(&name.node) else {
                    return Err(TypeError::new(
                        TypeErrorKind::UnknownSymbol(name.node),
                        name.span.clone(),
                    ));
                };
                if expected_type.node != val.ty {
                    return Err(TypeError::new(
                        TypeErrorKind::ExpectedType {
                            expected: expected_type.node.clone(),
                            found: val.ty,
                        },
                        expr.span.clone(),
                    ));
                }
                Ok(())
            }
            StmtKind::IfElse(if_else) => {
                self.analyze_expr(&mut if_else.cond)?;
                self.analyze_block(&mut if_else.then)?;
                if let Some(else_) = &mut if_else.else_ {
                    self.analyze_block(else_)?;
                }
                Ok(())
            }
            StmtKind::Call(call) => {
                self.analyze_call(call)?;
                Ok(())
            }
            StmtKind::Match(Match { scrutinee, arms }) => {
                let scrutinee_val = self.analyze_expr(scrutinee)?;
                for arm in arms {
                    match &arm.pat.node {
                        PatKind::Symbol(name) => {
                            self.bind_pattern(&arm.pat, &Type::synthetic(scrutinee_val.ty.clone()));
                        }
                        PatKind::Variant { name, bindings } => {
                            let TypeKind::Variant(vfields) = &scrutinee_val.ty else {
                                return Err(TypeError::new(
                                    TypeErrorKind::ExpectedVariant,
                                    scrutinee.span.clone(),
                                ));
                            };
                            let Some(variant) = vfields.iter().find(|v| v.name.node == *name)
                            else {
                                return Err(TypeError::new(
                                    TypeErrorKind::UnknownSymbol(*name),
                                    arm.pat.span.clone(),
                                ));
                            };
                            if bindings.len() != variant.adts.len() {
                                return Err(TypeError::new(
                                    TypeErrorKind::ExpectedVariant,
                                    arm.pat.span.clone(),
                                ));
                            }
                            for (binding, adt) in bindings.iter().zip(variant.adts.iter()) {
                                self.bind_pattern(binding, adt);
                            }
                        }
                        PatKind::Record(fields) => {
                            let TypeKind::Record(scru_fields) = &scrutinee_val.ty else {
                                return Err(TypeError::new(
                                    TypeErrorKind::ExpectedRecord,
                                    scrutinee.span.clone(),
                                ));
                            };
                            let pat_ty = TypeKind::Record(fields.clone());
                            self.structural_typecheck(
                                &scrutinee_val.ty,
                                &pat_ty,
                                scrutinee.span.clone(),
                            )?;
                            for field in fields {
                                self.type_map.insert(field.name, field.ty.clone().unwrap());
                            }
                        }
                        PatKind::Literal(lit) => match lit {
                            ValueKind::Int(_) => {
                                if scrutinee_val.ty != TypeKind::Int {
                                    return Err(TypeError::new(
                                        TypeErrorKind::ExpectedType {
                                            expected: TypeKind::Int,
                                            found: scrutinee_val.ty,
                                        },
                                        scrutinee.span.clone(),
                                    ));
                                }
                            }
                            _ => todo!(),
                        },
                        PatKind::Wildcard => {}
                        _ => todo!(),
                    }
                    self.analyze_block(&mut arm.body)?;
                }
                Ok(())
            }
            StmtKind::Return(expr) => {
                if let Some(expr) = expr {
                    self.analyze_expr(expr)?;
                }
                Ok(())
            }
        }
    }
}

pub fn typecheck(ctx: &mut Ctx, module: &mut Module) -> TypecheckResult<Module> {
    let mut analyzer = Analyzer::new(ctx);
    analyzer.typecheck_module(module)?;
    Ok(module.clone())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::{
        ast::{Pat, PatKind, TypeKind},
        ctx::Ctx,
        lex, parse,
        span::Spanned,
    };

    use super::*;

    fn tokenify(s: &str) -> (Ctx, VecDeque<Spanned<crate::lex::Token>>) {
        let mut ctx = Ctx::default();
        let tokens = lex::tokenize(&mut ctx, s).map(|t| t.unwrap()).collect();
        (ctx, tokens)
    }

    fn parseify(ctx: &mut Ctx, tokens: VecDeque<Spanned<crate::lex::Token>>) -> Module {
        parse::parse(ctx, tokens).unwrap()
    }

    fn typecheck_src(src: &str) -> (Ctx, TypecheckResult<Module>) {
        let (mut ctx, tokens) = tokenify(src);
        let mut module = parseify(&mut ctx, tokens);
        let result = typecheck(&mut ctx, &mut module);
        (ctx, result)
    }

    #[test]
    fn test_typecheck_simple_procedure() {
        let src = "foo :: (x: int): int { y : int = x }";
        let (_, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_infers_val_type() {
        let src = "foo :: (x: int): int { y := x }";
        let (_, result) = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { block, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        let StmtKind::ValDec { ty, .. } = &block.node.stmts[0].node else {
            panic!("expected val dec");
        };

        assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
    }

    #[test]
    fn test_typecheck_binop_expression() {
        let src = "foo :: (x: int, y: int): int { z : int = x + y }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_assignment() {
        let src = "foo :: (x: int): int { y : int = x \n y = x + 1 }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_if_else() {
        let src = "foo :: (x: int): int { if x { y : int = 1 } else { z : int = 2 } }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_constant() {
        let src = "MY_CONST :: 42";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Constant { ty, .. } = &module.declarations[0].node else {
            panic!("expected constant");
        };

        assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
    }

    #[test]
    fn test_typecheck_function_call() {
        let src = "bar :: (x: int): int { y : int = 1 } \n foo :: (a: int): int { bar(1) }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_function_call_with_return() {
        let src = "bar :: (x: int): int { y : int = 1 } \n foo :: (a: int): int { result : int = bar(1) }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_error_unknown_symbol_in_ident() {
        let src = "foo :: (): int { y : int = unknown }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }

    #[test]
    fn test_typecheck_error_unknown_symbol_in_assignment() {
        let src = "foo :: (): int { y : int = 1 \n y = unknown }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }

    #[test]
    fn test_typecheck_error_unknown_function() {
        let src = "foo :: (): int { unknown_func(1) }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }

    #[test]
    fn test_typecheck_nested_if() {
        let src = "foo :: (x: int): int { if 1 { if 1 { y : int = 1 } } }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_multiple_params() {
        let src = "foo :: (a: int, b: int, c: int): int { sum : int = 1 + 2 + 3 }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_procedure_infers_fn_type() {
        let src = "foo :: (x: int): int { y : int = 1 }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { fn_ty, sig, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        assert!(fn_ty.is_some());
        let TypeKind::Fn(fn_sig) = &fn_ty.as_ref().unwrap().node else {
            panic!("expected fn type");
        };
        assert_eq!(**fn_sig, sig.node);
    }

    #[test]
    fn test_typecheck_multiple_declarations() {
        let src = "CONST :: 10 \n bar :: (x: int): int { y : int = 1 } \n foo :: (a: int): int { bar(1) }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        assert_eq!(module.declarations.len(), 3);
    }

    #[test]
    fn test_typecheck_constant_with_type_annotation() {
        let src = "MY_CONST : int : 42";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Constant { ty, .. } = &module.declarations[0].node else {
            panic!("expected constant");
        };

        assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
    }

    #[test]
    fn test_typecheck_constant_with_binop() {
        let src = "MY_CONST :: 1 + 2 * 3";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Constant { ty, .. } = &module.declarations[0].node else {
            panic!("expected constant");
        };

        assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
    }

    #[test]
    fn test_typecheck_empty_procedure() {
        let src = "foo :: () {}";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_procedure_with_return_type() {
        let src = "foo :: (): int {}";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { sig, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        assert_eq!(
            sig.node.return_ty.as_ref().map(|t| &t.node),
            Some(&TypeKind::Int)
        );
    }

    #[test]
    fn test_typecheck_val_dec_infers_int_from_literal() {
        let src = "foo :: (): int { y := 42 }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { block, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        let StmtKind::ValDec { ty, .. } = &block.node.stmts[0].node else {
            panic!("expected val dec");
        };

        assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
    }

    #[test]
    fn test_typecheck_multiple_val_decs() {
        let src = "foo :: (): int { a : int = 1 \n b : int = 2 \n c : int = 3 }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_val_dec_uses_previous_val() {
        let src = "foo :: (): int { a : int = 1 \n b : int = a }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_assignment_to_declared_var() {
        let src = "foo :: (): int { a : int = 1 \n a = 2 }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_error_assignment_to_undeclared_var() {
        let src = "foo :: (): int { a = 1 }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }

    #[test]
    fn test_typecheck_variant_type() {
        let src = "foo :: (): int { x := .None }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { block, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
            panic!("expected val dec");
        };

        assert!(matches!(
            &expr.node,
            ExprKind::Allocation {
                kind: AllocKind::Variant(variant_name),
                elements,..

            } if *variant_name == Symbol(2) && elements.is_empty()
        ));
    }

    #[test]
    fn test_typecheck_variant_type_with_args() {
        let src = "foo :: (): int { x : .Some(int) | .None = .Some(1) }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let DeclKind::Procedure { block, .. } = &module.declarations[0].node else {
            panic!("expected procedure");
        };

        let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
            panic!("expected val dec");
        };

        assert!(matches!(
            &expr.node,
            ExprKind::Allocation {
                kind: AllocKind::Variant(variant_name),
                elements,..

            } if *variant_name == Symbol(2) && elements.len() == 1 && elements[0].node == ExprKind::Value(ValueKind::Int(1))
        ));
    }

    #[test]
    fn test_typecheck_subtyping_record() {
        let src = "foo :: () { x : { a: int } = { a := 1, b := 2 } }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_comptime_constant_evaluation() {
        let src = "MY_CONST :: 1 + 2 * 3";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_comptime_bool_evaluation() {
        let src = "TRUE_CONST :: true and false";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_monomorphization_basic_comptime_type() {
        let src = "identity :: (comptime T: type, x: T): T { return x }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
        let module = result.unwrap();

        let proc = module
            .declarations
            .iter()
            .find(|d| matches!(&d.node, DeclKind::Procedure { .. }))
            .expect("generic function not found");

        if let DeclKind::Procedure {
            sig, monomorph_of, ..
        } = &proc.node
        {
            assert!(
                monomorph_of.is_none(),
                "Original should not have monomorph_of"
            );
            let params = &sig.node.params.params;
            assert_eq!(params.len(), 2);
            assert!(params[0].is_comptime);
            assert!(matches!(params[0].ty.node, TypeKind::Type));
        }
    }

    #[test]
    fn test_monomorphization_multiple_comptime_params() {
        let src = "first :: (comptime T: type, comptime U: type, a: T, b: U): T { return a }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
        let module = result.unwrap();

        let proc = module
            .declarations
            .iter()
            .find(|d| matches!(&d.node, DeclKind::Procedure { .. }))
            .expect("generic function not found");

        if let DeclKind::Procedure { sig, .. } = &proc.node {
            let params = &sig.node.params.params;
            assert_eq!(params.len(), 4);
            assert!(params[0].is_comptime);
            assert!(params[1].is_comptime);
            assert!(!params[2].is_comptime);
            assert!(!params[3].is_comptime);
        }
    }

    #[test]
    fn test_monomorphization_mixed_params() {
        let src = "wrap :: (comptime T: type, x: T, y: int): T { return x }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok());
        let module = result.unwrap();

        let proc = module
            .declarations
            .iter()
            .find(|d| matches!(&d.node, DeclKind::Procedure { .. }))
            .expect("generic function not found");

        if let DeclKind::Procedure { sig, .. } = &proc.node {
            let params = &sig.node.params.params;
            assert_eq!(params.len(), 3);
            assert!(params[0].is_comptime);
            assert!(!params[1].is_comptime);
            assert!(!params[2].is_comptime);
        }
    }

    #[test]
    fn test_monomorphization_basic() {
        let src = "identity :: (comptime T: type, x: T): T { return x } \n main :: () { y : int = identity(int, 42) }";
        let (ctx, result) = typecheck_src(src);
        assert!(result.is_ok(), "typechecking failed: {:?}", result);
        let module = result.unwrap();

        let monomorph_decl = module.declarations.iter().find(|d| {
            if let DeclKind::Procedure { name, .. } = &d.node {
                ctx.resolve(name.node).starts_with("identity_TInt")
            } else {
                false
            }
        });
        assert!(
            monomorph_decl.is_some(),
            "monomorphized identity_TInt not found"
        );

        let monomorph = monomorph_decl.unwrap();
        if let DeclKind::Procedure {
            sig, monomorph_of, ..
        } = &monomorph.node
        {
            assert!(monomorph_of.is_some(), "monomorph_of should be set");
            let params = &sig.node.params.params;
            assert_eq!(params.len(), 2);
            assert!(!params[0].is_comptime, "T param should become runtime");
            assert!(!params[1].is_comptime, "x param should remain runtime");
        }
    }
}
