#![allow(unused)]

use std::collections::HashMap;

use crate::{
    ast::{
        self, AllocKind, Block, BlockInner, Call, Decl, DeclKind, Expr, ExprKind, IfElse, Module,
        Params, Pat, PatKind, Region, Signature, SignatureInner, Stmt, StmtKind, Type, TypeKind,
        ValueKind,
    },
    ctx::Symbol,
    span::{DUMMY_SPAN, Span, Spanned},
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

#[derive(Debug)]
pub enum TypeErrorKind {
    ExpectedType {
        expected: TypeKind,
        found: TypeKind,
    },
    FnTypeExpected,
    SignatureMismatch {
        expected: Box<SignatureInner>,
        found: Box<SignatureInner>,
    },
    UnknownSymbol(Symbol),
}

pub struct TypecheckCtx {
    type_map: HashMap<Symbol, Type>,
    function_sigs: HashMap<Symbol, Signature>,
}

impl TypecheckCtx {
    fn new() -> Self {
        let mut function_sigs = HashMap::new();

        // Register built-in functions
        // println takes one int argument and returns unit
        function_sigs.insert(
            Symbol(-1), // println
            Spanned::new(
                SignatureInner {
                    params: Params {
                        patterns: vec![Spanned::new(PatKind::Symbol(Symbol(-100)), DUMMY_SPAN)],
                        types: vec![Type::synthetic(TypeKind::Int)],
                    },
                    return_ty: None,
                },
                DUMMY_SPAN,
            ),
        );

        Self {
            type_map: HashMap::new(),
            function_sigs,
        }
    }
}

pub type TypecheckResult<T> = Result<T, TypeError>;

trait ResolveType {
    fn resolve_type(&self, ctx: &TypecheckCtx) -> Type;
}

impl ResolveType for Expr {
    fn resolve_type(&self, ctx: &TypecheckCtx) -> Type {
        match &self.node {
            ExprKind::Value(v) => match v {
                ValueKind::Int(_) => Type::synthetic(TypeKind::Int),
                ValueKind::Ident(i) => ctx.type_map.get(i).unwrap().clone(),
            },
            ExprKind::BinOp { lhs, rhs, .. } => {
                let lhs_ty = lhs.resolve_type(ctx);
                let rhs_ty = rhs.resolve_type(ctx);
                if lhs_ty.node != rhs_ty.node {
                    panic!("expected types to be equal");
                }
                lhs_ty
            }
            ExprKind::Call(Call { callee, args: _ }) => {
                let callee_sig = ctx.function_sigs.get(&callee.node).unwrap();
                match callee_sig.node.return_ty.clone() {
                    Some(ty) => ty,
                    None => Type::synthetic(TypeKind::Unit),
                }
            }
            ExprKind::Allocation {
                kind,
                region,
                elements,
            } => {
                // TODO: implement region handling
                // regions should be resolved by the time we get here
                let region_handle = region.unwrap_or(Region::Local);
                let kind = match kind {
                    AllocKind::Tuple(tys) => {
                        let mut types = Vec::new();
                        for elem in elements {
                            types.push(elem.resolve_type(ctx).node);
                        }

                        if tys.len() != 0 && tys.len() != types.len() && *tys != types {
                            panic!("expected tuple types to be equal");
                        }

                        // here we resolve the tuple types if they were not provided by a type hint
                        AllocKind::Tuple(types)
                    }
                    _ => kind.clone(),
                };

                Type::synthetic(TypeKind::Alloc(kind, region_handle))
            }
        }
    }
}

trait Typecheck {
    // INVARIANT: After typechecking, all types are either fully resolved, or an error is returned.
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()>;
}

fn type_check_call(call: &mut Call, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
    let Some(callee_sig) = ctx.function_sigs.get(&call.callee.node) else {
        return Err(TypeError::new(
            TypeErrorKind::UnknownSymbol(call.callee.node),
            call.callee.span.clone(),
        ));
    };

    // drop ctx &mut borrow here
    let callee_signature = callee_sig.clone();

    for arg in call.args.iter_mut() {
        arg.typecheck(ctx)?;
    }

    for (arg, ty) in call
        .args
        .iter()
        .zip(callee_signature.node.params.types.iter())
    {
        let arg_ty = arg.resolve_type(ctx);
        if ty.node != arg_ty.node {
            return Err(TypeError::new(
                TypeErrorKind::ExpectedType {
                    expected: ty.node.clone(),
                    found: arg_ty.node,
                },
                arg.span.clone(),
            ));
        }
    }

    Ok(())
}

impl Typecheck for Expr {
    #[allow(clippy::only_used_in_recursion)]
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        match &mut self.node {
            ExprKind::Value(v) => match v {
                ValueKind::Int(_) => Ok(()),
                ValueKind::Ident(i) => {
                    if !ctx.type_map.contains_key(i) {
                        return Err(TypeError::new(
                            TypeErrorKind::UnknownSymbol(*i),
                            self.span.clone(),
                        ));
                    }
                    Ok(())
                }
            },
            ExprKind::BinOp { lhs, rhs, .. } => {
                lhs.typecheck(ctx)?;
                rhs.typecheck(ctx)?;
                Ok(())
            }
            ExprKind::Call(c) => type_check_call(c, ctx),
            ExprKind::Allocation { elements, kind, .. } => {
                for e in &mut *elements {
                    e.typecheck(ctx)?;
                }

                match kind {
                    AllocKind::Tuple(tys) => {
                        let mut types = Vec::new();
                        for elem in elements {
                            types.push(elem.resolve_type(ctx).node);
                        }

                        if !tys.is_empty() && tys.len() != types.len() && *tys != types {
                            panic!("expected tuple types to be equal");
                        }

                        *tys = types;
                    }
                    AllocKind::Array(ty_kind, size) => {
                        assert_eq!(*size, elements.len());

                        assert!(
                            elements
                                .iter()
                                .all(|e| e.resolve_type(ctx).node == **ty_kind)
                        );
                    }
                    AllocKind::DynArray(ty_kind) => {
                        assert!(
                            elements
                                .iter()
                                .all(|e| e.resolve_type(ctx).node == **ty_kind)
                        );
                    }
                }
                Ok(())
            }
        }
    }
}

impl Typecheck for Stmt {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        match &mut self.node {
            StmtKind::ValDec { name, ty, expr } => {
                expr.typecheck(ctx)?;
                let expr_ty = expr.resolve_type(ctx);

                if let Some(declared_ty) = ty {
                    if declared_ty.node != expr_ty.node {
                        return Err(TypeError::new(
                            TypeErrorKind::ExpectedType {
                                expected: declared_ty.node.clone(),
                                found: expr_ty.node,
                            },
                            expr.span.clone(),
                        ));
                    }
                } else {
                    *ty = Some(expr_ty.clone());
                }
                ctx.type_map.insert(name.node, expr_ty);
            }
            StmtKind::Assign { name, expr } => {
                expr.typecheck(ctx)?;
                let expr_ty = expr.resolve_type(ctx);

                let Some(expected_type) = ctx.type_map.get(&name.node) else {
                    return Err(TypeError::new(
                        TypeErrorKind::UnknownSymbol(name.node),
                        name.span.clone(),
                    ));
                };

                if expected_type.node != expr_ty.node {
                    return Err(TypeError::new(
                        TypeErrorKind::ExpectedType {
                            expected: expected_type.node.clone(),
                            found: expr_ty.node,
                        },
                        expr.span.clone(),
                    ));
                }
            }
            StmtKind::IfElse(IfElse { cond, then, else_ }) => {
                // TODO: check that cond is a bool, right now we only have ints
                cond.typecheck(ctx)?;
                then.typecheck(ctx)?;
                if let Some(else_) = else_ {
                    else_.typecheck(ctx)?;
                }
            }
            StmtKind::Call(c) => type_check_call(c, ctx)?,
        }

        Ok(())
    }
}

impl Typecheck for Block {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        for stmt in self.node.stmts.iter_mut() {
            stmt.typecheck(ctx)?;
        }

        Ok(())
    }
}

impl Typecheck for Decl {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        match &mut self.node {
            DeclKind::Constant { name, ty, expr } => {
                expr.typecheck(ctx)?;
                let expr_ty = expr.resolve_type(ctx);

                if let Some(declared_ty) = ty {
                    if declared_ty.node != expr_ty.node {
                        return Err(TypeError::new(
                            TypeErrorKind::ExpectedType {
                                expected: declared_ty.node.clone(),
                                found: expr_ty.node,
                            },
                            expr.span.clone(),
                        ));
                    }
                } else {
                    *ty = Some(expr_ty);
                }
            }
            DeclKind::TypeDef { name: _, def: _ } => {}
            DeclKind::Procedure {
                name,
                fn_ty,
                sig,
                block,
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

                for (pat, ty) in sig
                    .node
                    .params
                    .patterns
                    .iter()
                    .zip(sig.node.params.types.iter())
                {
                    if let PatKind::Symbol(sym) = pat.node {
                        ctx.type_map.insert(sym, ty.clone());
                    }
                }

                ctx.function_sigs.insert(name.node, sig.clone());
                block.typecheck(ctx)?;
            }
        }

        Ok(())
    }
}

impl Typecheck for Module {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        for decl in self.declarations.iter_mut() {
            decl.typecheck(ctx)?;
        }

        Ok(())
    }
}

pub fn typecheck(module: &mut Module) -> TypecheckResult<()> {
    let mut ctx = TypecheckCtx::new();
    module.typecheck(&mut ctx)
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
        let mut ctx = Ctx::new();
        let tokens = lex::tokenize(&mut ctx, s).map(|t| t.unwrap()).collect();
        (ctx, tokens)
    }

    fn parseify(ctx: &mut Ctx, tokens: VecDeque<Spanned<crate::lex::Token>>) -> Module {
        parse::parse(ctx, tokens).unwrap()
    }

    fn typecheck_src(src: &str) -> TypecheckResult<Module> {
        let (mut ctx, tokens) = tokenify(src);
        let mut module = parseify(&mut ctx, tokens);
        typecheck(&mut module)?;
        Ok(module)
    }

    #[test]
    fn test_typecheck_simple_procedure() {
        let src = "foo :: (x: int): int { y : int = x }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_infers_val_type() {
        let src = "foo :: (x: int): int { y := x }";
        let result = typecheck_src(src);
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
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_assignment() {
        let src = "foo :: (x: int): int { y : int = x \n y = x + 1 }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_if_else() {
        let src = "foo :: (x: int): int { if x { y : int = 1 } else { z : int = 2 } }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_constant() {
        let src = "MY_CONST :: 42";
        let result = typecheck_src(src);
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
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_function_call_with_return() {
        let src = "bar :: (x: int): int { y : int = 1 } \n foo :: (a: int): int { result : int = bar(1) }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_error_unknown_symbol_in_ident() {
        let src = "foo :: (): int { y : int = unknown }";
        let result = typecheck_src(src);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }

    #[test]
    fn test_typecheck_error_unknown_symbol_in_assignment() {
        let src = "foo :: (): int { y : int = 1 \n y = unknown }";
        let result = typecheck_src(src);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }

    #[test]
    fn test_typecheck_error_unknown_function() {
        let src = "foo :: (): int { unknown_func(1) }";
        let result = typecheck_src(src);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }

    #[test]
    fn test_typecheck_nested_if() {
        let src = "foo :: (x: int): int { if 1 { if 1 { y : int = 1 } } }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_multiple_params() {
        let src = "foo :: (a: int, b: int, c: int): int { sum : int = 1 + 2 + 3 }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_procedure_infers_fn_type() {
        let src = "foo :: (x: int): int { y : int = 1 }";
        let result = typecheck_src(src);
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
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        assert_eq!(module.declarations.len(), 3);
    }

    #[test]
    fn test_typecheck_constant_with_type_annotation() {
        let src = "MY_CONST : int : 42";
        let result = typecheck_src(src);
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
        let result = typecheck_src(src);
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
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_procedure_with_return_type() {
        let src = "foo :: (): int {}";
        let result = typecheck_src(src);
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
        let result = typecheck_src(src);
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
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_val_dec_uses_previous_val() {
        let src = "foo :: (): int { a : int = 1 \n b : int = a }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_assignment_to_declared_var() {
        let src = "foo :: (): int { a : int = 1 \n a = 2 }";
        let result = typecheck_src(src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_typecheck_error_assignment_to_undeclared_var() {
        let src = "foo :: (): int { a = 1 }";
        let result = typecheck_src(src);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
    }
}
