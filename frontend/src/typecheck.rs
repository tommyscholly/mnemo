#![allow(unused)]

use std::collections::HashMap;

use crate::{
    ast::{self, Block, Call, Decl, Expr, IfElse, Module, Pat, Signature, Stmt, Type, Value},
    ctx::Symbol,
};

#[derive(Debug)]
pub enum TypeError {
    ExpectedType {
        expected: ast::Type,
        found: ast::Type,
        symbol: Symbol,
    },
    FnTypeExpected(Symbol),
    SignatureMismatch {
        expected: Box<ast::Signature>,
        found: Box<ast::Signature>,
        symbol: Symbol,
    },
    UnknownSymbol(Symbol),
}

pub struct TypecheckCtx {
    type_map: HashMap<Symbol, ast::Type>,
    function_sigs: HashMap<Symbol, ast::Signature>,
}

impl TypecheckCtx {
    fn new() -> Self {
        Self {
            type_map: HashMap::new(),
            function_sigs: HashMap::new(),
        }
    }
}

pub type TypecheckResult<T> = Result<T, TypeError>;

trait ResolveType {
    fn resolve_type(&self, ctx: &TypecheckCtx) -> ast::Type;
}

impl ResolveType for ast::Expr {
    fn resolve_type(&self, ctx: &TypecheckCtx) -> ast::Type {
        match self {
            ast::Expr::Value(v) => match v {
                Value::Int(_) => ast::Type::Int,
                Value::Ident(i) => ctx.type_map.get(i).unwrap().clone(),
            },
            ast::Expr::BinOp { lhs, rhs, .. } => {
                let lhs = lhs.resolve_type(ctx);
                let rhs = rhs.resolve_type(ctx);
                if lhs != rhs {
                    panic!("expected types to be equal");
                }
                lhs
            }
            ast::Expr::Call(Call { callee, args: _ }) => {
                let callee_sig = ctx.function_sigs.get(callee).unwrap();
                match callee_sig.return_ty.clone() {
                    Some(ty) => ty,
                    None => ast::Type::Unit,
                }
            }
        }
    }
}

trait Typecheck {
    // INVARIANT: After typechecking, all types are either fully resolved, or an error is returned.
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()>;
}

fn type_check_call(call: &mut Call, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
    let Some(callee_sig) = ctx.function_sigs.get(&call.callee) else {
        return Err(TypeError::UnknownSymbol(call.callee));
    };

    // drop ctx &mut borrow here
    let callee_signature = callee_sig.clone();

    for arg in call.args.iter_mut() {
        arg.typecheck(ctx)?;
    }

    for (arg, ty) in call.args.iter().zip(callee_signature.params.types.iter()) {
        if *ty != arg.resolve_type(ctx) {
            return Err(TypeError::ExpectedType {
                expected: ty.clone(),
                found: arg.resolve_type(ctx),
                symbol: call.callee,
            });
        }
    }

    Ok(())
}

impl Typecheck for Expr {
    #[allow(clippy::only_used_in_recursion)]
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        match self {
            Expr::Value(v) => match v {
                Value::Int(_) => Ok(()),
                Value::Ident(i) => {
                    let _ = ctx
                        .type_map
                        .get(i)
                        .map_or_else(|| Err(TypeError::UnknownSymbol(*i)), |ty| Ok(ty.clone()));

                    Ok(())
                }
            },
            Expr::BinOp { lhs, rhs, .. } => {
                lhs.typecheck(ctx)?;
                rhs.typecheck(ctx)?;
                Ok(())
            }
            Expr::Call(c) => type_check_call(c, ctx),
        }
    }
}

impl Typecheck for Stmt {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        match self {
            Stmt::ValDec { name, ty, expr } => {
                expr.typecheck(ctx)?;
                if let Some(ty) = ty {
                    if *ty != expr.resolve_type(ctx) {
                        return Err(TypeError::ExpectedType {
                            expected: ty.clone(),
                            found: expr.resolve_type(ctx),
                            symbol: *name,
                        });
                    }
                } else {
                    *ty = Some(expr.resolve_type(ctx));
                }
                ctx.type_map.insert(*name, expr.resolve_type(ctx));
            }
            Stmt::Assign { name, expr } => {
                expr.typecheck(ctx)?;
                let ty = expr.resolve_type(ctx);
                let Some(expected_type) = ctx.type_map.get(name) else {
                    return Err(TypeError::UnknownSymbol(*name));
                };

                if *expected_type != ty {
                    return Err(TypeError::ExpectedType {
                        expected: expected_type.clone(),
                        found: ty,
                        symbol: *name,
                    });
                }
            }
            Stmt::IfElse(IfElse { cond, then, else_ }) => {
                // TODO: check that cond is a bool, right now we only have ints
                cond.typecheck(ctx)?;
                then.typecheck(ctx)?;
                if let Some(else_) = else_ {
                    else_.typecheck(ctx)?;
                }
            }
            Stmt::Call(c) => type_check_call(c, ctx)?,
        }

        Ok(())
    }
}

impl Typecheck for Block {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        for stmt in self.stmts.iter_mut() {
            stmt.typecheck(ctx)?;
        }

        Ok(())
    }
}

impl Typecheck for Decl {
    fn typecheck(&mut self, ctx: &mut TypecheckCtx) -> TypecheckResult<()> {
        match self {
            Decl::Constant { name, ty, expr } => {
                expr.typecheck(ctx)?;
                if let Some(ty) = ty {
                    if *ty != expr.resolve_type(ctx) {
                        return Err(TypeError::ExpectedType {
                            expected: ty.clone(),
                            found: expr.resolve_type(ctx),
                            symbol: *name,
                        });
                    }
                } else {
                    *ty = Some(expr.resolve_type(ctx));
                }
            }
            Decl::TypeDef { name: _, def: _ } => {}
            Decl::Procedure {
                name,
                fn_ty,
                sig,
                block,
            } => {
                if let Some(fn_ty) = fn_ty {
                    let Type::Fn(fn_sig) = fn_ty else {
                        return Err(TypeError::FnTypeExpected(*name));
                    };

                    if *sig != **fn_sig {
                        return Err(TypeError::SignatureMismatch {
                            expected: fn_sig.clone(),
                            found: sig.clone().into(),
                            symbol: *name,
                        });
                    }
                } else {
                    *fn_ty = Some(Type::Fn(sig.clone().into()));
                }

                for (pat, ty) in sig.params.patterns.iter().zip(sig.params.types.iter()) {
                    #[allow(irrefutable_let_patterns)]
                    if let Pat::Symbol(name) = pat {
                        ctx.type_map.insert(*name, ty.clone());
                    }
                }

                ctx.function_sigs.insert(*name, sig.clone());
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
        ast::Pat,
        ctx::Ctx,
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
        let Decl::Procedure { block, .. } = &module.declarations[0] else {
            panic!("expected procedure");
        };

        let Stmt::ValDec { ty, .. } = &block.stmts[0] else {
            panic!("expected val dec");
        };

        assert_eq!(*ty, Some(Type::Int));
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
        let Decl::Constant { ty, .. } = &module.declarations[0] else {
            panic!("expected constant");
        };

        assert_eq!(*ty, Some(Type::Int));
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
        // Note: Currently resolve_type panics on unknown symbols rather than
        // returning an error. This test verifies the current behavior.
        let src = "foo :: (): int { y : int = unknown }";
        let result = std::panic::catch_unwind(|| typecheck_src(src));
        assert!(result.is_err(), "expected panic on unknown symbol");
    }

    #[test]
    fn test_typecheck_error_unknown_symbol_in_assignment() {
        // Note: Currently resolve_type panics on unknown symbols rather than
        // returning an error. This test verifies the current behavior.
        let src = "foo :: (): int { y : int = 1 \n y = unknown }";
        let result = std::panic::catch_unwind(|| typecheck_src(src));
        assert!(result.is_err(), "expected panic on unknown symbol");
    }

    #[test]
    fn test_typecheck_error_unknown_function() {
        let src = "foo :: (): int { unknown_func(1) }";
        let result = typecheck_src(src);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, TypeError::UnknownSymbol(_)));
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
        let Decl::Procedure { fn_ty, sig, .. } = &module.declarations[0] else {
            panic!("expected procedure");
        };

        assert!(fn_ty.is_some());
        let Type::Fn(fn_sig) = fn_ty.as_ref().unwrap() else {
            panic!("expected fn type");
        };
        assert_eq!(**fn_sig, *sig);
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
        let Decl::Constant { ty, .. } = &module.declarations[0] else {
            panic!("expected constant");
        };

        assert_eq!(*ty, Some(Type::Int));
    }

    #[test]
    fn test_typecheck_constant_with_binop() {
        let src = "MY_CONST :: 1 + 2 * 3";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let Decl::Constant { ty, .. } = &module.declarations[0] else {
            panic!("expected constant");
        };

        assert_eq!(*ty, Some(Type::Int));
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
        let Decl::Procedure { sig, .. } = &module.declarations[0] else {
            panic!("expected procedure");
        };

        assert_eq!(sig.return_ty, Some(Type::Int));
    }

    #[test]
    fn test_typecheck_val_dec_infers_int_from_literal() {
        let src = "foo :: (): int { y := 42 }";
        let result = typecheck_src(src);
        assert!(result.is_ok());

        let module = result.unwrap();
        let Decl::Procedure { block, .. } = &module.declarations[0] else {
            panic!("expected procedure");
        };

        let Stmt::ValDec { ty, .. } = &block.stmts[0] else {
            panic!("expected val dec");
        };

        assert_eq!(*ty, Some(Type::Int));
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
        assert!(matches!(err, TypeError::UnknownSymbol(_)));
    }
}
