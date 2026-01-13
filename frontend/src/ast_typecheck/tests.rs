#![cfg(test)]

use super::*;

use std::collections::VecDeque;

use crate::{ast::TypeKind, ctx::Ctx, lex, parse, span::Spanned};

fn tokenify(s: &str) -> (Ctx, VecDeque<Spanned<crate::lex::Token>>) {
    let mut ctx = Ctx::default();
    let tokens = lex::tokenize(&mut ctx, s).map(|t| t.unwrap()).collect();
    (ctx, tokens)
}

fn parseify(ctx: &mut Ctx, tokens: VecDeque<Spanned<crate::lex::Token>>) -> Module {
    parse::parse(ctx, tokens).unwrap()
}

fn typecheck_src(src: &str) -> (Ctx, TypecheckResult<()>, Module) {
    let (mut ctx, tokens) = tokenify(src);
    let mut module = parseify(&mut ctx, tokens);
    let result = typecheck(&mut ctx, &mut module);
    (ctx, result, module)
}

#[test]
fn test_typecheck_simple_procedure() {
    let src = "foo :: (x: int): int { y : int = x }";
    let (_, result, _) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_infers_val_type() {
    let src = "foo :: (x: int): int { y := x }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

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
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_assignment() {
    let src = "foo :: (x: int): int { y : int = x \n y = x + 1 }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_if_else() {
    let src = "foo :: (x: int): int { if x { y : int = 1 } else { z : int = 2 } }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_constant() {
    let src = "MY_CONST :: 42";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
    assert!(result.is_ok());

    let DeclKind::Constant { ty, .. } = &module.declarations[0].node else {
        panic!("expected constant");
    };

    assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
}

#[test]
fn test_typecheck_function_call() {
    let src = "bar :: (x: int): int { y : int = 1 } \n foo :: (a: int): int { bar(1) }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_function_call_with_return() {
    let src =
        "bar :: (x: int): int { y : int = 1 } \n foo :: (a: int): int { result : int = bar(1) }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_error_unknown_symbol_in_ident() {
    let src = "foo :: (): int { y : int = unknown }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
}

#[test]
fn test_typecheck_error_unknown_symbol_in_assignment() {
    let src = "foo :: (): int { y : int = 1 \n y = unknown }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
}

#[test]
fn test_typecheck_error_unknown_function() {
    let src = "foo :: (): int { unknown_func(1) }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
}

#[test]
fn test_typecheck_nested_if() {
    let src = "foo :: (x: int): int { if 1 { if 1 { y : int = 1 } } }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_multiple_params() {
    let src = "foo :: (a: int, b: int, c: int): int { sum : int = 1 + 2 + 3 }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_procedure_infers_fn_type() {
    let src = "foo :: (x: int): int { y : int = 1 }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

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
    let src =
        "CONST :: 10 \n bar :: (x: int): int { y : int = 1 } \n foo :: (a: int): int { bar(1) }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

    assert_eq!(module.declarations.len(), 3);
}

#[test]
fn test_typecheck_constant_with_type_annotation() {
    let src = "MY_CONST : int : 42";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

    let DeclKind::Constant { ty, .. } = &module.declarations[0].node else {
        panic!("expected constant");
    };

    assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
}

#[test]
fn test_typecheck_constant_with_binop() {
    let src = "MY_CONST :: 1 + 2 * 3";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

    let DeclKind::Constant { ty, .. } = &module.declarations[0].node else {
        panic!("expected constant");
    };

    assert_eq!(ty.as_ref().map(|t| &t.node), Some(&TypeKind::Int));
}

#[test]
fn test_typecheck_empty_procedure() {
    let src = "foo :: () {}";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_procedure_with_return_type() {
    let src = "foo :: (): int {}";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

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
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

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
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_val_dec_uses_previous_val() {
    let src = "foo :: (): int { a : int = 1 \n b : int = a }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_assignment_to_declared_var() {
    let src = "foo :: (): int { a : int = 1 \n a = 2 }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_error_assignment_to_undeclared_var() {
    let src = "foo :: (): int { a = 1 }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::UnknownSymbol(_)));
}

#[test]
fn test_typecheck_variant_type() {
    let src = "foo :: (): int { x := .None }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

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
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

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
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_comptime_constant_evaluation() {
    let src = "MY_CONST :: 1 + 2 * 3";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_comptime_bool_evaluation() {
    let src = "TRUE_CONST :: true and false";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_monomorphization_basic_comptime_type() {
    let src = "identity :: (comptime T: type, x: T): T { return x }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

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
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

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
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());

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
    let (ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok(), "typechecking failed: {:?}", result);

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
        assert_eq!(params.len(), 1);
        assert!(!params[0].is_comptime, "x param should remain runtime");
        assert_eq!(params[0].ty.node, TypeKind::Int);
    }
}

#[test]
fn test_typecheck_for_loop_array() {
    let src = "foo :: () { for x in [1, 2, 3] { y : int = x } }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_for_loop_range() {
    let src = "foo :: () { for i in 0..10 { y : int = i } }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_for_loop_with_call_in_body() {
    let src = "foo :: () { for x in [1, 2, 3] { y : int = x } }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_for_loop_nested() {
    let src = "foo :: () { for i in 0..3 { for j in 0..2 { sum : int = i + j } } }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_for_loop_binding_shadows() {
    let src = "foo :: (x: int) { for x in [1, 2, 3] { y : int = x } }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_ok());
}

#[test]
fn test_typecheck_for_loop_error_non_iterable() {
    let src = "foo :: () { for x in 5 { } }";
    let (_ctx, result, module) = typecheck_src(src);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::ExpectedIterable));
}
