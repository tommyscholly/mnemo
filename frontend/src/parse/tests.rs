#![cfg(test)]

use super::*;

use crate::{ast::TypeAliasDefinition, ctx::Symbol, lex};

fn tokenify(s: &str) -> (Ctx, VecDeque<SpannedToken>) {
    let mut ctx = Ctx::default();
    let tokens = lex::tokenize(&mut ctx, s).map(|t| t.unwrap()).collect();
    (ctx, tokens)
}

fn parse_decls_from(input: &str) -> ParseResult<Vec<Decl>> {
    let (mut ctx, mut tokens) = tokenify(input);
    parse_decls(&mut ctx, &mut tokens)
}

fn expect_parse_ok(input: &str, expected_decl_count: usize) -> Vec<Decl> {
    let decs = parse_decls_from(input).expect("expected parsing to succeed");
    assert_eq!(
        decs.len(),
        expected_decl_count,
        "expected {expected_decl_count} declarations from input `{input}`"
    );
    decs
}

fn assert_decl_is_constant(decl: &Decl, expected_name_id: u32, expected_has_ty: bool) {
    let DeclKind::Constant { name, ty, .. } = &decl.node else {
        panic!("expected Constant declaration");
    };
    assert_eq!(name.node, Symbol(expected_name_id));
    assert_eq!(ty.is_some(), expected_has_ty);
}

fn assert_decl_is_procedure(decl: &Decl, expected_name_id: u32, expected_param_count: usize) {
    let DeclKind::Procedure { name, sig, .. } = &decl.node else {
        panic!("expected Procedure declaration");
    };
    assert_eq!(name.node, Symbol(expected_name_id));
    assert_eq!(sig.node.params.params.len(), expected_param_count);
}

fn assert_decl_is_typedef(decl: &Decl, expected_name_id: u32) {
    let DeclKind::TypeDef { name, .. } = &decl.node else {
        panic!("expected TypeDef declaration");
    };
    assert_eq!(name.node, Symbol(expected_name_id));
}

fn expect_err_kind(input: &str, expected_kind: ParseErrorKind) {
    let result = parse_decls_from(input);
    assert!(result.is_err(), "expected error but got {:?}", result);
    assert_eq!(result.unwrap_err().kind, expected_kind);
}

#[test]
fn parse_type_as_value() {
    // 'type' is now a keyword, so this tests that we can parse a procedure with empty body
    let decs = expect_parse_ok("foo :: () { }", 1);
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        panic!("expected procedure");
    };

    assert!(block.node.stmts.is_empty());
}

#[test]
fn constant_parse_type_annot() {
    let decs = expect_parse_ok("foo: int : 1", 1);
    assert_decl_is_constant(&decs[0], 0, true);

    // Verify the type annotation
    let DeclKind::Constant { ty, expr, .. } = &decs[0].node else {
        unreachable!()
    };
    assert!(matches!(ty.as_ref().unwrap().node, TypeKind::Int));
    assert!(matches!(expr.node, ExprKind::Value(ValueKind::Int(1))));
}

#[test]
fn constant_bool_parse() {
    let decs = expect_parse_ok("foo :: false", 1);
    assert_decl_is_constant(&decs[0], 0, false);

    // Verify the expression is an int literal
    let DeclKind::Constant { expr, .. } = &decs[0].node else {
        unreachable!()
    };
    assert!(matches!(expr.node, ExprKind::Value(ValueKind::Bool(false))));
}

#[test]
fn constant_bool_parse_type_annot() {
    let decs = expect_parse_ok("foo: bool : true", 1);
    assert_decl_is_constant(&decs[0], 0, true);

    // Verify the type annotation
    let DeclKind::Constant { ty, expr, .. } = &decs[0].node else {
        unreachable!()
    };
    assert!(matches!(ty.as_ref().unwrap().node, TypeKind::Bool));
    assert!(matches!(expr.node, ExprKind::Value(ValueKind::Bool(true))));
}

#[test]
fn constant_err_no_expr() {
    expect_err_kind("foo ::", ParseErrorKind::UnexpectedEOF);
}

#[test]
fn proc_parse() {
    let decs = expect_parse_ok("foo :: () {}", 1);
    assert_decl_is_procedure(&decs[0], 0, 0);
}

#[test]
fn proc_parse_params_ret() {
    let decs = expect_parse_ok("foo :: (x: int, y: int): int {}", 1);
    assert_decl_is_procedure(&decs[0], 0, 2);

    let DeclKind::Procedure { sig, .. } = &decs[0].node else {
        unreachable!()
    };
    assert!(matches!(
        sig.node.return_ty.as_ref().unwrap().node,
        TypeKind::Int
    ));
}

#[test]
fn proc_parse_with_min_body() {
    let decs = expect_parse_ok(
        "foo :: (x: int, y: int): int { 
                z := 1
                x = 2
            }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 2);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };
    assert_eq!(block.node.stmts.len(), 2);
    assert!(matches!(block.node.stmts[0].node, StmtKind::ValDec { .. }));
    assert!(matches!(block.node.stmts[1].node, StmtKind::Assign { .. }));
}

#[test]
fn proc_parse_with_min_body_and_type_annot() {
    let decs = expect_parse_ok(
        "foo :: (x: int, y: int): int { 
                z : int = 1
                x = 2
            }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 2);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };
    assert_eq!(block.node.stmts.len(), 2);

    // Verify the type annotation on z
    let StmtKind::ValDec { ty, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };
    assert!(matches!(ty.as_ref().unwrap().node, TypeKind::Int));
}

#[test]
fn proc_parse_with_bin_op() {
    let decs = expect_parse_ok(
        "foo :: (x: int, y: int): int { 
                z : int = 1 + 2
                x = z + y
            }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 2);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };
    assert_eq!(block.node.stmts.len(), 2);

    // Verify first statement has binop expression
    let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };
    assert!(matches!(expr.node, ExprKind::BinOp { op: BinOp::Add, .. }));
}

#[test]
fn proc_parse_with_call() {
    let decs = expect_parse_ok(
        "foo :: (x: int, y: int): int { 
                z : int = x + y
                println(z)
            }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 2);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };
    assert_eq!(block.node.stmts.len(), 2);

    // Verify second statement is a call
    assert!(matches!(block.node.stmts[1].node, StmtKind::Call(_)));
}

#[test]
fn proc_parse_with_bad_binop() {
    expect_err_kind(
        "foo :: (x: int, y: int): int { 
                z : int = x +
            }",
        ParseErrorKind::ExpectedExpression,
    );
}

#[test]
fn proc_parse_with_ifelse() {
    let decs = expect_parse_ok(
        "foo :: (x: int, y: int): int { 
                if x {
                    println(x)
                } else {
                    println(y)
                }
            }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 2);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };
    assert_eq!(block.node.stmts.len(), 1);

    // Verify it's an if-else statement
    let StmtKind::IfElse(ifelse) = &block.node.stmts[0].node else {
        unreachable!()
    };
    assert!(ifelse.else_.is_some());
}

#[test]
fn parse_type_struct() {
    let decs = expect_parse_ok("T :: { field1: int, field2: int }", 1);
    assert_decl_is_typedef(&decs[0], 0);

    let DeclKind::TypeDef { def, .. } = &decs[0].node else {
        unreachable!()
    };
    let TypeAliasDefinition::Record(fields) = def else {
        panic!("expected struct")
    };
    assert_eq!(fields.len(), 2);
}

#[test]
fn parse_type_enum() {
    let decs = expect_parse_ok("Enum :: { X1, X2, X3 }", 1);
    assert_decl_is_typedef(&decs[0], 0);

    let DeclKind::TypeDef { def, .. } = &decs[0].node else {
        unreachable!()
    };
    let TypeAliasDefinition::Variant(fields) = def else {
        panic!("expected enum")
    };
    assert_eq!(fields.len(), 3);
}

#[test]
fn parse_type_enum_with_adts() {
    let decs = expect_parse_ok("Enum :: { X1(int), X2(int, int), X3(Enum) }", 1);
    assert_decl_is_typedef(&decs[0], 0);

    let DeclKind::TypeDef { def, .. } = &decs[0].node else {
        unreachable!()
    };
    let TypeAliasDefinition::Variant(fields) = def else {
        panic!("expected enum")
    };
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].adts.len(), 1);
    assert_eq!(fields[1].adts.len(), 2);
    assert_eq!(fields[2].adts.len(), 1);
}

#[test]
fn parse_array() {
    let decs = expect_parse_ok(
        "foo :: () { 
                x := [1, 2, 3]
            }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };
    assert_eq!(block.node.stmts.len(), 1);

    // Verify first is Array, second is DynArray
    let StmtKind::ValDec { expr: expr1, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };

    assert!(matches!(
        &expr1.node,
        ExprKind::Allocation {
            kind: AllocKind::Array(_, _),
            ..
        }
    ));
}

#[test]
fn parse_tuple() {
    let decs = expect_parse_ok(
        "foo :: () { 
                x := (1, 2, 3)
            }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };
    assert_eq!(block.node.stmts.len(), 1);

    let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };
    let ExprKind::Allocation { kind, elements, .. } = &expr.node else {
        panic!("expected allocation")
    };
    // types have not been resolved since we did not provide them
    assert_eq!(*kind, AllocKind::Tuple(vec![]));
    assert_eq!(elements.len(), 3);
}

#[test]
fn parse_record() {
    let decs = expect_parse_ok(
        "foo :: () { 
                x := { a := 1, b := 2 }
            }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };
    assert_eq!(block.node.stmts.len(), 1);

    let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };
    let ExprKind::Allocation { kind, elements, .. } = &expr.node else {
        panic!("expected allocation")
    };

    // types have not been resolved since we did not provide them
    assert_eq!(
        *kind,
        AllocKind::Record(vec![
            RecordField {
                name: Symbol(2),
                ty: None
            },
            RecordField {
                name: Symbol(3),
                ty: None
            }
        ])
    );
    assert_eq!(elements.len(), 2);
}

#[test]
fn parse_variant() {
    let decs = expect_parse_ok(
        "foo :: () { 
                x := .None
                y := .Some(1)
            }",
        1,
    );

    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };
    assert_eq!(block.node.stmts.len(), 2);

    let StmtKind::ValDec { expr: expr1, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };

    assert!(matches!(
        &expr1.node,
        ExprKind::Allocation {
            kind: AllocKind::Variant(variant_name),
            elements,..

        } if *variant_name == Symbol(2) && elements.is_empty()
    ));

    let StmtKind::ValDec { expr: expr2, .. } = &block.node.stmts[1].node else {
        unreachable!()
    };
    assert!(matches!(
        &expr2.node,
        ExprKind::Allocation {
            kind: AllocKind::Variant(variant_name),
            elements,..

        } if *variant_name == Symbol(4) && elements.len() == 1 && elements[0].node == ExprKind::Value(ValueKind::Int(1))
    ));
}

#[test]
fn parse_field_access_simple() {
    let decs = expect_parse_ok(
        "foo :: () { 
            x := y.field
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };
    assert_eq!(block.node.stmts.len(), 1);

    let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };

    let ExprKind::FieldAccess(base, field) = &expr.node else {
        panic!("expected field access, got {:?}", expr.node)
    };

    // base should be identifier 'y'
    assert!(matches!(base.node, ExprKind::Value(ValueKind::Ident(_))));
    // field should be 'field' (Symbol(3) after foo, x, y)
    assert_eq!(*field, Symbol(3));
}

#[test]
fn parse_field_access_chained() {
    let decs = expect_parse_ok(
        "foo :: () { 
            x := y.a.b.c
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };

    let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };

    // Should parse as ((y.a).b).c
    let ExprKind::FieldAccess(inner1, field_c) = &expr.node else {
        panic!("expected field access")
    };
    assert_eq!(*field_c, Symbol(5)); // 'c'

    let ExprKind::FieldAccess(inner2, field_b) = &inner1.node else {
        panic!("expected field access")
    };
    assert_eq!(*field_b, Symbol(4)); // 'b'

    let ExprKind::FieldAccess(base, field_a) = &inner2.node else {
        panic!("expected field access")
    };
    assert_eq!(*field_a, Symbol(3)); // 'a'

    assert!(matches!(base.node, ExprKind::Value(ValueKind::Ident(_))));
}

#[test]
fn parse_field_access_with_binop() {
    let decs = expect_parse_ok(
        "foo :: () { 
            x := a.x + b.y
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };

    let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };

    // Should parse as (a.x) + (b.y)
    let ExprKind::BinOp { op, lhs, rhs } = &expr.node else {
        panic!("expected binop, got {:?}", expr.node)
    };
    assert_eq!(*op, BinOp::Add);

    assert!(matches!(lhs.node, ExprKind::FieldAccess(_, _)));
    assert!(matches!(rhs.node, ExprKind::FieldAccess(_, _)));
}

#[test]
fn parse_field_access_with_multiply() {
    let decs = expect_parse_ok(
        "foo :: () { 
            x := a.x * b.y + c.z
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };

    let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };

    // Should parse as ((a.x) * (b.y)) + (c.z)
    let ExprKind::BinOp { op, lhs, rhs } = &expr.node else {
        panic!("expected binop")
    };
    assert_eq!(*op, BinOp::Add);

    // rhs should be c.z
    assert!(matches!(rhs.node, ExprKind::FieldAccess(_, _)));

    // lhs should be (a.x * b.y)
    let ExprKind::BinOp { op: inner_op, .. } = &lhs.node else {
        panic!("expected binop")
    };
    assert_eq!(*inner_op, BinOp::Mul);
}

#[test]
fn parse_field_access_on_call_result() {
    let decs = expect_parse_ok(
        "foo :: () { 
            x := bar().field
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };

    let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };

    let ExprKind::FieldAccess(base, _field) = &expr.node else {
        panic!("expected field access, got {:?}", expr.node)
    };

    assert!(matches!(base.node, ExprKind::Call(_)));
}

#[test]
fn parse_field_access_on_parens() {
    let decs = expect_parse_ok(
        "foo :: () { 
            x := (a).field
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };

    let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };

    assert!(matches!(expr.node, ExprKind::FieldAccess(_, _)));
}

#[test]
fn parse_field_access_missing_field_name() {
    expect_err_kind(
        "foo :: () { 
            x := y.
        }",
        ParseErrorKind::MalformedAccess,
    );
}

#[test]
fn parse_field_access_call() {
    let decs = expect_parse_ok(
        "foo :: () { 
            x := y.method()
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };

    let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };

    let ExprKind::Call(call) = &expr.node else {
        panic!("expected call, got {:?}", expr.node)
    };

    // should be a field access: y.method
    let ExprKind::FieldAccess(receiver, method_name) = &call.callee.node else {
        panic!(
            "expected field access as callee, got {:?}",
            call.callee.node
        )
    };

    // receiver should be identifier '
    assert!(matches!(
        receiver.node,
        ExprKind::Value(ValueKind::Ident(Symbol(2)))
    ));

    // name should be method
    assert_eq!(*method_name, Symbol(3));

    assert!(call.args.is_empty());
}

#[test]
fn parse_match() {
    let decs = expect_parse_ok(
        "foo :: () { 
            match x with
                .Some(x) => { y := x }
              | .None    => { y := 0 }
            }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        panic!("expected procedure");
    };

    let StmtKind::Match(Match { scrutinee, arms }) = &block.node.stmts[0].node else {
        panic!("expected match");
    };

    assert!(matches!(
        scrutinee.node,
        ExprKind::Value(ValueKind::Ident(Symbol(1)))
    ));

    assert_eq!(arms.len(), 2);

    let MatchArm { pat, body } = &arms[0];
    assert!(
        matches!(&pat.node, PatKind::Variant { name, bindings } if *name == Symbol(2) && bindings.len() == 1)
    );
    assert_eq!(body.node.stmts.len(), 1);

    let MatchArm { pat, body } = &arms[1];
    assert!(
        matches!(&pat.node, PatKind::Variant { name, bindings } if *name == Symbol(4) && bindings.is_empty())
    );
    assert_eq!(body.node.stmts.len(), 1);
}

#[test]
fn parse_region_alloc() {
    let decs = expect_parse_ok(
        "foo :: (comptime R: region) { 
            x := [ 1, 2 ] @ local
            y := (1, false) @ R
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 1);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        unreachable!()
    };

    let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
        unreachable!()
    };

    let ExprKind::Allocation { kind, region, .. } = &expr.node else {
        panic!("expected allocation")
    };

    assert_eq!(
        *kind,
        AllocKind::Array(TypeKind::Any.into(), ComptimeValue::Int(2).into())
    );
    assert_eq!(*region, Some(Region::Scoped(0)));

    let StmtKind::ValDec { expr, .. } = &block.node.stmts[1].node else {
        unreachable!()
    };

    let ExprKind::Allocation { kind, region, .. } = &expr.node else {
        panic!("expected allocation")
    };
    // types not resolved yet
    assert_eq!(*kind, AllocKind::Tuple(vec![]));
    assert_eq!(*region, Some(Region::Named(Symbol(1))));
}

#[test]
fn test_return_stmt() {
    let decs = expect_parse_ok("foo :: (): int { return 1 }", 1);
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        panic!("expected procedure");
    };

    let StmtKind::Return(expr) = &block.node.stmts[0].node else {
        panic!("expected return stmt");
    };

    assert!(expr.is_some());
    assert_eq!(
        expr.as_ref().unwrap().node,
        ExprKind::Value(ValueKind::Int(1))
    );
}

#[test]
fn test_empty_return_stmt() {
    let decs = expect_parse_ok("foo :: (): int { return }", 1);
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        panic!("expected procedure");
    };

    let StmtKind::Return(expr) = &block.node.stmts[0].node else {
        panic!("expected return stmt");
    };

    assert!(expr.is_none());
}

#[test]
fn parse_proc_with_comptime_param() {
    let decs = expect_parse_ok("foo :: (comptime T: type) {}", 1);
    assert_decl_is_procedure(&decs[0], 0, 1);

    let DeclKind::Procedure { sig, .. } = &decs[0].node else {
        panic!("expected procedure");
    };

    let param = &sig.node.params.params[0];
    assert!(param.is_comptime);
    assert!(matches!(param.ty.node, TypeKind::Type));
}

#[test]
fn parse_proc_with_multiple_comptime_params() {
    let decs = expect_parse_ok("foo :: (comptime T: type, comptime N: int) {}", 1);
    assert_decl_is_procedure(&decs[0], 0, 2);

    let DeclKind::Procedure { sig, .. } = &decs[0].node else {
        panic!("expected procedure");
    };

    assert!(sig.node.params.params[0].is_comptime);
    assert!(matches!(sig.node.params.params[0].ty.node, TypeKind::Type));
    assert!(sig.node.params.params[1].is_comptime);
    assert!(matches!(sig.node.params.params[1].ty.node, TypeKind::Int));
}

#[test]
fn parse_proc_with_mixed_params() {
    let decs = expect_parse_ok("foo :: (comptime T: type, x: int) {}", 1);
    assert_decl_is_procedure(&decs[0], 0, 2);

    let DeclKind::Procedure { sig, .. } = &decs[0].node else {
        panic!("expected procedure");
    };

    assert!(sig.node.params.params[0].is_comptime);
    assert!(!sig.node.params.params[1].is_comptime);
}

#[test]
fn parse_type_keyword() {
    let decs = expect_parse_ok("foo :: () { }", 1);
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        panic!("expected procedure");
    };

    assert!(block.node.stmts.is_empty());
}

#[test]
fn parse_extern_with_comptime_param() {
    let decs = expect_parse_ok("printf :: extern (comptime T: type, x: T)", 1);

    let DeclKind::Extern {
        sig,
        generic_params,
        ..
    } = &decs[0].node
    else {
        panic!("expected extern");
    };

    assert!(generic_params.is_none());
    assert_eq!(sig.node.params.params.len(), 2);
    assert!(sig.node.params.params[0].is_comptime);
}

#[test]
fn parse_generic_procedure() {
    let decs = expect_parse_ok("foo :: (comptime T: type) {}", 1);
    assert_decl_is_procedure(&decs[0], 0, 1);

    let DeclKind::Procedure {
        sig, monomorph_of, ..
    } = &decs[0].node
    else {
        panic!("expected procedure");
    };

    assert!(monomorph_of.is_none());
    assert!(sig.node.params.params[0].is_comptime);
}

#[test]
fn parse_variadic_param() {
    let decs = expect_parse_ok("printf :: extern (fmt: ^char, ...) : int", 1);

    let DeclKind::Extern { sig, .. } = &decs[0].node else {
        panic!("expected extern");
    };

    assert_eq!(sig.node.params.params.len(), 2);
    let last_param = &sig.node.params.params[1];
    assert_eq!(last_param.ty.node, TypeKind::Variadic);
}

#[test]
fn parse_variadic_in_procedure() {
    let decs = expect_parse_ok("foo :: (x: int, ...) {}", 1);
    assert_decl_is_procedure(&decs[0], 0, 2);

    let DeclKind::Procedure { sig, .. } = &decs[0].node else {
        panic!("expected procedure");
    };

    assert_eq!(sig.node.params.params.len(), 2);
    let last_param = &sig.node.params.params[1];
    assert_eq!(last_param.ty.node, TypeKind::Variadic);
}

#[test]
fn parse_for_loop_with_array() {
    let decs = expect_parse_ok(
        "foo :: () {
            for x in [1, 2, 3] {
                println(x)
            }
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        panic!("expected procedure");
    };
    assert_eq!(block.node.stmts.len(), 1);

    let StmtKind::For {
        binding,
        iter,
        body,
    } = &block.node.stmts[0].node
    else {
        panic!("expected for loop");
    };
    assert!(matches!(binding.node, PatKind::Symbol(_)));
    assert!(matches!(
        iter.node,
        ExprKind::Allocation {
            kind: AllocKind::Array(_, _),
            ..
        }
    ));
    assert_eq!(body.node.stmts.len(), 1);
}

#[test]
fn parse_for_loop_with_range() {
    let decs = expect_parse_ok(
        "foo :: () {
            for i in 0..10 {
                println(i)
            }
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        panic!("expected procedure");
    };
    assert_eq!(block.node.stmts.len(), 1);

    let StmtKind::For {
        binding,
        iter,
        body,
    } = &block.node.stmts[0].node
    else {
        panic!("expected for loop");
    };
    assert!(matches!(binding.node, PatKind::Symbol(_)));
    assert!(matches!(iter.node, ExprKind::Range { .. }));
    assert_eq!(body.node.stmts.len(), 1);
}

#[test]
fn parse_for_loop_with_wildcard() {
    let decs = expect_parse_ok(
        "foo :: () {
            for _ in [1, 2, 3] {
                do_something()
            }
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        panic!("expected procedure");
    };
    assert_eq!(block.node.stmts.len(), 1);

    let StmtKind::For { binding, .. } = &block.node.stmts[0].node else {
        panic!("expected for loop");
    };
    assert!(matches!(binding.node, PatKind::Wildcard));
}

#[test]
fn parse_for_loop_nested() {
    let decs = expect_parse_ok(
        "foo :: () {
            for i in 0..3 {
                for j in 0..2 {
                    println(i + j)
                }
            }
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        panic!("expected procedure");
    };
    assert_eq!(block.node.stmts.len(), 1);

    let StmtKind::For {
        body: outer_body, ..
    } = &block.node.stmts[0].node
    else {
        panic!("expected for loop");
    };
    assert_eq!(outer_body.node.stmts.len(), 1);

    let StmtKind::For { .. } = &outer_body.node.stmts[0].node else {
        panic!("expected nested for loop");
    };
}

#[test]
fn parse_for_loop_empty_body() {
    let decs = expect_parse_ok(
        "foo :: () {
            for x in [1, 2, 3] { }
        }",
        1,
    );
    assert_decl_is_procedure(&decs[0], 0, 0);

    let DeclKind::Procedure { block, .. } = &decs[0].node else {
        panic!("expected procedure");
    };
    assert_eq!(block.node.stmts.len(), 1);

    let StmtKind::For { body, .. } = &block.node.stmts[0].node else {
        panic!("expected for loop");
    };
    assert!(body.node.stmts.is_empty());
}

#[test]
fn parse_for_loop_missing_in_keyword() {
    expect_err_kind(
        "foo :: () {
            for x [1, 2, 3] { }
        }",
        ParseErrorKind::ExpectedToken(Token::Keyword(Keyword::In)),
    );
}

#[test]
fn parse_for_loop_missing_body() {
    expect_err_kind(
        "foo :: () {
            for x in [1, 2, 3]
        }",
        ParseErrorKind::ExpectedToken(Token::LBrace),
    );
}
