use std::collections::VecDeque;

use crate::ast::{
    AllocKind, Block, BlockInner, Call, Decl, DeclKind, EnumField, Expr, ExprKind, IfElse, Module,
    Params, PatKind, Signature, SignatureInner, Stmt, StmtKind, StructField, Type, TypeKind,
    UserDefinedType, ValueKind,
};
use crate::ctx::{Ctx, Symbol};
use crate::lex::{BinOp, Keyword, Token};
use crate::span::{Span, SpanExt, Spanned};

#[derive(Debug, PartialEq, Eq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseErrorKind {
    ExpectedIdentifier,
    ExpectedType,
    ExpectedExpression,
    ExpectedToken(Token),
    UnexpectedEOF,
}

impl ParseError {
    fn new(kind: ParseErrorKind, span: Span) -> Self {
        Self { kind, span }
    }

    fn eof() -> Self {
        Self::new(ParseErrorKind::UnexpectedEOF, 0..0)
    }
}

type ParseResult<T> = Result<T, ParseError>;

type SpannedToken = Spanned<Token>;

fn expect_next(tokens: &mut VecDeque<SpannedToken>, expected: Token) -> ParseResult<Span> {
    match tokens.pop_front() {
        Some(Spanned { node, span }) if node == expected => Ok(span),
        Some(Spanned { span, .. }) => Err(ParseError::new(
            ParseErrorKind::ExpectedToken(expected),
            span,
        )),
        None => Err(ParseError::eof()),
    }
}

fn expect_identifier(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Spanned<Symbol>> {
    match tokens.pop_front() {
        Some(Spanned {
            node: Token::Identifier(name),
            span,
        }) => Ok(Spanned::new(name, span)),
        Some(Spanned { span, .. }) => {
            Err(ParseError::new(ParseErrorKind::ExpectedIdentifier, span))
        }
        None => Err(ParseError::eof()),
    }
}

fn peek_token(tokens: &VecDeque<SpannedToken>) -> Option<&Token> {
    tokens.front().map(|t| &t.node)
}

fn peek_span(tokens: &VecDeque<SpannedToken>) -> Option<Span> {
    tokens.front().map(|t| t.span.clone())
}

fn parse_expr(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    parse_logical_or(tokens)
}

fn parse_logical_or(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let mut left = parse_logical_and(tokens)?;

    while let Some(Token::BinOp(BinOp::Or)) = peek_token(tokens) {
        let op_token = tokens.pop_front().unwrap();
        let right = parse_logical_and(tokens)?;
        let span = left.span.merge(&right.span);
        left = Spanned::new(
            ExprKind::BinOp {
                op: BinOp::Or,
                lhs: Box::new(left),
                rhs: Box::new(right),
            },
            span,
        );
    }

    Ok(left)
}

fn parse_logical_and(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let mut left = parse_comparison(tokens)?;

    while let Some(Token::BinOp(BinOp::And)) = peek_token(tokens) {
        let op_token = tokens.pop_front().unwrap();
        let right = parse_comparison(tokens)?;
        let span = left.span.merge(&right.span);
        left = Spanned::new(
            ExprKind::BinOp {
                op: BinOp::And,
                lhs: Box::new(left),
                rhs: Box::new(right),
            },
            span,
        );
    }

    Ok(left)
}

fn parse_comparison(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let mut left = parse_additive(tokens)?;

    while let Some(op) = peek_token(tokens) {
        match op {
            Token::BinOp(
                op @ (BinOp::EqEq | BinOp::NEq | BinOp::Gt | BinOp::GtEq | BinOp::Lt | BinOp::LtEq),
            ) => {
                let op = *op;
                tokens.pop_front();
                let right = parse_additive(tokens)?;
                let span = left.span.merge(&right.span);
                left = Spanned::new(
                    ExprKind::BinOp {
                        op,
                        lhs: Box::new(left),
                        rhs: Box::new(right),
                    },
                    span,
                );
            }
            _ => break,
        }
    }

    Ok(left)
}

fn parse_additive(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let mut left = parse_multiplicative(tokens)?;

    while let Some(op) = peek_token(tokens) {
        match op {
            Token::BinOp(BinOp::Add) | Token::BinOp(BinOp::Sub) => {
                let Spanned {
                    node: Token::BinOp(op),
                    ..
                } = tokens.pop_front().unwrap()
                else {
                    unreachable!()
                };
                let right = parse_multiplicative(tokens)?;
                let span = left.span.merge(&right.span);
                left = Spanned::new(
                    ExprKind::BinOp {
                        op,
                        lhs: Box::new(left),
                        rhs: Box::new(right),
                    },
                    span,
                );
            }
            _ => break,
        }
    }

    Ok(left)
}

fn parse_multiplicative(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let mut left = parse_primary(tokens)?;

    while let Some(op) = peek_token(tokens) {
        match op {
            Token::BinOp(BinOp::Mul) | Token::BinOp(BinOp::Div) | Token::BinOp(BinOp::Mod) => {
                let Spanned {
                    node: Token::BinOp(op),
                    ..
                } = tokens.pop_front().unwrap()
                else {
                    unreachable!()
                };
                let right = parse_primary(tokens)?;
                let span = left.span.merge(&right.span);
                left = Spanned::new(
                    ExprKind::BinOp {
                        op,
                        lhs: Box::new(left),
                        rhs: Box::new(right),
                    },
                    span,
                );
            }
            _ => break,
        }
    }

    Ok(left)
}

fn parse_identifier_expr(
    ident: Spanned<Symbol>,
    tokens: &mut VecDeque<SpannedToken>,
) -> ParseResult<Expr> {
    if let Some(Token::LParen) = peek_token(tokens) {
        let lparen_span = tokens.pop_front().unwrap().span;
        let mut args = Vec::new();

        loop {
            if let Some(Token::RParen) = peek_token(tokens) {
                break;
            }

            args.push(parse_expr(tokens)?);

            if let Some(Token::Comma) = peek_token(tokens) {
                tokens.pop_front();
            } else {
                break;
            }
        }

        let rparen_span = expect_next(tokens, Token::RParen)?;
        let span = ident.span.merge(&rparen_span);

        return Ok(Spanned::new(
            ExprKind::Call(Call {
                callee: ident,
                args,
            }),
            span,
        ));
    }

    let span = ident.span.clone();
    Ok(Spanned::new(
        ExprKind::Value(ValueKind::Ident(ident.node)),
        span,
    ))
}

fn parse_primary(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let Some(token) = tokens.pop_front() else {
        return Err(ParseError::eof());
    };

    match token.node {
        Token::Int(i) => Ok(Spanned::new(ExprKind::Value(ValueKind::Int(i)), token.span)),

        Token::Identifier(name) => parse_identifier_expr(Spanned::new(name, token.span), tokens),

        Token::LBracket => {
            let start_span = token.span;
            let next_tok = tokens.pop_front().ok_or_else(ParseError::eof)?;

            let alloc_kind = match next_tok.node {
                Token::RBracket => AllocKind::DynArray(Type::synthetic(TypeKind::Unit).into()),
                Token::Int(i) => {
                    expect_next(tokens, Token::RBracket)?;
                    AllocKind::Array(Type::synthetic(TypeKind::Int).into(), i as usize)
                }
                _ => {
                    return Err(ParseError::new(
                        ParseErrorKind::ExpectedExpression,
                        next_tok.span,
                    ));
                }
            };

            expect_next(tokens, Token::LBrace)?;
            let mut exprs = Vec::new();

            loop {
                if let Some(Token::RBrace) = peek_token(tokens) {
                    break;
                }

                exprs.push(parse_expr(tokens)?);

                if let Some(Token::Comma) = peek_token(tokens) {
                    tokens.pop_front();
                }
            }

            let end_span = expect_next(tokens, Token::RBrace)?;
            let span = start_span.merge(&end_span);

            Ok(Spanned::new(
                ExprKind::Allocation {
                    kind: alloc_kind,
                    elements: exprs,
                    region: None,
                },
                span,
            ))
        }

        Token::LParen => {
            let start_span = token.span;
            let expr = parse_expr(tokens)?;

            if let Some(Token::Comma) = peek_token(tokens) {
                tokens.pop_front();
                let mut exprs = vec![expr];

                loop {
                    if let Some(Token::RParen) = peek_token(tokens) {
                        break;
                    }

                    exprs.push(parse_expr(tokens)?);

                    if let Some(Token::Comma) = peek_token(tokens) {
                        tokens.pop_front();
                    }
                }

                let end_span = expect_next(tokens, Token::RParen)?;
                let span = start_span.merge(&end_span);

                Ok(Spanned::new(
                    ExprKind::Allocation {
                        kind: AllocKind::Tuple,
                        elements: exprs,
                        region: None,
                    },
                    span,
                ))
            } else {
                let end_span = expect_next(tokens, Token::RParen)?;
                // For grouped expressions, keep the inner expression's span
                // but we could also merge with parens if desired
                Ok(expr)
            }
        }

        _ => Err(ParseError::new(
            ParseErrorKind::ExpectedExpression,
            token.span,
        )),
    }
}

fn parse_params(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<(Params, Span)> {
    let start_span = expect_next(tokens, Token::LParen)?;
    let mut params = Params {
        patterns: Vec::new(),
        types: Vec::new(),
    };

    while !tokens.is_empty() {
        if let Some(Token::RParen) = peek_token(tokens) {
            break;
        }

        let name = expect_identifier(tokens)?;
        let ty = parse_type_annot(tokens)?
            .ok_or_else(|| ParseError::new(ParseErrorKind::ExpectedType, name.span.clone()))?;

        params
            .patterns
            .push(Spanned::new(PatKind::Symbol(name.node), name.span));
        params.types.push(ty);

        if let Some(Token::Comma) = peek_token(tokens) {
            tokens.pop_front();
        } else {
            break;
        }
    }

    let end_span = expect_next(tokens, Token::RParen)?;
    Ok((params, start_span.merge(&end_span)))
}

fn parse_proc_sig(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Signature> {
    let (params, params_span) = parse_params(tokens)?;

    let (return_ty, end_span) = if let Some(Token::Colon) = peek_token(tokens) {
        let ty = parse_type_annot(tokens)?;
        let end = ty
            .as_ref()
            .map(|t| t.span.clone())
            .unwrap_or(params_span.clone());
        (ty, end)
    } else {
        (None, params_span.clone())
    };

    let span = params_span.merge(&end_span);
    Ok(Spanned::new(SignatureInner { params, return_ty }, span))
}

fn parse_type_annot(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Option<Type>> {
    let Some(Spanned {
        node: Token::Colon,
        span: colon_span,
    }) = tokens.pop_front()
    else {
        return Err(ParseError::new(ParseErrorKind::ExpectedType, 0..0));
    };

    if let Some(Token::Keyword(_)) = peek_token(tokens) {
        let Spanned {
            node: Token::Keyword(type_name),
            span,
        } = tokens.pop_front().unwrap()
        else {
            unreachable!()
        };

        match type_name {
            Keyword::Int => Ok(Some(Spanned::new(TypeKind::Int, span))),
            // will be used here once more keywords are in
            #[allow(unreachable_patterns)]
            _ => Err(ParseError::new(ParseErrorKind::ExpectedType, span)),
        }
    } else if let Some(Token::Identifier(_)) = peek_token(tokens) {
        let Spanned {
            node: Token::Identifier(name),
            span,
        } = tokens.pop_front().unwrap()
        else {
            unreachable!()
        };
        Ok(Some(Spanned::new(TypeKind::UserDef(name), span)))
    } else {
        Ok(None)
    }
}

fn parse_ifelse(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Stmt> {
    let if_span = expect_next(tokens, Token::Keyword(Keyword::If))?;
    let cond = parse_expr(tokens)?;
    let then = parse_block(tokens)?;

    let (else_, end_span) = if let Some(Token::Keyword(Keyword::Else)) = peek_token(tokens) {
        expect_next(tokens, Token::Keyword(Keyword::Else))?;
        let else_block = parse_block(tokens)?;
        let span = else_block.span.clone();
        (Some(else_block), span)
    } else {
        (None, then.span.clone())
    };

    let span = if_span.merge(&end_span);
    Ok(Spanned::new(
        StmtKind::IfElse(IfElse { cond, then, else_ }),
        span,
    ))
}

fn parse_stmt(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Stmt> {
    match peek_token(tokens) {
        Some(Token::Keyword(Keyword::If)) => parse_ifelse(tokens),
        _ => {
            let name = expect_identifier(tokens)?;
            let start_span = name.span.clone();

            let stmt_kind = match peek_token(tokens) {
                Some(Token::Colon) => {
                    let ty = parse_type_annot(tokens)?;
                    expect_next(tokens, Token::Eq)?;
                    let expr = parse_expr(tokens)?;
                    let span = start_span.merge(&expr.span);
                    return Ok(Spanned::new(StmtKind::ValDec { name, ty, expr }, span));
                }
                Some(Token::Eq) => {
                    expect_next(tokens, Token::Eq)?;
                    let expr = parse_expr(tokens)?;
                    let span = start_span.merge(&expr.span);
                    return Ok(Spanned::new(StmtKind::Assign { name, expr }, span));
                }
                Some(Token::LParen) => {
                    let expr = parse_identifier_expr(name.clone(), tokens)?;
                    let ExprKind::Call(call) = expr.node else {
                        unreachable!()
                    };
                    let span = expr.span;
                    return Ok(Spanned::new(StmtKind::Call(call), span));
                }
                _ => {
                    return Err(ParseError::new(
                        ParseErrorKind::ExpectedExpression,
                        peek_span(tokens).unwrap_or(start_span),
                    ));
                }
            };
        }
    }
}

fn parse_block(tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Block> {
    let start_span = expect_next(tokens, Token::LBrace)?;
    let mut stmts = Vec::new();

    while !tokens.is_empty() {
        if let Some(Token::RBrace) = peek_token(tokens) {
            break;
        }

        stmts.push(parse_stmt(tokens)?);
    }

    let end_span = expect_next(tokens, Token::RBrace)?;
    let span = start_span.merge(&end_span);

    Ok(Spanned::new(BlockInner { stmts, expr: None }, span))
}

fn parse_procedure(
    name: Spanned<Symbol>,
    fn_ty: Option<Type>,
    tokens: &mut VecDeque<SpannedToken>,
) -> ParseResult<Decl> {
    let sig = parse_proc_sig(tokens)?;
    let block = parse_block(tokens)?;
    let span = name.span.merge(&block.span);

    Ok(Spanned::new(
        DeclKind::Procedure {
            name,
            fn_ty,
            sig,
            block,
        },
        span,
    ))
}

fn extract_typedef_fields(
    tokens: &mut VecDeque<SpannedToken>,
) -> ParseResult<Vec<VecDeque<SpannedToken>>> {
    let mut fields = Vec::new();
    expect_next(tokens, Token::LBrace)?;

    while !tokens.is_empty() {
        let mut field = VecDeque::new();
        let mut in_parens = false;

        loop {
            if let Some(Token::Comma) = peek_token(tokens) {
                tokens.pop_front();
                if !in_parens {
                    break;
                } else {
                    // Keep the comma token with a dummy span for now
                    field.push_back(Spanned::new(Token::Comma, 0..0));
                }
            } else if let Some(Token::RBrace) = peek_token(tokens) {
                break;
            } else {
                let token = tokens.pop_front().ok_or_else(ParseError::eof)?;
                if token.node == Token::LParen {
                    in_parens = true;
                } else if token.node == Token::RParen {
                    in_parens = false;
                }
                field.push_back(token);
            }
        }

        fields.push(field);

        if let Some(Token::RBrace) = peek_token(tokens) {
            break;
        }
    }

    expect_next(tokens, Token::RBrace)?;
    Ok(fields)
}

#[inline]
fn is_struct(fields: &[VecDeque<SpannedToken>]) -> bool {
    fields
        .iter()
        .all(|f| f.iter().any(|t| t.node == Token::Colon))
}

fn parse_struct_fields(fields: Vec<VecDeque<SpannedToken>>) -> Vec<StructField> {
    fields
        .into_iter()
        .map(|mut field_tokens| {
            let name = expect_identifier(&mut field_tokens).unwrap();
            let ty = parse_type_annot(&mut field_tokens)
                .expect("to parse")
                .expect("expected type annot on struct field");

            StructField { name, ty }
        })
        .collect()
}

fn parse_enum_fields(fields: Vec<VecDeque<SpannedToken>>) -> Vec<EnumField> {
    fields
        .into_iter()
        .map(|mut field_tokens| {
            let name = expect_identifier(&mut field_tokens).unwrap();

            let mut adts = Vec::new();
            if let Some(Token::LParen) = peek_token(&field_tokens) {
                field_tokens.pop_front();
                loop {
                    if let Some(Token::RParen) = peek_token(&field_tokens) {
                        break;
                    }

                    if let Some(Token::Keyword(Keyword::Int)) = peek_token(&field_tokens) {
                        let span = field_tokens.pop_front().unwrap().span;
                        adts.push(Spanned::new(TypeKind::Int, span));
                    } else if let Some(Token::Identifier(_)) = peek_token(&field_tokens) {
                        let Spanned {
                            node: Token::Identifier(type_name),
                            span,
                        } = field_tokens.pop_front().unwrap()
                        else {
                            unreachable!()
                        };
                        adts.push(Spanned::new(TypeKind::UserDef(type_name), span));
                    } else {
                        panic!("expected type name or int");
                    }

                    if let Some(Token::Comma) = peek_token(&field_tokens) {
                        field_tokens.pop_front();
                    }
                }

                field_tokens.pop_front();
            }

            EnumField { name, adts }
        })
        .collect()
}

fn parse_typedef(
    name: Spanned<Symbol>,
    _ty: Option<Type>,
    tokens: &mut VecDeque<SpannedToken>,
) -> ParseResult<Decl> {
    let start_span = name.span.clone();
    let fields = extract_typedef_fields(tokens)?;

    if is_struct(&fields) {
        let struct_fields = parse_struct_fields(fields);
        let end_span = struct_fields
            .last()
            .map(|f| f.ty.span.clone())
            .unwrap_or(start_span.clone());
        let span = start_span.merge(&end_span);

        Ok(Spanned::new(
            DeclKind::TypeDef {
                name,
                def: UserDefinedType::Struct(struct_fields),
            },
            span,
        ))
    } else {
        let enum_fields = parse_enum_fields(fields);
        let end_span = enum_fields
            .last()
            .map(|f| f.name.span.clone())
            .unwrap_or(start_span.clone());
        let span = start_span.merge(&end_span);

        Ok(Spanned::new(
            DeclKind::TypeDef {
                name,
                def: UserDefinedType::Enum(enum_fields),
            },
            span,
        ))
    }
}

fn parse_decls(_ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Vec<Decl>> {
    let mut decs = Vec::new();

    while !tokens.is_empty() {
        let name = expect_identifier(tokens)?;
        let start_span = name.span.clone();

        let ty = parse_type_annot(tokens)?;

        let Some(Spanned {
            node: Token::Colon, ..
        }) = tokens.pop_front()
        else {
            return Err(ParseError::new(
                ParseErrorKind::ExpectedExpression,
                peek_span(tokens).unwrap_or(start_span),
            ));
        };

        match peek_token(tokens) {
            Some(Token::LParen) => {
                decs.push(parse_procedure(name, ty, tokens)?);
            }
            Some(Token::LBrace) => {
                decs.push(parse_typedef(name, ty, tokens)?);
            }
            Some(_) => {
                let expr = parse_expr(tokens)?;
                let span = start_span.merge(&expr.span);
                decs.push(Spanned::new(DeclKind::Constant { name, ty, expr }, span));
            }
            None => return Err(ParseError::eof()),
        }
    }

    Ok(decs)
}

pub fn parse(ctx: &mut Ctx, tokens: VecDeque<SpannedToken>) -> ParseResult<Module> {
    let mut tokens = tokens;
    let decls = parse_decls(ctx, &mut tokens)?;

    let module = Module {
        declarations: decls,
    };

    Ok(module)
}

#[cfg(test)]
mod tests {
    use std::clone;

    use super::*;
    use crate::{
        ast::{Type, UserDefinedType},
        ctx::Symbol,
        lex,
    };

    fn tokenify(s: &str) -> (Ctx, VecDeque<SpannedToken>) {
        let mut ctx = Ctx::new();
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

    fn assert_decl_is_constant(decl: &Decl, expected_name_id: i32, expected_has_ty: bool) {
        let DeclKind::Constant { name, ty, .. } = &decl.node else {
            panic!("expected Constant declaration");
        };
        assert_eq!(name.node, Symbol(expected_name_id));
        assert_eq!(ty.is_some(), expected_has_ty);
    }

    fn assert_decl_is_procedure(decl: &Decl, expected_name_id: i32, expected_param_count: usize) {
        let DeclKind::Procedure { name, sig, .. } = &decl.node else {
            panic!("expected Procedure declaration");
        };
        assert_eq!(name.node, Symbol(expected_name_id));
        assert_eq!(sig.node.params.patterns.len(), expected_param_count);
    }

    fn assert_decl_is_typedef(decl: &Decl, expected_name_id: i32) {
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
    fn constant_parse() {
        let decs = expect_parse_ok("foo :: 1", 1);
        assert_decl_is_constant(&decs[0], 0, false);

        // Verify the expression is an int literal
        let DeclKind::Constant { expr, .. } = &decs[0].node else {
            unreachable!()
        };
        assert!(matches!(expr.node, ExprKind::Value(ValueKind::Int(1))));
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
        let UserDefinedType::Struct(fields) = def else {
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
        let UserDefinedType::Enum(fields) = def else {
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
        let UserDefinedType::Enum(fields) = def else {
            panic!("expected enum")
        };
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].adts.len(), 1);
        assert_eq!(fields[1].adts.len(), 2);
        assert_eq!(fields[2].adts.len(), 1);
    }

    #[test]
    fn parse_array_and_dyn_array() {
        let decs = expect_parse_ok(
            "foo :: () { 
                x := [3]{1, 2, 3}
                y := []{1, 2, 3}
            }",
            1,
        );
        assert_decl_is_procedure(&decs[0], 0, 0);

        let DeclKind::Procedure { block, .. } = &decs[0].node else {
            unreachable!()
        };
        assert_eq!(block.node.stmts.len(), 2);

        // Verify first is Array, second is DynArray
        let StmtKind::ValDec { expr: expr1, .. } = &block.node.stmts[0].node else {
            unreachable!()
        };

        assert!(matches!(
            &expr1.node,
            ExprKind::Allocation {
                kind: AllocKind::Array(_, 3),
                ..
            }
        ));

        let StmtKind::ValDec { expr: expr2, .. } = &block.node.stmts[1].node else {
            unreachable!()
        };
        assert!(matches!(
            &expr2.node,
            ExprKind::Allocation {
                kind: AllocKind::DynArray(_),
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

        // Verify it's a tuple allocation
        let StmtKind::ValDec { expr, .. } = &block.node.stmts[0].node else {
            unreachable!()
        };
        let ExprKind::Allocation { kind, elements, .. } = &expr.node else {
            panic!("expected allocation")
        };
        assert!(matches!(kind, AllocKind::Tuple));
        assert_eq!(elements.len(), 3);
    }
}
