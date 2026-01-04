use std::collections::VecDeque;

use crate::ast::{
    AllocKind, Block, BlockInner, Call, Decl, DeclKind, Expr, ExprKind, IfElse, Match, MatchArm,
    Module, Params, Pat, PatKind, RecordField, Signature, SignatureInner, Stmt, StmtKind, Type,
    TypeAliasDefinition, TypeKind, ValueKind, VariantField,
};
use crate::ctx::{Ctx, Symbol};
use crate::lex::{BinOp, Keyword, Token};
use crate::span::{Diagnostic, Span, SpanExt, Spanned};

#[derive(Debug, PartialEq, Eq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseErrorKind {
    ExpectedIdentifier,
    MalformedAccess,
    ExpectedType,
    ExpectedPattern,
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

impl Diagnostic for ParseError {
    fn span(&self) -> &Span {
        &self.span
    }

    fn message(&self) -> String {
        "parse error".to_string()
    }

    fn label(&self) -> Option<String> {
        Some(format!("{:?}", self.kind))
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

fn parse_expr(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let expr = parse_logical_or(ctx, tokens)?;
    Ok(expr)
}

fn parse_logical_or(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let mut left = parse_logical_and(ctx, tokens)?;

    while let Some(Token::BinOp(BinOp::Or)) = peek_token(tokens) {
        let _op_token = tokens.pop_front().unwrap();
        let right = parse_logical_and(ctx, tokens)?;
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

fn parse_logical_and(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let mut left = parse_comparison(ctx, tokens)?;

    while let Some(Token::BinOp(BinOp::And)) = peek_token(tokens) {
        let _op_token = tokens.pop_front().unwrap();
        let right = parse_comparison(ctx, tokens)?;
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

fn parse_comparison(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let mut left = parse_additive(ctx, tokens)?;

    while let Some(op) = peek_token(tokens) {
        match op {
            Token::BinOp(
                op @ (BinOp::EqEq | BinOp::NEq | BinOp::Gt | BinOp::GtEq | BinOp::Lt | BinOp::LtEq),
            ) => {
                let op = *op;
                tokens.pop_front();
                let right = parse_additive(ctx, tokens)?;
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

fn parse_additive(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let mut left = parse_multiplicative(ctx, tokens)?;

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
                let right = parse_multiplicative(ctx, tokens)?;
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

fn parse_postfix(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let mut expr = parse_primary(ctx, tokens)?;

    loop {
        match peek_token(tokens) {
            Some(Token::Dot) => {
                tokens.pop_front();
                if let Some(Token::Identifier(_)) = peek_token(tokens) {
                    let field = expect_identifier(tokens)?;

                    let span = expr.span.merge(&field.span);
                    expr = Spanned::new(ExprKind::FieldAccess(Box::new(expr), field.node), span);
                } else if let Some(Token::Int(_)) = peek_token(tokens) {
                    let Some(Spanned { node, span }) = tokens.pop_front() else {
                        unreachable!()
                    };

                    let Token::Int(i) = node else { unreachable!() };

                    let index: usize = match (i).try_into() {
                        Ok(i) => i,
                        Err(_) => {
                            return Err(ParseError::new(ParseErrorKind::MalformedAccess, span));
                        }
                    };

                    let span = expr.span.merge(&span);
                    expr = Spanned::new(ExprKind::TupleAccess(Box::new(expr), index), span);
                } else {
                    return Err(ParseError::new(
                        ParseErrorKind::MalformedAccess,
                        peek_span(tokens).unwrap_or(expr.span),
                    ));
                }
            }
            Some(Token::LBracket) => {
                tokens.pop_front();
                let index = parse_expr(ctx, tokens)?;
                expect_next(tokens, Token::RBracket)?;
                let span = expr.span.merge(&index.span);
                expr = Spanned::new(ExprKind::Index(Box::new(expr), Box::new(index)), span);
            }
            Some(Token::LParen) => {
                // Handle calls on any expression (method calls, etc.)
                tokens.pop_front();
                let mut args = Vec::new();

                loop {
                    if let Some(Token::RParen) = peek_token(tokens) {
                        break;
                    }
                    args.push(parse_expr(ctx, tokens)?);
                    if let Some(Token::Comma) = peek_token(tokens) {
                        tokens.pop_front();
                    } else {
                        break;
                    }
                }

                let end_span = expect_next(tokens, Token::RParen)?;
                let span = expr.span.merge(&end_span);

                // You'll need to decide how to represent this in your AST
                expr = Spanned::new(
                    ExprKind::Call(Call {
                        callee: Box::new(expr), // Requires changing Call struct
                        args,
                    }),
                    span,
                );
            }
            _ => break,
        }
    }

    Ok(expr)
}

fn parse_multiplicative(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let mut left = parse_postfix(ctx, tokens)?;

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
                let right = parse_postfix(ctx, tokens)?;
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
    ctx: &mut Ctx,
    ident: Spanned<Symbol>,
    tokens: &mut VecDeque<SpannedToken>,
) -> ParseResult<Expr> {
    let ident_name = ctx.resolve(ident.node);
    let first_char = ident_name.chars().next().unwrap();
    let is_variant = first_char.is_ascii_uppercase();
    if let Some(Token::LParen) = peek_token(tokens) {
        let lparen_span = tokens.pop_front().unwrap().span;
        let mut args = Vec::new();

        loop {
            if let Some(Token::RParen) = peek_token(tokens) {
                break;
            }

            args.push(parse_expr(ctx, tokens)?);

            if let Some(Token::Comma) = peek_token(tokens) {
                tokens.pop_front();
            } else {
                break;
            }
        }

        let rparen_span = expect_next(tokens, Token::RParen)?;
        let span = ident.span.merge(&lparen_span).merge(&rparen_span);

        if is_variant {
            // variant
            return Ok(Spanned::new(
                ExprKind::Allocation {
                    kind: AllocKind::Variant(ident.node),
                    elements: args,
                    region: None,
                },
                span,
            ));
        } else {
            let ident_expr =
                Spanned::new(ExprKind::Value(ValueKind::Ident(ident.node)), span.clone());

            return Ok(Spanned::new(
                ExprKind::Call(Call {
                    callee: Box::new(ident_expr),
                    args,
                }),
                span,
            ));
        }
    }

    let span = ident.span.clone();
    if is_variant {
        Ok(Spanned::new(
            ExprKind::Allocation {
                kind: AllocKind::Variant(ident.node),
                elements: vec![],
                region: None,
            },
            span,
        ))
    } else {
        Ok(Spanned::new(
            ExprKind::Value(ValueKind::Ident(ident.node)),
            span,
        ))
    }
}

fn parse_primary(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Expr> {
    let Some(token) = tokens.pop_front() else {
        return Err(ParseError::eof());
    };

    match token.node {
        Token::Int(i) => Ok(Spanned::new(ExprKind::Value(ValueKind::Int(i)), token.span)),
        Token::Keyword(Keyword::True) => Ok(Spanned::new(
            ExprKind::Value(ValueKind::Bool(true)),
            token.span,
        )),
        Token::Keyword(Keyword::False) => Ok(Spanned::new(
            ExprKind::Value(ValueKind::Bool(false)),
            token.span,
        )),

        Token::Identifier(name) => {
            parse_identifier_expr(ctx, Spanned::new(name, token.span), tokens)
        }

        Token::LBracket => {
            let start_span = token.span;
            let next_tok = tokens.pop_front().ok_or_else(ParseError::eof)?;

            let alloc_kind = match next_tok.node {
                Token::RBracket => AllocKind::DynArray(TypeKind::Int.into()),
                Token::Int(i) => {
                    expect_next(tokens, Token::RBracket)?;
                    AllocKind::Array(TypeKind::Int.into(), i as usize)
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

                exprs.push(parse_expr(ctx, tokens)?);

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
            let expr = parse_expr(ctx, tokens)?;

            if let Some(Token::Comma) = peek_token(tokens) {
                tokens.pop_front();
                let mut exprs = vec![expr];

                loop {
                    if let Some(Token::RParen) = peek_token(tokens) {
                        break;
                    }

                    exprs.push(parse_expr(ctx, tokens)?);

                    if let Some(Token::Comma) = peek_token(tokens) {
                        tokens.pop_front();
                    }
                }

                let end_span = expect_next(tokens, Token::RParen)?;
                let span = start_span.merge(&end_span);

                Ok(Spanned::new(
                    ExprKind::Allocation {
                        // filled in during typechecking
                        kind: AllocKind::Tuple(vec![]),
                        elements: exprs,
                        region: None,
                    },
                    span,
                ))
            } else {
                let _ = expect_next(tokens, Token::RParen)?;
                Ok(expr)
            }
        }

        Token::LBrace => {
            // parsing a record
            let start_span = token.span;
            let mut fields = Vec::new();
            let mut exprs = Vec::new();

            loop {
                if let Some(Token::RBrace) = peek_token(tokens) {
                    break;
                }

                let field_name = expect_identifier(tokens)?;
                let ty = parse_type_annot(ctx, tokens)?;
                expect_next(tokens, Token::Eq)?;
                let expr = parse_expr(ctx, tokens)?;

                fields.push(RecordField {
                    name: field_name.node,
                    ty,
                });
                exprs.push(expr);

                if let Some(Token::Comma) = peek_token(tokens) {
                    tokens.pop_front();
                }
            }

            let end_span = expect_next(tokens, Token::RBrace)?;
            let span = start_span.merge(&end_span);
            Ok(Spanned::new(
                ExprKind::Allocation {
                    kind: AllocKind::Record(fields),
                    elements: exprs,
                    region: None,
                },
                span,
            ))
        }

        _ => Err(ParseError::new(
            ParseErrorKind::ExpectedExpression,
            token.span,
        )),
    }
}

fn parse_params(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<(Params, Span)> {
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
        let ty = parse_type_annot(ctx, tokens)?
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

fn parse_proc_sig(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Signature> {
    let (params, params_span) = parse_params(ctx, tokens)?;

    let (return_ty, end_span) = if let Some(Token::Colon) = peek_token(tokens) {
        let ty = parse_type_annot(ctx, tokens)?;
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

fn parse_type(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Option<Type>> {
    let ty = match peek_token(tokens) {
        Some(Token::Keyword(Keyword::Int)) => {
            let span = tokens.pop_front().unwrap().span;

            Spanned::new(TypeKind::Int, span)
        }
        Some(Token::Keyword(Keyword::Bool)) => {
            let span = tokens.pop_front().unwrap().span;

            Spanned::new(TypeKind::Bool, span)
        }
        Some(Token::Keyword(Keyword::Char)) => {
            let span = tokens.pop_front().unwrap().span;

            Spanned::new(TypeKind::Char, span)
        }
        Some(Token::Identifier(name)) => {
            let name_str = ctx.resolve(*name);

            let front_letter = name_str.chars().next().unwrap();
            let is_uppercase = front_letter.is_ascii_uppercase();

            if is_uppercase {
                // variant
                // variants may have bars, such as Some(T) | None
                let mut fields = Vec::new();
                loop {
                    let variant_name = expect_identifier(tokens)?;
                    if let Some(Token::LParen) = peek_token(tokens) {
                        let _ = tokens.pop_front().unwrap();

                        let mut adts = Vec::new();
                        loop {
                            if let Some(Token::RParen) = peek_token(tokens) {
                                break;
                            }

                            adts.push(parse_type(ctx, tokens)?.unwrap());

                            if let Some(Token::Comma) = peek_token(tokens) {
                                tokens.pop_front();
                            }
                        }

                        expect_next(tokens, Token::RParen)?;
                        fields.push(VariantField {
                            name: variant_name,
                            adts,
                        });
                    } else {
                        fields.push(VariantField {
                            name: variant_name,
                            adts: Vec::new(),
                        });
                    }

                    if let Some(Token::Bar) = peek_token(tokens) {
                        tokens.pop_front();
                    } else {
                        break;
                    }
                }

                // first has to exist, and last can be first
                let first_span = fields.first().unwrap().name.span.clone();
                let span = fields.last().unwrap().name.span.merge(&first_span);

                Spanned::new(TypeKind::Variant(fields), span)
            } else {
                let Spanned {
                    node: Token::Identifier(name),
                    span,
                } = tokens.pop_front().unwrap()
                else {
                    unreachable!()
                };
                Spanned::new(TypeKind::TypeAlias(name), span)
            }
        }
        Some(Token::Caret) => {
            let caret_span = tokens.pop_front().unwrap().span;
            let ty = match parse_type(ctx, tokens)? {
                Some(ty) => ty,
                None => return Err(ParseError::new(ParseErrorKind::ExpectedType, caret_span)),
            };

            let span = caret_span.merge(&ty.span);
            Spanned::new(TypeKind::Ptr(Box::new(ty.node)), span)
        }
        Some(Token::LBrace) => {
            let fields = extract_typedef_fields(tokens)?;
            let record_fields = parse_record_fields(ctx, fields);

            let start_span = record_fields
                .first()
                .map(|f| f.ty.as_ref().unwrap().span.clone())
                .unwrap();

            let end_span = record_fields
                .last()
                .map(|f| f.ty.as_ref().unwrap().span.clone())
                .unwrap();

            let span = start_span.merge(&end_span);

            Spanned::new(TypeKind::Record(record_fields), span)
        }
        _ => {
            return Ok(None);
        }
    };

    Ok(Some(ty))
}

fn parse_type_annot(
    ctx: &mut Ctx,
    tokens: &mut VecDeque<SpannedToken>,
) -> ParseResult<Option<Type>> {
    let Some(Spanned {
        node: Token::Colon,
        span: _,
    }) = tokens.pop_front()
    else {
        return Err(ParseError::new(ParseErrorKind::ExpectedType, 0..0));
    };

    parse_type(ctx, tokens)
}

fn parse_ifelse(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Stmt> {
    let if_span = expect_next(tokens, Token::Keyword(Keyword::If))?;
    let cond = parse_expr(ctx, tokens)?;
    let then = parse_block(ctx, tokens)?;

    let (else_, end_span) = if let Some(Token::Keyword(Keyword::Else)) = peek_token(tokens) {
        expect_next(tokens, Token::Keyword(Keyword::Else))?;
        let else_block = parse_block(ctx, tokens)?;
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

fn parse_pat(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Pat> {
    let Some(token) = tokens.front() else {
        return Err(ParseError::eof());
    };

    match &token.node {
        Token::Identifier(name) => {
            let name_str = ctx.resolve(*name);

            if name_str == "_" {
                let span = tokens.pop_front().unwrap().span;
                return Ok(Spanned::new(PatKind::Wildcard, span));
            }

            let first_char = name_str.chars().next().unwrap();
            let is_variant = first_char.is_ascii_uppercase();
            let ident = expect_identifier(tokens)?;

            if is_variant {
                if let Some(Token::LParen) = peek_token(tokens) {
                    tokens.pop_front();
                    let mut bindings = Vec::new();

                    loop {
                        if let Some(Token::RParen) = peek_token(tokens) {
                            break;
                        }

                        bindings.push(parse_pat(ctx, tokens)?);

                        if let Some(Token::Comma) = peek_token(tokens) {
                            tokens.pop_front();
                        } else {
                            break;
                        }
                    }

                    let end_span = expect_next(tokens, Token::RParen)?;
                    let span = ident.span.merge(&end_span);

                    Ok(Spanned::new(
                        PatKind::Variant {
                            name: ident.node,
                            bindings,
                        },
                        span,
                    ))
                } else {
                    Ok(Spanned::new(
                        PatKind::Variant {
                            name: ident.node,
                            bindings: vec![],
                        },
                        ident.span,
                    ))
                }
            } else {
                Ok(Spanned::new(PatKind::Symbol(ident.node), ident.span))
            }
        }
        // TOOD: clean this up to a parse literal function
        Token::Int(i) => {
            let i = *i;
            let span = tokens.pop_front().unwrap().span;
            Ok(Spanned::new(PatKind::Literal(ValueKind::Int(i)), span))
        }
        Token::Keyword(Keyword::True) => {
            let span = tokens.pop_front().unwrap().span;
            Ok(Spanned::new(PatKind::Literal(ValueKind::Bool(true)), span))
        }
        Token::Keyword(Keyword::False) => {
            let span = tokens.pop_front().unwrap().span;
            Ok(Spanned::new(PatKind::Literal(ValueKind::Bool(false)), span))
        }
        Token::LParen => {
            let start_span = tokens.pop_front().unwrap().span;
            let mut patterns = Vec::new();

            loop {
                if let Some(Token::RParen) = peek_token(tokens) {
                    break;
                }

                patterns.push(parse_pat(ctx, tokens)?);

                if let Some(Token::Comma) = peek_token(tokens) {
                    tokens.pop_front();
                } else {
                    break;
                }
            }

            let end_span = expect_next(tokens, Token::RParen)?;
            let span = start_span.merge(&end_span);

            Ok(Spanned::new(PatKind::Tuple(patterns), span))
        }
        _ => Err(ParseError::new(
            ParseErrorKind::ExpectedPattern,
            token.span.clone(),
        )),
    }
}

fn parse_match_arm(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<MatchArm> {
    let pat = parse_pat(ctx, tokens)?;
    expect_next(tokens, Token::FatArrow)?;
    let body = parse_block(ctx, tokens)?;

    Ok(MatchArm { pat, body })
}

fn parse_match(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Stmt> {
    let match_span = expect_next(tokens, Token::Keyword(Keyword::Match))?;
    let scrutinee = parse_expr(ctx, tokens)?;
    expect_next(tokens, Token::Keyword(Keyword::With))?;

    let mut arms = Vec::new();

    loop {
        let arm = parse_match_arm(ctx, tokens)?;
        arms.push(arm);

        if let Some(Token::Bar) = peek_token(tokens) {
            tokens.pop_front();
        } else {
            break;
        }
    }

    let span = match_span.merge(&arms.last().unwrap().body.span);
    Ok(Spanned::new(
        StmtKind::Match(Match { scrutinee, arms }),
        span,
    ))
}

fn parse_stmt(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Stmt> {
    match peek_token(tokens) {
        Some(Token::Keyword(Keyword::If)) => parse_ifelse(ctx, tokens),
        Some(Token::Keyword(Keyword::Match)) => parse_match(ctx, tokens),
        // Some(Token::Keyword(Keyword::Return))
        _ => {
            let name = expect_identifier(tokens)?;
            let start_span = name.span.clone();

            match peek_token(tokens) {
                Some(Token::Colon) => {
                    let ty = parse_type_annot(ctx, tokens)?;
                    expect_next(tokens, Token::Eq)?;
                    let expr = parse_expr(ctx, tokens)?;
                    let span = start_span.merge(&expr.span);
                    Ok(Spanned::new(StmtKind::ValDec { name, ty, expr }, span))
                }
                Some(Token::Eq) => {
                    expect_next(tokens, Token::Eq)?;
                    let expr = parse_expr(ctx, tokens)?;
                    let span = start_span.merge(&expr.span);
                    Ok(Spanned::new(StmtKind::Assign { name, expr }, span))
                }
                Some(Token::LParen) => {
                    let expr = parse_identifier_expr(ctx, name.clone(), tokens)?;
                    let ExprKind::Call(call) = expr.node else {
                        unreachable!()
                    };
                    let span = expr.span;
                    Ok(Spanned::new(StmtKind::Call(call), span))
                }
                _ => Err(ParseError::new(
                    ParseErrorKind::ExpectedExpression,
                    peek_span(tokens).unwrap_or(start_span),
                )),
            }
        }
    }
}

fn parse_block(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Block> {
    let start_span = expect_next(tokens, Token::LBrace)?;
    let mut stmts = Vec::new();

    while !tokens.is_empty() {
        if let Some(Token::RBrace) = peek_token(tokens) {
            break;
        }

        stmts.push(parse_stmt(ctx, tokens)?);
    }

    let end_span = expect_next(tokens, Token::RBrace)?;
    let span = start_span.merge(&end_span);

    Ok(Spanned::new(BlockInner { stmts, expr: None }, span))
}

fn parse_procedure(
    ctx: &mut Ctx,
    name: Spanned<Symbol>,
    fn_ty: Option<Type>,
    tokens: &mut VecDeque<SpannedToken>,
) -> ParseResult<Decl> {
    let sig = parse_proc_sig(ctx, tokens)?;
    let block = parse_block(ctx, tokens)?;
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
fn is_record(fields: &[VecDeque<SpannedToken>]) -> bool {
    fields
        .iter()
        .all(|f| f.iter().any(|t| t.node == Token::Colon))
}

fn parse_record_fields(ctx: &mut Ctx, fields: Vec<VecDeque<SpannedToken>>) -> Vec<RecordField> {
    fields
        .into_iter()
        .map(|mut field_tokens| {
            let name = expect_identifier(&mut field_tokens).unwrap();
            let ty = parse_type_annot(ctx, &mut field_tokens)
                .expect("to parse")
                .expect("expected type annot on struct field");

            RecordField {
                name: name.node,
                ty: Some(ty),
            }
        })
        .collect()
}

fn parse_enum_fields(fields: Vec<VecDeque<SpannedToken>>) -> Vec<VariantField> {
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

                        adts.push(Spanned::new(TypeKind::TypeAlias(type_name), span));
                    } else {
                        panic!("expected type name or int");
                    }

                    if let Some(Token::Comma) = peek_token(&field_tokens) {
                        field_tokens.pop_front();
                    }
                }

                field_tokens.pop_front();
            }

            VariantField { name, adts }
        })
        .collect()
}

fn parse_typedef(
    ctx: &mut Ctx,
    name: Spanned<Symbol>,
    _ty: Option<Type>,
    tokens: &mut VecDeque<SpannedToken>,
) -> ParseResult<Decl> {
    let start_span = name.span.clone();
    let fields = extract_typedef_fields(tokens)?;

    if is_record(&fields) {
        let record_fields = parse_record_fields(ctx, fields);

        let end_span = record_fields
            .last()
            .map(|f| f.ty.as_ref().unwrap().span.clone())
            .unwrap_or(start_span.clone());
        let span = start_span.merge(&end_span);

        Ok(Spanned::new(
            DeclKind::TypeDef {
                name,
                def: TypeAliasDefinition::Record(record_fields),
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
                def: TypeAliasDefinition::Variant(enum_fields),
            },
            span,
        ))
    }
}

fn parse_decls(ctx: &mut Ctx, tokens: &mut VecDeque<SpannedToken>) -> ParseResult<Vec<Decl>> {
    let mut decs = Vec::new();

    while !tokens.is_empty() {
        let name = expect_identifier(tokens)?;
        let start_span = name.span.clone();

        let ty = parse_type_annot(ctx, tokens)?;

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
                decs.push(parse_procedure(ctx, name, ty, tokens)?);
            }
            Some(Token::LBrace) => {
                decs.push(parse_typedef(ctx, name, ty, tokens)?);
            }
            Some(Token::Keyword(Keyword::Extern)) => {
                expect_next(tokens, Token::Keyword(Keyword::Extern))?;
                let sig = parse_proc_sig(ctx, tokens)?;
                let span = name.span.merge(&sig.span);
                decs.push(Spanned::new(DeclKind::Extern { name, sig }, span));
            }
            Some(_) => {
                let expr = parse_expr(ctx, tokens)?;
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
        assert_eq!(sig.node.params.patterns.len(), expected_param_count);
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
                x := None
                y := Some(1)
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

        eprintln!("{:#?}", expr1);
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
                Some(x) => { y := x }
              | None    => { y := 0 }
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
}
