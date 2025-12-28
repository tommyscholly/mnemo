use std::collections::VecDeque;

use crate::ast::{
    AllocKind, Block, Call, Decl, EnumField, Expr, IfElse, Module, Params, Pat, Signature, Stmt,
    StructField, Type, UserDefinedType, Value,
};
use crate::ctx::{Ctx, Symbol};
use crate::lex::{BinOp, Keyword, Token};

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    Identifier,
    Type,
    Expression,
    Token,
}

type ParseResult<T> = Result<T, ParseError>;

fn expect_next(tokens: &mut VecDeque<Token>, token: Token) -> ParseResult<()> {
    eprintln!("expecting {token:?} in {tokens:?}");
    if let Some(next) = tokens.pop_front()
        && next == token
    {
        return Ok(());
    }

    Err(ParseError::Token)
}

fn expect_identifier(tokens: &mut VecDeque<Token>) -> ParseResult<Symbol> {
    let Some(Token::Identifier(name)) = tokens.pop_front() else {
        println!("{tokens:?}");
        return Err(ParseError::Identifier);
    };

    Ok(name)
}

// fn parse_expr(tokens: &mut VecDeque<Token>) -> ParseResult<Expr> {
//     match tokens.pop_front() {
//         Some(Token::Int(i)) => Ok(Expr::Value(Value::Int(i))),
//         t => {
//             println!("{t:?}");
//             Err(ParseError::Expression)
//         }
//     }
// }

fn parse_expr(tokens: &mut VecDeque<Token>) -> ParseResult<Expr> {
    parse_additive(tokens)
}

fn parse_additive(tokens: &mut VecDeque<Token>) -> ParseResult<Expr> {
    let mut left = parse_multiplicative(tokens)?;

    while let Some(op) = tokens.front() {
        match op {
            Token::BinOp(BinOp::Add) | Token::BinOp(BinOp::Sub) => {
                let op = tokens.pop_front().unwrap();
                let right = parse_multiplicative(tokens)?;
                left = Expr::BinOp {
                    op: match op {
                        Token::BinOp(BinOp::Add) => BinOp::Add,
                        Token::BinOp(BinOp::Sub) => BinOp::Sub,
                        _ => unreachable!(),
                    },
                    lhs: Box::new(left),
                    rhs: Box::new(right),
                };
            }
            _ => break,
        }
    }

    Ok(left)
}

fn parse_multiplicative(tokens: &mut VecDeque<Token>) -> ParseResult<Expr> {
    let mut left = parse_primary(tokens)?;

    while let Some(op) = tokens.front() {
        match op {
            Token::BinOp(BinOp::Mul) | Token::BinOp(BinOp::Div) => {
                let op = tokens.pop_front().unwrap();
                let right = parse_primary(tokens)?;
                left = Expr::BinOp {
                    op: match op {
                        Token::BinOp(BinOp::Mul) => BinOp::Mul,
                        Token::BinOp(BinOp::Div) => BinOp::Div,
                        _ => unreachable!(),
                    },
                    lhs: Box::new(left),
                    rhs: Box::new(right),
                };
            }
            _ => break,
        }
    }

    Ok(left)
}

fn parse_identifier(ident: Symbol, tokens: &mut VecDeque<Token>) -> ParseResult<Expr> {
    if let Some(Token::LParen) = tokens.front() {
        let mut args = Vec::new();
        tokens.pop_front();
        loop {
            if let Some(Token::RParen) = tokens.front() {
                break;
            }

            args.push(parse_expr(tokens)?);

            if let Some(Token::Comma) = tokens.front() {
                tokens.pop_front();
            } else {
                break;
            }
        }
        tokens.pop_front();
        return Ok(Expr::Call(Call {
            callee: ident,
            args,
        }));
    }

    Ok(Expr::Value(Value::Ident(ident)))
}

fn parse_primary(tokens: &mut VecDeque<Token>) -> ParseResult<Expr> {
    match tokens.pop_front() {
        Some(Token::Int(i)) => Ok(Expr::Value(Value::Int(i))),
        Some(Token::Identifier(name)) => parse_identifier(name, tokens),
        Some(Token::LBracket) => {
            let next_tok = tokens.pop_front().unwrap();
            let alloc_kind = match next_tok {
                Token::RBracket => AllocKind::DynArray,
                Token::Int(i) => {
                    expect_next(tokens, Token::RBracket)?;
                    AllocKind::Array(i as usize)
                }
                _ => return Err(ParseError::Expression),
            };

            expect_next(tokens, Token::LBrace)?;
            let mut exprs = Vec::new();
            loop {
                if let Some(Token::RBrace) = tokens.front() {
                    expect_next(tokens, Token::RBrace)?;
                    break;
                }

                exprs.push(parse_expr(tokens)?);

                if let Some(Token::Comma) = tokens.front() {
                    tokens.pop_front();
                }
            }

            Ok(Expr::Allocation {
                kind: alloc_kind,
                elements: exprs,
                region: None,
            })
        }
        Some(Token::LParen) => {
            let expr = parse_expr(tokens)?;
            if let Some(Token::Comma) = tokens.front() {
                tokens.pop_front();
                let mut exprs = vec![expr];
                loop {
                    if let Some(Token::RParen) = tokens.front() {
                        expect_next(tokens, Token::RParen)?;
                        break;
                    }

                    exprs.push(parse_expr(tokens)?);

                    if let Some(Token::Comma) = tokens.front() {
                        tokens.pop_front();
                    }
                }

                Ok(Expr::Allocation {
                    kind: AllocKind::Tuple,
                    elements: exprs,
                    region: None,
                })
            } else {
                expect_next(tokens, Token::RParen)?;
                Ok(expr)
            }
        }
        t => {
            println!("{t:?}");
            Err(ParseError::Expression)
        }
    }
}

fn parse_params(tokens: &mut VecDeque<Token>) -> ParseResult<Params> {
    let mut params = Params {
        patterns: Vec::new(),
        types: Vec::new(),
    };
    expect_next(tokens, Token::LParen)?;
    while !tokens.is_empty() {
        if let Some(Token::RParen) = tokens.front() {
            break;
        }

        let name = expect_identifier(tokens)?;

        let Some(ty) = parse_type_annot(tokens)? else {
            return Err(ParseError::Type);
        };

        params.patterns.push(Pat::Symbol(name));
        params.types.push(ty);

        if let Some(Token::Comma) = tokens.front() {
            tokens.pop_front();
        } else {
            break;
        }
    }
    expect_next(tokens, Token::RParen)?;

    Ok(params)
}

fn parse_proc_sig(tokens: &mut VecDeque<Token>) -> ParseResult<Signature> {
    let params = parse_params(tokens)?;
    let return_ty = if let Some(Token::Colon) = tokens.front() {
        parse_type_annot(tokens)?
    } else {
        None
    };

    Ok(Signature { params, return_ty })
}

fn parse_type_annot(tokens: &mut VecDeque<Token>) -> ParseResult<Option<Type>> {
    let Some(Token::Colon) = tokens.pop_front() else {
        return Err(ParseError::Type);
    };

    if let Some(Token::Keyword(_)) = tokens.front() {
        let Token::Keyword(type_name) = tokens.pop_front().unwrap() else {
            unreachable!()
        };

        match type_name {
            Keyword::Int => Ok(Some(Type::Int)),
            // will be used here once more keywords are in
            #[allow(unreachable_patterns)]
            _ => Err(ParseError::Type),
        }
    } else {
        Ok(None)
    }
}

fn parse_ifelse(tokens: &mut VecDeque<Token>) -> ParseResult<Stmt> {
    expect_next(tokens, Token::Keyword(Keyword::If))?;
    let cond = parse_expr(tokens)?;
    let then = parse_block(tokens)?;

    let else_ = if let Some(Token::Keyword(Keyword::Else)) = tokens.front() {
        expect_next(tokens, Token::Keyword(Keyword::Else))?;
        Some(parse_block(tokens)?)
    } else {
        None
    };

    Ok(Stmt::IfElse(IfElse { cond, then, else_ }))
}

fn parse_stmt(tokens: &mut VecDeque<Token>) -> ParseResult<Stmt> {
    // TODO: change this expect to something else
    match tokens.front() {
        Some(Token::Keyword(Keyword::If)) => parse_ifelse(tokens),
        _ => {
            let name = expect_identifier(tokens)?;
            let stmt = match tokens.front() {
                Some(Token::Colon) => {
                    let ty = parse_type_annot(tokens)?;
                    expect_next(tokens, Token::Eq)?;
                    let expr = parse_expr(tokens)?;
                    Stmt::ValDec { name, ty, expr }
                }
                Some(Token::Eq) => {
                    expect_next(tokens, Token::Eq)?;
                    let expr = parse_expr(tokens)?;
                    Stmt::Assign { name, expr }
                }
                Some(Token::LParen) => {
                    let Expr::Call(c) = parse_identifier(name, tokens)? else {
                        unreachable!()
                    };
                    Stmt::Call(c)
                }
                _ => unimplemented!(),
            };

            Ok(stmt)
        }
    }
}

fn parse_block(tokens: &mut VecDeque<Token>) -> ParseResult<Block> {
    let mut stmts = Vec::new();
    expect_next(tokens, Token::LBrace)?;
    while !tokens.is_empty() {
        if let Some(Token::RBrace) = tokens.front() {
            break;
        }

        stmts.push(parse_stmt(tokens)?);
    }
    expect_next(tokens, Token::RBrace)?;

    Ok(Block { stmts, expr: None })
}

fn parse_procedure(
    name: Symbol,
    fn_ty: Option<Type>,
    tokens: &mut VecDeque<Token>,
) -> ParseResult<Decl> {
    let sig = parse_proc_sig(tokens)?;
    let block = parse_block(tokens)?;

    Ok(Decl::Procedure {
        name,
        fn_ty,
        sig,
        block,
    })
}

fn extract_typedef_fields(tokens: &mut VecDeque<Token>) -> ParseResult<Vec<VecDeque<Token>>> {
    let mut fields = Vec::new();
    expect_next(tokens, Token::LBrace)?;
    while !tokens.is_empty() {
        let mut field = VecDeque::new();
        let mut in_parens = false;
        loop {
            if let Some(Token::Comma) = tokens.front() {
                tokens.pop_front();
                if !in_parens {
                    break;
                } else {
                    field.push_back(Token::Comma);
                }
            } else if let Some(Token::RBrace) = tokens.front() {
                break;
            } else {
                let token = tokens.pop_front().unwrap();
                if token == Token::LParen {
                    in_parens = true;
                } else if token == Token::RParen {
                    in_parens = false;
                }
                field.push_back(token);
            }
        }

        fields.push(field);

        if let Some(Token::RBrace) = tokens.front() {
            break;
        }
    }
    expect_next(tokens, Token::RBrace)?;

    Ok(fields)
}

#[inline]
fn is_struct(fields: &[VecDeque<Token>]) -> bool {
    fields.iter().all(|f| f.contains(&Token::Colon))
}

fn parse_struct_fields(fields: Vec<VecDeque<Token>>) -> Vec<StructField> {
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

fn parse_enum_fields(fields: Vec<VecDeque<Token>>) -> Vec<EnumField> {
    fields
        .into_iter()
        .map(|mut field_tokens| {
            let name = expect_identifier(&mut field_tokens).unwrap();

            let mut adts = Vec::new();
            if let Some(Token::LParen) = field_tokens.front() {
                field_tokens.pop_front();
                loop {
                    if let Some(Token::RParen) = field_tokens.front() {
                        break;
                    }

                    if let Token::Keyword(Keyword::Int) = field_tokens.front().unwrap() {
                        adts.push(Type::Int);
                        field_tokens.pop_front();
                    } else if let Some(Token::Identifier(type_name)) = field_tokens.front() {
                        adts.push(Type::UserDef(*type_name));
                        field_tokens.pop_front();
                    } else {
                        panic!("expected type name or int");
                    }

                    if let Some(Token::Comma) = field_tokens.front() {
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
    name: Symbol,
    // TOOD: verify that this type matches the parsed userdefinedtype
    _ty: Option<Type>,
    tokens: &mut VecDeque<Token>,
) -> ParseResult<Decl> {
    let fields = extract_typedef_fields(tokens)?;
    println!("{fields:?}");

    if is_struct(&fields) {
        let struct_fields = parse_struct_fields(fields);
        Ok(Decl::TypeDef {
            name,
            def: UserDefinedType::Struct(struct_fields),
        })
    } else {
        let enum_fields = parse_enum_fields(fields);
        Ok(Decl::TypeDef {
            name,
            def: UserDefinedType::Enum(enum_fields),
        })
    }
}

fn parse_decls(_ctx: &mut Ctx, tokens: &mut VecDeque<Token>) -> ParseResult<Vec<Decl>> {
    let mut decs = Vec::new();
    while !tokens.is_empty() {
        let name = expect_identifier(tokens)?;

        let ty = parse_type_annot(tokens)?;
        let Some(Token::Colon) = tokens.pop_front() else {
            return Err(ParseError::Expression);
        };

        match tokens.front() {
            Some(Token::LParen) => {
                decs.push(parse_procedure(name, ty, tokens)?);
            }
            Some(Token::LBrace) => {
                decs.push(parse_typedef(name, ty, tokens)?);
            }
            Some(_) => decs.push(Decl::Constant {
                name,
                ty,
                expr: parse_expr(tokens)?,
            }),
            None => return Err(ParseError::Expression),
        }
    }

    Ok(decs)
}

pub fn parse(ctx: &mut Ctx, mut tokens: VecDeque<Token>) -> ParseResult<Module> {
    let decls = parse_decls(ctx, &mut tokens)?;

    let module = Module {
        declarations: decls,
    };

    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{StructField, UserDefinedType},
        ctx::Symbol,
        lex,
    };

    fn tokenify(s: &str) -> (Ctx, VecDeque<Token>) {
        let mut ctx = Ctx::new();
        let tokens = lex::tokenize(&mut ctx, s).map(|t| t.unwrap().0).collect();
        (ctx, tokens)
    }

    fn parse_decls_from(input: &str) -> ParseResult<Vec<Decl>> {
        let (mut ctx, mut tokens) = tokenify(input);
        parse_decls(&mut ctx, &mut tokens)
    }

    fn expect_decl(input: &str, expected: Decl) {
        let decs = parse_decls_from(input).expect("expected parsing to succeed");
        assert!(
            !decs.is_empty(),
            "expected at least one declaration from input `{input}`"
        );
        assert_eq!(decs[0], expected);
    }

    fn expect_err(input: &str, error: ParseError) {
        assert_eq!(parse_decls_from(input), Err(error));
    }

    #[test]
    fn constant_parse() {
        expect_decl(
            "foo :: 1",
            Decl::Constant {
                name: Symbol(0),
                ty: None,
                expr: Expr::Value(Value::Int(1)),
            },
        );
    }

    #[test]
    fn constant_parse_type_annot() {
        expect_decl(
            "foo: int : 1",
            Decl::Constant {
                name: Symbol(0),
                ty: Some(Type::Int),
                expr: Expr::Value(Value::Int(1)),
            },
        );
    }

    #[test]
    fn constant_err_no_expr() {
        expect_err("foo ::", ParseError::Expression);
    }

    #[test]
    fn proc_parse() {
        expect_decl(
            "foo :: () {}",
            Decl::Procedure {
                name: Symbol(0),
                fn_ty: None,
                sig: Signature {
                    params: Params {
                        patterns: vec![],
                        types: vec![],
                    },
                    return_ty: None,
                },
                block: Block {
                    stmts: vec![],
                    expr: None,
                },
            },
        );
    }

    #[test]
    fn proc_parse_params_ret() {
        expect_decl(
            "foo :: (x: int, y: int): int {}",
            Decl::Procedure {
                name: Symbol(0),
                fn_ty: None,
                sig: Signature {
                    params: Params {
                        patterns: vec![Pat::Symbol(Symbol(1)), Pat::Symbol(Symbol(2))],
                        types: vec![Type::Int, Type::Int],
                    },
                    return_ty: Some(Type::Int),
                },
                block: Block {
                    stmts: vec![],
                    expr: None,
                },
            },
        );
    }

    #[test]
    fn proc_parse_with_min_body() {
        expect_decl(
            "foo :: (x: int, y: int): int { 
                z := 1
                x = 2
            }",
            Decl::Procedure {
                name: Symbol(0),
                fn_ty: None,
                sig: Signature {
                    params: Params {
                        patterns: vec![Pat::Symbol(Symbol(1)), Pat::Symbol(Symbol(2))],
                        types: vec![Type::Int, Type::Int],
                    },
                    return_ty: Some(Type::Int),
                },
                block: Block {
                    stmts: vec![
                        Stmt::ValDec {
                            name: Symbol(3),
                            ty: None,
                            expr: Expr::Value(Value::Int(1)),
                        },
                        Stmt::Assign {
                            name: Symbol(1),
                            expr: Expr::Value(Value::Int(2)),
                        },
                    ],
                    expr: None,
                },
            },
        );
    }

    #[test]
    fn proc_parse_with_min_body_and_type_annot() {
        expect_decl(
            "foo :: (x: int, y: int): int { 
                z : int = 1
                x = 2
            }",
            Decl::Procedure {
                name: Symbol(0),
                fn_ty: None,
                sig: Signature {
                    params: Params {
                        patterns: vec![Pat::Symbol(Symbol(1)), Pat::Symbol(Symbol(2))],
                        types: vec![Type::Int, Type::Int],
                    },
                    return_ty: Some(Type::Int),
                },
                block: Block {
                    stmts: vec![
                        Stmt::ValDec {
                            name: Symbol(3),
                            ty: Some(Type::Int),
                            expr: Expr::Value(Value::Int(1)),
                        },
                        Stmt::Assign {
                            name: Symbol(1),
                            expr: Expr::Value(Value::Int(2)),
                        },
                    ],
                    expr: None,
                },
            },
        );
    }

    #[test]
    fn proc_parse_with_bin_op() {
        expect_decl(
            "foo :: (x: int, y: int): int { 
                z : int = 1 + 2
                x = z + y
            }",
            Decl::Procedure {
                name: Symbol(0),
                fn_ty: None,
                sig: Signature {
                    params: Params {
                        patterns: vec![Pat::Symbol(Symbol(1)), Pat::Symbol(Symbol(2))],
                        types: vec![Type::Int, Type::Int],
                    },
                    return_ty: Some(Type::Int),
                },
                block: Block {
                    stmts: vec![
                        Stmt::ValDec {
                            name: Symbol(3),
                            ty: Some(Type::Int),
                            expr: Expr::BinOp {
                                op: BinOp::Add,
                                lhs: Box::new(Expr::Value(Value::Int(1))),
                                rhs: Box::new(Expr::Value(Value::Int(2))),
                            },
                        },
                        Stmt::Assign {
                            name: Symbol(1),
                            expr: Expr::BinOp {
                                op: BinOp::Add,
                                lhs: Box::new(Expr::Value(Value::Ident(Symbol(3)))),
                                rhs: Box::new(Expr::Value(Value::Ident(Symbol(2)))),
                            },
                        },
                    ],
                    expr: None,
                },
            },
        );
    }

    #[test]
    fn proc_parse_with_call() {
        expect_decl(
            "foo :: (x: int, y: int): int { 
                z : int = x + y
                println(z)
            }",
            Decl::Procedure {
                name: Symbol(0),
                fn_ty: None,
                sig: Signature {
                    params: Params {
                        patterns: vec![Pat::Symbol(Symbol(1)), Pat::Symbol(Symbol(2))],
                        types: vec![Type::Int, Type::Int],
                    },
                    return_ty: Some(Type::Int),
                },
                block: Block {
                    stmts: vec![
                        Stmt::ValDec {
                            name: Symbol(3),
                            ty: Some(Type::Int),
                            expr: Expr::BinOp {
                                op: BinOp::Add,
                                lhs: Box::new(Expr::Value(Value::Ident(Symbol(1)))),
                                rhs: Box::new(Expr::Value(Value::Ident(Symbol(2)))),
                            },
                        },
                        Stmt::Call(Call {
                            callee: Symbol(-1),
                            args: vec![Expr::Value(Value::Ident(Symbol(3)))],
                        }),
                    ],
                    expr: None,
                },
            },
        );
    }

    #[test]
    fn proc_parse_with_bad_binop() {
        expect_err(
            "foo :: (x: int, y: int): int { 
                z : int = x +
            }",
            ParseError::Expression,
        );
    }

    #[test]
    fn proc_parse_with_ifelse() {
        expect_decl(
            "foo :: (x: int, y: int): int { 
                if x {
                    println(x)
                } else {
                    println(y)
                }
            }",
            Decl::Procedure {
                name: Symbol(0),
                fn_ty: None,
                sig: Signature {
                    params: Params {
                        patterns: vec![Pat::Symbol(Symbol(1)), Pat::Symbol(Symbol(2))],
                        types: vec![Type::Int, Type::Int],
                    },
                    return_ty: Some(Type::Int),
                },
                block: Block {
                    stmts: vec![Stmt::IfElse(IfElse {
                        // cond: Expr::BinOp {
                        //     op: BinOp::Gt,
                        //     lhs: Box::new(Expr::Value(Value::Ident(Symbol(1)))),
                        //     rhs: Box::new(Expr::Value(Value::Ident(Symbol(2)))),
                        // },
                        cond: Expr::Value(Value::Ident(Symbol(1))),
                        then: Block {
                            stmts: vec![Stmt::Call(Call {
                                callee: Symbol(-1),
                                args: vec![Expr::Value(Value::Ident(Symbol(1)))],
                            })],
                            expr: None,
                        },
                        else_: Some(Block {
                            stmts: vec![Stmt::Call(Call {
                                callee: Symbol(-1),
                                args: vec![Expr::Value(Value::Ident(Symbol(2)))],
                            })],
                            expr: None,
                        }),
                    })],
                    expr: None,
                },
            },
        )
    }

    #[test]
    fn parse_type_struct() {
        expect_decl(
            "T :: { field1: int, field2: int }",
            Decl::TypeDef {
                name: Symbol(0),
                def: (UserDefinedType::Struct(vec![
                    StructField {
                        name: Symbol(1),
                        ty: Type::Int,
                    },
                    StructField {
                        name: Symbol(2),
                        ty: Type::Int,
                    },
                ])),
            },
        )
    }

    #[test]
    fn parse_type_enum() {
        expect_decl(
            "Enum :: { X1, X2, X3 }",
            Decl::TypeDef {
                name: Symbol(0),
                def: UserDefinedType::Enum(vec![
                    EnumField {
                        name: Symbol(1),
                        adts: Vec::new(),
                    },
                    EnumField {
                        name: Symbol(2),
                        adts: Vec::new(),
                    },
                    EnumField {
                        name: Symbol(3),
                        adts: Vec::new(),
                    },
                ]),
            },
        );
    }

    #[test]
    fn parse_type_enum_with_adts() {
        expect_decl(
            "Enum :: { X1(int), X2(int, int), X3(Enum) }",
            Decl::TypeDef {
                name: Symbol(0),
                def: UserDefinedType::Enum(vec![
                    EnumField {
                        name: Symbol(1),
                        adts: vec![Type::Int],
                    },
                    EnumField {
                        name: Symbol(2),
                        adts: vec![Type::Int, Type::Int],
                    },
                    EnumField {
                        name: Symbol(3),
                        adts: vec![Type::UserDef(Symbol(0))],
                    },
                ]),
            },
        );
    }

    #[test]
    fn parse_array_and_dyn_array() {
        expect_decl(
            "foo :: () { 
                x := [3]{1, 2, 3}
                y := []{1, 2, 3}
            }",
            Decl::Procedure {
                name: Symbol(0),
                fn_ty: None,
                sig: Signature {
                    params: Params {
                        patterns: vec![],
                        types: vec![],
                    },
                    return_ty: None,
                },
                block: Block {
                    stmts: vec![
                        Stmt::ValDec {
                            name: Symbol(1),
                            ty: None,
                            expr: Expr::Allocation {
                                kind: AllocKind::Array(3),
                                elements: vec![
                                    Expr::Value(Value::Int(1)),
                                    Expr::Value(Value::Int(2)),
                                    Expr::Value(Value::Int(3)),
                                ],
                                region: None,
                            },
                        },
                        Stmt::ValDec {
                            name: Symbol(2),
                            ty: None,
                            expr: Expr::Allocation {
                                kind: AllocKind::DynArray,
                                elements: vec![
                                    Expr::Value(Value::Int(1)),
                                    Expr::Value(Value::Int(2)),
                                    Expr::Value(Value::Int(3)),
                                ],
                                region: None,
                            },
                        },
                    ],
                    expr: None,
                },
            },
        );
    }

    #[test]
    fn parse_tuple() {
        expect_decl(
            "foo :: () { 
                x := (1, 2, 3)
            }",
            Decl::Procedure {
                name: Symbol(0),
                fn_ty: None,
                sig: Signature {
                    params: Params {
                        patterns: vec![],
                        types: vec![],
                    },
                    return_ty: None,
                },
                block: Block {
                    stmts: vec![Stmt::ValDec {
                        name: Symbol(1),
                        ty: None,
                        expr: Expr::Allocation {
                            kind: AllocKind::Tuple,
                            elements: vec![
                                Expr::Value(Value::Int(1)),
                                Expr::Value(Value::Int(2)),
                                Expr::Value(Value::Int(3)),
                            ],
                            region: None,
                        },
                    }],
                    expr: None,
                },
            },
        );
    }
}
