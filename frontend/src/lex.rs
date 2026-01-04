use crate::ctx::{Ctx, Symbol};
use crate::span::{Diagnostic, Span, Spanned};
use crate::{advance_single_token, handle_operator};
use std::fmt::Display;
use std::iter::Peekable;

#[allow(unused)]
#[derive(Debug)]
pub enum LexErrorKind {
    UnexpectedChar(LexItem),
    UnexpectedKeyword(String),
    UnexpectedEOF,
}

#[allow(unused)]
#[derive(Debug)]
pub struct LexError {
    kind: LexErrorKind,
    span: Span,
}

impl LexError {
    pub fn new(kind: LexErrorKind, span: Span) -> Self {
        Self { kind, span }
    }
}

impl Diagnostic for LexError {
    fn span(&self) -> &Span {
        &self.span
    }

    fn message(&self) -> String {
        "lex error".to_string()
    }

    fn label(&self) -> Option<String> {
        Some(format!("{:?}", self.kind))
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Keyword {
    Int,
    Return,
    Extern,
    Char,
    If,
    Else,
    Bool,
    True,
    False,
}

impl TryFrom<&str> for Keyword {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "int" => Ok(Keyword::Int),
            "return" => Ok(Keyword::Return),
            "extern" => Ok(Keyword::Extern),
            "char" => Ok(Keyword::Char),
            "if" => Ok(Keyword::If),
            "else" => Ok(Keyword::Else),
            "bool" => Ok(Keyword::Bool),
            "true" => Ok(Keyword::True),
            "false" => Ok(Keyword::False),
            _ => Err(()),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Token {
    Identifier(Symbol),
    Keyword(Keyword),
    BinOp(BinOp),
    Int(i32),
    Colon,
    Eq,
    /// (
    LParen,
    /// )
    RParen,
    /// {
    LBrace,
    /// }
    RBrace,
    /// [
    LBracket,
    /// ]
    RBracket,
    Comma,
    SemiColon,
    Arrow,
    At,
    Dot,
    Caret,
    Bar,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    And,
    Or,
    EqEq,
    NEq,
    Gt,
    GtEq,
    Lt,
    LtEq,
}

impl Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::And => write!(f, "and"),
            BinOp::Or => write!(f, "or"),
            BinOp::EqEq => write!(f, "=="),
            BinOp::NEq => write!(f, "!="),
            BinOp::Gt => write!(f, ">"),
            BinOp::GtEq => write!(f, ">="),
            BinOp::Lt => write!(f, "<"),
            BinOp::LtEq => write!(f, "<="),
        }
    }
}

impl TryFrom<&str> for BinOp {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "+" => Ok(BinOp::Add),
            "-" => Ok(BinOp::Sub),
            "*" => Ok(BinOp::Mul),
            "/" => Ok(BinOp::Div),
            "%" => Ok(BinOp::Mod),
            "==" => Ok(BinOp::EqEq),
            "!=" => Ok(BinOp::NEq),
            ">" => Ok(BinOp::Gt),
            ">=" => Ok(BinOp::GtEq),
            "<" => Ok(BinOp::Lt),
            "<=" => Ok(BinOp::LtEq),
            "and" => Ok(BinOp::And),
            "or" => Ok(BinOp::Or),
            _ => Err(()),
        }
    }
}

type LexItem = char;

pub struct Lexer<'a, T: Iterator<Item = LexItem>> {
    ctx: &'a mut Ctx,
    chars: Peekable<T>,
    start: usize,
    current: usize,
}

impl<'a, T: Iterator<Item = LexItem>> Lexer<'a, T> {
    pub fn new(chars: T, ctx: &'a mut Ctx) -> Self {
        Lexer {
            ctx,
            chars: chars.peekable(),
            start: 0,
            current: 0,
        }
    }

    fn next_token(&mut self) -> Result<Spanned<Token>, LexError> {
        while let Some(c) = self.chars.peek() {
            match c {
                '/' => {
                    if self.chars.peek() == Some(&'/') {
                        while let Some(&c) = self.chars.peek() {
                            if c == '\n' {
                                self.chars.next();
                                self.current += 1;
                                self.start = self.current;
                                break;
                            } else {
                                self.chars.next();
                                self.current += 1;
                            }
                        }
                    } else {
                        advance_single_token!(self, Token::BinOp(BinOp::Div))
                    }
                }
                '^' => {
                    advance_single_token!(self, Token::Caret)
                }
                '@' => {
                    advance_single_token!(self, Token::At)
                }
                ';' => {
                    advance_single_token!(self, Token::SemiColon)
                }
                ':' => {
                    advance_single_token!(self, Token::Colon)
                }
                '(' => {
                    advance_single_token!(self, Token::LParen)
                }
                ')' => {
                    advance_single_token!(self, Token::RParen)
                }
                '{' => {
                    advance_single_token!(self, Token::LBrace)
                }
                '}' => {
                    advance_single_token!(self, Token::RBrace)
                }
                '[' => {
                    advance_single_token!(self, Token::LBracket)
                }
                ']' => {
                    advance_single_token!(self, Token::RBracket)
                }
                ',' => {
                    advance_single_token!(self, Token::Comma)
                }
                '.' => {
                    advance_single_token!(self, Token::Dot)
                }
                '|' => {
                    advance_single_token!(self, Token::Bar)
                }
                '=' => {
                    handle_operator!(self, '=', '=', Token::Eq, Token::BinOp(BinOp::EqEq))
                }
                '>' => {
                    handle_operator!(
                        self,
                        '>',
                        '=',
                        Token::BinOp(BinOp::Gt),
                        Token::BinOp(BinOp::GtEq)
                    )
                }
                '<' => {
                    handle_operator!(
                        self,
                        '<',
                        '=',
                        Token::BinOp(BinOp::Lt),
                        Token::BinOp(BinOp::LtEq)
                    )
                }
                '-' => {
                    handle_operator!(self, '-', '>', Token::BinOp(BinOp::Sub), Token::Arrow)
                }
                ' ' | '\t' | '\n' | '\r' => {
                    self.chars.next();
                    self.start += 1;
                    self.current += 1;
                }
                c => {
                    if c.is_ascii_digit() {
                        let token = self.next_number();
                        self.start = self.current;
                        return token;
                    } else if c.is_alphabetic() || *c == '_' {
                        let token = self.next_kw_var();
                        self.start = self.current;
                        return token;
                    } else {
                        if let Ok(op) = BinOp::try_from(c.to_string().as_str()) {
                            advance_single_token!(self, Token::BinOp(op))
                        }

                        return Err(LexError::new(
                            LexErrorKind::UnexpectedChar(*c),
                            self.start..self.current,
                        ));
                    }
                }
            }
        }

        Err(LexError::new(
            LexErrorKind::UnexpectedEOF,
            self.current..self.current,
        ))
    }

    fn next_number(&mut self) -> Result<Spanned<Token>, LexError> {
        let mut number = String::new();
        while let Some(c) = self.chars.peek() {
            if c.is_ascii_digit() {
                number.push(*c);
                self.chars.next();
                self.current += 1;
            } else {
                break;
            }
        }

        Ok(Spanned::new(
            Token::Int(number.parse().unwrap()),
            self.start..self.current,
        ))
    }

    fn next_kw_var(&mut self) -> Result<Spanned<Token>, LexError> {
        let mut kw_var = String::new();
        while let Some(c) = self.chars.peek() {
            if c.is_alphanumeric() || *c == '_' {
                kw_var.push(*c);
                self.chars.next();
                self.current += 1;
            } else {
                break;
            }
        }

        match Keyword::try_from(kw_var.as_str()) {
            Ok(kw) => Ok(Spanned::new(Token::Keyword(kw), self.start..self.current)),
            Err(_) => match BinOp::try_from(kw_var.as_str()) {
                Ok(op) => Ok(Spanned::new(Token::BinOp(op), self.start..self.current)),
                Err(_) => Ok(Spanned::new(
                    Token::Identifier(self.ctx.intern(&kw_var)),
                    self.start..self.current,
                )),
            },
        }
    }
}

impl<'a, T: Iterator<Item = LexItem>> Iterator for Lexer<'a, T> {
    type Item = Result<Spanned<Token>, LexError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.chars.peek().is_none() {
            None
        } else {
            let tok = self.next_token();
            match tok {
                Ok(token) => Some(Ok(token)),
                Err(err) => Some(Err(err)),
            }
        }
    }
}

pub fn tokenize<'a>(ctx: &'a mut Ctx, src: &str) -> Lexer<'a, impl Iterator<Item = LexItem>> {
    let chars = src.chars();
    Lexer::new(chars, ctx)
}
