use crate::ctx::Symbol;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Value {
    Int(i32),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    Int,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Expr {
    Value(Value),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Pat {
    Symbol(Symbol),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Params {
    pub patterns: Vec<Pat>,
    pub types: Vec<Type>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Stmt {
    ValDec {
        name: Symbol,
        ty: Option<Type>,
        expr: Expr,
    },

    Assign {
        name: Symbol,
        expr: Expr,
    },
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    // optional return
    pub expr: Option<Expr>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Signature {
    pub params: Params,
    pub return_ty: Option<Type>,
    // TODO: add effect with handling here
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Decl {
    // name :: int
    Constant {
        name: Symbol,
        ty: Option<Type>,
        expr: Expr,
    },

    Procedure {
        name: Symbol,
        // TODO: implement fn_tys
        fn_ty: Option<Type>,
        sig: Signature,
        block: Block,
    },
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Module {
    pub declarations: Vec<Decl>,
}
