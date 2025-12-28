use crate::{ctx::Symbol, lex::BinOp};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Value {
    Int(i32),
    Ident(Symbol),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct StructField {
    pub name: Symbol,
    pub ty: Type,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct EnumField {
    pub name: Symbol,
    // enums can have associated data of N types, or just be single names
    // NormalEnum :: { One, Two, Foo, Bar }
    // ADTEnum :: { One(int), Two(int, int), Foo(T), Bar(T) }
    pub adts: Vec<Type>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum UserDefinedType {
    Struct(Vec<StructField>),
    Enum(Vec<EnumField>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    Unit,
    Int,
    UserDef(Symbol),
    Fn(Box<Signature>),
    Alloc(AllocKind, Region),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Call {
    pub callee: Symbol,
    pub args: Vec<Expr>,
}

// possibly make this both a statement and an expression
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IfElse {
    pub cond: Expr,
    pub then: Block,
    pub else_: Option<Block>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AllocKind {
    DynArray,
    Array(usize),
    Tuple,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Region {
    // need to track where a region originates from
    Local,
    Generic(Symbol),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Expr {
    Value(Value),
    Allocation {
        kind: AllocKind,
        // some allocations may not have elements, we just leave this empty
        elements: Vec<Expr>,
        // without a region, this is allocated in the function's implicit contextual region
        region: Option<Region>,
    },
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    // a function call that is used in an assignment or declaration
    Call(Call),
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
    // a function call with no assignment
    Call(Call),
    IfElse(IfElse),
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

    TypeDef {
        name: Symbol,
        def: UserDefinedType,
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
