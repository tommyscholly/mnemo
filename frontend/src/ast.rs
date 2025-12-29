use crate::ctx::Symbol;
use crate::lex::BinOp;
use crate::span::{DUMMY_SPAN, Spanned};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ValueKind {
    Int(i32),
    Ident(Symbol),
}

pub type Value = Spanned<ValueKind>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct StructField {
    pub name: Spanned<Symbol>,
    pub ty: Type,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct EnumField {
    pub name: Spanned<Symbol>,
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
pub enum TypeKind {
    Unit,
    Int,
    Bool,
    UserDef(Symbol),
    Fn(Box<SignatureInner>),
    Alloc(AllocKind, Region),
}

pub type Type = Spanned<TypeKind>;

impl Type {
    /// Create an unspanned type (for synthesized/inferred types).
    pub fn synthetic(kind: TypeKind) -> Self {
        Spanned::new(kind, DUMMY_SPAN)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum AllocKind {
    DynArray(Box<Type>),
    Array(Box<Type>, usize),
    Tuple,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Region {
    // need to track where a region originates from
    Local,
    Generic(Symbol),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Call {
    pub callee: Spanned<Symbol>,
    pub args: Vec<Expr>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ExprKind {
    Value(ValueKind),
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

pub type Expr = Spanned<ExprKind>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum PatKind {
    Symbol(Symbol),
}

pub type Pat = Spanned<PatKind>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Params {
    pub patterns: Vec<Pat>,
    pub types: Vec<Type>,
}

// possibly make this both a statement and an expression
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IfElse {
    pub cond: Expr,
    pub then: Block,
    pub else_: Option<Block>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StmtKind {
    ValDec {
        name: Spanned<Symbol>,
        ty: Option<Type>,
        expr: Expr,
    },
    Assign {
        name: Spanned<Symbol>,
        expr: Expr,
    },
    // a function call with no assignment
    Call(Call),
    IfElse(IfElse),
}

pub type Stmt = Spanned<StmtKind>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BlockInner {
    pub stmts: Vec<Stmt>,
    // optional return expression
    pub expr: Option<Expr>,
}

pub type Block = Spanned<BlockInner>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SignatureInner {
    pub params: Params,
    pub return_ty: Option<Type>,
    // TODO: add effect with handling here
}

pub type Signature = Spanned<SignatureInner>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum DeclKind {
    // name :: int
    Constant {
        name: Spanned<Symbol>,
        ty: Option<Type>,
        expr: Expr,
    },

    TypeDef {
        name: Spanned<Symbol>,
        def: UserDefinedType,
    },

    Procedure {
        name: Spanned<Symbol>,
        // TODO: implement fn_tys
        fn_ty: Option<Type>,
        sig: Signature,
        block: Block,
    },
}

pub type Decl = Spanned<DeclKind>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Module {
    pub declarations: Vec<Decl>,
}
