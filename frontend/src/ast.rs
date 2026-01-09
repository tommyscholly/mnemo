use std::hash::{Hash, Hasher};

use crate::Span;
use crate::ctx::Symbol;
use crate::lex::BinOp;
use crate::span::{DUMMY_SPAN, Spanned};

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ValueKind {
    Bool(bool),
    Ident(Symbol),
    Int(i32),
    Type(TypeKind),
}

#[allow(unused)]
pub type Value = Spanned<ValueKind>;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct RecordField {
    pub name: Symbol,
    pub ty: Option<Type>,
}

#[derive(Debug, Clone)]
pub struct VariantField {
    pub name: Spanned<Symbol>,
    // enums can have associated data of N types, or just be single names
    // NormalEnum :: { One, Two, Foo, Bar }
    // ADTEnum :: { One(int), Two(int, int), Foo(T), Bar(T) }
    pub adts: Vec<Type>,
}

// requires manual hash cause manual impl of PartialEq
impl Hash for VariantField {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.node.hash(state);
        for adt in &self.adts {
            adt.hash(state);
        }
    }
}

impl PartialEq for VariantField {
    fn eq(&self, other: &Self) -> bool {
        let adts = self.adts.iter().map(|t| &t.node).collect::<Vec<_>>();
        let other_adts = other.adts.iter().map(|t| &t.node).collect::<Vec<_>>();
        self.name.node == other.name.node && adts == other_adts
    }
}

impl Eq for VariantField {}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TypeAliasDefinition {
    Record(Vec<RecordField>),
    Variant(Vec<VariantField>),
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum TypeKind {
    Alloc(AllocKind, Region),
    Bool,
    Char,
    ComptimeInt,
    Fn(Box<SignatureInner>),
    Int,
    // all ptrs must be qualified with a region (at some point)
    Ptr(Box<TypeKind> /*, Region */),
    Record(Vec<RecordField>),
    Resolved(Box<TypeKind>),
    Type,
    // for both type aliases and generic type parameters
    TypeAlias(Symbol),
    Unit,
    Variadic,
    Variant(Vec<VariantField>),
}

pub type Type = Spanned<TypeKind>;

impl Type {
    pub fn synthetic(kind: TypeKind) -> Self {
        Spanned::new(kind, DUMMY_SPAN)
    }

    pub fn with_span(kind: TypeKind, span: Span) -> Self {
        Spanned::new(kind, span)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum AllocKind {
    Array(Box<TypeKind>, usize),
    DynArray(Box<TypeKind>),
    Record(Vec<RecordField>),
    Str(String),
    Tuple(Vec<TypeKind>),
    Variant(Symbol),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Region {
    // need to track where a region originates from
    Stack,
    Local,
    Generic(Symbol),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Call {
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
    // filled in during typechecking
    pub returned_ty: Option<Type>,
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
    FieldAccess(Box<Expr>, Symbol),
    TupleAccess(Box<Expr>, usize),
    // a function call that is used in an assignment or declaration
    Call(Call),
    Index(Box<Expr>, Box<Expr>),
}

pub type Expr = Spanned<ExprKind>;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum PatKind {
    Symbol(Symbol),
    Wildcard,
    Literal(ValueKind),
    Variant {
        name: Symbol,
        // x and y are the patterns
        // Some(x, y)
        bindings: Vec<Pat>,
    },
    Tuple(Vec<Pat>),
    Record(Vec<RecordField>),
}

pub type Pat = Spanned<PatKind>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MatchArm {
    pub pat: Pat,
    pub body: Block,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Match {
    pub scrutinee: Expr,
    pub arms: Vec<MatchArm>,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Params {
    pub params: Vec<Param>,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Param {
    pub pattern: Pat,
    pub ty: Type,
    pub is_comptime: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ComptimeValue {
    Int(i128),
    Bool(bool),
    Type(TypeKind),
    Array(Vec<ComptimeValue>),
    Unit,
}

#[derive(Debug, Clone)]
pub struct TypedValue {
    pub ty: TypeKind,
    pub comptime_value: Option<ComptimeValue>,
}

impl TypedValue {
    pub fn runtime(ty: TypeKind) -> Self {
        Self {
            ty,
            comptime_value: None,
        }
    }

    pub fn comptime(ty: TypeKind, value: ComptimeValue) -> Self {
        Self {
            ty,
            comptime_value: Some(value),
        }
    }

    pub fn is_comptime(&self) -> bool {
        self.comptime_value.is_some()
    }

    pub fn unwrap_type(self) -> TypeKind {
        self.ty
    }
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
        is_comptime: bool,
    },
    Assign {
        name: Spanned<Symbol>,
        expr: Expr,
    },
    // a function call that is used in an assignment or declaration
    Call(Call),
    IfElse(Box<IfElse>),
    Match(Match),
    Return(Option<Expr>),
}

pub type Stmt = Spanned<StmtKind>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BlockInner {
    pub stmts: Vec<Stmt>,
    // optional return expression
    pub expr: Option<Expr>,
}

pub type Block = Spanned<BlockInner>;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct SignatureInner {
    pub params: Params,
    pub return_ty: Option<Type>,
    // TODO: add effect with handling here
}

pub type Signature = Spanned<SignatureInner>;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Constraint {
    Allocates(Region),
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct MonomorphKey {
    pub base_fn: Symbol,
    pub comptime_args: Vec<ComptimeValue>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum DeclKind {
    Extern {
        name: Spanned<Symbol>,
        sig: Signature,
        generic_params: Option<Vec<Param>>,
    },
    Constant {
        name: Spanned<Symbol>,
        ty: Option<Type>,
        expr: Expr,
        is_comptime: bool,
    },
    TypeDef {
        name: Spanned<Symbol>,
        def: TypeAliasDefinition,
    },
    Procedure {
        name: Spanned<Symbol>,
        // TODO: implement fn_tys
        fn_ty: Option<Type>,
        sig: Signature,
        constraints: Vec<Constraint>,
        block: Block,
        monomorph_of: Option<MonomorphKey>,
        // filled in during typechecking, to skip over during lowering to MIR
        is_comptime: bool,
    },
}

pub type Decl = Spanned<DeclKind>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Module {
    pub declarations: Vec<Decl>,
}
