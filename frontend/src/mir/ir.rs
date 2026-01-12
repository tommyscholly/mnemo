use std::fmt::Display;

use crate::lex::BinOp;

// corresponds to locals in the defining function
pub type LocalId = usize;
pub type BlockId = usize;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Ty {
    Array(Box<Ty>, usize),
    Bool,
    Char,
    DynArray(Box<Ty>),
    Int,
    Ptr(Box<Ty>),
    Record(Vec<Ty>),
    Str,
    TaggedUnion(Vec<(u8, Ty)>),
    Tuple(Vec<Ty>),
    Unit,
    Variadic,
}

impl Ty {
    pub fn bytes(&self) -> usize {
        match self {
            Ty::Bool => 1,
            Ty::Char => 1,
            Ty::Str => 1,
            Ty::Int => 4,
            Ty::Unit => 0,
            Ty::Array(ty, len) => ty.bytes() * *len,
            Ty::DynArray(ty) => ty.bytes(),
            Ty::Tuple(tys) => tys.iter().map(|t| t.bytes()).sum(),
            Ty::Ptr(_) => 4,
            Ty::TaggedUnion(tags_tys) => {
                let max_payload_size = tags_tys.iter().map(|(_tag, ty)| ty.bytes()).max().unwrap();
                let max_payload_align = tags_tys
                    .iter()
                    .map(|(_tag, ty)| ty.align())
                    .max()
                    .unwrap_or(1);
                let base_size = 1 + max_payload_size;
                let padding = if max_payload_align > 1 {
                    (max_payload_align - (base_size % max_payload_align)) % max_payload_align
                } else {
                    0
                };
                base_size + padding
            }
            Ty::Record(tys) => tys.iter().map(|t| t.bytes()).sum(),
            Ty::Variadic => 1,
        }
    }

    pub fn align(&self) -> usize {
        match self {
            Ty::Bool => 1,
            Ty::Char => 1,
            Ty::Str => 1,
            Ty::Int => 4,
            Ty::Unit => 1,
            Ty::Array(ty, _) => ty.align(),
            Ty::DynArray(ty) => ty.align(),
            Ty::Tuple(tys) => tys.iter().map(|t| t.align()).max().unwrap_or(1),
            Ty::Ptr(_) => 4,
            Ty::TaggedUnion(tags_tys) => {
                let max_payload_align = tags_tys
                    .iter()
                    .map(|(_tag, ty)| ty.align())
                    .max()
                    .unwrap_or(1);
                std::cmp::max(1, max_payload_align)
            }
            Ty::Record(tys) => tys.iter().map(|t| t.align()).max().unwrap_or(1),
            // TODO: is this correct?
            Ty::Variadic => 1,
        }
    }
}

impl Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ty::Bool => write!(f, "bool"),
            Ty::Char => write!(f, "char"),
            Ty::Int => write!(f, "int"),
            Ty::Str => write!(f, "str"),
            Ty::Unit => write!(f, "unit"),
            Ty::Array(ty, len) => write!(f, "[{}]{{ {} }}", ty, len),
            Ty::DynArray(ty) => write!(f, "dyn{{ {} }}", ty),
            Ty::Tuple(tys) => write!(
                f,
                "({})",
                tys.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Ty::Ptr(ty) => write!(f, "^{}", ty),
            Ty::TaggedUnion(tags_tys) => {
                let tags_tys_str = tags_tys
                    .iter()
                    .map(|(tag, ty)| format!("{}:<{}>", tag, ty))
                    .collect::<Vec<String>>()
                    .join("|");

                write!(f, "{}", tags_tys_str)
            }

            Ty::Record(tys) => write!(
                f,
                "{{ {} }}",
                tys.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Ty::Variadic => write!(f, "..."),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum PlaceKind {
    /// array index operations
    // this was local id, now i'm making this an rvalue
    Index(LocalId),
    /// field idx, and a type for validation
    Field(usize, Ty),
    Deref,
}

#[derive(Debug, PartialEq, Eq, Clone)]
// places are memory locations
pub struct Place {
    pub local: LocalId,
    // we use the place kind to determine where we are looking in the local
    // for example, if we are looking at a field, we need to know the field index
    // if we are looking at an array index, we need to know the array index
    pub kind: PlaceKind,
}

impl Place {
    pub fn new(local: LocalId, kind: PlaceKind) -> Self {
        Self { local, kind }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Constant {
    Int(i32),
    Bool(bool),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Operand {
    Constant(Constant),
    Copy(Place),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum AllocKind {
    Array(Ty),
    DynArray(Ty),
    Record(Vec<Ty>),
    Str(String),
    Tuple(Vec<Ty>),
    Variant(u8, Ty),
}

impl Display for AllocKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocKind::Array(ty) => write!(f, "array<{}>", ty),
            AllocKind::Tuple(tys) => {
                let ty_str = tys
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<String>>()
                    .join(" * ");

                write!(f, "{}", ty_str)
            }
            AllocKind::DynArray(ty) => write!(f, "dyn_array<{}>", ty),
            AllocKind::Str(s) => write!(f, "str<{}>", s),
            AllocKind::Record(tys) => {
                let ty_str = tys
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<String>>()
                    .join(",");

                write!(f, "{{{}}}", ty_str)
            }
            AllocKind::Variant(tag, ty) => write!(f, "Variant({})<{}>", tag, ty),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum RValue {
    Use(Operand),
    BinOp(BinOp, Operand, Operand),
    Alloc(AllocKind, Vec<Operand>),
}

impl RValue {
    pub fn place(&self) -> Option<Place> {
        match self {
            RValue::Use(op) => match op {
                Operand::Constant(_) => None,
                Operand::Copy(p) => Some(p.clone()),
            },
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Statement {
    // This is in SSA form, so assigning defines a new local
    Assign(LocalId, RValue),
    Store(Place, RValue),
    Phi(LocalId, Vec<LocalId>),
    Call {
        function_name: String,
        args: Vec<RValue>,
        destination: Option<LocalId>,
    },
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct JumpTable {
    pub default: BlockId,
    pub cases: Vec<(i32, BlockId)>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Terminator {
    Return(Option<LocalId>),
    Br(BlockId),
    // usize is index of the local for the condition
    BrIf(LocalId, BlockId, BlockId),
    BrTable(LocalId, JumpTable),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BasicBlock {
    pub block_id: BlockId,
    pub stmts: Vec<Statement>,
    // TODO: should this possibly be an optional at some points
    pub terminator: Terminator,
}

impl BasicBlock {
    pub fn new(block_id: BlockId) -> Self {
        Self {
            block_id,
            stmts: Vec::new(),
            terminator: Terminator::Return(None),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Local {
    pub id: LocalId,
    pub ty: Ty,
}

impl Local {
    pub fn new(local_id: LocalId, ty: Ty) -> Self {
        Self { id: local_id, ty }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Function {
    pub name: String,
    pub blocks: Vec<BasicBlock>,
    pub parameters: usize,
    pub return_ty: Ty,
    pub locals: Vec<Local>,
}

impl IntoIterator for Function {
    type Item = BasicBlock;
    type IntoIter = std::vec::IntoIter<BasicBlock>;

    // TOOD: make this a proper CFG traversal
    fn into_iter(self) -> Self::IntoIter {
        self.blocks.into_iter()
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Extern {
    pub name: String,
    pub params: Vec<Ty>,
    pub return_ty: Ty,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Module {
    pub functions: Vec<Function>,
    pub constants: Vec<RValue>,
    pub externs: Vec<Extern>,
}
