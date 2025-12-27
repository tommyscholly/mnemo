#![allow(unused)]

pub mod visualize;

use std::fmt::Display;

use crate::lex::BinOp;

// corresponds to locals in the defining function
pub type LocalId = usize;
pub type FunctionId = usize;
pub type BlockId = usize;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Ty {
    Bool,
    Char,
    Int,
}

impl Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ty::Bool => write!(f, "bool"),
            Ty::Char => write!(f, "char"),
            Ty::Int => write!(f, "int"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Operand {
    Constant(i32),
    Local(LocalId),
}

#[derive(Debug, PartialEq, Eq)]
pub enum RValue {
    Use(Operand),
    BinOp(BinOp, Box<RValue>, Box<RValue>),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Statement {
    // This is in SSA form, so assigning defines a new local
    Assign(LocalId, RValue),
    Phi(LocalId, Vec<LocalId>),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Terminator {
    Return,
    Br(BlockId),
    // usize is index of the local for the condition
    BrIf(usize, BlockId, BlockId),
    Call {
        function_id: FunctionId,
        args: Vec<RValue>,
        destination: Option<LocalId>,
        target: BlockId,
    },
}

#[derive(Debug, PartialEq, Eq)]
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
            terminator: Terminator::Return,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Local {
    pub id: LocalId,
    pub ty: Ty,
}

impl Local {
    pub fn new(local_id: LocalId, ty: Ty) -> Self {
        Self { id: local_id, ty }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Function {
    pub function_id: FunctionId,
    pub blocks: Vec<BasicBlock>,
    pub parameters: u32,
    pub locals: Vec<Local>,
}
