use crate::lex::BinOp;

// corresponds to locals in the defining function
pub type LocalId = usize;
pub type FunctionId = usize;
pub type BlockId = usize;

#[derive(Debug)]
pub enum Ty {
    Bool,
    Char,
    Int,
}

#[derive(Debug)]
pub enum Operand {
    Constant(i32),
    Local(LocalId),
}

#[derive(Debug)]
pub enum RValue {
    Use(Operand),
    BinOp(BinOp, Box<RValue>, Box<RValue>),
}

#[derive(Debug)]
pub enum Statement {
    // This is in SSA form, so assigning defines a new local
    Assign(LocalId, RValue),
}

#[derive(Debug)]
pub enum Terminator {
    Return,
    Br(BlockId, Vec<LocalId>),
    // usize is index of the local for the condition
    BrIf(usize, BlockId, BlockId),
    Call {
        function_id: FunctionId,
        args: Vec<RValue>,
        destination: Option<LocalId>,
        target: BlockId,
    },
}

#[derive(Debug)]
pub struct BasicBlock {
    pub block_id: BlockId,
    pub params: u32,
    pub stmts: Vec<Statement>,
    // TODO: should this possibly be an optional at some points
    pub terminator: Terminator,
}

impl BasicBlock {
    pub fn new(block_id: BlockId) -> Self {
        Self {
            block_id,
            params: 0,
            stmts: Vec::new(),
            terminator: Terminator::Return,
        }
    }
}

#[derive(Debug)]
pub struct Local {
    pub ty: Ty,
}

#[derive(Debug)]
pub struct Function {
    pub function_id: FunctionId,
    pub blocks: Vec<BasicBlock>,
    pub parameters: u32,
    pub locals: Vec<Local>,
}
