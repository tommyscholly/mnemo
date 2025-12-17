pub(super) enum Ty {
    Bool,
    Char,
    Int,
}

pub(super) enum Statement {}

pub(super) enum Terminator {}

pub(super) struct BasicBlock {
    stmts: Vec<Statement>,
    // TODO: should this possibly be an optional at some points
    terminator: Terminator,
}

pub(super) struct Local {
    ty: Ty,
}

pub(super) struct Function {
    blocks: Vec<BasicBlock>,
    parameters: u32,
    locals: Vec<Local>,
}
