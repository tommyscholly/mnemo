use crate::mir::*;

pub fn visit_block_succs<F: FnMut(BlockId)>(_func: &Function, block: &BasicBlock, mut visit: F) {
    match &block.terminator {
        Terminator::Return(_) => {}
        Terminator::Br(block_id) => {
            visit(*block_id);
        }
        Terminator::BrIf(_, then_, else_) => {
            visit(*then_);
            visit(*else_);
        }
        Terminator::BrTable(_, jump_table) => {
            for (_, target) in jump_table.cases.iter() {
                visit(*target);
            }
            visit(jump_table.default);
        }
    }
}
