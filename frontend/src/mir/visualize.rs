use crate::{ast_visitor::MIRCtx, mir::*};

pub trait MIRVisualizer {
    fn visualize(&self, ctx: &MIRCtx, indent: usize);
}

impl MIRVisualizer for Operand {
    fn visualize(&self, _ctx: &MIRCtx, _indent: usize) {
        match self {
            Operand::Local(local_id) => print!("%{local_id}"),
            Operand::Constant(i) => print!("{i}"),
        }
    }
}

impl MIRVisualizer for RValue {
    fn visualize(&self, ctx: &MIRCtx, indent: usize) {
        match self {
            RValue::Use(op) => op.visualize(ctx, indent),
            RValue::BinOp(op, lhs, rhs) => {
                lhs.visualize(ctx, indent);
                print!(" {op} ");
                rhs.visualize(ctx, indent);
            }
        }
    }
}

impl MIRVisualizer for Terminator {
    fn visualize(&self, ctx: &MIRCtx, indent: usize) {
        let tabs = "\t".repeat(indent);
        print!("{tabs}");
        match self {
            Terminator::Return => println!("return"),
            Terminator::Br(block_id) => println!("br {}", block_id),
            Terminator::BrIf(cond_local_id, then_block_id, else_block_id) => {
                println!(
                    "br_if {} {} {}",
                    cond_local_id, then_block_id, else_block_id
                )
            }
            Terminator::Call {
                function_id,
                args,
                destination,
                target,
            } => {
                let dest = destination.map_or("None".to_string(), |f| format!("{}", f));
                print!("call {} (", function_id);
                args.iter().for_each(|a| a.visualize(ctx, indent));
                println!(") {} {}", dest, target)
            }
        }
    }
}

impl MIRVisualizer for Statement {
    fn visualize(&self, ctx: &MIRCtx, indent: usize) {
        let tabs = "\t".repeat(indent);
        print!("{tabs}");
        match self {
            Statement::Assign(local_id, expr) => {
                print!("assign {} = ", local_id);
                expr.visualize(ctx, indent);
                println!()
            }
            Statement::Phi(local_id, ids) => {
                println!("phi {} = {:?}", local_id, ids);
            }
        }
    }
}

impl MIRVisualizer for BasicBlock {
    fn visualize(&self, ctx: &MIRCtx, indent: usize) {
        let tabs = "\t".repeat(indent);
        print!("{tabs}");
        println!("block {}:", self.block_id);
        for stmt in self.stmts.iter() {
            stmt.visualize(ctx, indent + 1);
        }
        self.terminator.visualize(ctx, indent + 1);
    }
}

impl MIRVisualizer for Function {
    fn visualize(&self, ctx: &MIRCtx, indent: usize) {
        let tabs = "\t".repeat(indent);
        print!("{tabs}");
        print!("fn {}(", self.function_id);
        for (i, local) in self.locals.iter().enumerate() {
            print!("{}: {}", i, local.ty);
            if i != self.locals.len() - 1 {
                print!(", ");
            }
        }
        println!("):");

        for block in self.blocks.iter() {
            block.visualize(ctx, indent + 1);
        }
    }
}
