use crate::mir::*;

pub trait MIRVisualizer {
    fn visualize(&self, indent: usize);
}

impl MIRVisualizer for Operand {
    fn visualize(&self, indent: usize) {
        match self {
            Operand::Local(local_id) => print!("%{local_id}"),
            Operand::Constant(i) => print!("{i}"),
        }
    }
}

impl MIRVisualizer for RValue {
    fn visualize(&self, indent: usize) {
        match self {
            RValue::Use(op) => op.visualize(indent),
            RValue::BinOp(op, lhs, rhs) => {
                lhs.visualize(indent);
                print!(" {op} ");
                rhs.visualize(indent);
            }
            RValue::Alloc(kind, ops) => {
                print!("alloc {} {{", kind);
                for (i, op) in ops.iter().enumerate() {
                    op.visualize(indent);
                    if i != ops.len() - 1 {
                        print!(", ");
                    }
                }
                println!("}}");
            }
        }
    }
}

impl MIRVisualizer for Terminator {
    fn visualize(&self, indent: usize) {
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
                print!("call ${} (", function_id);
                args.iter().for_each(|a| a.visualize(indent));
                println!(") {} {}", dest, target)
            }
        }
    }
}

impl MIRVisualizer for Statement {
    fn visualize(&self, indent: usize) {
        let tabs = "\t".repeat(indent);
        print!("{tabs}");
        match self {
            Statement::Assign(local_id, expr) => {
                print!("%{} = ", local_id);
                expr.visualize(indent);
                println!()
            }
            Statement::Phi(local_id, ids) => {
                println!("phi %{} = {:?}", local_id, ids);
            }
        }
    }
}

impl MIRVisualizer for BasicBlock {
    fn visualize(&self, indent: usize) {
        let tabs = "\t".repeat(indent);
        print!("{tabs}");
        println!("bb {}:", self.block_id);
        for stmt in self.stmts.iter() {
            stmt.visualize(indent + 1);
        }
        self.terminator.visualize(indent + 1);
    }
}

impl MIRVisualizer for Function {
    fn visualize(&self, indent: usize) {
        let tabs = "\t".repeat(indent);
        print!("{tabs}");
        print!("fn ${}(", self.function_id);
        for (i, local) in self.locals.iter().enumerate().take(self.parameters) {
            print!("{}: {}", i, local.ty);
            if i != self.locals.len() - 1 {
                print!(", ");
            }
        }
        println!("):");

        for block in self.blocks.iter() {
            block.visualize(indent + 1);
        }
    }
}

impl MIRVisualizer for Module {
    fn visualize(&self, indent: usize) {
        println!("module:");
        for constant in self.constants.iter() {
            print!("constant: ");
            constant.visualize(indent + 1);
            println!();
        }

        for function in self.functions.iter() {
            function.visualize(indent + 1);
        }
    }
}
