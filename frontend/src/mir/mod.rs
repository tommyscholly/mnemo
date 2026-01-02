mod graph;
pub mod ir;
mod visit;
pub mod visualize;
mod domtree;

pub use ir::*;

#[allow(unused)]
trait MirPass {}

pub fn run_passes(module: Module, output_mir: bool) -> Module {
    let mut cfg = graph::FlowGraph::new();

    for func in &module.functions {
        cfg.compute(func);

        if output_mir {
            println!("{}", cfg.to_dot_string());
        }
    }

    module
}
