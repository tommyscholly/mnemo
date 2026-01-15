mod domtree;
mod graph;
pub mod ir;
mod visit;
pub mod visualize;

pub use ir::*;

#[allow(unused)]
trait MirPass {}

pub fn run_passes(module: Module, output_mir: bool) -> Module {
    let module = erase_regions(module);

    let mut cfg = graph::FlowGraph::new();

    for func in &module.functions {
        cfg.compute(func);

        if output_mir {
            println!("{}", cfg.to_dot_string());
        }
    }

    module
}

fn erase_regions(module: Module) -> Module {
    Module {
        functions: module
            .functions
            .into_iter()
            .map(erase_function_regions)
            .collect(),
        constants: module.constants,
        externs: module.externs,
    }
}

fn erase_function_regions(func: Function) -> Function {
    Function {
        name: func.name,
        blocks: func.blocks,
        parameters: func.parameters,
        return_ty: erase_type_regions(&func.return_ty),
        locals: func
            .locals
            .into_iter()
            .map(|local| Local {
                id: local.id,
                ty: erase_type_regions(&local.ty),
            })
            .collect(),
        region_params: Vec::new(),
        region_outlives: Vec::new(),
    }
}

fn erase_type_regions(ty: &Ty) -> Ty {
    match ty {
        Ty::Array(inner, len) => Ty::Array(Box::new(erase_type_regions(inner)), *len),
        Ty::Bool => Ty::Bool,
        Ty::Char => Ty::Char,
        Ty::DynArray(inner) => Ty::DynArray(Box::new(erase_type_regions(inner))),
        Ty::Int => Ty::Int,
        Ty::Ptr(inner, _) => Ty::Ptr(Box::new(erase_type_regions(inner)), STATIC_REGION),
        Ty::Record(tys) => Ty::Record(tys.iter().map(erase_type_regions).collect()),
        Ty::Str => Ty::Str,
        Ty::TaggedUnion(tags_tys) => Ty::TaggedUnion(
            tags_tys
                .iter()
                .map(|(tag, ty)| (*tag, erase_type_regions(ty)))
                .collect(),
        ),
        Ty::Tuple(tys) => Ty::Tuple(tys.iter().map(erase_type_regions).collect()),
        Ty::Unit => Ty::Unit,
        Ty::Variadic => Ty::Variadic,
    }
}
