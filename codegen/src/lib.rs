mod llvm;

use anyhow::Result;
use frontend::{Ctx, mir::Module};

pub enum CodegenBackend {
    SelfHost,
    LLVM,
    Cranelift,
}

trait Compiler {
    fn compile(self, module: Module, ctx: Ctx) -> Result<()>;
}

pub fn codegen(backend: CodegenBackend, module: Module, ctx: Ctx) -> Result<()> {
    match backend {
        CodegenBackend::SelfHost => todo!(),
        CodegenBackend::LLVM => llvm::compile(module, ctx),
        CodegenBackend::Cranelift => todo!(),
    }
}
