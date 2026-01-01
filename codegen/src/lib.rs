mod llvm;

use std::process::Command;

use anyhow::Result;
use clap::ValueEnum;
use frontend::{Ctx, mir::Module};

#[derive(ValueEnum, Clone, Debug)]
pub enum CodegenBackend {
    SelfHost,
    LLVM,
    Cranelift,
}

trait Compiler {
    fn compile(self, module: Module, ctx: Ctx, debug: bool) -> Result<()>;
}

pub fn codegen(backend: CodegenBackend, module: Module, ctx: Ctx, debug: bool) -> Result<()> {
    match backend {
        CodegenBackend::SelfHost => todo!(),
        CodegenBackend::LLVM => llvm::compile(module, ctx, debug)?,
        CodegenBackend::Cranelift => todo!(),
    }

    let _ = Command::new("cc").arg("a.o").arg("lib.c").status()?;

    Ok(())
}
