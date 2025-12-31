use clap::Parser;
use codegen::{CodegenBackend, codegen};
use frontend::do_frontend;
use frontend::visualize::MIRVisualizer;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Options {
    file: String,
}

fn main() {
    let options = Options::parse();
    let (module, ctx) = do_frontend(&options.file);

    module.visualize(0);

    codegen(CodegenBackend::LLVM, module, ctx).unwrap();
}
