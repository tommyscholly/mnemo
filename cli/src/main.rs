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
    let (module, ctx) = match do_frontend(&options.file) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    module.visualize(0);

    codegen(CodegenBackend::LLVM, module, ctx).unwrap();
}
