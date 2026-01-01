use clap::Parser;
use codegen::{CodegenBackend, codegen};
use frontend::do_frontend;
use frontend::visualize::MIRVisualizer;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Options {
    file: String,
    #[arg(short, long, default_value = "LLVM")]
    backend: CodegenBackend,
    #[arg(short, long)]
    verbose: bool,
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

    if options.verbose {
        module.visualize(0);
    }

    codegen(CodegenBackend::LLVM, module, ctx, options.verbose).unwrap();
}
