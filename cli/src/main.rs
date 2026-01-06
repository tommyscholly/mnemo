use clap::Parser;
use codegen::{CodegenBackend, codegen};
use frontend::{do_frontend, visualize::MIRVisualizer};

#[derive(Parser, Debug)]
#[command(author, version)]
struct Options {
    file: String,
    #[arg(short, long, default_value = "LLVM")]
    backend: CodegenBackend,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long)]
    mir: bool,
    #[arg(short, long)]
    graphviz: bool,
}

fn main() {
    let options = Options::parse();
    let (module, ctx) = match do_frontend(&options.file, options.graphviz) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    if options.mir {
        module.visualize(0);
    }

    codegen(CodegenBackend::LLVM, module, ctx, options.verbose).unwrap();
}
