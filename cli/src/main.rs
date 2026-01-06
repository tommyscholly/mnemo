use clap::Parser;
use codegen::{CodegenBackend, codegen};
use frontend::do_frontend;

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
}

fn main() {
    let options = Options::parse();
    let (module, ctx) = match do_frontend(&options.file, options.mir) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    codegen(CodegenBackend::LLVM, module, ctx, options.verbose).unwrap();
}
