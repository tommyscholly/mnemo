use clap::Parser;
use frontend::do_frontend;
use frontend::visualize::MIRVisualizer;

#[derive(Parser, Debug)]
#[command(author, version)]
struct Options {
    file: String,
}

fn main() {
    let options = Options::parse();
    let module = do_frontend(&options.file);

    module.visualize(0);
}
