//! A Rust implementation of Andrej Karpathy's micrograd - a tiny autograd engine
//! that implements backpropagation (reverse-mode automatic differentiation) over
//! a dynamically built DAG. This allows for training neural networks with a
//! minimal yet feature-complete implementation.

use anyhow::Result;
use nn::Value;

mod engine;
mod nn;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let a = Value::new(2.0, None, None);
    let b = Value::new(3.0, None, None);
    let c = a.clone() + b.clone();
    let d = Value::new(4.0, None, None);
    let e = c.clone() * d.clone();

    // Print the computation graph
    println!("{}", e.draw_ascii());

    // Or save to file
    // c.render_ascii("graph.txt").unwrap();

    Ok(())
}
