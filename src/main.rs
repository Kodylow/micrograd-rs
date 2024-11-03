//! A Rust implementation of Andrej Karpathy's micrograd - a tiny autograd engine
//! that implements backpropagation (reverse-mode automatic differentiation) over
//! a dynamically built DAG. This allows for training neural networks with a
//! minimal yet feature-complete implementation.

use anyhow::Result;
use nn::Value;

mod draw;
mod engine;
mod nn;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // inputs x1, x2
    let x1 = Value::new(2.0, None, "x1".to_string(), None);
    let x2 = Value::new(0.0, None, "x2".to_string(), None);

    // weights w1, w2
    let w1 = Value::new(-3.0, None, "w1".to_string(), None);
    let w2 = Value::new(1.0, None, "w2".to_string(), None);

    // bias of the neuron
    let b = Value::new(6.7, None, "b".to_string(), None);

    // compute neuron activation
    let mut x1w1 = &x1 * &w1;
    x1w1.set_label("x1*w1".to_string());
    let mut x2w2 = &x2 * &w2;
    x2w2.set_label("x2*w2".to_string());

    let mut x1w1x2w2 = &x1w1 + &x2w2;
    x1w1x2w2.set_label("x1w1 + x2w2".to_string());

    let mut n = &x1w1x2w2 + &b;
    n.set_label("n".to_string());

    let mut o = n.tanh();
    o.set_label("o".to_string());

    // Print the computation graph
    println!("{}", o.draw_ascii());

    Ok(())
}
