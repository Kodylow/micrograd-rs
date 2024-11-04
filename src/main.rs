//! A Rust implementation of Andrej Karpathy's micrograd - a tiny autograd engine
//! that implements backpropagation (reverse-mode automatic differentiation) over
//! a dynamically built DAG. This allows for training neural networks with a
//! minimal yet feature-complete implementation.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use viz::BackpropViz;

use engine::Value;
use rand::prelude::SliceRandom;

use rand::thread_rng;
use viz::load_training_data;

use crate::nn::Module;

mod draw;
mod engine;
mod nn;
mod viz;

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(value_enum)]
    mode: Mode,

    #[arg(short, long)]
    visualize: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Mode {
    Val,
    Nn,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    match args.mode {
        Mode::Val => run_values_example(args.visualize)?,
        Mode::Nn => run_nn_example()?,
    }
    Ok(())
}

fn run_values_example(visualize: bool) -> Result<()> {
    // inputs x1, x2
    let x1 = Value::new(2.0, None, "x1".to_string(), None);
    let x2 = Value::new(0.0, None, "x2".to_string(), None);

    // weights w1, w2
    let w1 = Value::new(-3.0, None, "w1".to_string(), None);
    let w2 = Value::new(1.0, None, "w2".to_string(), None);

    // bias of the neuron
    let b = Value::new(6.8813735870195432, None, "b".to_string(), None);

    // compute neuron activation
    let x1w1 = &x1 * &w1;
    x1w1.set_label("x1*w1".to_string());
    let x2w2 = &x2 * &w2;
    x2w2.set_label("x2*w2".to_string());

    let x1w1x2w2 = &x1w1 + &x2w2;
    x1w1x2w2.set_label("x1w1 + x2w2".to_string());

    let n = &x1w1x2w2 + &b;
    n.set_label("n".to_string());

    let o = n.tanh();
    o.set_label("o".to_string());
    // do/dn = 1 - tanh(n)^2 = 1 - o^2

    // Print the computation graph before backprop
    println!("Before backprop:");
    println!("{}", o.draw_ascii());

    if visualize {
        let mut viz = BackpropViz::new();
        o.set_grad(1.0);
        o.backward_with_viz(&mut viz);
    } else {
        o.set_grad(1.0);
        o.backward();
    }

    // Print the computation graph after backprop
    println!("After backprop:");
    println!("{}", o.draw_ascii());
    Ok(())
}

fn run_nn_example() -> Result<()> {
    // Load dataset from CSV
    let mut xs = load_training_data("xor_data.csv")?;

    // Shuffle the dataset
    xs.shuffle(&mut thread_rng());

    // Calculate split index (80% train, 20% test)
    let split_idx = (xs.len() as f64 * 0.8) as usize;
    let (train_data, test_data) = xs.split_at(split_idx);

    // Create a 2-layer neural network (2->4->1)
    let mut model = nn::MLP::new(2, &vec![4, 1]);
    let mut losses: Vec<f64> = Vec::new();

    // Training loop
    for epoch in 0..100 {
        let mut epoch_loss = 0.0;

        for (x, y) in train_data {
            let pred = model.forward(x.to_vec())[0].clone();
            let loss = (&pred - y).pow(2.0);
            epoch_loss += loss.data();
            model.zero_grad();
            loss.backward();
            model.update_weights(0.1);
        }

        epoch_loss /= train_data.len() as f64;
        losses.push(epoch_loss);

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, epoch_loss);
        }
    }

    viz::plot_losses(&losses, "training_loss.png")?;

    // Evaluate on test set
    println!("\n--- Test Set Evaluation ---");
    let mut test_correct = 0;
    let mut test_error = 0.0;
    for (x, y) in test_data {
        let pred = model.forward(x.to_vec())[0].clone();
        let error = (pred.data() - y.data()).abs();
        test_error += error;
        if error < 0.5 {
            test_correct += 1;
        }
        println!(
            "Input: ({:.1}, {:.1}), Target: {:.1}, Predicted: {:.1}",
            x[0].data(),
            x[1].data(),
            y.data(),
            pred.data()
        );
    }
    println!(
        "\nTest Accuracy: {:.1}%",
        (test_correct as f64 / test_data.len() as f64) * 100.0
    );
    println!(
        "Test Average Error: {:.4}",
        test_error / test_data.len() as f64
    );

    Ok(())
}
