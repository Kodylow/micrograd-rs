//! A Rust implementation of Andrej Karpathy's micrograd - a tiny autograd engine
//! that implements backpropagation (reverse-mode automatic differentiation) over
//! a dynamically built DAG. This allows for training neural networks with a
//! minimal yet feature-complete implementation.

use anyhow::Result;
use engine::Value;
use plotters::prelude::*;

use crate::nn::Module;

mod draw;
mod engine;
mod nn;
mod viz;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    run_values_example()?;
    run_nn_example()?;
    Ok(())
}

fn run_values_example() -> Result<()> {
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

    // Backpropagate the gradient of o with respect to n
    // Set the gradient of o to 1.0 as the base case
    o.set_grad(1.0);
    o.backward();

    // Print the computation graph after backprop
    println!("After backprop:");
    println!("{}", o.draw_ascii());
    Ok(())
}

fn run_nn_example() -> Result<()> {
    // Create a simple dataset: XOR problem
    let xs = vec![
        (
            vec![
                Value::new(0.0, None, "x0".to_string(), None),
                Value::new(0.0, None, "x1".to_string(), None),
            ],
            Value::new(0.0, None, "y".to_string(), None),
        ),
        (
            vec![
                Value::new(0.0, None, "x0".to_string(), None),
                Value::new(1.0, None, "x1".to_string(), None),
            ],
            Value::new(1.0, None, "y".to_string(), None),
        ),
        (
            vec![
                Value::new(1.0, None, "x0".to_string(), None),
                Value::new(0.0, None, "x1".to_string(), None),
            ],
            Value::new(1.0, None, "y".to_string(), None),
        ),
        (
            vec![
                Value::new(1.0, None, "x0".to_string(), None),
                Value::new(1.0, None, "x1".to_string(), None),
            ],
            Value::new(0.0, None, "y".to_string(), None),
        ),
    ];

    // Create a 2-layer neural network (2->4->1)
    let mut model = nn::MLP::new(2, &vec![4, 1]);
    let mut losses: Vec<f64> = Vec::new();

    // Training loop
    for epoch in 0..100 {
        let mut epoch_loss = 0.0;

        for (x, y) in &xs {
            // Forward pass
            let pred = model.forward(x.to_vec())[0].clone();

            // Calculate loss (MSE)
            let loss = (&pred - y).pow(2.0);
            epoch_loss += loss.data();

            // Backward pass
            model.zero_grad();
            loss.backward();

            // Update weights (SGD)
            model.update_weights(0.1);
        }

        epoch_loss /= xs.len() as f64;
        losses.push(epoch_loss);

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, epoch_loss);
        }
    }

    // Plot the loss curve
    plot_losses(&losses, "training_loss.png")?;

    // Test the model
    for (x, y) in &xs {
        let pred = model.forward(x.to_vec())[0].clone();
        println!(
            "Input: {:?}, Target: {}, Prediction: {:.4}",
            x,
            y,
            pred.data()
        );
    }

    Ok(())
}

fn plot_losses(losses: &[f64], filename: &str) -> Result<()> {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0f64..losses.len() as f64,
            0f64..losses.iter().copied().fold(0. / 0., f64::max),
        )?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        losses.iter().enumerate().map(|(i, &v)| (i as f64, v)),
        &RED,
    ))?;

    Ok(())
}
