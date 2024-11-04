# micrograd-rs 🧠

A Rust implementation of Andrej Karpathy's micrograd - a tiny autograd engine that implements backpropagation over a dynamically built computation graph.

## Features 🚀

- Automatic differentiation with reverse-mode backpropagation
- Dynamic computation graph construction
- Basic neural network operations:
  - Binary ops (+, -, \*, /)
  - Activation functions (tanh, ReLU)
  - Power function
- Interactive visualization of backpropagation
- Neural network implementation with configurable layers
- Training visualization with loss plots

## Example Usage 💡

```rust
// Create computation nodes
let x1 = Value::new(2.0, None, "x1".to_string(), None);
let w1 = Value::new(-3.0, None, "w1".to_string(), None);
// Perform operations
let prod = &x1 &w1;
let act = prod.tanh();
// Backpropagate gradients
act.backward();
// Access gradients
println!("x1 gradient: {}", x1.grad());
```

## Neural Network Training 🎯

```rust
// Create a 2-layer neural network (2->4->1)
let mut model = nn::MLP::new(2, &vec![4, 1]);
// Training loop
for epoch in 0..100 {
  for (x, y) in train_data {
    let pred = model.forward(x)[0].clone();
    let loss = (&pred - y).pow(2.0);
    model.zero_grad();
    loss.backward();
    model.update_weights(0.1);
  }
}
```

## Visualization 📊

Run with visualization enabled:

```bash
cargo run val --visualize
```

This shows step-by-step backpropagation through the computation graph with color-coded nodes:

- 🟡 Active node being processed
- 🟢 Completed nodes
- ⚪ Unprocessed nodes

## Implementation Details 🔧

The core `Value` type wraps a computation node that tracks:

- Scalar value
- Gradient
- Operation history
- Backward function for gradient computation

References to code structure:
rust
/// A node in the computation graph that tracks both forward computation and gradients for backprop.
/// Each Value represents a scalar value and its gradient with respect to some loss function. #[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInternal>>);
struct ValueInternal {
data: f64,
grad: f64,
prev: Vec<Value>,
op: String,
pub label: String,
backward_fn: Option<BackwardFn>,
}

## Credits 🙏

Based on Andrej Karpathy's micrograd educational project, reimplemented in Rust with additional features and visualizations.

## License 📄

MIT
