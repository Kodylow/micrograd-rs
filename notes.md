# Neural Networks & Derivatives Notes

Based on Andrej Karpathy's micrograd & neural nets course

## Key Concepts

### Derivatives & Backpropagation

- Derivative = slope/rate of change at a point

  ```rust
  // Calculate derivative as slope using small change in x
  let h = 0.0001; // Small delta

  // Initial values
  let x = 2.0;
  let y = -3.0;
  let c = 10.0;

  // Calculate f(x) at two nearby points
  let f1 = x * y + c;
  let f2 = x * (y + h) + c;

  // Slope = change in y / change in x
  let derivative = (f2 - f1) / h;
  println!("Derivative (slope) at x={}: {}", x, derivative);
  ```

- Chain rule: derivative of composite functions multiplies derivatives of each part
- Backpropagation = applying chain rule backwards through computation graph
- Each node in graph stores:
  - Forward pass value
  - Gradient (derivative) for backward pass
  - References to input nodes

### Neural Network Basics

- Neurons = units that:
  1. Take weighted inputs
  2. Sum them
  3. Apply activation function (like tanh)
- Layers = groups of neurons
- Forward pass = computing outputs
- Training loop:
  1. Forward pass
  2. Compute loss
  3. Backward pass (backprop)
  4. Update weights

### Implementation Details

- Value object tracks:
  - Data (number)
  - Gradient
  - Operation that created it
  - Backward function
- Neural net components:
  - Neurons: weighted sum + nonlinearity
  - Layers: groups of neurons
  - MLP: stack of layers

### Training Process

1. Initialize weights randomly
2. Forward pass through network
3. Compare output to desired (loss)
4. Backprop to get gradients
5. Update weights (w -= learning_rate \* w.grad)
6. Repeat

### Key Functions

- Forward pass:
  ```rust
  out = w * x + b  // Linear
  out = tanh(out)  // Nonlinear
  ```
- Backward pass:
  - Recursively apply chain rule
  - Accumulate gradients at each node

## Implementation Notes for Rust

- Need graph structure for backprop
- Use Rc/RefCell for mutable sharing
- Consider using traits for operations
- Track dependencies for proper cleanup
