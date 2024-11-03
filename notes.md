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

- Backpropagation explained:

  1. Forward pass computes values through network
  2. Backward pass computes gradients by:
     - Starting at output node with gradient 1.0
     - Working backwards through graph
     - At each node:
       - Compute local gradient (derivative of operation)
       - Multiply by upstream gradient (chain rule)
       - Pass gradients to input nodes

  Example with multiplication node:

  ```rust
  // Forward: z = x * y
  let x = 2.0;
  let y = 3.0;
  let z = x * y; // z = 6.0

  // Backward:
  // dz/dx = y = 3.0
  // dz/dy = x = 2.0
  // If upstream gradient is 1.0:
  // x.grad = 1.0 * dz/dx = 3.0
  // y.grad = 1.0 * dz/dy = 2.0
  ```

  Key points:

  - Local gradients depend on operation type:
    - Add: gradient = 1.0 for both inputs
    - Multiply: gradient = other input's value
    - Divide: more complex derivatives
  - Chain rule multiplies local gradient by upstream gradient
  - Gradients flow backwards and accumulate
  - Each node stores its gradient for optimization

  This allows automatic computation of derivatives through arbitrary computation graphs, enabling neural network training.

- Example backpropagation calculation:

  ```rust
  // Forward pass
  let a = Value::new(2.0, None, "a".to_string(), None);  // a = 2
  let b = Value::new(-3.0, None, "b".to_string(), None); // b = -3
  let c = a.clone() * b.clone();                         // c = a * b = -6
  let d = Value::new(10.0, None, "d".to_string(), None); // d = 10
  let L = c + d;                                         // L = c + d = 4

  // Computation graph:
  //       a(2)    b(-3)
  //          \    /
  //           c(-6)     d(10)
  //               \    /
  //                L(4)

  // Backward pass (chain rule):
  // 1. dL/dL = 1.0 (derivative of output wrt itself)

  // 2. Derivatives wrt direct inputs to L:
  //    dL/dd = 1.0 * 1.0 = 1.0  (through + operation)
  //    dL/dc = 1.0 * 1.0 = 1.0  (through + operation)

  // 3. Derivatives wrt inputs to c:
  //    dL/da = dL/dc * dc/da = 1.0 * b = -3.0
  //    dL/db = dL/dc * dc/db = 1.0 * a = 2.0

  // Verify with numerical approximation:
  let h = 0.0001;

  // Check dL/dd
  let L1 = (a.clone() * b.clone()) + Value::new(10.0, None, "d".to_string(), None);
  let L2 = (a.clone() * b.clone()) + Value::new(10.0 + h, None, "d".to_string(), None);
  let dL_dd = (L2.data - L1.data) / h;  // Should be ~1.0

  // Check dL/da
  let L1 = (Value::new(2.0, None, "a".to_string(), None) * b.clone()) + d.clone();
  let L2 = (Value::new(2.0 + h, None, "a".to_string(), None) * b.clone()) + d.clone();
  let dL_da = (L2.data - L1.data) / h;  // Should be ~-3.0
  ```

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
