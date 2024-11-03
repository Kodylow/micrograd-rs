# Neural Networks & Derivatives

_Andrej Karpathy's Neural Networks: Zero to Hero_

## Backpropagation Deep Dive

### 1. The Core Idea

Backpropagation = Chain rule applied backwards through a computation graph

### 2. Simple Example: z = x \* y

```rust
// Forward Pass
let x = 2.0;    // Input
let y = 3.0;    // Input
let z = x * y;  // z = 6.0

// Backward Pass
// dz/dx = y = 3.0    (derivative of z with respect to x is y)
// dz/dy = x = 2.0    (derivative of z with respect to y is x)

// If upstream gradient is 1.0:
x.grad = 1.0 * y  // = 3.0
y.grad = 1.0 * x  // = 2.0
```

### 3. Multi-Step Example

```
     a(2)    b(-3)
        \    /
         c(-6)     d(10)
             \    /
              L(4)
```

#### Forward Pass

```rust
let a = Value::new(2.0);   // a = 2
let b = Value::new(-3.0);  // b = -3
let c = a * b;             // c = -6
let d = Value::new(10.0);  // d = 10
let L = c + d;             // L = 4
```

#### Backward Pass (Step by Step)

1. Start at L (output)

   ```rust
   // L = c + d
   // dL/dL = 1.0 (starting gradient)
   L.grad = 1.0
   ```

2. Addition Node (L = c + d)

   ```rust
   // Addition distributes gradient equally
   // dL/dc = 1.0
   // dL/dd = 1.0
   c.grad = L.grad * 1.0  // = 1.0
   d.grad = L.grad * 1.0  // = 1.0
   ```

3. Multiplication Node (c = a \* b)

   ```rust
   // Local derivatives:
   // dc/da = b = -3.0
   // dc/db = a = 2.0

   // Chain rule:
   a.grad = c.grad * b    // = 1.0 * (-3.0) = -3.0
   b.grad = c.grad * a    // = 1.0 * 2.0 = 2.0
   ```

### 4. Common Patterns

#### Addition (+)

```rust
// z = x + y
// dz/dx = 1.0
// dz/dy = 1.0

// Gradient flows through unchanged
x.grad = upstream_grad * 1.0
y.grad = upstream_grad * 1.0
```

#### Multiplication (\*)

```rust
// z = x * y
// dz/dx = y
// dz/dy = x

// Gradient is scaled by other input
x.grad = upstream_grad * y
y.grad = upstream_grad * x
```

#### Tanh (Common Activation)

```rust
// z = tanh(x)
// dz/dx = 1 - tanh²(x)

// Example:
let x = 2.0;
let z = x.tanh();
// If x = 2.0, tanh(2.0) ≈ 0.964
// dz/dx = 1 - 0.964² ≈ 0.071

x.grad = upstream_grad * (1.0 - z*z)
```

### 5. Visual Intuition

```
Forward Pass →
[Input] → [Hidden] → [Output]
   x    →    h     →    y

← Backward Pass
grad_x ← grad_h  ← grad_y=1.0
```

Each gradient represents:

- How much would the output change
- If we tweaked this value a tiny bit
- While keeping everything else constant

### 6. Key Insights

1. **Local Computation**

   - Each node only needs to know its immediate operation
   - Complex derivatives emerge from simple local rules

2. **Gradient Flow**

   - Gradients flow backward
   - Each step multiplies by local derivative
   - Products can get very small (vanishing gradient)
   - Products can get very large (exploding gradient)

3. **Practical Implementation**

```rust
struct Value {
    data: f64,
    grad: f64,
    backward: Box<dyn Fn()>,
    // Each Value stores its backward function
    // When called, updates gradients of inputs
}
```

### 7. Common Gotchas

- **Initialize gradients to zero** before backward pass
- **Accumulate gradients** when a value is used multiple times
- **Topological order** matters during backward pass
- **Clear gradients** between training steps
