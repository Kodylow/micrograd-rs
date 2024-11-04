//! This module implements a simple autograd engine for training neural networks.
//! It provides automatic differentiation through a dynamically built computation graph,
//! enabling backpropagation for gradient-based optimization.

use std::{
    borrow::Borrow,
    collections::HashSet,
    hash::{Hash, Hasher},
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

use crate::viz::BackpropViz;

/// A node in the computation graph that tracks both forward computation and gradients for backprop.
/// Each Value represents a scalar value and its gradient with respect to some loss function.
pub struct Value {
    /// The actual scalar value stored at this node
    pub(crate) data: f64,
    /// References to input Values that were used to compute this Value
    pub(crate) prev: Vec<Value>,
    /// The operation that produced this Value (e.g. "+", "*", "tanh")
    pub(crate) op: String,
    /// A human-readable label for debugging and visualization
    pub(crate) label: String,
    /// The gradient of the final output with respect to this Value (∂out/∂self)
    pub(crate) grad: f64,
    /// Function to compute gradients during backprop using the chain rule
    pub(crate) backward_fn: Option<Arc<dyn Fn(&mut Value) + Send + Sync>>,
}

impl Value {
    /// Creates a new Value node in the computation graph.
    ///
    /// # Arguments
    /// * `data` - The scalar value to store
    /// * `children` - Optional input Values used to compute this Value
    /// * `label` - Human-readable name for debugging
    /// * `op` - Optional operation that produced this Value
    pub fn new(data: f64, children: Option<Vec<Value>>, label: String, op: Option<String>) -> Self {
        Self {
            data,
            prev: children.unwrap_or_default(),
            op: op.unwrap_or_default(),
            label,
            grad: 0.0, // Gradients start at zero before backprop
            backward_fn: None,
        }
    }

    /// Updates the node's label
    pub fn set_label(&mut self, label: String) {
        self.label = label;
    }

    /// Sets the gradient for this node
    pub fn set_grad(&mut self, grad: f64) {
        self.grad = grad;
    }

    /// Initiates backpropagation from this node.
    /// This computes ∂self/∂x for all nodes x in the graph.
    pub fn backward(&mut self) {
        let mut viz = BackpropViz::new();
        self.backward_with_viz(&mut viz)
    }

    /// Internal implementation of backprop that includes visualization.
    /// Uses the chain rule to propagate gradients backward through the graph:
    /// If y = f(x) and x = g(w), then ∂L/∂w = (∂L/∂y)(∂y/∂x)(∂x/∂w)
    fn backward_with_viz(&mut self, viz: &mut BackpropViz) {
        let ptr = self as *const Value as usize;
        viz.active_nodes.insert(ptr);

        if let Some(backward_fn) = self.backward_fn.take() {
            let desc = format!(
                "Computing gradient for node '{}'\n\
                Current value: {:.4}\n\
                Current gradient: {:.4}\n\
                Operation: {}",
                self.label, self.data, self.grad, self.op
            );
            viz.draw_step(self, &desc);

            backward_fn(self);
            self.backward_fn = Some(backward_fn);

            viz.completed_nodes.insert(ptr);
            viz.active_nodes.remove(&ptr);
        }

        for child in &mut self.prev {
            child.backward_with_viz(viz);
        }
    }

    /// Implements binary operations (+, -, *, /) between Values.
    /// Each operation stores its inputs and a closure for computing gradients.
    fn binary_op(left: impl Borrow<Value>, right: impl Borrow<Value>, op: &str) -> Value {
        let left = left.borrow();
        let right = right.borrow();
        let data = match op {
            "+" => left.data + right.data,
            "*" => left.data * right.data,
            "-" => left.data - right.data,
            "/" => left.data / right.data,
            _ => unreachable!(),
        };

        let mut out = Value::new(
            data,
            Some(vec![left.clone(), right.clone()]),
            format!("({}_{}_{}", left.label, op, right.label),
            Some(op.to_string()),
        );

        // Each operation must implement its own gradient computation based on calculus rules
        out.backward_fn = match op {
            "+" => Some(Arc::new(|out: &mut Value| {
                // Addition: z = x + y
                // ∂z/∂x = 1, ∂z/∂y = 1
                out.prev[0].grad += out.grad; // ∂z/∂x = 1
                out.prev[1].grad += out.grad; // ∂z/∂y = 1
            })),
            "*" => Some(Arc::new(|out: &mut Value| {
                // Multiplication: z = x * y
                // ∂z/∂x = y, ∂z/∂y = x
                out.prev[0].grad += out.prev[1].data * out.grad; // ∂z/∂x = y
                out.prev[1].grad += out.prev[0].data * out.grad; // ∂z/∂y = x
            })),
            "-" => Some(Arc::new(|out: &mut Value| {
                // Subtraction: z = x - y
                // ∂z/∂x = 1, ∂z/∂y = -1
                out.prev[0].grad += out.grad; // ∂z/∂x = 1
                out.prev[1].grad -= out.grad; // ∂z/∂y = -1
            })),
            "/" => Some(Arc::new(|out: &mut Value| {
                // Division: z = x/y
                // ∂z/∂x = 1/y
                // ∂z/∂y = -x/y²
                out.prev[0].grad += out.grad * (1.0 / out.prev[1].data); // ∂z/∂x = 1/y
                out.prev[1].grad += out.grad * (-out.prev[0].data / out.prev[1].data.powi(2));
                // ∂z/∂y = -x/y²
            })),
            _ => None,
        };

        out
    }

    /// Implements hyperbolic tangent activation function.
    /// tanh(x) = (e^x - e^-x)/(e^x + e^-x)
    pub fn tanh(&self) -> Value {
        let x = self.data;
        let exp_pos = x.exp();
        let exp_neg = (-x).exp();
        let t = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        let mut out = Value::new(
            t,
            Some(vec![self.clone()]),
            format!("tanh({})", self.label),
            Some("tanh".to_string()),
        );

        out.backward_fn = Some(Arc::new(|out: &mut Value| {
            // For tanh(x), the derivative is:
            // ∂tanh(x)/∂x = 1 - tanh²(x)
            let t = out.data; // t is already the tanh result
            out.prev[0].grad += (1.0 - t * t) * out.grad;
        }));
        out
    }

    /// Builds a topologically sorted list of all nodes in the graph.
    /// This ensures that when we process nodes, all dependencies are handled first.
    pub fn build_topo(&self) -> Vec<Value> {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo_recursive(v: &Value, topo: &mut Vec<Value>, visited: &mut HashSet<usize>) {
            let ptr = v as *const Value as usize;
            if !visited.contains(&ptr) {
                visited.insert(ptr);

                // Visit all dependencies first
                for child in &v.prev {
                    build_topo_recursive(child, topo, visited);
                }

                // Then add this node
                topo.push(v.clone());
            }
        }

        build_topo_recursive(self, &mut topo, &mut visited);
        topo
    }

    /// Implements power function x^n.
    pub fn pow(&self, exponent: f64) -> Value {
        let mut out = Value::new(
            self.data.powf(exponent),
            Some(vec![self.clone()]),
            format!("{}^{}", self.label, exponent),
            Some("pow".to_string()),
        );

        out.backward_fn = Some(Arc::new(move |out: &mut Value| {
            // Power rule: ∂(x^n)/∂x = n * x^(n-1)
            out.prev[0].grad += exponent * out.prev[0].data.powf(exponent - 1.0) * out.grad;
        }));
        out
    }

    /// Implements ReLU (Rectified Linear Unit) activation function.
    /// ReLU(x) = max(0, x)
    pub fn relu(&self) -> Value {
        let mut out = Value::new(
            if self.data > 0.0 { self.data } else { 0.0 },
            Some(vec![self.clone()]),
            format!("relu({})", self.label),
            Some("relu".to_string()),
        );

        out.backward_fn = Some(Arc::new(|out: &mut Value| {
            // ReLU derivative:
            // ∂ReLU(x)/∂x = 1 if x > 0, else 0
            out.prev[0].grad += if out.data > 0.0 { out.grad } else { 0.0 };
        }));
        out
    }
}

/// Macro to implement binary operations for Value types
macro_rules! impl_binary_op {
    ($trait:ident, $fn:ident, $op:expr) => {
        impl $trait for Value {
            type Output = Value;
            fn $fn(self, rhs: Value) -> Value {
                Value::binary_op(self, rhs, $op)
            }
        }
        impl $trait for &Value {
            type Output = Value;
            fn $fn(self, rhs: &Value) -> Value {
                Value::binary_op(self, rhs, $op)
            }
        }
    };
}

impl_binary_op!(Add, add, "+");
impl_binary_op!(Mul, mul, "*");
impl_binary_op!(Sub, sub, "-");
impl_binary_op!(Div, div, "/");

// Standard trait implementations for Value type
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.prev == other.prev
            && self.op == other.op
            && self.label == other.label
            && self.grad == other.grad
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.prev.hash(state);
        self.op.hash(state);
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Value(data={}, prev={:?}, op={}, label={})",
            self.data, self.prev, self.op, self.label
        )
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.data)
            .field("prev", &self.prev)
            .field("op", &self.op)
            .field("label", &self.label)
            .field("grad", &self.grad)
            .finish()
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            prev: self.prev.clone(),
            op: self.op.clone(),
            label: self.label.clone(),
            grad: self.grad,
            backward_fn: self.backward_fn.clone(),
        }
    }
}

// Implementations for scalar operations with Value
impl Add<f64> for Value {
    type Output = Value;
    fn add(self, rhs: f64) -> Value {
        self + Value::new(rhs, None, rhs.to_string(), None)
    }
}

impl Add<Value> for f64 {
    type Output = Value;
    fn add(self, rhs: Value) -> Value {
        Value::new(self, None, self.to_string(), None) + rhs
    }
}
