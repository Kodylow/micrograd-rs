//! This module implements a simple autograd engine for training neural networks.
//! It provides automatic differentiation through a dynamically built computation graph,
//! enabling backpropagation for gradient-based optimization.

use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

use crate::viz::BackpropViz;

/// A node in the computation graph that tracks both forward computation and gradients for backprop.
/// Each Value represents a scalar value and its gradient with respect to some loss function.
#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInternal>>);

struct ValueInternal {
    data: f64,
    grad: f64,
    prev: Vec<Value>,
    op: String,
    pub label: String,
    backward_fn: Option<BackwardFn>,
}

type BackwardFn = Box<dyn Fn(&ValueInternal) + 'static>;

impl Value {
    /// Creates a new Value node in the computation graph.
    ///
    /// # Arguments
    /// * `data` - The scalar value to store
    /// * `children` - Optional input Values used to compute this Value
    /// * `label` - Human-readable name for debugging
    /// * `op` - Optional operation that produced this Value
    pub fn new(data: f64, children: Option<Vec<Value>>, label: String, op: Option<String>) -> Self {
        Value(Rc::new(RefCell::new(ValueInternal {
            data,
            grad: 0.0,
            prev: children.unwrap_or_default(),
            op: op.unwrap_or_default(),
            label,
            backward_fn: None,
        })))
    }

    /// Returns the scalar value stored in this node
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    /// Sets the scalar value stored in this node
    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    /// Returns the label of this node
    pub fn label(&self) -> String {
        self.0.borrow().label.clone()
    }

    /// Returns the operation of this node
    pub fn op(&self) -> String {
        self.0.borrow().op.clone()
    }

    /// Returns the gradient of this node
    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    /// Sets the gradient of this node
    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    /// Returns the previous nodes of this node
    pub fn prev(&self) -> Vec<Value> {
        self.0.borrow().prev.clone()
    }

    /// Updates the node's label
    pub fn set_label(&self, label: String) {
        self.0.borrow_mut().label = label;
    }

    /// Initiates backpropagation from this node.
    /// This computes ∂self/∂x for all nodes x in the graph.
    pub fn backward(&self) {
        self.0.borrow_mut().grad = 1.0;
        let mut visited = HashSet::new();
        self.backward_internal(&mut visited);
    }

    /// Internal implementation of backprop that includes visualization.
    /// Uses the chain rule to propagate gradients backward through the graph:
    /// If y = f(x) and x = g(w), then ∂L/∂w = (∂L/∂y)(∂y/∂x)(∂x/∂w)
    fn backward_internal(&self, visited: &mut HashSet<usize>) {
        let ptr = Rc::as_ptr(&self.0) as usize;
        if visited.insert(ptr) {
            let internal = self.0.borrow();
            if let Some(ref backward_fn) = internal.backward_fn {
                backward_fn(&internal);
            }

            // Clone prev to avoid borrow issues
            let prev = internal.prev.clone();
            drop(internal);

            for child in prev {
                child.backward_internal(visited);
            }
        }
    }

    /// Implements binary operations (+, -, *, /) between Values.
    /// Each operation stores its inputs and a closure for computing gradients.
    fn binary_op(left: &Value, right: &Value, op: &str) -> Value {
        let data = match op {
            "+" => left.data() + right.data(),
            "*" => left.data() * right.data(),
            "-" => left.data() - right.data(),
            "/" => left.data() / right.data(),
            _ => unreachable!(),
        };

        let out = Value::new(
            data,
            Some(vec![left.clone(), right.clone()]),
            format!(
                "({}_{}_{}",
                left.0.borrow().label,
                op,
                right.0.borrow().label
            ),
            Some(op.to_string()),
        );

        // Set backward functions with proper closure captures
        let backward_fn: Option<BackwardFn> = match op {
            "+" => Some(Box::new(move |out| {
                let g = out.grad;
                out.prev[0].0.borrow_mut().grad += g;
                out.prev[1].0.borrow_mut().grad += g;
            })),
            "*" => Some(Box::new(move |out| {
                let g = out.grad;
                out.prev[0].0.borrow_mut().grad += out.prev[1].data() * g;
                out.prev[1].0.borrow_mut().grad += out.prev[0].data() * g;
            })),
            "-" => Some(Box::new(move |out| {
                let g = out.grad;
                out.prev[0].0.borrow_mut().grad += g;
                out.prev[1].0.borrow_mut().grad -= g;
            })),
            "/" => Some(Box::new(move |out| {
                let g = out.grad;
                out.prev[0].0.borrow_mut().grad += g * (1.0 / out.prev[1].data());
                out.prev[1].0.borrow_mut().grad +=
                    out.grad * (-out.prev[0].data() / out.prev[1].data().powi(2));
            })),
            _ => None,
        };

        out.0.borrow_mut().backward_fn = backward_fn;
        out
    }

    /// Implements hyperbolic tangent activation function.
    /// tanh(x) = (e^x - e^-x)/(e^x + e^-x)
    pub fn tanh(&self) -> Value {
        let x = self.data();
        let exp_pos = x.exp();
        let exp_neg = (-x).exp();
        let t = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        let out = Value::new(
            t,
            Some(vec![self.clone()]),
            format!("tanh({})", self.0.borrow().label),
            Some("tanh".to_string()),
        );

        out.0.borrow_mut().backward_fn = Some(Box::new(move |out| {
            // For tanh(x), the derivative is:
            // ∂tanh(x)/∂x = 1 - tanh²(x)
            let t = out.data; // t is already the tanh result
            out.prev[0].0.borrow_mut().grad += (1.0 - t * t) * out.grad;
        }));
        out
    }

    /// Builds a topologically sorted list of all nodes in the graph.
    /// This ensures that when we process nodes, all dependencies are handled first.
    pub fn build_topo(&self) -> Vec<Value> {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo_recursive(v: &Value, topo: &mut Vec<Value>, visited: &mut HashSet<usize>) {
            let ptr = Rc::as_ptr(&v.0) as usize;
            if !visited.contains(&ptr) {
                visited.insert(ptr);

                // Visit all dependencies first
                for child in &v.0.borrow().prev {
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
        let out = Value::new(
            self.data().powf(exponent),
            Some(vec![self.clone()]),
            format!("{}^{}", self.0.borrow().label, exponent),
            Some("pow".to_string()),
        );

        out.0.borrow_mut().backward_fn = Some(Box::new(move |out| {
            // Power rule: ∂(x^n)/∂x = n * x^(n-1)
            out.prev[0].0.borrow_mut().grad +=
                exponent * out.prev[0].data().powf(exponent - 1.0) * out.grad;
        }));
        out
    }

    /// Implements ReLU (Rectified Linear Unit) activation function.
    /// ReLU(x) = max(0, x)
    pub fn relu(&self) -> Value {
        let out = Value::new(
            if self.data() > 0.0 { self.data() } else { 0.0 },
            Some(vec![self.clone()]),
            format!("relu({})", self.0.borrow().label),
            Some("relu".to_string()),
        );

        out.0.borrow_mut().backward_fn = Some(Box::new(move |out| {
            // ReLU derivative:
            // ∂ReLU(x)/∂x = 1 if x > 0, else 0
            out.prev[0].0.borrow_mut().grad += if out.data > 0.0 { out.grad } else { 0.0 };
        }));
        out
    }

    /// Initiates backpropagation from this node with visualization.
    /// This computes ∂self/∂x for all nodes x in the graph.
    pub fn backward_with_viz(&self, viz: &mut BackpropViz) {
        self.0.borrow_mut().grad = 1.0;
        let mut visited = HashSet::new();
        self.backward_internal_with_viz(&mut visited, viz);
    }

    /// Internal implementation of backprop that includes visualization.
    /// Uses the chain rule to propagate gradients backward through the graph:
    /// If y = f(x) and x = g(w), then ∂L/∂w = (∂L/∂y)(∂y/∂x)(∂x/∂w)
    fn backward_internal_with_viz(&self, visited: &mut HashSet<usize>, viz: &mut BackpropViz) {
        let ptr = Rc::as_ptr(&self.0) as usize;
        if visited.insert(ptr) {
            viz.active_nodes.insert(ptr);
            let desc = format!(
                "Computing gradient for node '{}'\n\
                Current value: {:.4}\n\
                Current gradient: {:.4}\n\
                Operation: {}",
                self.label(),
                self.data(),
                self.grad(),
                self.op()
            );
            viz.draw_step(self, &desc);

            let internal = self.0.borrow();
            if let Some(ref backward_fn) = internal.backward_fn {
                backward_fn(&internal);
            }

            let prev = internal.prev.clone();
            drop(internal);

            viz.completed_nodes.insert(ptr);
            viz.active_nodes.remove(&ptr);

            for child in prev {
                child.backward_internal_with_viz(visited, viz);
            }
        }
    }
}

// Implement ops traits
impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Value {
        Value::binary_op(self, rhs, "+")
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: &Value) -> Value {
        Value::binary_op(self, rhs, "-")
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: &Value) -> Value {
        Value::binary_op(self, rhs, "*")
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, rhs: &Value) -> Value {
        Value::binary_op(self, rhs, "/")
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data: {}, grad: {})", self.data(), self.grad())
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data: {}, grad: {})", self.data(), self.grad())
    }
}
