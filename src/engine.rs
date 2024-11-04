use std::{
    borrow::Borrow,
    collections::HashSet,
    hash::{Hash, Hasher},
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

use crate::viz::BackpropViz;

pub struct Value {
    pub(crate) data: f64,
    pub(crate) prev: Vec<Value>,
    pub(crate) op: String,
    pub(crate) label: String,
    pub(crate) grad: f64,
    pub(crate) backward_fn: Option<Arc<dyn Fn(&mut Value) + Send + Sync>>,
}

impl Value {
    pub fn new(data: f64, children: Option<Vec<Value>>, label: String, op: Option<String>) -> Self {
        Self {
            data,
            prev: children.unwrap_or_default(),
            op: op.unwrap_or_default(),
            label,
            grad: 0.0,
            backward_fn: None,
        }
    }

    pub fn set_label(&mut self, label: String) {
        self.label = label;
    }

    pub fn set_grad(&mut self, grad: f64) {
        self.grad = grad;
    }

    pub fn backward(&mut self) {
        let mut viz = BackpropViz::new();
        self.backward_with_viz(&mut viz)
    }

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

        // Set the backprop function based on operation
        out.backward_fn = match op {
            "+" => Some(Arc::new(|out: &mut Value| {
                // For addition, gradient flows equally to both inputs
                // ∂(a+b)/∂a = 1
                out.prev[0].grad += out.grad;
                // ∂(a+b)/∂b = 1
                out.prev[1].grad += out.grad;
            })),
            "*" => Some(Arc::new(|out: &mut Value| {
                // For multiplication, gradient flows to each input scaled by the other input
                // ∂(a*b)/∂a = b
                out.prev[0].grad += out.prev[1].data * out.grad;
                // ∂(a*b)/∂b = a
                out.prev[1].grad += out.prev[0].data * out.grad;
            })),
            "-" => Some(Arc::new(|out: &mut Value| {
                // For subtraction, gradient flows to the first input and subtracts from the second
                // ∂(a-b)/∂a = 1
                out.prev[0].grad += out.grad;
                // ∂(a-b)/∂b = -1
                out.prev[1].grad -= out.grad;
            })),
            "/" => Some(Arc::new(|out: &mut Value| {
                // For division (a/b), gradient flows according to quotient rule
                // ∂(a/b)/∂a = 1/b
                out.prev[0].grad += out.grad * (1.0 / out.prev[1].data);
                // ∂(a/b)/∂b = -a/b²
                out.prev[1].grad += out.grad * (-out.prev[0].data / out.prev[1].data.powi(2));
            })),
            _ => None,
        };

        out
    }

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
            // For tanh(x), gradient is 1 - tanh(x)^2
            let t = out.data; // t is already the tanh result
            out.prev[0].grad += (1.0 - t * t) * out.grad;
        }));
        out
    }

    pub fn build_topo(&self) -> Vec<Value> {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo_recursive(v: &Value, topo: &mut Vec<Value>, visited: &mut HashSet<usize>) {
            let ptr = v as *const Value as usize;
            if !visited.contains(&ptr) {
                visited.insert(ptr);

                // Recursively visit all children first
                for child in &v.prev {
                    build_topo_recursive(child, topo, visited);
                }

                // Add current node after all children
                topo.push(v.clone());
            }
        }

        build_topo_recursive(self, &mut topo, &mut visited);
        topo
    }

    pub fn pow(&self, exponent: f64) -> Value {
        let mut out = Value::new(
            self.data.powf(exponent),
            Some(vec![self.clone()]),
            format!("{}^{}", self.label, exponent),
            Some("pow".to_string()),
        );

        out.backward_fn = Some(Arc::new(move |out: &mut Value| {
            // For power rule: ∂(x^n)/∂x = n * x^(n-1)
            out.prev[0].grad += exponent * out.prev[0].data.powf(exponent - 1.0) * out.grad;
        }));
        out
    }

    pub fn relu(&self) -> Value {
        let mut out = Value::new(
            if self.data > 0.0 { self.data } else { 0.0 },
            Some(vec![self.clone()]),
            format!("relu({})", self.label),
            Some("relu".to_string()),
        );

        out.backward_fn = Some(Arc::new(|out: &mut Value| {
            out.prev[0].grad += if out.data > 0.0 { out.grad } else { 0.0 };
        }));
        out
    }
}

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
