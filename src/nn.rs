use std::{
    borrow::Borrow,
    hash::{Hash, Hasher},
    ops::{Add, Div, Mul, Sub},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    pub(crate) data: f64,
    pub(crate) prev: Vec<Value>,
    pub(crate) op: String,
    pub(crate) label: String,
    pub(crate) grad: f64,
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.prev.hash(state);
        self.op.hash(state);
    }
}

impl Value {
    pub fn new(data: f64, children: Option<Vec<Value>>, label: String, op: Option<String>) -> Self {
        Self {
            data,
            prev: children.unwrap_or_default(),
            op: op.unwrap_or_default(),
            label,
            grad: 0.0,
        }
    }

    pub fn set_label(&mut self, label: String) {
        self.label = label;
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
        Value::new(
            data,
            Some(vec![left.clone(), right.clone()]),
            format!("({}_{}_{}", left.label, op, right.label),
            Some(op.to_string()),
        )
    }

    pub fn tanh(&self) -> Value {
        let x = self.data;
        let exp_pos = x.exp();
        let exp_neg = (-x).exp();
        let t = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        Value::new(
            t,
            Some(vec![self.clone()]),
            format!("tanh({})", self.label),
            Some("tanh".to_string()),
        )
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

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Value(data={}, prev={:?}, op={}, label={})",
            self.data, self.prev, self.op, self.label
        )
    }
}
