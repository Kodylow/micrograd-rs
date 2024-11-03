use std::{
    borrow::Borrow,
    hash::{Hash, Hasher},
    ops::{Add, Div, Mul, Sub},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    data: f64,
    prev: Vec<Value>,
    op: String,
    label: String,
    grad: f64,
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

    pub fn draw_ascii(&self) -> String {
        let mut result = String::new();
        let mut visited = std::collections::HashSet::new();
        self.draw_ascii_recursive(&mut result, &mut visited, "", true);
        result
    }

    fn draw_ascii_recursive(
        &self,
        result: &mut String,
        visited: &mut std::collections::HashSet<usize>,
        prefix: &str,
        is_last: bool,
    ) {
        let ptr = self as *const Value as usize;
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        // Draw current node
        result.push_str(&format!(
            "{}{} {}\n",
            prefix,
            format!("[{:.4}, {:.4}]", self.data, self.grad),
            self.label
        ));

        if !self.prev.is_empty() {
            let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

            // Only draw one connector with the operation
            result.push_str(&format!("{}└─ {}\n", new_prefix, self.op));

            // Draw children
            let child_prefix = format!("{}    ", new_prefix);
            for (i, child) in self.prev.iter().enumerate() {
                let connector = if i == self.prev.len() - 1 {
                    "└──"
                } else {
                    "├──"
                };
                child.draw_ascii_recursive(
                    result,
                    visited,
                    &format!("{}{}", child_prefix, connector),
                    i == self.prev.len() - 1,
                );
            }
        }
    }

    #[allow(dead_code)]
    pub fn render_ascii(&self, output_file: &str) -> std::io::Result<()> {
        std::fs::write(output_file, self.draw_ascii())
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
