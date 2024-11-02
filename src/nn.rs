use std::hash::{Hash, Hasher};

#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    data: f64,
    prev: Vec<Value>,
    op: String,
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let bits = self.data.to_bits();
        bits.hash(state);
        self.prev.hash(state);
        self.op.hash(state);
    }
}

impl Value {
    pub fn new(data: f64, children: Option<Vec<Value>>, op: Option<String>) -> Self {
        Self {
            data,
            prev: children.unwrap_or(vec![]),
            op: op.unwrap_or("".to_string()),
        }
    }

    pub fn draw_ascii(&self) -> String {
        let mut result = String::new();
        let mut visited = std::collections::HashSet::new();
        self.draw_ascii_recursive(&mut result, &mut visited, 0);
        result
    }

    fn draw_ascii_recursive(
        &self,
        result: &mut String,
        visited: &mut std::collections::HashSet<usize>,
        depth: usize,
    ) {
        let ptr = self as *const Value as usize;
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        let indent = "      ".repeat(depth);

        // Draw children first
        if !self.prev.is_empty() {
            for child in &self.prev {
                child.draw_ascii_recursive(result, visited, depth + 1);
            }

            // After children, draw the operation
            result.push_str(&indent);
            result.push_str(&format!("  {}\n", self.op));
        }

        // Draw current node
        result.push_str(&indent);
        result.push_str(&format!("{:.4}\n", self.data));
    }

    #[allow(dead_code)]
    pub fn render_ascii(&self, output_file: &str) -> std::io::Result<()> {
        let ascii = self.draw_ascii();
        std::fs::write(output_file, ascii)?;
        Ok(())
    }
}

// Implement operator overloading for +
impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        Value::new(
            self.data + rhs.data,
            Some(vec![self, rhs]),
            Some("+".to_string()),
        )
    }
}

// Implement operator overloading for *
impl std::ops::Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        Value::new(
            self.data * rhs.data,
            Some(vec![self, rhs]),
            Some("*".to_string()),
        )
    }
}

// Implement operator overloading for -
impl std::ops::Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        Value::new(
            self.data - rhs.data,
            Some(vec![self, rhs]),
            Some("-".to_string()),
        )
    }
}

// Implement operator overloading for /
impl std::ops::Div for Value {
    type Output = Value;

    fn div(self, rhs: Value) -> Value {
        Value::new(
            self.data / rhs.data,
            Some(vec![self, rhs]),
            Some("/".to_string()),
        )
    }
}

// Implement Display trait for pretty printing
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Value(data={}, prev={:?}, op={})",
            self.data, self.prev, self.op
        )
    }
}
