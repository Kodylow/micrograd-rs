#[derive(Clone, Debug)]
pub struct Value {
    data: f64,
    prev: Vec<Value>,
    op: String,
}

impl Value {
    pub fn new(data: f64, children: Option<Vec<Value>>, op: Option<String>) -> Self {
        Self {
            data,
            prev: children.unwrap_or(vec![]),
            op: op.unwrap_or("".to_string()),
        }
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
