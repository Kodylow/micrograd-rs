#[derive(Clone)]
pub struct Value {
    data: f64,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self { data }
    }
}

// Implement operator overloading for +
impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        Value::new(self.data + rhs.data)
    }
}

// Implement Display trait for pretty printing
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}
