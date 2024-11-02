use std::collections::HashMap;
use std::collections::HashSet;
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

    pub fn trace(&self) -> (HashSet<&Value>, HashSet<(&Value, &Value)>) {
        let mut nodes = HashSet::new();
        let mut edges = HashSet::new();

        fn build<'a>(
            v: &'a Value,
            nodes: &mut HashSet<&'a Value>,
            edges: &mut HashSet<(&'a Value, &'a Value)>,
        ) {
            if !nodes.contains(&v) {
                nodes.insert(v);
                for child in &v.prev {
                    edges.insert((child, v));
                    build(child, nodes, edges);
                }
            }
        }

        build(self, &mut nodes, &mut edges);
        (nodes, edges)
    }

    pub fn draw_ascii(&self) -> String {
        let (nodes, _) = self.trace();
        let mut result = String::new();

        // Create a mapping of nodes to their levels
        let mut levels: HashMap<*const Value, usize> = HashMap::new();

        // Helper function to determine node levels
        fn assign_levels(
            v: &Value,
            level: usize,
            levels: &mut HashMap<*const Value, usize>,
        ) -> usize {
            let ptr = v as *const Value;
            let current_level = levels.entry(ptr).or_insert(level);
            *current_level = (*current_level).max(level);

            let mut max_child_level = *current_level;
            for child in &v.prev {
                max_child_level = max_child_level.max(assign_levels(child, level + 1, levels));
            }
            max_child_level
        }

        let max_level = assign_levels(self, 0, &mut levels);

        // Sort nodes by levels
        let mut nodes_by_level: Vec<Vec<&Value>> = vec![Vec::new(); max_level + 1];
        for node in nodes {
            let level = levels[&(node as *const Value)];
            nodes_by_level[level].push(node);
        }

        // Only print operation nodes with their connections
        for level_nodes in nodes_by_level.iter() {
            for node in level_nodes {
                if !node.op.is_empty() {
                    let label = format!("{} ({:.4})", node.op, node.data);
                    result.push_str(&format!("{}\n", label));

                    for child in &node.prev {
                        result.push_str(&format!("└─> {:.4}\n", child.data));
                    }
                }
            }
        }

        result
    }

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
