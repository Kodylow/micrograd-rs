use crate::Value;

impl Value {
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

        result.push_str(&format!(
            "{}{} {}\n",
            prefix,
            format!("[{:.4}, {:.4}]", self.data(), self.grad()),
            self.label()
        ));

        if !self.prev().is_empty() {
            let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });
            result.push_str(&format!("{}└─ {}\n", new_prefix, self.op()));

            let child_prefix = format!("{}    ", new_prefix);
            for (i, child) in self.prev().iter().enumerate() {
                let connector = if i == self.prev().len() - 1 {
                    "└──"
                } else {
                    "├──"
                };
                child.draw_ascii_recursive(
                    result,
                    visited,
                    &format!("{}{}", child_prefix, connector),
                    i == self.prev().len() - 1,
                );
            }
        }
    }
}
