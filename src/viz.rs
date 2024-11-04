use crate::Value;
use colored::*;
use std::collections::HashSet;

pub struct BackpropViz {
    pub active_nodes: HashSet<usize>,
    pub completed_nodes: HashSet<usize>,
}

impl BackpropViz {
    pub fn new() -> Self {
        Self {
            active_nodes: HashSet::new(),
            completed_nodes: HashSet::new(),
        }
    }

    pub fn draw_step(&self, value: &Value, step_desc: &str) {
        println!("\n{}", "Current Operation:".bright_blue().bold());
        println!("{}", step_desc);
        println!("\n{}", "Computation Graph:".bright_green().bold());
        self.draw_graph(value);
        println!("\nPress Enter to continue...");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
    }

    fn draw_graph(&self, value: &Value) {
        let mut result = String::new();
        let mut visited = HashSet::new();
        self.draw_node_recursive(value, &mut result, &mut visited, "", true);
        println!("{}", result);
    }

    fn draw_node_recursive(
        &self,
        value: &Value,
        result: &mut String,
        visited: &mut HashSet<usize>,
        prefix: &str,
        is_last: bool,
    ) {
        let ptr = value as *const Value as usize;
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        let node_str = format!(
            "{}{} {} [data={:.4}, grad={:.4}]",
            prefix,
            if is_last { "└─" } else { "├─" },
            value.label,
            value.data,
            value.grad
        );

        let colored_str = if self.active_nodes.contains(&ptr) {
            node_str.bright_yellow().bold().to_string()
        } else if self.completed_nodes.contains(&ptr) {
            node_str.bright_green().to_string()
        } else {
            node_str.normal().to_string()
        };

        result.push_str(&format!("{}\n", colored_str));

        if !value.prev.is_empty() {
            let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

            for (i, child) in value.prev.iter().enumerate() {
                self.draw_node_recursive(
                    child,
                    result,
                    visited,
                    &new_prefix,
                    i == value.prev.len() - 1,
                );
            }
        }
    }
}
