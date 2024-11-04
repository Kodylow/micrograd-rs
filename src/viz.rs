use crate::Value;
use colored::*;
use std::collections::HashSet;

use anyhow::Result;
use csv::Reader;
use plotters::prelude::*;
use std::fs::File;

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
            value.label(),
            value.data(),
            value.grad()
        );

        let colored_str = if self.active_nodes.contains(&ptr) {
            node_str.bright_yellow().bold().to_string()
        } else if self.completed_nodes.contains(&ptr) {
            node_str.bright_green().to_string()
        } else {
            node_str.normal().to_string()
        };

        result.push_str(&format!("{}\n", colored_str));

        if !value.prev().is_empty() {
            let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

            for (i, child) in value.prev().iter().enumerate() {
                self.draw_node_recursive(
                    child,
                    result,
                    visited,
                    &new_prefix,
                    i == value.prev().len() - 1,
                );
            }
        }
    }
}

pub fn plot_losses(losses: &[f64], filename: &str) -> Result<()> {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0f64..losses.len() as f64,
            0f64..losses.iter().copied().fold(0. / 0., f64::max),
        )?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        losses.iter().enumerate().map(|(i, &v)| (i as f64, v)),
        &RED,
    ))?;

    Ok(())
}

pub fn load_training_data(filename: &str) -> Result<Vec<(Vec<Value>, Value)>> {
    let file = File::open(filename)?;
    let mut reader = Reader::from_reader(file);
    let mut training_data = Vec::new();

    for result in reader.records() {
        let record = result?;
        let x0: f64 = record[0].parse()?;
        let x1: f64 = record[1].parse()?;
        let target: f64 = record[2].parse()?;

        training_data.push((
            vec![
                Value::new(x0, None, "x0".to_string(), None),
                Value::new(x1, None, "x1".to_string(), None),
            ],
            Value::new(target, None, "y".to_string(), None),
        ));
    }

    Ok(training_data)
}
