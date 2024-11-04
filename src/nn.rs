use std::rc::Rc;

use crate::engine::{Value, ValueRef};
use rand::Rng;

/// Base trait for neural network modules
pub trait Module {
    fn parameters(&self) -> Vec<ValueRef>;

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.borrow_mut().grad = 0.0;
        }
    }
}

/// Single neuron with weights, bias, and optional nonlinearity
pub struct Neuron {
    w: Vec<ValueRef>,
    b: ValueRef,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::thread_rng();
        let w = (0..nin)
            .map(|i| Value::new(rng.gen_range(-1.0..1.0), None, format!("w{}", i), None))
            .collect();
        let b = Value::new(0.0, None, "b".to_string(), None);
        Self { w, b, nonlin }
    }

    pub fn forward(&self, x: &[ValueRef]) -> ValueRef {
        let act = self
            .w
            .iter()
            .zip(x.iter())
            .fold(Rc::clone(&self.b), |sum, (wi, xi)| {
                &sum + &(wi.borrow().clone() * xi.borrow().clone())
            });
        if self.nonlin {
            Value::relu(&act)
        } else {
            act
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<ValueRef> {
        let mut params = self.w.clone();
        params.push(Rc::clone(&self.b));
        params
    }
}

/// Layer of neurons
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin, nonlin)).collect();
        Self { neurons }
    }

    pub fn forward(&self, x: &[ValueRef]) -> Vec<ValueRef> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<ValueRef> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

/// Multi-layer perceptron
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let sizes = std::iter::once(&nin)
            .chain(nouts.iter())
            .collect::<Vec<_>>();
        let layers = sizes
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                Layer::new(
                    *w[0],
                    *w[1],
                    i != nouts.len() - 1, // Nonlinearity except for last layer
                )
            })
            .collect();
        Self { layers }
    }

    pub fn forward(&self, x: Vec<ValueRef>) -> Vec<ValueRef> {
        let mut output = x;
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn layer_outputs(&self, x: Vec<ValueRef>) -> Vec<Vec<ValueRef>> {
        let mut outputs = Vec::new();
        let mut current = x;
        outputs.push(current.clone());

        for layer in &self.layers {
            current = layer.forward(&current);
            outputs.push(current.clone());
        }

        outputs
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<ValueRef> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
