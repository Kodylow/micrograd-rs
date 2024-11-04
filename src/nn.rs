use crate::engine::Value;
use rand::Rng;

/// Base trait for neural network modules
pub trait Module {
    fn parameters(&self) -> Vec<Value>;

    fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.set_grad(0.0);
        }
    }
}

/// Single neuron with weights, bias, and optional nonlinearity
pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            w: (0..nin)
                .map(|i| Value::new(rng.gen_range(-1.0..1.0), None, format!("w{}", i), None))
                .collect(),
            b: Value::new(0.0, None, "b".to_string(), None),
            nonlin,
        }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
        let mut act = self.b.clone();
        for (wi, xi) in self.w.iter().zip(x.iter()) {
            act = &act + &(wi * xi);
        }
        if self.nonlin {
            act.relu()
        } else {
            act
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

/// Layer of neurons
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Self {
        Self {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect(),
        }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

/// Multi-layer perceptron
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut sizes = vec![nin];
        sizes.extend_from_slice(nouts);

        let layers = (0..nouts.len())
            .map(|i| {
                Layer::new(
                    sizes[i],
                    sizes[i + 1],
                    i != nouts.len() - 1, // nonlin=false for last layer
                )
            })
            .collect();

        Self { layers }
    }

    pub fn forward(&self, mut x: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    pub fn update_weights(&mut self, learning_rate: f64) {
        for layer in &mut self.layers {
            for neuron in &mut layer.neurons {
                // Update weights
                for w in &mut neuron.w {
                    let grad = w.grad();
                    w.set_data(w.data() - learning_rate * grad);
                }
                // Update bias
                let grad = neuron.b.grad();
                neuron.b.set_data(neuron.b.data() - learning_rate * grad);
            }
        }
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
