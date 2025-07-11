# Neural Network Library in Go

A comprehensive neural network implementation in Go featuring back-propagation, multiple activation functions, and real-time training visualization.

## Features

- 🧠 **Multi-layer Neural Networks** with configurable architecture
- 🔄 **Back-propagation** with gradient descent optimization
- 🎯 **Multiple Activation Functions**: Sigmoid, ReLU, Tanh, Softmax
- 📊 **Real-time Training Visualization** in terminal
- 💾 **Model Persistence** - save and load trained models
- 📈 **Training Metrics** - loss tracking and performance monitoring
- 🎲 **Secure Random Initialization** using cryptographic randomness
- 🧪 **Comprehensive Testing** with unit tests and benchmarks
- 📚 **Well-documented API** with examples

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/neural-network-go
cd neural-network-go
go mod tidy
```

### Basic Usage

```go
package main

import (
    "fmt"
    "github.com/yourusername/neural-network-go/pkg/network"
)

func main() {
    // Create a neural network with 3 input, 4 hidden, and 2 output neurons
    nn := network.NewNeuralNetwork([]int{3, 4, 2})
    
    // Training data
    inputs := [][]float64{
        {0, 0, 1},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 1},
    }
    targets := [][]float64{
        {0, 1},
        {1, 1},
        {1, 0},
        {0, 0},
    }
    
    // Train the network
    nn.Train(inputs, targets, 1000, 0.1, true)
    
    // Make predictions
    prediction := nn.Predict([]float64{1, 0, 1})
    fmt.Printf("Prediction: %v\n", prediction)
}
```

## Project Structure

```
├── cmd/
│   └── demo/                 # Demo application with visualization
├── pkg/
│   ├── network/             # Core neural network implementation
│   ├── activation/          # Activation functions
│   ├── loss/               # Loss functions
│   ├── optimizer/          # Optimization algorithms
│   └── utils/              # Utility functions
├── examples/               # Example applications
├── tests/                  # Test files
├── docs/                   # Documentation
└── scripts/                # Build and utility scripts
```

## Core Components

### Neural Network

The main `NeuralNetwork` struct provides:

- **Configurable Architecture**: Any number of layers with custom sizes
- **Multiple Activation Functions**: Choose from Sigmoid, ReLU, Tanh, Softmax
- **Training Methods**: Batch and stochastic gradient descent
- **Model Persistence**: Save/load trained models to/from files

### Activation Functions

- **Sigmoid**: `σ(x) = 1 / (1 + e^(-x))`
- **ReLU**: `f(x) = max(0, x)`
- **Tanh**: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- **Softmax**: `f(x_i) = e^(x_i) / Σ(e^(x_j))`

### Loss Functions

- **Mean Squared Error (MSE)**
- **Cross-Entropy Loss**
- **Binary Cross-Entropy**

### Optimizers

- **Stochastic Gradient Descent (SGD)**
- **Adam Optimizer** (planned)
- **RMSprop** (planned)

## Advanced Features

### Real-time Visualization

Watch the network train in real-time with detailed visualizations:

```go
nn.TrainWithVisualization(inputs, targets, epochs, learningRate)
```

### Model Persistence

Save and load trained models:

```go
// Save model
err := nn.SaveModel("my_model.json")

// Load model
nn, err := network.LoadModel("my_model.json")
```

### Custom Training

```go
// Custom training loop
for epoch := 0; epoch < epochs; epoch++ {
    for i, input := range inputs {
        output := nn.Forward(input)
        loss := nn.CalculateLoss(output, targets[i])
        nn.Backward(input, targets[i])
        
        if epoch%100 == 0 {
            fmt.Printf("Epoch %d, Loss: %.4f\n", epoch, loss)
        }
    }
}
```

## Examples

### XOR Problem

```go
// Solve the XOR problem
inputs := [][]float64{
    {0, 0}, {0, 1}, {1, 0}, {1, 1},
}
targets := [][]float64{
    {0}, {1}, {1}, {0},
}

nn := network.NewNeuralNetwork([]int{2, 4, 1})
nn.Train(inputs, targets, 10000, 0.1, false)
```

### Image Classification (MNIST-style)

```go
// Load and preprocess data
data := utils.LoadMNISTData("data/mnist.csv")

// Create network for 10-class classification
nn := network.NewNeuralNetwork([]int{784, 128, 64, 10})
nn.SetActivationFunction("relu", "hidden")
nn.SetActivationFunction("softmax", "output")

// Train
nn.Train(data.TrainImages, data.TrainLabels, 100, 0.01, true)
```

## Performance

- **Fast Training**: Optimized matrix operations
- **Memory Efficient**: Minimal memory footprint
- **Concurrent Training**: Support for parallel processing (planned)

## Testing

Run the test suite:

```bash
go test ./...
go test -v -bench=. ./...
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Roadmap

- [ ] Adam optimizer implementation
- [ ] Convolutional Neural Networks (CNN)
- [ ] Recurrent Neural Networks (RNN)
- [ ] GPU acceleration support
- [ ] Web interface for training visualization
- [ ] Model export to ONNX format
- [ ] Distributed training support

## Acknowledgments

- Inspired by the back-propagation algorithm
- Built with Go's excellent concurrency features
- Uses secure random number generation for initialization


