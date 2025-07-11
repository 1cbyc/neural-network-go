package main

import (
	"flag"
	"fmt"
	"math"
	"time"

	"github.com/1cbyc/neural-network-go/pkg/network"
	"github.com/1cbyc/neural-network-go/pkg/utils"
)

func main() {
	var (
		example     = flag.String("example", "xor", "Example to run: xor, classification, regression, visualization")
		epochs      = flag.Int("epochs", 1000, "Number of training epochs")
		learningRate = flag.Float64("lr", 0.1, "Learning rate")
		optimizer   = flag.String("optimizer", "sgd", "Optimizer: sgd, adam, rmsprop")
		visualize   = flag.Bool("visualize", false, "Enable real-time visualization")
	)
	flag.Parse()

	fmt.Println("🧠 Neural Network Demo")
	fmt.Println("======================")
	fmt.Println()

	switch *example {
	case "xor":
		runXORExample(*epochs, *learningRate, *optimizer, *visualize)
	case "classification":
		runClassificationExample(*epochs, *learningRate, *optimizer, *visualize)
	case "regression":
		runRegressionExample(*epochs, *learningRate, *optimizer, *visualize)
	case "visualization":
		runVisualizationExample(*epochs, *learningRate, *optimizer)
	default:
		fmt.Printf("Unknown example: %s\n", *example)
		fmt.Println("Available examples: xor, classification, regression, visualization")
	}
}

func runXORExample(epochs int, learningRate float64, optimizer string, visualize bool) {
	fmt.Println("🔀 XOR Problem Example")
	fmt.Println("=====================")
	fmt.Println()

	// Create neural network
	nn := network.NewNeuralNetwork([]int{2, 4, 1})
	nn.SetActivationFunction("relu", "hidden")
	nn.SetActivationFunction("sigmoid", "output")
	nn.SetLossFunction("binary_crossentropy")
	nn.SetOptimizer(optimizer, learningRate)
	nn.PrintSummary()

	// XOR training data
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	fmt.Println("Training Data:")
	for i, input := range inputs {
		fmt.Printf("  Input: %v -> Target: %v\n", input, targets[i])
	}
	fmt.Println()

	// Train the network
	startTime := time.Now()
	if visualize {
		nn.TrainWithVisualization(inputs, targets, epochs, learningRate)
	} else {
		nn.Train(inputs, targets, epochs, learningRate, true)
	}
	duration := time.Since(startTime)

	fmt.Printf("\nTraining completed in %s\n", utils.FormatDuration(duration))

	// Test the network
	fmt.Println("\nTest Results:")
	fmt.Println("=============")
	for i, input := range inputs {
		prediction := nn.Predict(input)
		fmt.Printf("  Input: %v -> Prediction: %.4f (Target: %.0f)\n", 
			input, prediction[0], targets[i][0])
	}

	// Calculate accuracy
	loss, accuracy := nn.Evaluate(inputs, targets)
	fmt.Printf("\nFinal Loss: %.6f\n", loss)
	fmt.Printf("Accuracy: %.2f%%\n", accuracy*100)
}

func runClassificationExample(epochs int, learningRate float64, optimizer string, visualize bool) {
	fmt.Println("🎯 Multi-class Classification Example")
	fmt.Println("===================================")
	fmt.Println()

	// Create neural network for 3-class classification
	nn := network.NewNeuralNetwork([]int{4, 8, 6, 3})
	nn.SetActivationFunction("relu", "hidden")
	nn.SetActivationFunction("softmax", "output")
	nn.SetLossFunction("categorical_crossentropy")
	nn.SetOptimizer(optimizer, learningRate)
	nn.PrintSummary()

	// Create synthetic classification data
	inputs := [][]float64{
		{1.0, 0.5, 0.2, 0.8}, // Class 0
		{0.3, 0.9, 0.1, 0.4}, // Class 0
		{0.8, 0.2, 0.9, 0.1}, // Class 1
		{0.1, 0.7, 0.8, 0.3}, // Class 1
		{0.4, 0.1, 0.3, 0.9}, // Class 2
		{0.9, 0.6, 0.2, 0.5}, // Class 2
		{0.2, 0.8, 0.4, 0.7}, // Class 0
		{0.7, 0.3, 0.9, 0.2}, // Class 1
		{0.5, 0.4, 0.1, 0.8}, // Class 2
		{0.1, 0.9, 0.6, 0.3}, // Class 0
	}

	targets := [][]float64{
		{1, 0, 0}, // Class 0
		{1, 0, 0}, // Class 0
		{0, 1, 0}, // Class 1
		{0, 1, 0}, // Class 1
		{0, 0, 1}, // Class 2
		{0, 0, 1}, // Class 2
		{1, 0, 0}, // Class 0
		{0, 1, 0}, // Class 1
		{0, 0, 1}, // Class 2
		{1, 0, 0}, // Class 0
	}

	fmt.Printf("Training on %d samples...\n", len(inputs))

	// Train the network
	startTime := time.Now()
	if visualize {
		nn.TrainWithVisualization(inputs, targets, epochs, learningRate)
	} else {
		nn.Train(inputs, targets, epochs, learningRate, true)
	}
	duration := time.Since(startTime)

	fmt.Printf("\nTraining completed in %s\n", utils.FormatDuration(duration))

	// Test the network
	fmt.Println("\nTest Results:")
	fmt.Println("=============")
	correct := 0
	for i, input := range inputs {
		prediction := nn.Predict(input)
		
		// Find predicted class
		predClass := 0
		for j, val := range prediction {
			if val > prediction[predClass] {
				predClass = j
			}
		}
		
		// Find actual class
		actualClass := 0
		for j, val := range targets[i] {
			if val > targets[i][actualClass] {
				actualClass = j
			}
		}
		
		if predClass == actualClass {
			correct++
		}
		
		fmt.Printf("  Input: %v -> Predicted: Class %d (%.3f, %.3f, %.3f) | Actual: Class %d\n",
			input, predClass, prediction[0], prediction[1], prediction[2], actualClass)
	}

	accuracy := float64(correct) / float64(len(inputs))
	fmt.Printf("\nAccuracy: %.2f%% (%d/%d correct)\n", accuracy*100, correct, len(inputs))
}

func runRegressionExample(epochs int, learningRate float64, optimizer string, visualize bool) {
	fmt.Println("📈 Regression Example")
	fmt.Println("====================")
	fmt.Println()

	// Create neural network for regression
	nn := network.NewNeuralNetwork([]int{1, 8, 4, 1})
	nn.SetActivationFunction("relu", "hidden")
	nn.SetActivationFunction("linear", "output")
	nn.SetLossFunction("mse")
	nn.SetOptimizer(optimizer, learningRate)
	nn.PrintSummary()

	// Create synthetic regression data (y = 2x + 1 + noise)
	inputs := [][]float64{}
	targets := [][]float64{}
	
	for i := 0; i < 50; i++ {
		x := float64(i) / 10.0
		y := 2*x + 1 + (utils.SecureRandom()-0.5)*0.2 // Add some noise
		inputs = append(inputs, []float64{x})
		targets = append(targets, []float64{y})
	}

	fmt.Printf("Training on %d samples...\n", len(inputs))

	// Train the network
	startTime := time.Now()
	if visualize {
		nn.TrainWithVisualization(inputs, targets, epochs, learningRate)
	} else {
		nn.Train(inputs, targets, epochs, learningRate, true)
	}
	duration := time.Since(startTime)

	fmt.Printf("\nTraining completed in %s\n", utils.FormatDuration(duration))

	// Test the network
	fmt.Println("\nTest Results:")
	fmt.Println("=============")
	
	testInputs := [][]float64{
		{0.5}, {1.0}, {1.5}, {2.0}, {2.5}, {3.0}, {3.5}, {4.0}, {4.5}, {5.0},
	}
	
	totalError := 0.0
	for _, input := range testInputs {
		prediction := nn.Predict(input)
		expected := 2*input[0] + 1
		error := math.Abs(prediction[0] - expected)
		totalError += error
		
		fmt.Printf("  Input: %.1f -> Prediction: %.3f (Expected: %.3f, Error: %.3f)\n",
			input[0], prediction[0], expected, error)
	}
	
	avgError := totalError / float64(len(testInputs))
	fmt.Printf("\nAverage Error: %.3f\n", avgError)
	
	// Calculate R-squared
	loss, _ := nn.Evaluate(inputs, targets)
	fmt.Printf("Final Loss (MSE): %.6f\n", loss)
}

func runVisualizationExample(epochs int, learningRate float64, optimizer string) {
	fmt.Println("🎨 Real-time Training Visualization")
	fmt.Println("==================================")
	fmt.Println()

	// Create a simple network for visualization
	nn := network.NewNeuralNetwork([]int{3, 4, 2})
	nn.SetActivationFunction("sigmoid", "all")
	nn.SetLossFunction("mse")
	nn.SetOptimizer(optimizer, learningRate)
	nn.PrintSummary()

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

	fmt.Println("Starting real-time visualization...")
	fmt.Println("Press Ctrl+C to stop")
	fmt.Println()

	// Train with visualization
	nn.TrainWithVisualization(inputs, targets, epochs, learningRate)
} 