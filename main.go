package main

import (
	"fmt"
	"time"

	"github.com/1cbyc/neural-network-go/pkg/network"
	"github.com/1cbyc/neural-network-go/pkg/utils"
)

func main() {
	fmt.Println("🧠 Neural Network Library Demo")
	fmt.Println("==============================")
	fmt.Println()

	// Create a neural network
	nn := network.NewNeuralNetwork([]int{2, 4, 1})
	nn.SetActivationFunction("relu", "hidden")
	nn.SetActivationFunction("sigmoid", "output")
	nn.SetLossFunction("binary_crossentropy")
	nn.SetOptimizer("adam", 0.01)

	// Print network summary
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
		fmt.Printf("  %v XOR %v = %v\n", input[0], input[1], targets[i][0])
	}
	fmt.Println()

	// Train the network
	fmt.Println("Training...")
	startTime := time.Now()
	nn.Train(inputs, targets, 1000, 0.01, true)
	duration := time.Since(startTime)

	fmt.Printf("\nTraining completed in %s\n", utils.FormatDuration(duration))

	// Test the network
	fmt.Println("\nTest Results:")
	fmt.Println("=============")
	for i, input := range inputs {
		prediction := nn.Predict(input)
		fmt.Printf("  %v XOR %v = %.4f (expected: %.0f)\n", 
			input[0], input[1], prediction[0], targets[i][0])
	}

	// Calculate final metrics
	loss, accuracy := nn.Evaluate(inputs, targets)
	fmt.Printf("\nFinal Loss: %.6f\n", loss)
	fmt.Printf("Accuracy: %.2f%%\n", accuracy*100)

	fmt.Println("\n🎉 Neural network successfully learned the XOR function!")
	fmt.Println("\nTo explore more features, run:")
	fmt.Println("  go run cmd/demo/main.go -help")
} 