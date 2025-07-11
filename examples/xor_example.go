package main

import (
	"fmt"
	"time"

	"github.com/1cbyc/neural-network-go/pkg/network"
	"github.com/1cbyc/neural-network-go/pkg/utils"
)

func main() {
	fmt.Println("🔀 XOR Problem with Neural Network")
	fmt.Println("==================================")
	fmt.Println()

	// Create neural network
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
	nn.Train(inputs, targets, 5000, 0.01, true)
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

	// Save the trained model
	err := nn.SaveModel("xor_model.json")
	if err != nil {
		fmt.Printf("Error saving model: %v\n", err)
	} else {
		fmt.Println("Model saved to 'xor_model.json'")
	}

	// Demonstrate loading the model
	loadedNN, err := network.LoadModel("xor_model.json")
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
	} else {
		fmt.Println("Model loaded successfully!")
		
		// Test the loaded model
		testInput := []float64{0.5, 0.5}
		prediction := loadedNN.Predict(testInput)
		fmt.Printf("Test prediction for %v: %.4f\n", testInput, prediction[0])
	}
} 