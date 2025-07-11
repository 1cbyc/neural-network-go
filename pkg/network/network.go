package network

import (
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/yourusername/neural-network-go/pkg/activation"
	"github.com/yourusername/neural-network-go/pkg/loss"
	"github.com/yourusername/neural-network-go/pkg/optimizer"
	"github.com/yourusername/neural-network-go/pkg/utils"
)

// NeuralNetwork represents a multi-layer neural network
type NeuralNetwork struct {
	LayerSizes         []int                    `json:"layer_sizes"`
	Weights            [][][]float64            `json:"weights"`
	Biases             [][]float64              `json:"biases"`
	ActivationFunctions []string                 `json:"activation_functions"`
	LossFunction       string                   `json:"loss_function"`
	Optimizer          optimizer.Optimizer      `json:"-"`
	TrainingHistory    *TrainingHistory         `json:"training_history"`
}

// TrainingHistory tracks training metrics
type TrainingHistory struct {
	Losses     []float64 `json:"losses"`
	Accuracies []float64 `json:"accuracies"`
	Epochs     []int     `json:"epochs"`
}

// NewNeuralNetwork creates a new neural network with the specified layer sizes
func NewNeuralNetwork(layerSizes []int) *NeuralNetwork {
	if len(layerSizes) < 2 {
		panic("neural network must have at least 2 layers (input and output)")
	}

	nn := &NeuralNetwork{
		LayerSizes:         layerSizes,
		Weights:            make([][][]float64, len(layerSizes)-1),
		Biases:             make([][]float64, len(layerSizes)-1),
		ActivationFunctions: make([]string, len(layerSizes)-1),
		LossFunction:       "mse",
		TrainingHistory:    &TrainingHistory{},
	}

	// Initialize weights and biases
	for i := 0; i < len(layerSizes)-1; i++ {
		nn.Weights[i] = utils.InitializeWeights(layerSizes[i], layerSizes[i+1])
		nn.Biases[i] = utils.InitializeBiases(layerSizes[i+1])
		nn.ActivationFunctions[i] = "sigmoid" // Default activation
	}

	// Set output layer activation based on loss function
	nn.ActivationFunctions[len(nn.ActivationFunctions)-1] = "sigmoid"

	return nn
}

// SetActivationFunction sets the activation function for a specific layer
func (nn *NeuralNetwork) SetActivationFunction(activation string, layerType string) {
	switch layerType {
	case "all":
		for i := range nn.ActivationFunctions {
			nn.ActivationFunctions[i] = activation
		}
	case "hidden":
		for i := 0; i < len(nn.ActivationFunctions)-1; i++ {
			nn.ActivationFunctions[i] = activation
		}
	case "output":
		nn.ActivationFunctions[len(nn.ActivationFunctions)-1] = activation
	default:
		// Try to parse as layer index
		var layerIndex int
		if _, err := fmt.Sscanf(layerType, "%d", &layerIndex); err == nil {
			if layerIndex >= 0 && layerIndex < len(nn.ActivationFunctions) {
				nn.ActivationFunctions[layerIndex] = activation
			}
		}
	}
}

// SetLossFunction sets the loss function
func (nn *NeuralNetwork) SetLossFunction(lossFunction string) {
	nn.LossFunction = lossFunction
}

// SetOptimizer sets the optimizer
func (nn *NeuralNetwork) SetOptimizer(optimizerName string, learningRate float64) {
	nn.Optimizer = optimizer.GetOptimizer(optimizerName, learningRate)
}

// Forward performs forward propagation through the network
func (nn *NeuralNetwork) Forward(input []float64) [][]float64 {
	if len(input) != nn.LayerSizes[0] {
		panic(fmt.Sprintf("input size %d does not match network input size %d", len(input), nn.LayerSizes[0]))
	}

	activations := make([][]float64, len(nn.LayerSizes))
	activations[0] = make([]float64, len(input))
	copy(activations[0], input)

	// Forward propagate through each layer
	for i := 0; i < len(nn.Weights); i++ {
		// Calculate weighted sum
		weightedSum := make([]float64, nn.LayerSizes[i+1])
		for j := range weightedSum {
			sum := 0.0
			for k := range activations[i] {
				sum += activations[i][k] * nn.Weights[i][k][j]
			}
			weightedSum[j] = sum + nn.Biases[i][j]
		}

		// Apply activation function
		activationFunc, _ := activation.GetActivationFunction(nn.ActivationFunctions[i])
		activations[i+1] = activation.ApplyActivation(weightedSum, activationFunc)
	}

	return activations
}

// Predict performs a forward pass and returns the output layer
func (nn *NeuralNetwork) Predict(input []float64) []float64 {
	activations := nn.Forward(input)
	return activations[len(activations)-1]
}

// Backward performs backpropagation to calculate gradients
func (nn *NeuralNetwork) Backward(input []float64, target []float64) ([][][]float64, [][]float64) {
	// Forward pass
	activations := nn.Forward(input)

	// Initialize gradients
	weightGradients := make([][][]float64, len(nn.Weights))
	biasGradients := make([][]float64, len(nn.Biases))

	for i := range weightGradients {
		weightGradients[i] = make([][]float64, len(nn.Weights[i]))
		for j := range weightGradients[i] {
			weightGradients[i][j] = make([]float64, len(nn.Weights[i][j]))
		}
		biasGradients[i] = make([]float64, len(nn.Biases[i]))
	}

	// Calculate output layer error
	output := activations[len(activations)-1]
	lossFunc, lossDeriv := loss.GetLossFunction(nn.LossFunction)
	
	// For softmax with cross-entropy, use special derivative
	if nn.ActivationFunctions[len(nn.ActivationFunctions)-1] == "softmax" && nn.LossFunction == "categorical_crossentropy" {
		outputErrors := activation.SoftmaxDerivative(output, target)
		activations[len(activations)-1] = outputErrors
	} else {
		outputErrors := lossDeriv(output, target)
		activationDeriv, _ := activation.GetActivationFunction(nn.ActivationFunctions[len(nn.ActivationFunctions)-1])
		outputDeltas := activation.ApplyActivationDerivative(output, activationDeriv)
		
		// Element-wise multiplication of errors and activation derivatives
		for i := range outputErrors {
			outputDeltas[i] *= outputErrors[i]
		}
		activations[len(activations)-1] = outputDeltas
	}

	// Backpropagate through hidden layers
	for i := len(nn.Weights) - 1; i >= 0; i-- {
		// Calculate gradients for current layer
		for j := range nn.Weights[i] {
			for k := range nn.Weights[i][j] {
				weightGradients[i][j][k] = activations[i][j] * activations[i+1][k]
			}
		}

		// Calculate bias gradients
		for j := range nn.Biases[i] {
			biasGradients[i][j] = activations[i+1][j]
		}

		// Calculate error for previous layer (if not input layer)
		if i > 0 {
			prevErrors := make([]float64, len(activations[i]))
			for j := range prevErrors {
				for k := range activations[i+1] {
					prevErrors[j] += activations[i+1][k] * nn.Weights[i][j][k]
				}
			}

			// Apply activation derivative
			activationDeriv, _ := activation.GetActivationFunction(nn.ActivationFunctions[i-1])
			prevDeltas := activation.ApplyActivationDerivative(activations[i], activationDeriv)
			
			for j := range prevErrors {
				prevDeltas[j] *= prevErrors[j]
			}
			activations[i] = prevDeltas
		}
	}

	return weightGradients, biasGradients
}

// Train trains the neural network on the provided data
func (nn *NeuralNetwork) Train(inputs, targets [][]float64, epochs int, learningRate float64, verbose bool) {
	if nn.Optimizer == nil {
		nn.Optimizer = optimizer.GetOptimizer("sgd", learningRate)
	}

	startTime := time.Now()
	
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		predictions := make([][]float64, len(inputs))

		// Forward and backward pass for each training example
		for i, input := range inputs {
			// Forward pass
			predictions[i] = nn.Predict(input)

			// Calculate loss
			lossFunc, _ := loss.GetLossFunction(nn.LossFunction)
			loss := lossFunc(predictions[i], targets[i])
			totalLoss += loss

			// Backward pass
			weightGradients, biasGradients := nn.Backward(input, targets[i])

			// Update weights and biases using optimizer
			for j := range nn.Weights {
				nn.Weights[j] = nn.Optimizer.UpdateWeights(nn.Weights[j], weightGradients[j], j)
				nn.Biases[j] = nn.Optimizer.UpdateBiases(nn.Biases[j], biasGradients[j], j)
			}
		}

		// Calculate average loss and accuracy
		averageLoss := totalLoss / float64(len(inputs))
		accuracy := utils.CalculateAccuracy(predictions, targets)

		// Store training history
		nn.TrainingHistory.Losses = append(nn.TrainingHistory.Losses, averageLoss)
		nn.TrainingHistory.Accuracies = append(nn.TrainingHistory.Accuracies, accuracy)
		nn.TrainingHistory.Epochs = append(nn.TrainingHistory.Epochs, epoch)

		// Print progress
		if verbose && (epoch%100 == 0 || epoch == epochs-1) {
			utils.PrintProgress(epoch+1, epochs, averageLoss)
			fmt.Printf(" | Accuracy: %.4f", accuracy)
			if epoch == epochs-1 {
				fmt.Println()
			}
		}
	}

	if verbose {
		duration := time.Since(startTime)
		fmt.Printf("Training completed in %s\n", utils.FormatDuration(duration))
	}
}

// TrainWithVisualization trains the network with real-time visualization
func (nn *NeuralNetwork) TrainWithVisualization(inputs, targets [][]float64, epochs int, learningRate float64) {
	if nn.Optimizer == nil {
		nn.Optimizer = optimizer.GetOptimizer("sgd", learningRate)
	}

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		for i, input := range inputs {
			// Forward pass
			activations := nn.Forward(input)
			output := activations[len(activations)-1]

			// Calculate loss
			lossFunc, _ := loss.GetLossFunction(nn.LossFunction)
			loss := lossFunc(output, targets[i])
			totalLoss += loss

			// Backward pass
			weightGradients, biasGradients := nn.Backward(input, targets[i])

			// Update weights and biases
			for j := range nn.Weights {
				nn.Weights[j] = nn.Optimizer.UpdateWeights(nn.Weights[j], weightGradients[j], j)
				nn.Biases[j] = nn.Optimizer.UpdateBiases(nn.Biases[j], biasGradients[j], j)
			}

			// Visualize network state
			nn.visualizeNetwork(input, activations, targets[i], loss, epoch)
			time.Sleep(100 * time.Millisecond)
		}

		averageLoss := totalLoss / float64(len(inputs))
		fmt.Printf("\nEpoch %d, Average Loss: %.6f\n", epoch, averageLoss)
		time.Sleep(500 * time.Millisecond)
	}
}

// visualizeNetwork displays the network state during training
func (nn *NeuralNetwork) visualizeNetwork(input []float64, activations [][]float64, target []float64, loss float64, epoch int) {
	fmt.Print("\033[2J\033[H") // Clear screen
	fmt.Printf("Epoch: %d | Loss: %.4f\n\n", epoch, loss)

	// Display input layer
	fmt.Println("Input Layer:")
	for i, val := range input {
		fmt.Printf("  x%d: %.3f\n", i, val)
	}

	// Display hidden layers
	for i := 1; i < len(activations)-1; i++ {
		fmt.Printf("\nHidden Layer %d:\n", i)
		for j, val := range activations[i] {
			fmt.Printf("  h%d: %.3f\n", j, val)
		}
	}

	// Display output layer
	fmt.Printf("\nOutput Layer:\n")
	for i, val := range activations[len(activations)-1] {
		fmt.Printf("  y%d: %.3f (target: %.3f)\n", i, val, target[i])
	}

	// Display some weight information
	fmt.Printf("\nWeight Statistics:\n")
	for i, layerWeights := range nn.Weights {
		min, max := layerWeights[0][0], layerWeights[0][0]
		sum := 0.0
		count := 0
		
		for _, row := range layerWeights {
			for _, weight := range row {
				if weight < min {
					min = weight
				}
				if weight > max {
					max = weight
				}
				sum += weight
				count++
			}
		}
		
		avg := sum / float64(count)
		fmt.Printf("  Layer %d: min=%.3f, max=%.3f, avg=%.3f\n", i+1, min, max, avg)
	}
}

// Evaluate evaluates the network on test data
func (nn *NeuralNetwork) Evaluate(inputs, targets [][]float64) (float64, float64) {
	predictions := make([][]float64, len(inputs))
	totalLoss := 0.0

	for i, input := range inputs {
		predictions[i] = nn.Predict(input)
		lossFunc, _ := loss.GetLossFunction(nn.LossFunction)
		loss := lossFunc(predictions[i], targets[i])
		totalLoss += loss
	}

	averageLoss := totalLoss / float64(len(inputs))
	accuracy := utils.CalculateAccuracy(predictions, targets)

	return averageLoss, accuracy
}

// SaveModel saves the neural network to a file
func (nn *NeuralNetwork) SaveModel(filename string) error {
	return utils.SaveModel(nn, filename)
}

// LoadModel loads a neural network from a file
func LoadModel(filename string) (*NeuralNetwork, error) {
	var nn NeuralNetwork
	err := utils.LoadModel(filename, &nn)
	if err != nil {
		return nil, err
	}
	return &nn, nil
}

// GetTrainingHistory returns the training history
func (nn *NeuralNetwork) GetTrainingHistory() *TrainingHistory {
	return nn.TrainingHistory
}

// PrintSummary prints a summary of the neural network
func (nn *NeuralNetwork) PrintSummary() {
	utils.PrintNetworkSummary(nn.LayerSizes, nn.ActivationFunctions)
	fmt.Printf("Loss Function: %s\n", nn.LossFunction)
	if nn.Optimizer != nil {
		fmt.Printf("Optimizer: %s\n", nn.Optimizer.GetName())
	}
	fmt.Printf("Total Parameters: %d\n", nn.GetParameterCount())
}

// GetParameterCount returns the total number of parameters in the network
func (nn *NeuralNetwork) GetParameterCount() int {
	count := 0
	for i := range nn.Weights {
		count += len(nn.Weights[i]) * len(nn.Weights[i][0])
		count += len(nn.Biases[i])
	}
	return count
}

// Clone creates a deep copy of the neural network
func (nn *NeuralNetwork) Clone() *NeuralNetwork {
	// Serialize and deserialize to create a deep copy
	data, err := json.Marshal(nn)
	if err != nil {
		panic("failed to clone neural network: " + err.Error())
	}

	var clone NeuralNetwork
	err = json.Unmarshal(data, &clone)
	if err != nil {
		panic("failed to clone neural network: " + err.Error())
	}

	return &clone
} 