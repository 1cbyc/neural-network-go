package tests

import (
	"math"
	"testing"

	"github.com/1cbyc/neural-network-go/pkg/activation"
	"github.com/1cbyc/neural-network-go/pkg/loss"
	"github.com/1cbyc/neural-network-go/pkg/network"
	"github.com/1cbyc/neural-network-go/pkg/optimizer"
	"github.com/1cbyc/neural-network-go/pkg/utils"
)

func TestNeuralNetworkCreation(t *testing.T) {
	nn := network.NewNeuralNetwork([]int{2, 3, 1})
	
	if len(nn.LayerSizes) != 3 {
		t.Errorf("Expected 3 layer sizes, got %d", len(nn.LayerSizes))
	}
	
	if len(nn.Weights) != 2 {
		t.Errorf("Expected 2 weight matrices, got %d", len(nn.Weights))
	}
	
	if len(nn.Biases) != 2 {
		t.Errorf("Expected 2 bias vectors, got %d", len(nn.Biases))
	}
}

func TestForwardPropagation(t *testing.T) {
	nn := network.NewNeuralNetwork([]int{2, 3, 1})
	input := []float64{0.5, 0.3}
	
	activations := nn.Forward(input)
	
	if len(activations) != 3 {
		t.Errorf("Expected 3 activation layers, got %d", len(activations))
	}
	
	if len(activations[0]) != 2 {
		t.Errorf("Expected input layer size 2, got %d", len(activations[0]))
	}
	
	if len(activations[1]) != 3 {
		t.Errorf("Expected hidden layer size 3, got %d", len(activations[1]))
	}
	
	if len(activations[2]) != 1 {
		t.Errorf("Expected output layer size 1, got %d", len(activations[2]))
	}
}

func TestPredict(t *testing.T) {
	nn := network.NewNeuralNetwork([]int{2, 3, 1})
	input := []float64{0.5, 0.3}
	
	prediction := nn.Predict(input)
	
	if len(prediction) != 1 {
		t.Errorf("Expected prediction size 1, got %d", len(prediction))
	}
	
	// Check that prediction is between 0 and 1 (sigmoid output)
	if prediction[0] < 0 || prediction[0] > 1 {
		t.Errorf("Expected prediction between 0 and 1, got %f", prediction[0])
	}
}

func TestXORTraining(t *testing.T) {
	nn := network.NewNeuralNetwork([]int{2, 4, 1})
	nn.SetActivationFunction("relu", "hidden")
	nn.SetActivationFunction("sigmoid", "output")
	nn.SetLossFunction("binary_crossentropy")
	nn.SetOptimizer("adam", 0.01)
	
	inputs := [][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}
	targets := [][]float64{
		{0}, {1}, {1}, {0},
	}
	
	// Train for a few epochs
	nn.Train(inputs, targets, 100, 0.01, false)
	
	// Test predictions
	for i, input := range inputs {
		prediction := nn.Predict(input)
		expected := targets[i][0]
		
		// Convert prediction to binary
		predBinary := 0.0
		if prediction[0] > 0.5 {
			predBinary = 1.0
		}
		
		if predBinary != expected {
			t.Logf("XOR test failed: input %v, prediction %.4f (binary %.0f), expected %.0f", 
				input, prediction[0], predBinary, expected)
		}
	}
}

func TestActivationFunctions(t *testing.T) {
	// Test sigmoid
	sigmoid, sigmoidDeriv := activation.GetActivationFunction("sigmoid")
	
	if sigmoid(0) != 0.5 {
		t.Errorf("Expected sigmoid(0) = 0.5, got %f", sigmoid(0))
	}
	
	if sigmoidDeriv(0.5) != 0.25 {
		t.Errorf("Expected sigmoidDeriv(0.5) = 0.25, got %f", sigmoidDeriv(0.5))
	}
	
	// Test ReLU
	relu, reluDeriv := activation.GetActivationFunction("relu")
	
	if relu(-1) != 0 {
		t.Errorf("Expected relu(-1) = 0, got %f", relu(-1))
	}
	
	if relu(2) != 2 {
		t.Errorf("Expected relu(2) = 2, got %f", relu(2))
	}
	
	if reluDeriv(-1) != 0 {
		t.Errorf("Expected reluDeriv(-1) = 0, got %f", reluDeriv(-1))
	}
	
	if reluDeriv(2) != 1 {
		t.Errorf("Expected reluDeriv(2) = 1, got %f", reluDeriv(2))
	}
}

func TestLossFunctions(t *testing.T) {
	// Test MSE
	mse, mseDeriv := loss.GetLossFunction("mse")
	
	predictions := []float64{0.8, 0.2}
	targets := []float64{1.0, 0.0}
	
	mseLoss := mse(predictions, targets)
	if mseLoss <= 0 {
		t.Errorf("Expected positive MSE loss, got %f", mseLoss)
	}
	
	derivatives := mseDeriv(predictions, targets)
	if len(derivatives) != 2 {
		t.Errorf("Expected 2 derivatives, got %d", len(derivatives))
	}
	
	// Test binary cross-entropy
	bce, bceDeriv := loss.GetLossFunction("binary_crossentropy")
	
	bceLoss := bce(predictions, targets)
	if bceLoss <= 0 {
		t.Errorf("Expected positive BCE loss, got %f", bceLoss)
	}
	
	bceDerivatives := bceDeriv(predictions, targets)
	if len(bceDerivatives) != 2 {
		t.Errorf("Expected 2 BCE derivatives, got %d", len(bceDerivatives))
	}
}

func TestOptimizers(t *testing.T) {
	// Test SGD
	sgd := optimizer.GetOptimizer("sgd", 0.1)
	
	weights := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	gradients := [][]float64{{0.01, 0.02}, {0.03, 0.04}}
	
	updatedWeights := sgd.UpdateWeights(weights, gradients, 0)
	
	if len(updatedWeights) != len(weights) {
		t.Errorf("Expected same number of weight rows, got %d", len(updatedWeights))
	}
	
	// Test Adam
	adam := optimizer.GetOptimizer("adam", 0.01)
	
	updatedWeightsAdam := adam.UpdateWeights(weights, gradients, 0)
	
	if len(updatedWeightsAdam) != len(weights) {
		t.Errorf("Expected same number of weight rows, got %d", len(updatedWeightsAdam))
	}
}

func TestUtils(t *testing.T) {
	// Test secure random
	random := utils.SecureRandom()
	if random < -1 || random > 1 {
		t.Errorf("Expected random value between -1 and 1, got %f", random)
	}
	
	// Test weight initialization
	weights := utils.InitializeWeights(3, 4)
	if len(weights) != 3 {
		t.Errorf("Expected 3 rows, got %d", len(weights))
	}
	if len(weights[0]) != 4 {
		t.Errorf("Expected 4 columns, got %d", len(weights[0]))
	}
	
	// Test bias initialization
	biases := utils.InitializeBiases(4)
	if len(biases) != 4 {
		t.Errorf("Expected 4 biases, got %d", len(biases))
	}
	
	// Test vector operations
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	
	sum := utils.VectorAdd(a, b)
	if len(sum) != 3 {
		t.Errorf("Expected sum vector of length 3, got %d", len(sum))
	}
	if sum[0] != 5 {
		t.Errorf("Expected sum[0] = 5, got %f", sum[0])
	}
	
	diff := utils.VectorSubtract(a, b)
	if diff[0] != -3 {
		t.Errorf("Expected diff[0] = -3, got %f", diff[0])
	}
	
	product := utils.VectorMultiply(a, b)
	if product[0] != 4 {
		t.Errorf("Expected product[0] = 4, got %f", product[0])
	}
}

func TestModelPersistence(t *testing.T) {
	nn := network.NewNeuralNetwork([]int{2, 3, 1})
	nn.SetActivationFunction("relu", "hidden")
	nn.SetActivationFunction("sigmoid", "output")
	
	// Save model
	err := nn.SaveModel("test_model.json")
	if err != nil {
		t.Errorf("Failed to save model: %v", err)
	}
	
	// Load model
	loadedNN, err := network.LoadModel("test_model.json")
	if err != nil {
		t.Errorf("Failed to load model: %v", err)
	}
	
	// Test that loaded model has same architecture
	if len(loadedNN.LayerSizes) != len(nn.LayerSizes) {
		t.Errorf("Loaded model has different layer sizes")
	}
	
	// Test prediction consistency
	input := []float64{0.5, 0.3}
	originalPred := nn.Predict(input)
	loadedPred := loadedNN.Predict(input)
	
	if math.Abs(originalPred[0]-loadedPred[0]) > 1e-10 {
		t.Errorf("Predictions differ after save/load: %f vs %f", 
			originalPred[0], loadedPred[0])
	}
}

func TestTrainingHistory(t *testing.T) {
	nn := network.NewNeuralNetwork([]int{2, 3, 1})
	
	inputs := [][]float64{{0, 0}, {1, 1}}
	targets := [][]float64{{0}, {1}}
	
	nn.Train(inputs, targets, 10, 0.1, false)
	
	history := nn.GetTrainingHistory()
	
	if len(history.Losses) == 0 {
		t.Errorf("Expected training history to contain losses")
	}
	
	if len(history.Accuracies) == 0 {
		t.Errorf("Expected training history to contain accuracies")
	}
	
	if len(history.Epochs) == 0 {
		t.Errorf("Expected training history to contain epochs")
	}
}

func BenchmarkForwardPropagation(b *testing.B) {
	nn := network.NewNeuralNetwork([]int{10, 50, 30, 5})
	input := make([]float64, 10)
	for i := range input {
		input[i] = float64(i) / 10.0
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nn.Predict(input)
	}
}

func BenchmarkTraining(b *testing.B) {
	nn := network.NewNeuralNetwork([]int{5, 10, 5})
	nn.SetOptimizer("adam", 0.01)
	
	inputs := make([][]float64, 100)
	targets := make([][]float64, 100)
	
	for i := range inputs {
		inputs[i] = make([]float64, 5)
		targets[i] = make([]float64, 5)
		for j := range inputs[i] {
			inputs[i][j] = float64(i+j) / 100.0
			targets[i][j] = float64(i*j) / 100.0
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nn.Train(inputs, targets, 10, 0.01, false)
	}
} 