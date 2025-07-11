package utils

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"
)

// SecureRandom generates a cryptographically secure random float64 between -1 and 1
func SecureRandom() float64 {
	var b [8]byte
	_, err := rand.Read(b[:])
	if err != nil {
		panic("cannot generate random number: " + err.Error())
	}
	// Convert to [0,1)
	r := float64(binary.LittleEndian.Uint64(b[:])) / float64(^uint64(0))
	// Scale to [-1,1)
	return r*2 - 1
}

// InitializeWeights initializes weights with Xavier/Glorot initialization
func InitializeWeights(rows, cols int) [][]float64 {
	weights := make([][]float64, rows)
	limit := math.Sqrt(6.0 / float64(rows+cols))

	for i := range weights {
		weights[i] = make([]float64, cols)
		for j := range weights[i] {
			weights[i][j] = (SecureRandom() - 0.5) * 2 * limit
		}
	}
	return weights
}

// InitializeBiases initializes biases with small random values
func InitializeBiases(size int) []float64 {
	biases := make([]float64, size)
	for i := range biases {
		biases[i] = (SecureRandom() - 0.5) * 0.1
	}
	return biases
}

// MatrixMultiply performs matrix multiplication: result = a * b
func MatrixMultiply(a [][]float64, b [][]float64) [][]float64 {
	if len(a) == 0 || len(b) == 0 || len(a[0]) != len(b) {
		return nil
	}

	rows, cols := len(a), len(b[0])
	result := make([][]float64, rows)

	for i := range result {
		result[i] = make([]float64, cols)
		for j := range result[i] {
			sum := 0.0
			for k := range a[i] {
				sum += a[i][k] * b[k][j]
			}
			result[i][j] = sum
		}
	}

	return result
}

// VectorAdd adds two vectors element-wise
func VectorAdd(a, b []float64) []float64 {
	if len(a) != len(b) {
		return nil
	}

	result := make([]float64, len(a))
	for i := range result {
		result[i] = a[i] + b[i]
	}
	return result
}

// VectorSubtract subtracts two vectors element-wise
func VectorSubtract(a, b []float64) []float64 {
	if len(a) != len(b) {
		return nil
	}

	result := make([]float64, len(a))
	for i := range result {
		result[i] = a[i] - b[i]
	}
	return result
}

// VectorMultiply multiplies two vectors element-wise
func VectorMultiply(a, b []float64) []float64 {
	if len(a) != len(b) {
		return nil
	}

	result := make([]float64, len(a))
	for i := range result {
		result[i] = a[i] * b[i]
	}
	return result
}

// VectorScale scales a vector by a scalar
func VectorScale(vector []float64, scalar float64) []float64 {
	result := make([]float64, len(vector))
	for i := range result {
		result[i] = vector[i] * scalar
	}
	return result
}

// MatrixTranspose transposes a matrix
func MatrixTranspose(matrix [][]float64) [][]float64 {
	if len(matrix) == 0 {
		return nil
	}

	rows, cols := len(matrix), len(matrix[0])
	result := make([][]float64, cols)

	for i := range result {
		result[i] = make([]float64, rows)
		for j := range result[i] {
			result[i][j] = matrix[j][i]
		}
	}

	return result
}

// OuterProduct computes the outer product of two vectors
func OuterProduct(a, b []float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(b))
		for j := range result[i] {
			result[i][j] = a[i] * b[j]
		}
	}
	return result
}

// CalculateMean calculates the mean of a slice of float64 values
func CalculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// CalculateStdDev calculates the standard deviation of a slice of float64 values
func CalculateStdDev(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	mean := CalculateMean(values)
	sum := 0.0
	for _, v := range values {
		diff := v - mean
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(values)))
}

// NormalizeData normalizes data using z-score normalization
func NormalizeData(data [][]float64) [][]float64 {
	if len(data) == 0 {
		return nil
	}

	cols := len(data[0])
	means := make([]float64, cols)
	stds := make([]float64, cols)

	// Calculate means and standard deviations for each column
	for j := 0; j < cols; j++ {
		column := make([]float64, len(data))
		for i := range data {
			column[i] = data[i][j]
		}
		means[j] = CalculateMean(column)
		stds[j] = CalculateStdDev(column)
		if stds[j] == 0 {
			stds[j] = 1 // Avoid division by zero
		}
	}

	// Normalize data
	normalized := make([][]float64, len(data))
	for i := range data {
		normalized[i] = make([]float64, cols)
		for j := range data[i] {
			normalized[i][j] = (data[i][j] - means[j]) / stds[j]
		}
	}

	return normalized
}

// MinMaxNormalize normalizes data to [0, 1] range
func MinMaxNormalize(data [][]float64) [][]float64 {
	if len(data) == 0 {
		return nil
	}

	cols := len(data[0])
	mins := make([]float64, cols)
	maxs := make([]float64, cols)

	// Initialize with first row
	for j := range data[0] {
		mins[j] = data[0][j]
		maxs[j] = data[0][j]
	}

	// Find min and max for each column
	for i := range data {
		for j := range data[i] {
			if data[i][j] < mins[j] {
				mins[j] = data[i][j]
			}
			if data[i][j] > maxs[j] {
				maxs[j] = data[i][j]
			}
		}
	}

	// Normalize data
	normalized := make([][]float64, len(data))
	for i := range data {
		normalized[i] = make([]float64, cols)
		for j := range data[i] {
			if maxs[j] == mins[j] {
				normalized[i][j] = 0.5 // Avoid division by zero
			} else {
				normalized[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j])
			}
		}
	}

	return normalized
}

// ShuffleData shuffles the data and targets together
func ShuffleData(data, targets [][]float64) ([][]float64, [][]float64) {
	if len(data) != len(targets) {
		return data, targets
	}

	// Create indices
	indices := make([]int, len(data))
	for i := range indices {
		indices[i] = i
	}

	// Fisher-Yates shuffle
	for i := len(indices) - 1; i > 0; i-- {
		j := int(SecureRandom() * float64(i+1))
		indices[i], indices[j] = indices[j], indices[i]
	}

	// Shuffle data and targets
	shuffledData := make([][]float64, len(data))
	shuffledTargets := make([][]float64, len(targets))

	for i, idx := range indices {
		shuffledData[i] = data[idx]
		shuffledTargets[i] = targets[idx]
	}

	return shuffledData, shuffledTargets
}

// SplitData splits data into training and validation sets
func SplitData(data, targets [][]float64, validationRatio float64) ([][]float64, [][]float64, [][]float64, [][]float64) {
	if len(data) != len(targets) {
		return nil, nil, nil, nil
	}

	// Shuffle data first
	shuffledData, shuffledTargets := ShuffleData(data, targets)

	// Calculate split point
	splitIndex := int(float64(len(data)) * (1 - validationRatio))

	// Split data
	trainData := shuffledData[:splitIndex]
	trainTargets := shuffledTargets[:splitIndex]
	valData := shuffledData[splitIndex:]
	valTargets := shuffledTargets[splitIndex:]

	return trainData, trainTargets, valData, valTargets
}

// SaveModel saves a neural network model to a JSON file
func SaveModel(model interface{}, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	return encoder.Encode(model)
}

// LoadModel loads a neural network model from a JSON file
func LoadModel(filename string, model interface{}) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	return decoder.Decode(model)
}

// PrintProgress prints training progress with a progress bar
func PrintProgress(epoch, totalEpochs int, loss float64) {
	progress := float64(epoch) / float64(totalEpochs)
	barLength := 50
	filledLength := int(progress * float64(barLength))

	bar := strings.Repeat("█", filledLength) + strings.Repeat("░", barLength-filledLength)
	fmt.Printf("\rEpoch %d/%d [%s] Loss: %.6f", epoch, totalEpochs, bar, loss)
}

// PrintNetworkSummary prints a summary of the neural network architecture
func PrintNetworkSummary(layerSizes []int, activationFunctions []string) {
	fmt.Println("Neural Network Architecture:")
	fmt.Println("============================")

	for i, size := range layerSizes {
		layerType := "Hidden"
		if i == 0 {
			layerType = "Input"
		} else if i == len(layerSizes)-1 {
			layerType = "Output"
		}

		activation := "sigmoid"
		if i < len(activationFunctions) {
			activation = activationFunctions[i]
		}

		fmt.Printf("Layer %d (%s): %d neurons, Activation: %s\n", i, layerType, size, activation)
	}
	fmt.Println()
}

// FormatDuration formats a duration in a human-readable way
func FormatDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	} else if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	} else {
		minutes := int(d.Minutes())
		seconds := int(d.Seconds()) % 60
		return fmt.Sprintf("%dm %ds", minutes, seconds)
	}
}

// CalculateAccuracy calculates classification accuracy
func CalculateAccuracy(predictions, targets [][]float64) float64 {
	if len(predictions) != len(targets) {
		return 0
	}

	correct := 0
	total := 0

	for i := range predictions {
		if len(predictions[i]) != len(targets[i]) {
			continue
		}

		// Find predicted and actual classes
		predClass := 0
		actualClass := 0

		for j := range predictions[i] {
			if predictions[i][j] > predictions[i][predClass] {
				predClass = j
			}
			if targets[i][j] > targets[i][actualClass] {
				actualClass = j
			}
		}

		if predClass == actualClass {
			correct++
		}
		total++
	}

	if total == 0 {
		return 0
	}

	return float64(correct) / float64(total)
}
