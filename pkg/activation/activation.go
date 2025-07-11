package activation

import "math"

// ActivationFunction represents a function that can be applied to neural network layers
type ActivationFunction func(x float64) float64

// DerivativeFunction represents the derivative of an activation function
type DerivativeFunction func(x float64) float64

// Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidDerivative returns the derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
func SigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

// ReLU activation function: f(x) = max(0, x)
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// ReLUDerivative returns the derivative of ReLU: f'(x) = 1 if x > 0, else 0
func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Tanh activation function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// TanhDerivative returns the derivative of tanh: f'(x) = 1 - tanh²(x)
func TanhDerivative(x float64) float64 {
	return 1 - x*x
}

// LeakyReLU activation function: f(x) = max(αx, x) where α is typically 0.01
func LeakyReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * x
}

// LeakyReLUDerivative returns the derivative of LeakyReLU
func LeakyReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0.01
}

// ELU activation function: f(x) = x if x > 0, else α * (e^x - 1)
func ELU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * (math.Exp(x) - 1)
}

// ELUDerivative returns the derivative of ELU
func ELUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0.01 * math.Exp(x)
}

// Softmax applies softmax to a slice of values
func Softmax(values []float64) []float64 {
	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}

	expSum := 0.0
	expValues := make([]float64, len(values))
	
	for i, v := range values {
		expValues[i] = math.Exp(v - max)
		expSum += expValues[i]
	}

	result := make([]float64, len(values))
	for i, expVal := range expValues {
		result[i] = expVal / expSum
	}
	
	return result
}

// SoftmaxDerivative returns the derivative of softmax (simplified for cross-entropy)
func SoftmaxDerivative(output, target []float64) []float64 {
	derivatives := make([]float64, len(output))
	for i := range output {
		derivatives[i] = output[i] - target[i]
	}
	return derivatives
}

// Linear activation function: f(x) = x
func Linear(x float64) float64 {
	return x
}

// LinearDerivative returns the derivative of linear: f'(x) = 1
func LinearDerivative(x float64) float64 {
	return 1
}

// GetActivationFunction returns the activation function by name
func GetActivationFunction(name string) (ActivationFunction, DerivativeFunction) {
	switch name {
	case "sigmoid":
		return Sigmoid, SigmoidDerivative
	case "relu":
		return ReLU, ReLUDerivative
	case "tanh":
		return Tanh, TanhDerivative
	case "leaky_relu":
		return LeakyReLU, LeakyReLUDerivative
	case "elu":
		return ELU, ELUDerivative
	case "linear":
		return Linear, LinearDerivative
	default:
		return Sigmoid, SigmoidDerivative
	}
}

// ApplyActivation applies an activation function to a slice of values
func ApplyActivation(values []float64, activation ActivationFunction) []float64 {
	result := make([]float64, len(values))
	for i, v := range values {
		result[i] = activation(v)
	}
	return result
}

// ApplyActivationDerivative applies the derivative of an activation function
func ApplyActivationDerivative(values []float64, derivative DerivativeFunction) []float64 {
	result := make([]float64, len(values))
	for i, v := range values {
		result[i] = derivative(v)
	}
	return result
} 