package loss

import "math"

// LossFunction represents a function that calculates the loss between predictions and targets
type LossFunction func(predictions, targets []float64) float64

// LossDerivative represents the derivative of a loss function
type LossDerivative func(predictions, targets []float64) []float64

// MeanSquaredError calculates the mean squared error loss
func MeanSquaredError(predictions, targets []float64) float64 {
	if len(predictions) != len(targets) {
		return 0
	}
	
	sum := 0.0
	for i, pred := range predictions {
		diff := pred - targets[i]
		sum += diff * diff
	}
	return sum / float64(len(predictions))
}

// MeanSquaredErrorDerivative returns the derivative of MSE
func MeanSquaredErrorDerivative(predictions, targets []float64) []float64 {
	if len(predictions) != len(targets) {
		return nil
	}
	
	derivatives := make([]float64, len(predictions))
	for i := range predictions {
		derivatives[i] = 2.0 * (predictions[i] - targets[i]) / float64(len(predictions))
	}
	return derivatives
}

// BinaryCrossEntropy calculates binary cross-entropy loss
func BinaryCrossEntropy(predictions, targets []float64) float64 {
	if len(predictions) != len(targets) {
		return 0
	}
	
	sum := 0.0
	for i, pred := range predictions {
		// Clip predictions to avoid log(0)
		if pred <= 0 {
			pred = 1e-15
		} else if pred >= 1 {
			pred = 1 - 1e-15
		}
		
		sum += -(targets[i]*math.Log(pred) + (1-targets[i])*math.Log(1-pred))
	}
	return sum / float64(len(predictions))
}

// BinaryCrossEntropyDerivative returns the derivative of binary cross-entropy
func BinaryCrossEntropyDerivative(predictions, targets []float64) []float64 {
	if len(predictions) != len(targets) {
		return nil
	}
	
	derivatives := make([]float64, len(predictions))
	for i, pred := range predictions {
		// Clip predictions to avoid division by zero
		if pred <= 0 {
			pred = 1e-15
		} else if pred >= 1 {
			pred = 1 - 1e-15
		}
		
		derivatives[i] = -(targets[i]/pred - (1-targets[i])/(1-pred)) / float64(len(predictions))
	}
	return derivatives
}

// CategoricalCrossEntropy calculates categorical cross-entropy loss
func CategoricalCrossEntropy(predictions, targets []float64) float64 {
	if len(predictions) != len(targets) {
		return 0
	}
	
	sum := 0.0
	for i, pred := range predictions {
		// Clip predictions to avoid log(0)
		if pred <= 0 {
			pred = 1e-15
		}
		
		sum += -targets[i] * math.Log(pred)
	}
	return sum
}

// CategoricalCrossEntropyDerivative returns the derivative of categorical cross-entropy
func CategoricalCrossEntropyDerivative(predictions, targets []float64) []float64 {
	if len(predictions) != len(targets) {
		return nil
	}
	
	derivatives := make([]float64, len(predictions))
	for i, pred := range predictions {
		// Clip predictions to avoid division by zero
		if pred <= 0 {
			pred = 1e-15
		}
		
		derivatives[i] = -targets[i] / pred
	}
	return derivatives
}

// HingeLoss calculates hinge loss for binary classification
func HingeLoss(predictions, targets []float64) float64 {
	if len(predictions) != len(targets) {
		return 0
	}
	
	sum := 0.0
	for i, pred := range predictions {
		// Convert targets from [0,1] to [-1,1]
		target := 2*targets[i] - 1
		loss := math.Max(0, 1-target*pred)
		sum += loss
	}
	return sum / float64(len(predictions))
}

// HingeLossDerivative returns the derivative of hinge loss
func HingeLossDerivative(predictions, targets []float64) []float64 {
	if len(predictions) != len(targets) {
		return nil
	}
	
	derivatives := make([]float64, len(predictions))
	for i, pred := range predictions {
		// Convert targets from [0,1] to [-1,1]
		target := 2*targets[i] - 1
		
		if target*pred < 1 {
			derivatives[i] = -target / float64(len(predictions))
		} else {
			derivatives[i] = 0
		}
	}
	return derivatives
}

// HuberLoss calculates Huber loss (combination of MSE and MAE)
func HuberLoss(predictions, targets []float64) float64 {
	return HuberLossWithDelta(predictions, targets, 1.0)
}

// HuberLossWithDelta calculates Huber loss with custom delta parameter
func HuberLossWithDelta(predictions, targets []float64, delta float64) float64 {
	if len(predictions) != len(targets) {
		return 0
	}
	
	sum := 0.0
	for i, pred := range predictions {
		diff := pred - targets[i]
		absDiff := math.Abs(diff)
		
		if absDiff <= delta {
			sum += 0.5 * diff * diff
		} else {
			sum += delta*absDiff - 0.5*delta*delta
		}
	}
	return sum / float64(len(predictions))
}

// HuberLossDerivative returns the derivative of Huber loss
func HuberLossDerivative(predictions, targets []float64) []float64 {
	return HuberLossDerivativeWithDelta(predictions, targets, 1.0)
}

// HuberLossDerivativeWithDelta returns the derivative of Huber loss with custom delta
func HuberLossDerivativeWithDelta(predictions, targets []float64, delta float64) []float64 {
	if len(predictions) != len(targets) {
		return nil
	}
	
	derivatives := make([]float64, len(predictions))
	for i, pred := range predictions {
		diff := pred - targets[i]
		absDiff := math.Abs(diff)
		
		if absDiff <= delta {
			derivatives[i] = diff / float64(len(predictions))
		} else {
			if diff > 0 {
				derivatives[i] = delta / float64(len(predictions))
			} else {
				derivatives[i] = -delta / float64(len(predictions))
			}
		}
	}
	return derivatives
}

// GetLossFunction returns the loss function by name
func GetLossFunction(name string) (LossFunction, LossDerivative) {
	switch name {
	case "mse":
		return MeanSquaredError, MeanSquaredErrorDerivative
	case "binary_crossentropy":
		return BinaryCrossEntropy, BinaryCrossEntropyDerivative
	case "categorical_crossentropy":
		return CategoricalCrossEntropy, CategoricalCrossEntropyDerivative
	case "hinge":
		return HingeLoss, HingeLossDerivative
	case "huber":
		return HuberLoss, HuberLossDerivative
	default:
		return MeanSquaredError, MeanSquaredErrorDerivative
	}
}

// CalculateAccuracy calculates classification accuracy
func CalculateAccuracy(predictions, targets []float64) float64 {
	if len(predictions) != len(targets) {
		return 0
	}
	
	correct := 0
	for i, pred := range predictions {
		predClass := 0
		if pred >= 0.5 {
			predClass = 1
		}
		
		targetClass := 0
		if targets[i] >= 0.5 {
			targetClass = 1
		}
		
		if predClass == targetClass {
			correct++
		}
	}
	
	return float64(correct) / float64(len(predictions))
} 