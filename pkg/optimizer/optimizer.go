package optimizer

import "math"

// Optimizer interface defines methods for updating weights and biases
type Optimizer interface {
	UpdateWeights(weights [][]float64, gradients [][]float64, layerIndex int) [][]float64
	UpdateBiases(biases []float64, gradients []float64, layerIndex int) []float64
	GetName() string
}

// SGD implements Stochastic Gradient Descent optimizer
type SGD struct {
	learningRate float64
	momentum     float64
	velocities   map[int][][]float64 // layer index -> velocity matrix
	biasVelocities map[int][]float64 // layer index -> bias velocity
}

// NewSGD creates a new SGD optimizer
func NewSGD(learningRate, momentum float64) *SGD {
	return &SGD{
		learningRate:   learningRate,
		momentum:       momentum,
		velocities:     make(map[int][][]float64),
		biasVelocities: make(map[int][]float64),
	}
}

// UpdateWeights updates weights using SGD with momentum
func (s *SGD) UpdateWeights(weights [][]float64, gradients [][]float64, layerIndex int) [][]float64 {
	rows, cols := len(weights), len(weights[0])
	
	// Initialize velocity if not exists
	if _, exists := s.velocities[layerIndex]; !exists {
		s.velocities[layerIndex] = make([][]float64, rows)
		for i := range s.velocities[layerIndex] {
			s.velocities[layerIndex][i] = make([]float64, cols)
		}
	}
	
	// Update weights with momentum
	for i := range weights {
		for j := range weights[i] {
			s.velocities[layerIndex][i][j] = s.momentum*s.velocities[layerIndex][i][j] + s.learningRate*gradients[i][j]
			weights[i][j] -= s.velocities[layerIndex][i][j]
		}
	}
	
	return weights
}

// UpdateBiases updates biases using SGD with momentum
func (s *SGD) UpdateBiases(biases []float64, gradients []float64, layerIndex int) []float64 {
	// Initialize bias velocity if not exists
	if _, exists := s.biasVelocities[layerIndex]; !exists {
		s.biasVelocities[layerIndex] = make([]float64, len(biases))
	}
	
	// Update biases with momentum
	for i := range biases {
		s.biasVelocities[layerIndex][i] = s.momentum*s.biasVelocities[layerIndex][i] + s.learningRate*gradients[i]
		biases[i] -= s.biasVelocities[layerIndex][i]
	}
	
	return biases
}

// GetName returns the optimizer name
func (s *SGD) GetName() string {
	return "SGD"
}

// Adam implements Adam optimizer
type Adam struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	m            map[int][][]float64 // first moment
	v            map[int][][]float64 // second moment
	mBias        map[int][]float64   // bias first moment
	vBias        map[int][]float64   // bias second moment
	t            int                  // time step
}

// NewAdam creates a new Adam optimizer
func NewAdam(learningRate, beta1, beta2, epsilon float64) *Adam {
	return &Adam{
		learningRate: learningRate,
		beta1:        beta1,
		beta2:        beta2,
		epsilon:      epsilon,
		m:            make(map[int][][]float64),
		v:            make(map[int][][]float64),
		mBias:        make(map[int][]float64),
		vBias:        make(map[int][]float64),
		t:            0,
	}
}

// UpdateWeights updates weights using Adam optimizer
func (a *Adam) UpdateWeights(weights [][]float64, gradients [][]float64, layerIndex int) [][]float64 {
	rows, cols := len(weights), len(weights[0])
	
	// Initialize moments if not exists
	if _, exists := a.m[layerIndex]; !exists {
		a.m[layerIndex] = make([][]float64, rows)
		a.v[layerIndex] = make([][]float64, rows)
		for i := range a.m[layerIndex] {
			a.m[layerIndex][i] = make([]float64, cols)
			a.v[layerIndex][i] = make([]float64, cols)
		}
	}
	
	a.t++
	
	// Update weights using Adam
	for i := range weights {
		for j := range weights[i] {
			// Update biased first moment estimate
			a.m[layerIndex][i][j] = a.beta1*a.m[layerIndex][i][j] + (1-a.beta1)*gradients[i][j]
			
			// Update biased second raw moment estimate
			a.v[layerIndex][i][j] = a.beta2*a.v[layerIndex][i][j] + (1-a.beta2)*gradients[i][j]*gradients[i][j]
			
			// Compute bias-corrected first moment estimate
			mHat := a.m[layerIndex][i][j] / (1 - math.Pow(a.beta1, float64(a.t)))
			
			// Compute bias-corrected second raw moment estimate
			vHat := a.v[layerIndex][i][j] / (1 - math.Pow(a.beta2, float64(a.t)))
			
			// Update weights
			weights[i][j] -= a.learningRate * mHat / (math.Sqrt(vHat) + a.epsilon)
		}
	}
	
	return weights
}

// UpdateBiases updates biases using Adam optimizer
func (a *Adam) UpdateBiases(biases []float64, gradients []float64, layerIndex int) []float64 {
	// Initialize bias moments if not exists
	if _, exists := a.mBias[layerIndex]; !exists {
		a.mBias[layerIndex] = make([]float64, len(biases))
		a.vBias[layerIndex] = make([]float64, len(biases))
	}
	
	// Update biases using Adam
	for i := range biases {
		// Update biased first moment estimate
		a.mBias[layerIndex][i] = a.beta1*a.mBias[layerIndex][i] + (1-a.beta1)*gradients[i]
		
		// Update biased second raw moment estimate
		a.vBias[layerIndex][i] = a.beta2*a.vBias[layerIndex][i] + (1-a.beta2)*gradients[i]*gradients[i]
		
		// Compute bias-corrected first moment estimate
		mHat := a.mBias[layerIndex][i] / (1 - math.Pow(a.beta1, float64(a.t)))
		
		// Compute bias-corrected second raw moment estimate
		vHat := a.vBias[layerIndex][i] / (1 - math.Pow(a.beta2, float64(a.t)))
		
		// Update biases
		biases[i] -= a.learningRate * mHat / (math.Sqrt(vHat) + a.epsilon)
	}
	
	return biases
}

// GetName returns the optimizer name
func (a *Adam) GetName() string {
	return "Adam"
}

// RMSprop implements RMSprop optimizer
type RMSprop struct {
	learningRate float64
	rho          float64
	epsilon      float64
	v            map[int][][]float64 // moving average of squared gradients
	vBias        map[int][]float64   // bias moving average
}

// NewRMSprop creates a new RMSprop optimizer
func NewRMSprop(learningRate, rho, epsilon float64) *RMSprop {
	return &RMSprop{
		learningRate: learningRate,
		rho:          rho,
		epsilon:      epsilon,
		v:            make(map[int][][]float64),
		vBias:        make(map[int][]float64),
	}
}

// UpdateWeights updates weights using RMSprop optimizer
func (r *RMSprop) UpdateWeights(weights [][]float64, gradients [][]float64, layerIndex int) [][]float64 {
	rows, cols := len(weights), len(weights[0])
	
	// Initialize moving average if not exists
	if _, exists := r.v[layerIndex]; !exists {
		r.v[layerIndex] = make([][]float64, rows)
		for i := range r.v[layerIndex] {
			r.v[layerIndex][i] = make([]float64, cols)
		}
	}
	
	// Update weights using RMSprop
	for i := range weights {
		for j := range weights[i] {
			// Update moving average of squared gradients
			r.v[layerIndex][i][j] = r.rho*r.v[layerIndex][i][j] + (1-r.rho)*gradients[i][j]*gradients[i][j]
			
			// Update weights
			weights[i][j] -= r.learningRate * gradients[i][j] / (math.Sqrt(r.v[layerIndex][i][j]) + r.epsilon)
		}
	}
	
	return weights
}

// UpdateBiases updates biases using RMSprop optimizer
func (r *RMSprop) UpdateBiases(biases []float64, gradients []float64, layerIndex int) []float64 {
	// Initialize bias moving average if not exists
	if _, exists := r.vBias[layerIndex]; !exists {
		r.vBias[layerIndex] = make([]float64, len(biases))
	}
	
	// Update biases using RMSprop
	for i := range biases {
		// Update moving average of squared gradients
		r.vBias[layerIndex][i] = r.rho*r.vBias[layerIndex][i] + (1-r.rho)*gradients[i]*gradients[i]
		
		// Update biases
		biases[i] -= r.learningRate * gradients[i] / (math.Sqrt(r.vBias[layerIndex][i]) + r.epsilon)
	}
	
	return biases
}

// GetName returns the optimizer name
func (r *RMSprop) GetName() string {
	return "RMSprop"
}

// GetOptimizer returns an optimizer by name
func GetOptimizer(name string, learningRate float64) Optimizer {
	switch name {
	case "sgd":
		return NewSGD(learningRate, 0.9)
	case "adam":
		return NewAdam(learningRate, 0.9, 0.999, 1e-8)
	case "rmsprop":
		return NewRMSprop(learningRate, 0.9, 1e-8)
	default:
		return NewSGD(learningRate, 0.0)
	}
} 