# Activation Functions in Neural Networks

This document explains **Activation Functions** used in Neural Networks in a **clear, exam-ready, and GitHub-friendly** manner.  
All formulas are written in **LaTeX-style** for better readability on GitHub.

---

## 1. What is an Activation Function?

An **activation function** is a mathematical function applied inside each neuron of a neural network.  
It decides whether a neuron should **activate and pass information forward** or not.

### Neuron Equation

$$
z = g\left( \sum_{i=1}^{n} w_i x_i + b \right)
$$

Where:
- $w$ → weights  
- $x$ → inputs  
- $b$ → bias  
- $g(\cdot)$ → activation function  

---

## 2. Why Activation Functions Are Needed

- Without activation functions, a neural network behaves like a **linear model**
- Real-world data is **non-linear**
- Activation functions help neural networks **learn complex patterns**
- Therefore, activation functions **must be non-linear**

---

## 3. Ideal Activation Function

An ideal activation function should satisfy:

1. **Non-linear**
2. **Differentiable** (for gradient descent & backpropagation)
3. **Computationally inexpensive**
4. **Zero-centered**
5. **Non-saturating** (avoids vanishing gradient problem)

---

## 4. Sigmoid Activation Function

### Definition

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### Range
$$
(0, 1)
$$

### Usage
- Used in **output layer**
- Used for **binary classification**
- Output represents **probability**

### Advantages
- Smooth and non-linear
- Differentiable
- Probabilistic interpretation

### Disadvantages
- Saturates at extreme values
- Suffers from **vanishing gradient problem**
- Not zero-centered
- Computationally expensive

---

## 5. Tanh Activation Function

### Definition

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

### Range
$$
(-1, 1)
$$

### Derivative

$$
\frac{d}{dx} (\tanh x) = 1 - \tanh^2(x)
$$

### Advantages
- Zero-centered
- Differentiable
- Stronger gradients than sigmoid

### Disadvantages
- Saturating function
- Vanishing gradient problem
- Computationally expensive

---

## 6. ReLU Activation Function (Rectified Linear Unit)

### Definition

$$
f(x) = \max(0, x)
$$

### Advantages
- Computationally efficient
- Non-saturating for $x > 0$
- Faster convergence
- Most widely used activation function

### Disadvantages
- Not zero-centered
- Not differentiable at $x = 0$
- **Dying ReLU problem**

---

## 7. Comparison Table

| Activation | Mathematical Form | Range | Zero-Centered | Common Usage |
|-----------|------------------|-------|---------------|--------------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | (0,1) | ❌ | Binary output |
| Tanh | $\tanh(x)$ | (-1,1) | ✅ | Hidden layers |
| ReLU | $\max(0,x)$ | [0,∞) | ❌ | Hidden layers |

---

## 8. Final Summary

- **Sigmoid** → Binary classification output layer  
- **Tanh** → Better than sigmoid but still saturates  
- **ReLU** → Default choice for hidden layers  

---

## Author
Prepared for Machine Learning & Deep Learning study and revision.
