# Activation Functions in Neural Networks

This document explains **Activation Functions** used in Neural Networks in a **clear, exam‑ready, and beginner‑friendly** manner.  
It is suitable for **Machine Learning / Deep Learning study, revision, and GitHub documentation**.

---

## 1. What is an Activation Function?

An **activation function** is a mathematical function applied inside each neuron of a neural network.  
It decides whether a neuron should **activate and pass information forward** or not.

### Neuron Equation
z = g(w₁x₁ + w₂x₂ + … + wₙxₙ + b)

Where:
- w → weights  
- x → inputs  
- b → bias  
- g(·) → activation function  

---

## 2. Why Activation Functions Are Needed

- Without activation functions, a neural network behaves like a **linear model**
- Real‑world data is **non‑linear**
- Activation functions help neural networks **learn complex patterns**
- Therefore, activation functions **must be non‑linear**

---

## 3. Ideal Activation Function

An ideal activation function should have the following properties:

1. Non‑linear  
2. Differentiable (for gradient descent & backpropagation)  
3. Computationally inexpensive  
4. Zero‑centered (better optimization)  
5. Non‑saturating (avoids vanishing gradient problem)  

---

## 4. Sigmoid Activation Function

### Definition
σ(x) = 1 / (1 + e⁻ˣ)

### Range
(0, 1)

### Usage
- Commonly used in **output layer**
- Used for **binary classification**
- Output can be interpreted as **probability**

### Advantages
- Probability‑based output
- Non‑linear
- Differentiable

### Disadvantages
- Saturating function
- Vanishing gradient problem
- Not zero‑centered
- Computationally expensive

---

## 5. Tanh Activation Function

### Definition
tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)

### Range
(−1, 1)

### Derivative
d/dx (tanh x) = 1 − tanh²(x)

### Advantages
- Non‑linear
- Differentiable
- Zero‑centered (better than sigmoid)

### Disadvantages
- Saturating function
- Vanishing gradient problem
- Computationally expensive

---

## 6. ReLU Activation Function (Rectified Linear Unit)

### Definition
f(x) = max(0, x)

### Advantages
- Non‑linear
- Non‑saturating in positive region
- Computationally inexpensive
- Faster convergence
- Most widely used activation function

### Disadvantages
- Not zero‑centered
- Not differentiable at x = 0
- Dying ReLU problem

---

## 7. Quick Comparison

| Activation | Range | Zero‑Centered | Saturation | Usage |
|-----------|------|---------------|------------|-------|
| Sigmoid | (0,1) | ❌ | Yes | Binary output |
| Tanh | (−1,1) | ✅ | Yes | Hidden layers |
| ReLU | [0,∞) | ❌ | No (positive side) | Hidden layers |

---

## 8. Final Summary

- Use **Sigmoid** → Binary classification output layer  
- Use **Tanh** → Better than sigmoid but still saturates  
- Use **ReLU** → Best default choice for hidden layers  

---

## Author
Prepared for Machine Learning & Deep Learning study and revision.


