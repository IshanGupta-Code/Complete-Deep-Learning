# Activation Functions in Neural Networks

This README explains **Activation Functions** with definitions, formulas, advantages, disadvantages, and visual intuition using graphs.

---

## What is an Activation Function?

An activation function decides whether a neuron should be activated or not.

$$
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$

$$
a = g(z)
$$

---

## Ideal Properties of an Activation Function

- Non-linear  
- Differentiable  
- Computationally efficient  
- Zero-centered  
- Non-saturating  

---

## 1. Sigmoid Activation Function

### Formula
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### Graph
![Sigmoid Function](images/sigmoid.png)

### Advantages
- Smooth and differentiable  
- Probability output  

### Disadvantages
- Vanishing gradient  
- Not zero-centered  

---

## 2. Tanh Activation Function

### Formula
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

### Graph
![Tanh Function](images/tanh.png)

### Advantages
- Zero-centered  
- Better than sigmoid  

### Disadvantages
- Vanishing gradient  

---

## 3. ReLU (Rectified Linear Unit)

### Formula
$$
\text{ReLU}(x) = \max(0, x)
$$

### Graph
![ReLU Function](images/relu.png)

### Advantages
- Fast computation  
- Faster convergence  

### Disadvantages
- Dying ReLU problem  

---

## Dying ReLU Problem

When ReLU neurons output zero constantly due to negative inputs, learning stops.

### Solutions
- Lower learning rate  
- Positive bias  
- Use ReLU variants  

---

## 3.1 Leaky ReLU

### Formula
$$
f(x) =
\begin{cases}
x, & x > 0 \\
0.01x, & x \le 0
\end{cases}
$$

### Graph
![Leaky ReLU Function](images/leaky_relu.png)

---

## 3.2 Parametric ReLU (PReLU)

### Formula
$$
f(x) =
\begin{cases}
x, & x > 0 \\
ax, & x \le 0
\end{cases}
$$

### Graph
![PReLU Function](images/prelu.png)

---

## 3.3 ELU (Exponential Linear Unit)

### Formula
$$
f(x) =
\begin{cases}
x, & x > 0 \\
\alpha(e^x - 1), & x \le 0
\end{cases}
$$

### Graph
![ELU Function](images/elu.png)

---

## 3.4 SELU (Scaled ELU)

### Formula
$$
f(x) = \lambda
\begin{cases}
x, & x > 0 \\
\alpha(e^x - 1), & x \le 0
\end{cases}
$$

### Graph
![SELU Function](images/selu.png)

---

## Summary Table

| Function | Zero-Centered | Non-Saturating | Key Issue |
|--------|--------------|----------------|----------|
| Sigmoid | ❌ | ❌ | Vanishing Gradient |
| Tanh | ✅ | ❌ | Vanishing Gradient |
| ReLU | ❌ | ✅ | Dying ReLU |
| Leaky ReLU | ❌ | ✅ | Parameter tuning |
| PReLU | ❌ | ✅ | Extra parameters |
| ELU | ✅ | ❌ | Expensive |
| SELU | ✅ | ❌ | Special constraints |

---

## Conclusion

Activation functions introduce non-linearity into neural networks. ReLU is widely used, while its variants overcome limitations like dying neurons and poor convergence.
