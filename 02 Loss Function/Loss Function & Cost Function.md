# Loss Function, Cost Function & Error Function (Deep Learning)

This README explains **Loss Functions, Cost Functions, and Error Functions** used in **Regression and Classification** problems in Deep Learning.  


---

## 1. Loss Function vs Cost Function

- **Loss Function**  
  Measures the error for **one single data point**.
  
  Example:
  L = 1/2 (y - ŷ)²

- **Cost Function**  
  Measures the **average loss over the entire dataset (batch)**.
  
  Example:
  C = Σ (1/t) L

---

## 2. Types of Machine Learning Problems

1. Regression – Continuous numerical output  
2. Classification – Discrete class output (Categorical Data)

---

## 3. Loss Functions for Regression

### 3.1 Mean Squared Error (MSE)

Loss:
L = (y - ŷ)²

Cost:
C = Σ (1/t) (y - ŷ)²

#### Advantages
- Quadratic in nature
- Smooth gradient descent
- Only one global minimum
- Penalizes large errors heavily

#### Disadvantage
- Not robust to outliers

---

### 3.2 Mean Absolute Error (MAE)

Loss:
L = |y - ŷ|

Cost:
C = Σ (1/t) |y - ŷ|

#### Advantages
- More robust to outliers than MSE

#### Disadvantages
- Non-smooth gradient
- Can have local minima
- Optimization is computationally difficult

---

### 3.3 Huber Loss

A combination of **MSE and MAE**.

Loss:
If |y - ŷ| ≤ δ  
→ 1/2 (y - ŷ)²  

Else  
→ δ|y - ŷ| − 1/2 δ²  

Where δ is a **hyperparameter**.

#### Advantage
- Stable optimization
- Robust to outliers

---

## 4. Loss Functions for Classification

### 4.1 Binary Cross Entropy (Log Loss)

Used in **Binary Classification** (Logistic Regression).

Loss:
− y log(ŷ) − (1 − y) log(1 − ŷ)

Cases:
- If y = 1 → −log(ŷ)
- If y = 0 → −log(1 − ŷ)

Activation Function:
Sigmoid  
ŷ = 1 / (1 + e⁻ᶻ)

---

### 4.2 Categorical Cross Entropy (Multi-Class)

Used for **Multi-Class Classification**.

Loss:
L(xᵢ, yᵢ) = − Σ yᵢⱼ log(ŷᵢⱼ)



 One-Hot Encoding Example

Classes: Good, Bad, Neutral

- Good → [1, 0, 0]
- Bad → [0, 1, 0]
- Neutral → [0, 0, 1]

Softmax Activation Function

Used in **multi-class classification**.

Softmax:
ŷ = eᶻⁱ / Σ eᶻʲ

Properties:
- Converts raw scores into probabilities
- Sum of probabilities = 1

---

## 5. Quick Summary

- Regression:
  - MSE
  - MAE
  - Huber Loss

- Binary Classification:
  - Binary Cross Entropy + Sigmoid

- Multi-Class Classification:
  - Categorical Cross Entropy + Softmax

- Loss Function → Single data point  
- Cost Function → Entire dataset  

---

## Author
**Ishan Gupta**
