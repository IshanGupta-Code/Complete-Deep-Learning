# IMDB Sentiment Analysis with Keras

Sentiment analysis is performed on the **IMDB movie reviews dataset** that comes bundled with **Keras**. The dataset contains **50,000 labeled reviews**, split into **25,000 training samples** (with a validation split) and **25,000 test samples**. Reviews are integer-encoded, where only the **10,000 most frequent words** are retained to build the vocabulary.

Self-trained word embeddings are learned using the **Keras Embedding layer**, followed by a neural network–based architecture for binary sentiment classification.

---

## Dataset Details

- Source: IMDB dataset (Keras built-in)  
- Total samples: 50,000  
- Training samples: 25,000  
- Test samples: 25,000  
- Vocabulary size: **10,000 most frequent words**  
- Classes: Positive / Negative  

---

## Model Architecture

The implemented model follows a deep learning approach with:

- Embedding layer for word representation  
- Fully connected (Dense) layers with ReLU activation  
- Dropout layers to reduce overfitting  
- Sigmoid output layer for binary classification  

The model is trained using:
- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy, Precision, Recall, F1-score  

---

## Results

The model achieves the following performance on the test dataset:

- **Accuracy:** **85.54%**
- Balanced precision and recall across both sentiment classes
- Stable predictions confirmed through confusion matrix analysis

---

## Evaluation Strategy

Since this task involves **binary classification**, regression metrics such as R² were avoided. Instead, classification-focused metrics were used to ensure meaningful and reliable evaluation.

---

## Implementation Details

- Framework: Keras (TensorFlow backend)  
- Programming Language: Python  
- Environment: Jupyter Notebook  
- Vocabulary size limited to **10,000 words**  
- Compatible with both CPU and GPU execution  

---

## References

- *Deep Learning with Python* — François Chollet  
- Keras Official Documentation  

---

## Author

**Ishan Gupta**  
Date: 2025  
