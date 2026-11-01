# 🧠 Deep Learning Journey & Implementations

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg?logo=keras&logoColor=white)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A structured collection of notebooks, experiments, and projects tracking my journey through Deep Learning — from implementing backpropagation and neural nets from scratch to advanced Convolutional, Recurrent, and Attention-based architectures.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Modules & Projects](#-modules--projects)
  - [1. Foundations from Scratch](#1-foundations-from-scratch-andrejipynb)
  - [2. Artificial Neural Networks (ANN)](#2-artificial-neural-networks-ann)
  - [3. Convolutional Neural Networks (CNN)](#3-convolutional-neural-networks-cnn)
  - [4. Transfer Learning](#4-transfer-learning)
  - [5. Advanced Network Design (Functional API)](#5-advanced-network-design-functional-api)
  - [6. Hyperparameter Optimization (Keras Tuner)](#6-hyperparameter-optimization-keras-tuner)
  - [7. Recurrent Neural Networks & NLP](#7-recurrent-neural-networks--nlp)
  - [8. Attention & Encoder-Decoder Architectures](#8-attention--encoder-decoder-architectures)
- [Tech Stack & Tools](#-tech-stack--tools)
- [Getting Started](#-getting-started)
- [How to Use & Contribute](#-how-to-use--contribute)

---

## 🔍 Overview

This repository documents practical deep learning workflows across multiple problem domains:
- **Foundations**: Building an autograd engine (`Value` class) and backpropagation graph from first principles.
- **Tabular & Vision**: Customer churn classification, MNIST digit classification, and Cats vs Dogs image classification.
- **Architectures**: LeNet/CNNs, Transfer Learning with VGG16, multi-input functional models, and Recurrent Neural Networks (RNN / LSTM).
- **Modern NLP**: Tokenization, embedding spaces, next-word prediction, and attention mechanisms in sequence-to-sequence setups.

---

## 📂 Repository Structure

```plaintext
Deep Learning/
├── andrej.ipynb                               # Autograd engine & backprop from scratch
├── ANN/                                       # Artificial Neural Networks
│   ├── Churn problem/
│   │   └── churn.ipynb                        # Customer churn prediction on tabular data
│   └── Mnist data set/
│       ├── mnist.ipynb                        # MNIST handwritten digit classification
│       └── mnist_prediction.ipynb             # Model inference & evaluation
├── CNN/                                       # Convolutional Neural Networks
│   ├── CNN_Architecture.ipynb                 # CNN layers, feature maps & LeNet-style architecture
│   └── padding.ipynb                          # Valid vs same padding, strides & pooling
├── catsvsdogs/
│   └── prediction.ipynb                       # Binary image classification with CNNs
├── TransferLearning/
│   └── predict.ipynb                          # Transfer learning via pre-trained VGG16
├── Functional_models/
│   ├── frist.ipynb                            # Multi-input / multi-branch models with Keras Functional API
│   └── model.png                              # Computational graph visualization
├── Keras Tuner/
│   └── tuner.ipynb                            # Automated hyperparameter search & optimization
├── RNN/                                       # Recurrent Neural Networks & Sequence Models
│   ├── Next_Word_Predictor/
│   │   └── app/
│   │       └── main.ipynb                     # Next-word language model
│   └── practice/
│       ├── rnn.ipynb                          # SimpleRNN layer mechanics and parameter counts
│       ├── embeddings.ipynb                   # Word representations & Keras Embedding layer
│       ├── imbd_data.ipynb                    # IMDB sentiment dataset preprocessing
│       └── imbd_embedding.ipynb               # Sentiment classification using embeddings
├── Attention_and_encoder_decoder/             # Seq2Seq & Attention Mechanisms
│   └── Architecture/
│       ├── Requirements.md                    # Module requirements
│       └── model.ipynb                        # Bidirectional LSTM & Attention mechanism
└── README.md
```

---

### 6. Hyperparameter Optimization (`Keras Tuner/`)
- **[Automated Tuning](file:///c:/Users/Krish/Downloads/Deep%20Learning/Keras%20Tuner/tuner.ipynb)**:
  - Systematic search over model hyperparameters (number of layers, units per layer, learning rates, activation functions).
  - Integration with Scikit-learn preprocessing pipelines (`ColumnTransformer`, `SimpleImputer`).

### 7. Recurrent Neural Networks & NLP (`RNN/`)
- **[RNN Mechanics](file:///c:/Users/Krish/Downloads/Deep%20Learning/RNN/practice/rnn.ipynb)**: Understanding `SimpleRNN` cell state transitions, unrolling over timesteps, and weights/bias parameter derivation.
- **[Word Embeddings](file:///c:/Users/Krish/Downloads/Deep%20Learning/RNN/practice/embeddings.ipynb)**: Tokenization, vocabulary construction, sequence padding (`pad_sequences`), and dense vector representations via the `Embedding` layer.
- **[IMDB Sentiment Analysis](file:///c:/Users/Krish/Downloads/Deep%20Learning/RNN/practice/imbd_embedding.ipynb)**: Binary text classification over movie reviews using embedding layers and recurrent nets.
- **[Next-Word Predictor](file:///c:/Users/Krish/Downloads/Deep%20Learning/RNN/Next_Word_Predictor/app/main.ipynb)**: Language modeling using sequences to generate and predict text step-by-step.

### 8. Attention & Encoder-Decoder Architectures (`Attention_and_encoder_decoder/`)
- **[Sequence-to-Sequence Modeling](file:///c:/Users/Krish/Downloads/Deep%20Learning/Attention_and_encoder_decoder/Architecture/model.ipynb)**:
  - Encoder-Decoder paradigm with `Bidirectional(LSTM)`.
  - Attention mechanisms allowing the decoder to focus dynamically across input tokens.
  - Tokenization, lemmatization (`nltk`), and parquet dataset pipelines.

---

## 🛠 Tech Stack & Tools

- **Core Frameworks**: TensorFlow 2.x, Keras
- **Machine Learning & Preprocessing**: Scikit-Learn
- **Natural Language Processing**: NLTK, Tokenizer, Pad Sequences
- **Computer Vision**: OpenCV, PIL / Pillow
- **Data Manipulation & Analysis**: NumPy, Pandas, PyArrow (Parquet)
- **Tuning & Visualization**: Keras Tuner, Matplotlib, Seaborn

---

## ⚡ Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed. A GPU (CUDA-enabled) is recommended for training CNNs and sequence models but not required.

### 1. Clone the Repository
```bash
git clone https://github.com/Krish14-ai/DeepLearning.git
cd DeepLearning
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv dl_env
.\dl_env\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv dl_env
source dl_env/bin/activate
```

