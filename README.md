# Traffic-sign-recognition-model-
This repository contains two classical ML model based implementation of Traffic sign recognition aswell as deep learning model based implementation. This repository also compares and evaluates these models

# Traffic Sign Recognition using Deep Learning and Classical Machine Learning

**Course:** Machine Learning  
**Project Type:** Image Classification  
**Dataset:** German Traffic Sign Dataset (GTSRB-style subset)

**Team Members:**  
- Haseeb Ur Rehman


## Abstract

Traffic sign recognition has played a huge role in the pervalence of autonomous driving. This project not only implements traffic sign recognition using deep learning, and classical machine learning algoithim but it also compares them. The Convolutional Neural Network is able to learn directly from images, but the classical models which in this project are Support Vector Machine and Random Forest are trained after feature extraction from images using dense neural network. The models made in this project are evaluated using many evalutaiion metrics, confusion matrices and statistical significance testing. After comparison of models it can clearly be deduced that deep  learning significantly outperforms classical approaches. 

---

## 1. Introduction

Traffic sign recognition involves classifying images of road signs into  categories such as speed limits and warnings. Accurate recognition is essential for road safety and driver assistance systems. The objective of this project is to:

- Implement a CNN-based classifier 
- Implement and evaluate classical machine learning algorithms on extracted image features
- Perform a comprehensive comparative analysis between all three models

---

## 2. Dataset Description

### Dataset Source
The dataset is derived from the German Traffic Sign Recognition Benchmark (GTSRB) format.


### Dataset Characteristics
- Multi-class classification 
- RGB images of varying resolutions
- Feature scaled images

### Preprocessing
- Images resized 
- Pixel values normalized 
- Stratified train/validation/test split
- Reproducibility ensured using fixed random seeds

---

## 3. Methodology

### 3.1 Deep Learning Approach


**Key architectural components:**
- Convolutional layers with ReLU activation
- Batch Normalization for stable training
- Dropout for regularization
- MaxPooling for spatial downsampling
- Flattening for input handling 

**Training strategies:**
- Adam optimizer
- Early stopping
- Learning rate scheduling

---

### 3.2 Classical Machine Learning Approaches

Classical Machine Learning approaches are not capable of learning well from images so feature extraction is used before implementing classical aproaches

#### Feature Extraction
- Resolution of images is decreased 
- Flattened into 1D vectors
- Dense neural encoder for feature extraction

The activations from the second-to-last dense layer are used as final feature vectors.

#### Classical Models Implemented
- **Support Vector Machine (SVM)** with RBF kernel
- **Random Forest Classifier**

#### Hyperparameter Tuning
- GridSearchCV
- cross-validation


---

## 4. Results & Analysis

### 4.1 Evaluation Metrics

The following metrics are reported:
- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1-score (macro-averaged)
- Confusion Matrix

### 4.2 Performance Comparison
<img width="795" height="168" alt="image" src="https://github.com/user-attachments/assets/9ace9fc5-ec12-4ddb-a8d9-8d55bcde833a" />


### 4.3 Statistical Significance Testing

To determine if the differences in perfromances are statistically meaningful:

- **McNemar’s Test** is applied between model predictions
- **Permutation testing** is used to validate accuracy differences

Results indicate that the CNN’s performance improvement over classical models is statistically significant .

---

## 5. Results Visualization

- Confusion matrices for all models
- Comparative bar plots of evaluation metrics
- Statistical test summaries


---

## 6. Business & Practical Impact

Accurate traffic sign recognition systems can:
- Enhance autonomous driving safety
- Reduce driver workload
- Enable real-time traffic regulation compliance
- Improve intelligent transportation systems in smart cities

---

## 7. Conclusion & Future Work

### Conclusion
This study demonstrates that CNN-based deep learning models significantly outperform classical machine learning methods on raw image data. However, classical models can still achieve reasonable performance when paired with learned feature representations.

### Future Work
- Use larger and more diverse datasets
- Apply data augmentation techniques
- Evaluate model under adverse conditions such as blur pictures or dimly lit settingf

---

## 8. References 

https://www.geeksforgeeks.org/deep-learning/traffic-signs-recognition-using-cnn-and-keras-in-python/
https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/
https://youtu.be/KuXjwB4LzSA?si=wqf9hAi3ea1etZyi
https://www.geeksforgeeks.org/deep-learning/tensorflow/
https://www.geeksforgeeks.org/python/opencv-python-tutorial/
https://www.w3schools.com/python/pandas/default.asp
https://www.w3schools.com/python/matplotlib_intro.asp
https://www.w3schools.com/python/numpy/default.asp

