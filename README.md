# Parkinson's Disease Detection using CNN-LSTM

## Overview
This work utilizes a deep learning model for the detection of Parkinson's disease depending on the vocal features of the patients. The model utilizes feature extraction through Convolutional Neural Networks (CNN) with sequence pattern recognition using Bidirectional Long Short Term Memory (Bi-LSTM) networks. The pre-processing of data includes normalization and dimensionality reduction by using UMAP. Model evaluation will carry out with the use of accuracy, classification reports, confusion matrices, and ROC curves.

## Dataset
The dataset used is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data). It is a dataset obtained from multiple vocal features of patients with the column `status`, which shows if they are diagnosed with or are without Parkinson's disease (1 and 0, respectively).

## Features Used
- **Vocal features** obtained from the patients.
- **Normalization** through StandardScaler.
- **Dimensionality reduction using** UMAP.
- **Feature visualization** via KDE plots and UMAP projections.

## Model Architecture
The deep learning model comprises:
- **Convolutional layers (CNN)** for feature extraction.
- **Batch normalization and dropout** for regularization.
- **Global Average Pooling** for dimension reduction.
- **Bidirectional LSTM layers** for pattern recognition in sequence.
- **Final dense layers** for classification output.
- **Sigmoid activation function** providing binary classification.

## Installation
Run the required dependencies to execute this project:
```bash
pip install numpy pandas seaborn tensorflow shap umap-learn scikit-learn matplotlib
```

## Usage
1. **Execute the script**:
```bash
python parkinsons_diesease_detection.py
```
2. The resulting model would be stored in `manual_tuned_parkinsons_model.h5`.
3. The script will print accuracy, classification report, confusion matrix, and visualization.

## Evaluation Metrics
- **Accuracy**
- **Classification Report containing Precision, Recall, and F1-Score**
- **Confusion Matrix**
- **ROC Curve with Threshold Markings**

## Model Performance
Here is how the model is expected to perform after evaluation on the test dataset, along with proper visualization for classification accuracy and feature distribution.

## Visualizations
- **Feature Distribution Graphs** (Kernel Density Estimation for `PPE` feature)
- **UMAP Features Projection**
- **Heatmap Confusion Matrix**
- **Thresholds on ROC Curve**

## Output
After finishing execution, it provides:
- A trained CNN-LSTM model saved as `manual_tuned_parkinsons_model.h5`
- Performance metrics
- Visual representations for feature study and classification analysis

## License
This is an open-source project and free for research and educational usage.

## Author
SANJAI.S
