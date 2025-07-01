# Rice Leaf Disease Classification using CNN

This project implements a Convolutional Neural Network (CNN) for classifying rice leaf diseases. The model aims to accurately identify different diseases affecting rice plants, which can aid in early detection and effective disease management.

## Table of Contents

  - [Introduction](https://www.google.com/search?q=%23introduction)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Data Preparation](https://www.google.com/search?q=%23data-preparation)
  - [Model Architecture](https://www.google.com/search?q=%23model-architecture)
  - [Training](https://www.google.com/search?q=%23training)
  - [Evaluation](https://www.google.com/search?q=%23evaluation)
  - [Challenges Faced](https://www.google.com/search?q=%23challenges-faced)
  - [Conclusion](https://www.google.com/search?q=%23conclusion)

## Introduction

Rice is a staple food for a large portion of the world's population. Diseases can significantly impact rice yield and quality. This project leverages deep learning, specifically CNNs, to build a robust system for automated rice leaf disease detection.

## Dataset

The dataset used for this project contains images of rice leaves categorized by disease. The original dataset directory is specified as `C:\Users\mvara\Desktop\datamites\Data_rice`.

## Data Preparation

### Data Split Strategy

The dataset was split into training, validation, and testing sets to ensure fair model evaluation and prevent data leakage.

  * **Training Set:** 60% of the data.
  * **Validation Set:** 20% of the data.
  * **Test Set:** 20% of the data.

A Python function `split_dataset` was used to perform this split, copying images into respective directories under `C:\Users\mvara\Desktop\datamites\split_data_rice`.

### Image Preprocessing and Augmentation

`ImageDataGenerator` from `tensorflow.keras.preprocessing.image` was used for image processing.

  * Images were resized to `224x224` pixels (`IMG_SIZE`).
  * Pixel values were rescaled to a range of 0-1 (by dividing by 255).
  * **Training Data Augmentation:** Horizontal flipping and a zoom range of 0.2 were applied to the training data to increase dataset variability and reduce overfitting.
  * **Validation and Test Data:** Only rescaling was applied.
  * **Batch Size:** A `BATCH_SIZE` of 32 was used for balanced memory usage and training stability on CPU. This helped maintain efficient processing without overloading system resources.

## Model Architecture

The model utilizes transfer learning with `MobileNetV2` as the base model.

  * The `MobileNetV2` base was loaded without its top classification layer, and its layers were frozen for initial training.
  * Custom layers were added on top:
      * `Flatten` layer to convert the 2D feature maps into a 1D vector.
      * `Dense` layers with ReLU activation for feature learning.
      * `Dropout` layers (e.g., 0.5 dropout rate) to prevent overfitting.
      * A final `Dense` layer with `softmax` activation for classification into 3 classes.

## Training

  * **Optimizer:** Adam optimizer with a learning rate of 0.0001.
  * **Loss Function:** Categorical cross-entropy, suitable for multi-class classification.
  * **Metrics:** Accuracy was monitored during training.
  * **Callbacks:**
      * `EarlyStopping`: Monitored validation loss and stopped training if it didn't improve for 10 consecutive epochs.
      * `ModelCheckpoint`: Saved the best model weights based on validation accuracy.

## Evaluation

The model achieved:

  * **Training Accuracy:** 98.59%.
  * **Validation Accuracy:** 97.14%.
  * **Test Accuracy:** 96.77%.

The confusion matrix indicated effective learning across all classes with strong generalization and no major overfitting. Minor confusion was observed between class 1 and others during validation, and balanced precision and recall were achieved across all classes.

## Challenges Faced

  * Limited and imbalanced dataset across classes.
  * Risk of overfitting with a deep model.
  * Initial label mismatch due to `shuffle=False` missing.
  * Difficulty tuning layers during transfer learning.
  * Balancing effective data augmentation without distortion.

## Conclusion

The model is accurate, reliable, and suitable for real-world deployment or integration into a disease detection system. Further improvements can be made by increasing dataset size or fine-tuning the base model.
