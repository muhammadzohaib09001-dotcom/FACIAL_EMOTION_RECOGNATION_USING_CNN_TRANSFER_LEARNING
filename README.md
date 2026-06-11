# FACIAL_EMOTION_RECOGNATION_USING_CNN_TRANSFER_LEARNING

# Facial Emotion Recognition Using CNN and Transfer Learning

This project focuses on **facial emotion recognition from images** using deep learning techniques, including custom Convolutional Neural Networks (CNNs), hyperparameter-tuned CNNs, transfer learning, and Vision Transformer (ViT)-based models.

The aim of the project is to compare how different preprocessing methods, augmentation strategies, CNN architectures, and transfer learning models perform when classifying facial emotions.

## Project Overview

Facial emotion recognition is an important computer vision task used in areas such as human-computer interaction, affective computing, security, mental health analysis, and intelligent user interfaces.

In this project, multiple deep learning models are trained and evaluated on a facial emotion image dataset containing six emotion classes:

* Ahegao
* Angry
* Happy
* Neutral
* Sad
* Surprise

The project compares the performance of:

* Simple Custom CNN
* Hyperparameter-Tuned CNN
* Transfer Learning Model
* Fine-Tuned Transfer Learning Model
* Vision Transformer Model
* Fine-Tuned Vision Transformer Model

## Research Questions

This project investigates the following questions:

1. How do different data preprocessing and augmentation methods affect the performance of facial emotion recognition models?
2. How do different CNN architectures compare in extracting facial features for emotion recognition?
3. How does the performance of CNN-based models compare with transfer learning models for facial emotion recognition?
4. How do CNN and transfer learning models differ in classification performance and error patterns when evaluated using standard metrics?

## Dataset

The dataset contains **15,454 images** belonging to six emotion classes.

### Class Distribution

| Emotion Class | Number of Images |
| ------------- | ---------------: |
| Ahegao        |            1,206 |
| Angry         |            1,313 |
| Happy         |            3,740 |
| Neutral       |            4,027 |
| Sad           |            3,934 |
| Surprise      |            1,234 |

The dataset was split into:

* **70% training data**
* **15% validation data**
* **15% test data**

## Technologies Used

The project was developed using Python and deep learning libraries.

### Main Libraries

* TensorFlow
* Keras
* Keras Tuner
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Project Structure

```text
FACIAL_EMOTION_RECOGNATION_USING_CNN_TRANSFER_LEARNING/
│
├── FacialEmotionRecognition_Project.ipynb
├── README.md
└── models/
    └── saved trained models
```

## Methodology

### 1. Environment Setup

The required libraries were imported, including TensorFlow, Keras, Keras Tuner, NumPy, Matplotlib, Seaborn, and Scikit-learn.

A fixed random seed was used to improve reproducibility.

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

### 2. Dataset Loading

The dataset was loaded using Keras:

```python
keras.utils.image_dataset_from_directory()
```

Images were resized to:

```python
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
```

### 3. Exploratory Data Analysis

Exploratory Data Analysis was performed to:

* Display sample images from each class
* Check the number of images in each emotion category
* Visualize class distribution using bar charts

### 4. Data Preprocessing

Images were normalized using a `Rescaling` layer:

```python
layers.Rescaling(1./255)
```

Class imbalance was handled using class weights calculated with Scikit-learn:

```python
compute_class_weight()
```

### 5. Data Augmentation

Data augmentation was applied to improve generalization and reduce overfitting.

The augmentation pipeline included:

* Random horizontal flip
* Random rotation
* Random zoom
* Random translation

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])
```

## Models Implemented

### Model 1: Simple Custom CNN

A baseline CNN model was created using convolutional layers, max pooling, dropout, and dense layers.

Main architecture:

* Conv2D layer with 16 filters
* Conv2D layer with 32 filters
* Conv2D layer with 64 filters
* MaxPooling layers
* Flatten layer
* Dense layer with 128 neurons
* Dropout layer
* Softmax output layer

The simple CNN achieved approximately:

```text
Test Accuracy: 0.5803
Test Loss: 1.0262
```

### Model 2: Hyperparameter-Tuned CNN

A deeper CNN model was created and tuned using **Keras Tuner RandomSearch**.

The tuned parameters included:

* Dense units
* Dropout rate
* Learning rate

Best hyperparameters found:

```text
Dense Units: 128
Dropout Rate: 0.4
Learning Rate: 0.0005
```

### Model 3: Transfer Learning Model

A transfer learning approach was implemented using a pretrained deep learning model.

Transfer learning helps improve performance by using pretrained feature extraction layers instead of training all feature representations from scratch.

### Model 4: Fine-Tuned Transfer Learning Model

The transfer learning model was further fine-tuned by unfreezing selected layers and training them on the facial emotion dataset.

This allows the pretrained model to adapt more closely to the target emotion recognition task.

### Model 5: Vision Transformer Model

A Vision Transformer model was also applied to compare transformer-based image classification with CNN-based approaches.

### Model 6: Fine-Tuned Vision Transformer Model

The Vision Transformer model was fine-tuned to improve classification performance on the facial emotion dataset.

## Evaluation Metrics

The models were evaluated using the following metrics:

* Accuracy
* Loss
* Precision
* Recall
* F1-score
* Confusion matrix
* Classification report

For the simple CNN model, the classification report showed that some classes such as **Ahegao** and **Happy** performed better, while classes such as **Angry**, **Sad**, and **Neutral** had more classification confusion.

## Results Summary

The simple CNN model achieved a test accuracy of around **58%**. The model showed reasonable performance but also struggled with some visually similar emotion classes.

The project demonstrates that:

* Data augmentation helps improve generalization.
* Class weighting is useful for handling imbalanced datasets.
* Hyperparameter tuning can improve CNN performance.
* Transfer learning can extract stronger visual features.
* Fine-tuning pretrained models can further improve emotion classification.
* Confusion matrices are useful for identifying misclassified emotion classes.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/FACIAL_EMOTION_RECOGNATION_USING_CNN_TRANSFER_LEARNING.git
cd FACIAL_EMOTION_RECOGNATION_USING_CNN_TRANSFER_LEARNING
```

### 2. Install Required Libraries

```bash
pip install tensorflow keras-tuner numpy matplotlib seaborn scikit-learn
```

### 3. Open the Notebook

```bash
jupyter notebook FacialEmotionRecognition_Project.ipynb
```

Or open it in Google Colab.

### 4. Set Dataset Path

Update the dataset path in the notebook:

```python
dataset_path = "/path/to/your/dataset/"
```

The dataset folder should contain class folders like this:

```text
dataset/
├── Ahegao/
├── Angry/
├── Happy/
├── Neutral/
├── Sad/
└── Surprise/
```

### 5. Run All Cells

Run the notebook cells from top to bottom to:

* Load the dataset
* Perform EDA
* Preprocess images
* Train models
* Evaluate model performance
* Display predictions
* Save trained models

## Saved Models

The trained models can be saved using:

```python
model.save("simple_cnn_emotion_model.keras")
```

You can later load a saved model using:

```python
model = keras.models.load_model("simple_cnn_emotion_model.keras")
```

## Future Improvements

Possible improvements include:

* Using a larger and more balanced dataset
* Applying stronger augmentation techniques
* Testing additional pretrained models such as EfficientNet, ResNet, or MobileNet
* Using face detection before classification
* Improving classification of visually similar emotions
* Deploying the model using Streamlit or Flask
* Creating a real-time webcam emotion recognition system

## Conclusion

This project demonstrates the use of CNNs, hyperparameter tuning, transfer learning, and Vision Transformers for facial emotion recognition. The results show that deep learning models can classify facial emotions with reasonable accuracy, but performance depends heavily on dataset quality, class balance, preprocessing, augmentation, and model architecture.

## Author

**Muhammad Zohaib Khan**

## License

This project is for academic and educational purposes.
