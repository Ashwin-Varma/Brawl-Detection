# Brawl Detection in Muggle Surveillance Footage

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Approach and Methodology](#approach-and-methodology)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Implementation Details](#implementation-details)
6. [Execution Instructions](#execution-instructions)
7. [Results and Performance](#results-and-performance)
8. [Bonus Task: Counting People in Brawls](#bonus-task-counting-people-in-brawls)
9. [Repository Structure](#repository-structure)
10. [References](#references)

## Introduction

The wizarding world faces a dire threat as Noctifera and the Veilborn attempt to disrupt the peace. To combat this, the Ministry of Magic has launched Operation Vigilate, a surveillance initiative that monitors Muggle spaces for disturbances. Our task is to develop an AI-powered magical detection system that identifies brawls in Muggle surveillance videos and, as an additional challenge, estimates the number of people involved in a brawl.

## Dataset Description

The dataset consists of 1,600 labeled surveillance videos divided into two categories:

- **Peace Videos (800):** Footage showing individuals engaging in routine activities.
- **Brawl Videos (800):** Footage depicting physical altercations with rapid and chaotic movements.

Each video is between 1 to 10 seconds long. The dataset is used to train a deep learning model for automated brawl detection.

## Approach and Methodology

Our approach leverages deep learning-based action recognition using I3D (Inflated 3D ConvNet), a powerful video classification model trained on the Kinetics dataset. The workflow includes:

- **Preprocessing:** Extracting frames from videos and resizing them to fit the I3D model's input.
- **Feature Extraction:** Using a pre-trained I3D model from TensorFlow Hub to obtain high-dimensional feature vectors.
- **Training a Classifier:** Feeding the extracted features into a classifier (e.g., SVM or fully connected neural network) to distinguish between peace and brawl videos.
- **Evaluation:** Computing the F1-score to measure classification performance.

## Model Training and Evaluation

### Feature Extraction

We use OpenCV to load and preprocess videos, ensuring each clip has a uniform number of frames. The I3D model extracts spatiotemporal features from the videos. These features are used to train a supervised classifier.

### Classification Model

We experimented with different classifiers (e.g., SVM, neural networks) to achieve optimal performance. The final model was selected based on cross-validation results.

### Performance Metric

The primary evaluation metric is F1-score, ensuring a balance between precision and recall.

## Implementation Details

### Libraries Used

- **TensorFlow & TensorFlow Hub:** For I3D feature extraction.
- **OpenCV:** For video processing.
- **NumPy & Pandas:** For data manipulation.
- **Scikit-learn:** For classification models.

### Preprocessing Pipeline

1. Extract 64 frames per video (or pad if fewer frames exist).
2. Resize frames to (224, 224) pixels.
3. Convert frames to RGB format.
4. Extract deep learning features using I3D.
5. Train a classifier on extracted features.

## Execution Instructions

We implemented a pipeline that processes video data, extracts features using I3D, and classifies the videos into peace or brawl categories. The training process involved:

1. Preprocessing the dataset and extracting 64 frames per video.
2. Using a pre-trained I3D model to extract feature vectors.
3. Training a classifier (SVM) on these feature vectors.
4. Evaluating performance using precision, recall, and F1-score.
5. Implementing YOLO-based person detection for counting people in brawl videos.

## Results and Performance

### Classification Report

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| peace      | 0.99      | 0.90   | 0.94     | 200     |
| brawl      | 0.91      | 0.99   | 0.95     | 200     |

- **Accuracy:** 0.94
- **Macro Average:** Precision 0.95, Recall 0.95, F1-score 0.94
- **Weighted Average:** Precision 0.95, Recall 0.94, F1-score 0.94

**F1 Score:** 0.9474

## Bonus Task: Counting People in Brawls

For the bonus task, we employ YOLOv5 to detect and count individuals in brawl videos. Steps:

1. Run YOLOv5 on detected brawl videos.
2. Count the number of bounding boxes corresponding to people.
3. Store the results alongside classification outputs.

## Repository Structure

brawl-detection/
├── Dataset/
│ ├── peace/
│ ├── brawl/
├── train.ipynb # Training script
├── predict.csv # Prediction script
├── count_people.ipynb # Bonus task script
├── README.md # Project documentation


## References

- Carreira, J., & Zisserman, A. (2017). Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset.
- Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement.
