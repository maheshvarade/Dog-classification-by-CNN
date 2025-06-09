# 🐕 Dog Breed Classification Using MobileNet 📱🧠
This project focuses on building a deep learning model to classify images into 120 dog breeds using the MobileNet architecture. The dataset is highly imbalanced, so special care was taken to handle class distribution and improve model generalization.

Datasets:'https://www.kaggle.com/datasets/amandam1/120-dog-breeds-breed-classification'
📁 Project Overview
Objective: Classify dog images into 120 breed categories.

Dataset: Pre-split into train, validation, and test folders (image data).

Input Size: 224x224x3 (resized all images for model compatibility).

Model: MobileNet (transfer learning with fine-tuning).

Classes: 120 unique dog breeds.

Framework: TensorFlow / Keras.

⚙️ Key Steps
1. Data Preprocessing
Loaded train, validation, and test datasets using ImageDataGenerator.

Resized all images to 224x224 with RGB channels.

Encoded class labels automatically from folder structure.

Used class_weight to balance training across classes.

2. Handling Class Imbalance
Checked for class distribution in train/val/test sets.

Used class weighting to prioritize learning from under-represented breeds.

Explored augmentation options, but stuck to class weights for simplicity and efficiency.

3. Model Building – MobileNet
python
Copy code
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
Loaded pre-trained MobileNet with imagenet weights (without top layers).

Added:

GlobalAveragePooling2D

Dense(256, relu)

Dropout(0.3)

Final Dense(120, softmax) layer

Compiled with Adam(learning_rate=0.0001) and categorical_crossentropy.

4. Training & Callbacks
Used EarlyStopping and ModelCheckpoint to prevent overfitting and save the best model.

Trained with class_weight parameter.

📊 Evaluation
Top-1 Accuracy: X.XX (replace with actual)

Top-3 Accuracy: 0.93

Loss/Accuracy Curves: Tracked with history.history.

Confusion Matrix
Created color-coded confusion matrix to visualize misclassified classes.

Highlighted errors using a diverging colormap (coolwarm) with the diagonal masked.


✅ Key Learnings
MobileNet is fast and lightweight, ideal for deployment and efficient training.

Class imbalance can be effectively handled using class_weight when augmentation is limited.

Top-3 accuracy is a better metric when classes are highly similar.

Error heatmaps are great for identifying confused classes.

📦 Future Work
Experiment with fine-tuning more layers of MobileNet.

Use data augmentation for rare breeds.

Deploy the model as a web app or API.

Compare performance with EfficientNet and ResNet.
