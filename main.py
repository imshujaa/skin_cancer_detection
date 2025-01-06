# Import system libraries
import os
import time
import shutil
import itertools

# Import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

# Import deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print('Modules loaded')

# Load the dataset
data_dir = 'hmnist_28_28_RGB.csv'
data = pd.read_csv(data_dir)
print("First five rows of the dataset:")
print(data.head())

data.isnull().sum()

data = data[:5000]

#data.shape

data.dropna(inplace=True)

# Separate labels and features
Label = data["label"]
Data = data.drop(columns=["label"])

# Check unique values in the Label column to ensure they are correct
print("Unique labels in the dataset:", Label.unique())

# Count the occurrences of each label to understand distribution
label_counts = Label.value_counts()
print("\nLabel counts before Resampling:\n", label_counts)

# Set plot style for better aesthetics
plt.figure(figsize=(10, 6))
sns.countplot(x=Label)
plt.title("Class Distribution Before Resampling")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Plotting a pie chart for class distribution
label_counts = Label.value_counts()
plt.figure(figsize=(7, 7))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Class Distribution Before Resampling')
plt.show()

from collections import Counter
print("Class distribution after Resampling:", Counter(Label))

# Handle class imbalance using RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler

Resampling = RandomUnderSampler(random_state=49)
Data, Label = Resampling.fit_resample(Data, Label)

# Updated class distribution plot
plt.figure(figsize=(10, 3))
sns.countplot(x=Label, palette='viridis')
plt.title("Class Distribution After Resampling")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.xticks(rotation=90)
plt.grid(True)
plt.yscale('log')
plt.tight_layout()
plt.show()

# Plotting a pie chart for class distribution
label_counts = Label.value_counts()
plt.figure(figsize=(7, 7))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Class Distribution After Resampling')
plt.show()

# Convert data to numpy arrays and reshape into images
Data = np.array(Data).reshape(-1, 28, 28, 3)
Label = np.array(Label)

print('Shape of Data:', Data.shape)

# Define class mapping
classes = {4: ('nv', 'melanocytic nevi'),
           6: ('mel', 'melanoma'),
           2: ('bkl', 'benign keratosis-like lesions'),
           1: ('bcc', 'basal cell carcinoma'),
           5: ('vasc', 'pyogenic granulomas and hemorrhage'),
           0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
           3: ('df', 'dermatofibroma')}

# Reshape data for flattened feature analysis
Data_flattened = Data.reshape(len(Data), -1)

# Normalize the flattened data for PCA analysis
scaler = StandardScaler()
Data_scaled = scaler.fit_transform(Data_flattened)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)
Data_pca = pca.fit_transform(Data_scaled)

# Visualize explained variance ratio by PCA components
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
plt.title('Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=0.25, random_state=49)

# Convert labels to categorical
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
testgen = ImageDataGenerator(rescale=1./255)

# Ensure generators output data in the correct format
def data_generator(generator, X, y):
    for X_batch, y_batch in generator.flow(X, y, batch_size=128):
        yield tf.convert_to_tensor(X_batch, dtype=tf.float32), tf.convert_to_tensor(y_batch, dtype=tf.float32)

# Compute class weights
# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(Label),
#     y=Label
# )
# class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
# print("Class weights:", class_weights_dict)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 3)),
    MaxPooling2D(),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),
    BatchNormalization(),

    Flatten(),
    Dropout(0.2),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adamax(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Learning Rate Reduction Callback
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=2,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)

# Train the model
history = model.fit(
    data_generator(datagen, X_train, y_train),
    steps_per_epoch=len(X_train) // 128,
    epochs=25,
    validation_data=data_generator(testgen, X_test, y_test),
    validation_steps=len(X_test) // 128,
    callbacks=[learning_rate_reduction],
    # class_weight=class_weights_dict
)

# Plot training metrics
def plot_training(hist):
    plt.figure(figsize=(20, 8))
    plt.plot(hist.history['accuracy'], label='Training Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
plot_training(history)

# Evaluate the model
train_score = model.evaluate(data_generator(testgen, X_train, y_train), steps=len(X_train) // 128)
test_score = model.evaluate(data_generator(testgen, X_test, y_test), steps=len(X_test) // 128)
print(f"Train Accuracy: {train_score[1]}\nTest Accuracy: {test_score[1]}")

# Save the model
model.save('Skin_Cancer_im.h5')
