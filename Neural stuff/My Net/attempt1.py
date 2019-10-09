# This is based off https://www.tensorflow.org/tutorials/images/classification
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


print(f"Tensor flow version: {tf.__version__}")

# This is the old dataset stuff
# # Assemble the data set
# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
# print(f"Data is in {path_to_zip}")
# # Get the path for the training and validation data
# train_dir = os.path.join(PATH, 'train')
# validation_dir = os.path.join(PATH, 'validation')

train_dir = "C:\\Users\\Ben\\PycharmProjects\\SpannersAndScrewdrivers\\Dataset\\train"
validation_dir = "C:\\Users\\Ben\\PycharmProjects\\SpannersAndScrewdrivers\\Dataset\\Validation"

# Get the path for each of the classes' train and validation
train_screwdrivers_dir = os.path.join(train_dir, 'Screwdriver')  # directory with our training cat pictures
train_wrenches_dir = os.path.join(train_dir, 'Wrench')  # directory with our training dog pictures
validation_screwdrivers_dir = os.path.join(validation_dir, 'Screwdriver')  # directory with our validation cat pictures
validation_wrenches_dir = os.path.join(validation_dir, 'Wrench')  # directory with our validation dog pictures

# Print out a few cool numbers to understand how much data is actually available
num_screwdrivers_tr = len(os.listdir(train_screwdrivers_dir))
num_wrenches_tr = len(os.listdir(train_wrenches_dir))

num_screwdrivers_val = len(os.listdir(validation_screwdrivers_dir))
num_wrenches_val = len(os.listdir(validation_wrenches_dir))

total_train = num_screwdrivers_tr + num_wrenches_tr
total_val = num_screwdrivers_val + num_wrenches_val

print('total training screwdriver images:', num_screwdrivers_tr)
print('total training wrench images:', num_wrenches_tr)

print('total validation screwdriver images:', num_screwdrivers_val)
print('total validation wrench images:', num_wrenches_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

# Pre-setup variables to process stuff
batch_size = 20
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Make the image generator do random things with the images
image_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

# The evaluation set stays the same
image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')


# sample_training_images, sample_training_classes = next(train_data_gen)

# # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20,20))
#     axes = axes.flatten()
#     for img, ax in zip( images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#
#
# # Run the function
# plotImages(sample_training_images[:5])

# # Print out what things should be
# sample_training_classes = ["Screwdriver" if i==0 else "Wrench" for i in sample_training_classes]
# print(f"Classes in example are as follows: {sample_training_classes[:5]}")

# Build the model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print out a summary of the model
model.summary()

# Train the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

# Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save_weights('attempt1.h5')