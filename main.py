import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam

#path to images
train_path = '/home/kevin/ML_Projects/cats&dogs_projects/PetImages/Train'
test_path = '/home/kevin/ML_Projects/cats&dogs_projects/PetImages/Test'
valid_path = '/home/kevin/ML_Projects/cats&dogs_projects/PetImages/Valid'

#preprocessing images

train_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(train_path, target_size=(224, 224), class_mode='binary', batch_size=10, shuffle=True)
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(test_path, target_size=(224, 224), class_mode='binary', batch_size=10, shuffle=True)
valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(valid_path, target_size=(224, 224), class_mode='binary' , batch_size=10 , shuffle=False)


#model creation
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate = 0.003), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_batches, validation_data=valid_batches, epochs=3, verbose=2, shuffle=True)
pred = np.array(model.predict(test_batches))

print(pred.round())


