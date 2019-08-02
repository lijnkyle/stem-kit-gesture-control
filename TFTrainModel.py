from keras.utils import to_categorical
import time
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import PIL
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras import layers
from keras import optimizers

train_dir = os.path.join('data', 'train')
test_dir = os.path.join('data', 'test')

# use the pretrained convolutional base
pre_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(150,150,3))
pre_model.summary()

datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 20

# extract festures by pre-trained VGG19
def extract_features(directory, sample_count):
    features=np.zeros(shape=(sample_count, 4, 4, 512))
    labels=np.zeros(shape=(sample_count))
    generator=datagen.flow_from_directory(directory,target_size=(150,150),batch_size=batch_size,class_mode='sparse')
    i=0
    for inputs_batch, labels_batch in generator:
        features_batch=pre_model.predict(inputs_batch)
        if len(features_batch)!=batch_size:
            features[i*batch_size:i*batch_size+len(features_batch)]=features_batch
            labels[i*batch_size:i*batch_size+len(features_batch)]=labels_batch
        else:
            features[i*batch_size:(i+1)*batch_size]=features_batch
            labels[i*batch_size:(i+1)*batch_size]=labels_batch
        i+=1
        if i*batch_size>=sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2320)
test_features, test_labels = extract_features(test_dir, 93)

#flatten the features
train_features = np.reshape(train_features, (2320,4*4*512))
test_features = np.reshape(test_features, (93,4*4*512))

# convent labels into one-hot vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(train_labels)
print(train_labels.shape)

# build a densely-connected neural network classifier
model=models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

#train model
model.compile(optimizer=optimizers.RMSprop(lr=1e-5), loss='categorical_crossentropy', metrics=['acc'])
history=model.fit(train_features, train_labels, epochs=10, batch_size=32, validation_data = (test_features, test_labels))

model.summary()

#save model
model.save('TrainedModel/GestureRecogModel.h5')

