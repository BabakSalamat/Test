'''
Classifiying ECG data using deep learning approach

Written by Babak Salamat 
Date : 25.11.2021

Implementation steps:
    1- Initialization and data prepration (Train/Validation/Test)
    2- Creating layers and model
    3- Setting training parameters (loss and optimization functions)
    4- Train the model
 '''
#=====================================================
# Initialization
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import np_utils

def plot_history(net_history):
    losses = history['loss']
    val_losses = history['val_loss']
    accuracies = history['accuracy']
    val_accuracies = history['val_accuracy']
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['Loss', 'Val_loss'])
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['Accuracy', 'Val_accuracy'])

# import dataset
train_df = pd.read_csv("E:\Project_final/mitbih_train.csv", header = None)
test_df = pd.read_csv("E:\Project_final/mitbih_test.csv", header = None)

# Resampling
from sklearn.utils import resample
df_1=train_df[train_df[187]==1]
df_2=train_df[train_df[187]==2]
df_3=train_df[train_df[187]==3]
df_4=train_df[train_df[187]==4]
df_0=(train_df[train_df[187]==0]).sample(n=20000,random_state=42)

df_1_upsample=resample(df_1,replace=True,n_samples=20000,random_state=123)
df_2_upsample=resample(df_2,replace=True,n_samples=20000,random_state=124)
df_3_upsample=resample(df_3,replace=True,n_samples=20000,random_state=125)
df_4_upsample=resample(df_4,replace=True,n_samples=20000,random_state=126)

train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])

classes=train_df.groupby(187,group_keys=False).apply(lambda train_df : train_df.sample(1))

# peek on classes
classes

from keras.losses import CategoricalCrossentropy
from keras.utils.np_utils import to_categorical

target_train=train_df[187]
target_test=test_df[187]
y_train=to_categorical(target_train)
y_test=to_categorical(target_test)
X_train=train_df.iloc[:,:186].values
X_test=test_df.iloc[:,:186].values

#=====================================================
# Creating our model

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

myModel = Sequential()
myModel.add(Dense(200, activation='relu',input_shape=(186,)))
myModel.add(Dropout(0.2))
myModel.add(Dense(50, activation='relu'))
myModel.add(Dropout(0.1))
myModel.add(Dense(5, activation='softmax'))

myModel.compile(optimizer=Adam(lr = 0.01), loss = CategoricalCrossentropy(), metrics=['accuracy'])

#=====================================================
# Train our model
net_history = myModel.fit(X_train, y_train, batch_size=128, 
                          epochs=6, validation_split=0.2)
history = net_history.history
#=====================================================
# Results for training

import matplotlib.pyplot as plt

plot_history(history)

#=====================================================
# Evaluation
test_loss, test_accuracy = myModel.evaluate(X_test, y_test)




