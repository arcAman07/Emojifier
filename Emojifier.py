#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


from tensorflow.python.saved_model import builder as pb_builder


# In[3]:


import pandas as pd


# In[4]:


from PIL import Image


# In[5]:


train_data = pd.read_csv("D:/ML/Dataset/Emotion Detection/train.csv")


# In[6]:


test_data = pd.read_csv("D:/ML/Dataset/Emotion Detection/test.csv")


# In[7]:


# DATA EXPLORATION


# In[8]:


train_data.head()


# In[9]:


len(train_data)


# In[10]:


len(test_data)


# In[11]:


test_data.head()


# In[12]:


train_data.shape


# In[13]:


test_data.shape


# In[14]:


data = train_data.iloc[0,1]


# In[15]:


len(data)


# In[16]:


L = data.split()


# In[17]:


len(L)


# In[18]:


L


# In[19]:


48*48


# In[20]:


final_img = [[0] * 48 for i in range(48)]


# In[21]:


starting_point = 0
for i in range(0,48):
    for j in range(0,48):
        final_img[i][j] = L[starting_point]
        starting_point += 1
        


# In[22]:


final_img


# In[23]:


array = np.array(final_img, dtype=np.uint8)


# In[24]:


array


# In[25]:


new_image = Image.fromarray(array)


# In[26]:


new_image.save('brandNew.png')


# In[27]:


# DATA PREPROCESSING - TRAINING DATA IS CREATED


# In[28]:


CATEGORIES = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]


# In[29]:


# CORRESPODING INDEX NUMBER OF THE CATEGORIES DEPICT THEIR LABELS


# In[30]:


training_data = []


# In[38]:


def create_training_data():
    for i in range(len(train_data)):
        initial_list = train_data.iloc[i,1].split()
        final_img = [[0] * 48 for i in range(48)]
        label = train_data.iloc[i,0]
        starting_point = 0
        for i in range(0,48):
            for j in range(0,48):
                final_img[i][j] = initial_list[starting_point]
                starting_point += 1
        array = np.array(final_img, dtype=np.uint8)
        initial_image = cv2.resize(array,(48,48))
#         upgrade_image = cv2.cvtColor(initial_image,cv2.COLOR_GRAY2RGB)
        training_data.append([initial_image,label])
    
    


# In[39]:


create_training_data()


# In[40]:


len(training_data)


# In[41]:


training_data


# In[42]:


plt.imshow(training_data[2][0])


# In[43]:


training_data[2][0].shape


# In[44]:


initial_image = cv2.resize(array,(120,120))


# In[45]:


upgrade_img = cv2.cvtColor(initial_image,cv2.COLOR_GRAY2RGB)


# In[46]:


upgrade_img.shape


# In[47]:


plt.imshow(upgrade_img)


# In[48]:


training_data[2][1]


# In[49]:


# CREATING THE VALIDATION DATA


# In[50]:


validation_data = []


# In[51]:


def create_validation_data():
    for i in range(len(train_data)-20000):
        initial_list = train_data.iloc[i,1].split()
        final_img = [[0] * 48 for i in range(48)]
        label = train_data.iloc[i,0]
        starting_point = 0
        for i in range(0,48):
            for j in range(0,48):
                final_img[i][j] = initial_list[starting_point]
                starting_point += 1
        array = np.array(final_img, dtype=np.uint8)
        initial_image = cv2.resize(array,(48,48))
#         upgrade_image = cv2.cvtColor(initial_image,cv2.COLOR_GRAY2RGB)
        validation_data.append([initial_image,label])


# In[52]:


create_validation_data()


# In[53]:


len(validation_data)


# In[54]:


# CREATING THE TESTING DATA


# In[55]:


testing_data  = []


# In[56]:


def create_testing_data():
    for i in range(0,len(test_data)):
        initial_list = test_data.iloc[i,0].split()
        final_img = [[0] * 48 for i in range(48)]
        label = 0
        starting_point = 0
        for i in range(0,48):
            for j in range(0,48):
                final_img[i][j] = initial_list[starting_point]
                starting_point += 1
        array = np.array(final_img, dtype=np.uint8)
        initial_image = cv2.resize(array,(48,48))
#         upgrade_image = cv2.cvtColor(initial_image,cv2.COLOR_GRAY2RGB)
        testing_data.append([initial_image,label])


# In[57]:


create_testing_data()


# In[58]:


len(testing_data)


# In[59]:


training_data


# In[60]:


testing_data


# In[61]:


import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd


# In[62]:


from tensorflow.keras.optimizers import RMSprop


# In[63]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[64]:


train_datagen = ImageDataGenerator(rescale=1.0/255.,)

# NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE 
# TRAIN GENERATOR.

 
validation_datagen = ImageDataGenerator(rescale=1.0/255.)

# NOTE: YOU MUST USE A BACTH SIZE OF 10 (batch_size=10) FOR THE 
# VALIDATION GENERATOR.


# In[65]:


import random
random.shuffle(training_data)


# In[66]:


X_train = []
y_train = []
X_test = []
y_test = []
X_valid = []
y_valid = []


# In[67]:


for features,label in training_data:
    X_train.append(features)
    y_train.append(label)
X_train = np.array(X_train).reshape(-1,48,48,1)
y_train = np.array(y_train)


# In[68]:


for features,label in validation_data:
    X_valid.append(features)
    y_valid.append(label)
X_valid = np.array(X_valid).reshape(-1,48,48,1)
y_valid = np.array(y_valid)


# In[69]:


for features,label in testing_data:
    X_test.append(features)
    y_test.append(label)
X_test = np.array(X_test).reshape(-1,48,48,1)
y_test = np.array(y_test)


# In[70]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D,BatchNormalization


# In[71]:


X_train


# In[72]:


X_train.shape[1:]


# In[73]:


X_train = X_train/255.0


# In[74]:


# vgg19 = keras.applications.VGG19(weights='imagenet', include_top=False)


# In[75]:


path_inception = "D:/ML/Dataset/Emotion Detection/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Import the inception model  
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = path_inception
pre_trained_model = InceptionV3(input_shape = (75, 75, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable=False
  # Your Code Here

  
# Print the model summary
pre_trained_model.summary()

# Expected Output is extremely large, but should end with:

# batch_normalization_v1_281 (Bat (None, 3, 3, 192)    576         conv2d_281[0][0]                 
# __________________________________________________________________________________________________
# activation_273 (Activation)     (None, 3, 3, 320)    0           batch_normalization_v1_273[0][0] 
# __________________________________________________________________________________________________
# mixed9_1 (Concatenate)          (None, 3, 3, 768)    0           activation_275[0][0]             
#                                                                 activation_276[0][0]             
# __________________________________________________________________________________________________
# concatenate_5 (Concatenate)     (None, 3, 3, 768)    0           activation_279[0][0]             
#                                                                 activation_280[0][0]             
# __________________________________________________________________________________________________
# activation_281 (Activation)     (None, 3, 3, 192)    0           batch_normalization_v1_281[0][0] 
# __________________________________________________________________________________________________
# mixed10 (Concatenate)           (None, 3, 3, 2048)   0           activation_273[0][0]             
#                                                                 mixed9_1[0][0]                   
#                                                                 concatenate_5[0][0]              
#                                                                 activation_281[0][0]             
# ==================================================================================================
# Total params: 21,802,784
# Trainable params: 0
# Non-trainable params: 21,802,784


# In[76]:


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Expected Output:
# ('last layer output shape: ', (None, 7, 7, 768))


# In[77]:


import seaborn as sns


# In[78]:


import matplotlib.pyplot as plt


# In[79]:


from tensorflow.keras.layers import Bidirectional,LSTM, Dense, Embedding, Dropout, GlobalAveragePooling1D


# In[80]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[115]:


model = Sequential()
# model.add(Conv2D(128, (3, 3), input_shape=X_train.shape[1:]))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(256, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(512, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(256, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(512, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(tf.keras.layers.Flatten(input_shape = (48,48)))

model.add(tf.keras.Input(shape=(48,48)))

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

# model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.2))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

# model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Dense(128))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Dense(64))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Dense(48))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

model.add(Dense(7))
model.add(Activation('softmax'))



# In[116]:


from tensorflow.keras import layers
from tensorflow.keras import Model


# In[117]:


# # Flatten the output layer to 1 dimension
# input_layer = tf.keras.Input(shape=(48, 48))
# x = layers.Flatten()(input_layer)
# # Add a fully connected layer with 1,024 hidden units and ReLU activation
# x = layers.Dense(512,activation='relu')(x)
# # Add a dropout rate of 0.2
# x = layers.Dropout(0.2)(x)
# x = layers.Dense(512,activation='relu')(x)
# # Add a dropout rate of 0.2
# x = layers.Dropout(0.2)(x)
# x = layers.Dense(512,activation='relu')(x)
# # Add a dropout rate of 0.2
# x = layers.Dropout(0.2)(x)
# x = layers.Dense(256,activation='relu')(x)
# # Add a dropout rate of 0.2
# x = layers.Dropout(0.2)(x)
# x = layers.Dense(128,activation='relu')(x)
# # Add a dropout rate of 0.2
# x = layers.Dropout(0.2)(x)
# # Add a final sigmoid layer for classification
# output_layer = layers.Dense(7,activation='softmax')(x)

# model = Model(inputs=input_layer, outputs=output_layer)


# In[118]:


# model = Model(pre_trained_model.input, x) 


# In[119]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[120]:


model.summary()


# In[121]:


model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))


# In[122]:


import os
import zipfile
import shutil


# In[123]:


# Define a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') is not None and logs.get('accuracy')>0.97):
            print("\nReached 97.0% accuracy so cancelling training!")
            self.model.stop_training = True


# In[124]:


callbacks = myCallback()


# In[125]:


# callbacks=[callbacks]


# In[128]:


# %matplotlib inline
# import matplotlib.pyplot as plt
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc=0)
# plt.figure()


# plt.show()


# In[129]:


# MAKING PREDICTIONS : BEFORE TESTING PURPOSE


# In[130]:


import cv2


# In[131]:


happy_pred = cv2.imread("C:/Users/amans/Downloads/happy1.jpg")


# In[132]:


type(happy_pred)


# In[133]:


happy_pred.shape


# In[134]:


plt.imshow(happy_pred)


# In[141]:


new_img = cv2.cvtColor(happy_pred,cv2.COLOR_BGR2GRAY)


# In[142]:


plt.imshow(new_img)


# In[143]:


new_img.shape


# In[144]:


img_pred = cv2.resize(new_img,(48,48))


# In[145]:


img_pred = img_pred.reshape((48,48,1))


# In[146]:


img_pred.shape


# In[147]:


# Import Module
from tkinter import *
from PIL import Image, ImageTk


# In[148]:


image = Image.open("C:/Users/amans/Downloads/happy1.jpg")


# In[149]:


width = 48
height = 48


# In[150]:


resized_image = image.resize((width, height))


# In[151]:


plt.imshow(resized_image)


# In[152]:


resized_image.save('happyPred.jpeg')


# In[153]:


happy_prediction = cv2.imread("happyPred.jpeg",cv2.IMREAD_GRAYSCALE)


# In[154]:


happy_prediction.shape


# In[155]:


plt.imshow(happy_prediction)


# In[156]:


happy_prediction = happy_prediction.reshape((1,48,48))


# In[157]:


happy_prediction.shape


# In[158]:


model.predict(happy_prediction)


# In[159]:


validation_data


# In[160]:


check_data = 0


# In[161]:


# for features,label in validation_data:
#     check_data += 1
#     features = features.reshape(1,48,48)
#     prediction = model.predict(features)
#     print(np.argmax(prediction))
#     print (label)
#     print(prediction)
    
#     max_pred = prediction[0].index(max(prediction[0]))
#     print(max_pred)
#     break


# In[162]:


total_pred = 0
correct_pred = 0


# In[163]:


for features,label in validation_data:
    total_pred += 1
    features = features.reshape(1,48,48)
    prediction = model.predict(features)
    final_pred = np.argmax(prediction)
    if (final_pred == label):
        correct_pred += 1


# In[164]:


train_accuracy = correct_pred/total_pred


# In[165]:


train_accuracy


# In[166]:


valid_data = pd.read_csv("D:/ML/Dataset/Emotion Detection/icml_face_data.csv")


# In[167]:


valid_data.head()


# In[168]:


valid_data.shape


# In[169]:


len(valid_data)


# In[170]:


# PUBLIC TEST STARTS FROM INDEX NUMBER 28709


# In[171]:


valid_data.iloc[28709,1]


# In[172]:


test_case = []


# In[173]:


def create_test_data():
    for i in range(28709,len(valid_data)):
        initial_list = valid_data.iloc[i,2].split()
        final_img = [[0] * 48 for i in range(48)]
        label = valid_data.iloc[i,0]
        starting_point = 0
        for i in range(0,48):
            for j in range(0,48):
                final_img[i][j] = initial_list[starting_point]
                starting_point += 1
        array = np.array(final_img, dtype=np.uint8)
        test_case.append([array,label])


# In[174]:


create_test_data()


# In[175]:


total_pred1 = 0
correct_pred1 = 0


# In[176]:


for features,label in test_case:
    total_pred1 += 1
    features = features.reshape(1,48,48)
    prediction = model.predict(features)
    final_pred = np.argmax(prediction)
    if (final_pred == label):
        correct_pred1 += 1


# In[177]:


validation_accuracy = correct_pred1/total_pred1


# In[178]:


validation_accuracy


# In[179]:


# Saving the model for Future Inferences

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


# In[ ]:




