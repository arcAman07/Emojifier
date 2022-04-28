#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
import os


# In[2]:


from gtts import gTTS


# In[3]:


L = ["CALM DOWN","DONT BE DISGUSTED","THERE IS NOTHING TO BE AFRAID OF","ALWAYS REMAIN HAPPY AND ENJOY YOUR DAY","CHEER UP, DONT BE SAD","WHY ARE U SO SURPRISED","THAT'S THE PERFECT EXPRESSION"]


# In[4]:


tts = gTTS(L[0])
tts1 = gTTS(L[1])
tts2 = gTTS(L[2])
tts3 = gTTS(L[3])
tts4 = gTTS(L[4])
tts5 = gTTS(L[5])
tts6 = gTTS(L[6])


# In[5]:


tts.save('h1.mp3')
tts1.save('h2.mp3')
tts2.save('h3.mp3')
tts3.save('h4.mp3')
tts4.save('h5.mp3')
tts5.save('h6.mp3')
tts6.save('h7.mp3')


# In[6]:


os.system("h1.mp3")


# In[7]:


face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[8]:


def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
    
    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),thickness=5)
        return face_img


# In[9]:


L = ["CALM DOWN","DONT BE DISGUSTED","THERE IS NOTHING TO BE AFRAID OF","ALWAYS REMAIN HAPPY AND ENJOY YOUR DAY","CHEER UP, DONT BE SAD","WHY ARE U SO SURPRISED","THAT'S THE PERFECT EXPRESSION"]


# In[10]:


# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()


# In[11]:


width = 48
height = 48


# In[ ]:


video_capture = cv2.VideoCapture(0)
# video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,48);
# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,48);
while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
#     prediction = new_model.predict(gray)
#     print(np.argmax(prediction))

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# In[ ]:


os.system("h4.mp3")


# In[ ]:




