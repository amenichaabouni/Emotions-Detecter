#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import matplotlib.pyplot as plt


# In[2]:


img = cv.imread("facefinal.jpg")
plt.imshow(img)                          


# In[3]:


img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  
plt.imshow(img_rgb)


# In[ ]:


#importing deepface library and DeepFace
from deepface import DeepFace


#this analyses the given image and gives values
#when we use this for 1st time, it may give many errors and some google drive links to download some '.h5' and zip files, download and save them in the location where it shows that files are missing.
prediction = DeepFace.analyze(img_rgb)


# In[ ]:


#check what all the things DeepFace.analyze() function has analyzed 
prediction


# In[ ]:


#loading our xml file into faceCascade using cv2.CascadeClassifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

#detecting face in color_image and getting 4 points(x,y,u,v) around face from the image, and assigning those values to 'faces' variable 
faces = faceCascade.detectMultiScale(img_rgb, 1.1, 4)

#using that 4 points to draw a rectangle around face in the image
for (x, y, u, v) in faces:
    cv2.rectangle(img_rgb, (x,y), (x+u, y+v), (0, 0, 225), 2)
    
plt.imshow(img_rgb)


# In[ ]:


#choose font for text
font = cv2.FONT_HERSHEY_PLAIN

#for showing emotion on image
cv2.putText(img_rgb, prediction['dominant_emotion'], (0, 50), font, 1, 
(225,0,0), 2, cv2.LINE_4)

#for showing race on image
cv2.putText(img_rgb, prediction['dominant_race'], (40, 100),font,  0.5,
(0,0,0), 2, cv2.LINE_4)

#finally displaying image
plt.imshow(img_rgb)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




