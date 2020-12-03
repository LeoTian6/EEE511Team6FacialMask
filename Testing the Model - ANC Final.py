#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing the necessary packages
from keras.models import load_model
import cv2
import numpy as np


# In[ ]:


model = load_model('model-008.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Opening the video webcam of your PC
source=cv2.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}


# In[ ]:


#Importing the necessary 
while(True):

    #Reading the Images from the live videocam
    ret,img=source.read()
    print(ret)
    
    #Converting the color images to grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #To extract the Region of Interest(ROI) from the extracted image
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
        
        #Giving the Rectangle color block for the face image
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
     
    
    cv2.imshow('Face Mask Detector- ANC Final',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()

