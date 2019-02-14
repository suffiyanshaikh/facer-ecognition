#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 00:36:59 2019

@author: suffiyan
"""
""
""" importing libraries"""
import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import feature
import sqlite3
import io

""" array cannot be direct store in to data base,sql data base doesnt support array data type,function is define to convert array into suitable data types which can be accepted by database here we are converting tha arrays in 'blob' """

def adapt_array(feature):
    out=io.BytesIO()
    np.save(out,feature)
    out.seek(0)
    return sqlite3.Binary(out.read())


""" creating or calling existing data base"""

sqlite3.register_adapter(np.ndarray,adapt_array)
conn=sqlite3.connect('/home/suffiyan/Desktop/project/imgdb.sqlite',detect_types=sqlite3.PARSE_DECLTYPES)
cur=conn.cursor()


#Intializing camera
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.namedWindow("cap",cv2.WINDOW_NORMAL)
    cv2.imshow("cap",frame)
    k=cv2.waitKey(1)
   
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    
    elif k == ord('s'):
        cap.release()
        cv2.destroyAllWindows()
        break

# Our operations on the frame come here
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray=cv2.resize(gray,(600,600))

#hist equalization
equ=cv2.equalizeHist(gray)

#image denoising filtering
kernel_5x5=np.ones((5,5),np.float32)/32
k5=scipy.ndimage.convolve(equ,kernel_5x5)
flt=cv2.filter2D(k5,-1,kernel_5x5)

"""Create a cascade classifier object/load haar classifier """
face_cascade=cv2.CascadeClassifier("/home/suffiyan/Desktop/project/haarcascade_frontalface_default.xml")
#inputimage=cv2.imread("/home/suffiyan/Desktop/faces/gray.jpg",0)


""" detect multiscale"""
faces=face_cascade.detectMultiScale(flt,1.05,3)
print(faces)

""" create rectangles on the image"""
for x,y,w,h in faces:
    rectimg=cv2.rectangle(flt,(x,y),(x+w,y+h),(0,255,0),3)

#output=cv2.resize(rectimg,(int(rectimg.shape[1]),int(rectimg.shape[0])))

#crop the img
facecnt = len(faces)
print("Detected faces: %d" % facecnt)
i = 0
height, width = flt.shape[:2]

for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = rectimg[ny:ny+nr, nx:nx+nr]
            break    
        
"""finding lbp of image """
eps=1e-7
lbp = feature.local_binary_pattern(faceimg,8,
1, method="uniform")
(hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0,8+ 3),range=(0,8+ 2))
 
		# normalize the histogram
hist = hist.astype("float")
hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patter
print(hist)
plt.subplot(1,2,1)
plt.imshow(gray)
plt.title('Input image')

plt.subplot(1,2,2)
plt.imshow(equ)
plt.title('Equalized Image')
plt.show()

plt.subplot(1,2,1)
plt.imshow(flt)
plt.title('Filtered Image')

plt.subplot(1,2,2)
plt.imshow(rectimg)
plt.title("Face detected")
plt.show()

plt.subplot(1,2,1)
plt.imshow(faceimg)
plt.title("Crop Image")

plt.subplot(1,2,2)
plt.plot(hist)
plt.title(" Normalized Histogram")
plt.show()

total=sum(hist)
#adding features to data base
if total==sum(hist):
  cur.execute('INSERT INTO facefeatures(name,feature)values(?, ?)',('suffiyan',hist))
  conn.commit()
  conn.close()
  print(total)

    
