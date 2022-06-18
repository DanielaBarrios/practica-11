##practica 11
import numpy as np
from matplotlib import pyplot as plt
import math as ma
import cv2 #opencv

########################### RECONOCIMIENTO   ###########################

query_img = cv2.imread('gato.png')
train_img = cv2.imread('gato2.png')

# Convert it to grayscale
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
  
# Initialize the ORB detector algorithm
orb = cv2.ORB_create()
  
# Now detect the keypoints and compute the descriptors for the query image and train image
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)
 
# Initialize the Matcher for matching the keypoints and then match the keypoints
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors,trainDescriptors)
  
# draw the matches to the final image containing both the images the drawMatches() function takes both images and keypoints and outputs the matched query image with
# its train image
final_img = cv2.drawMatches(query_img, queryKeypoints,
train_img, trainKeypoints, matches[:20],None)
  
final_img = cv2.resize(final_img, (1000,650))
 
# Show the final image
cv2.imshow("Matches", final_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

############################ Deteccion   ###########################
cap = cv2.VideoCapture('video.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    kernel = np.ones((5,5),np.uint8)
    fgmask = fgbg.apply(frame)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    res_ope = cv2.bitwise_and(frame,frame, mask= opening)
    res_r = cv2.bitwise_and(frame,frame, mask= fgmask)
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)
    cv2.imshow('x',res_r)
    cv2.imshow('res_open',res_ope)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
