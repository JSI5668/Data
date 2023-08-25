import cv2
import numpy as np
import math
import time
import sys
from omnicv import fisheyeImgConv

# Import equirectangular image
IMAGE_PATH = 'E:/Dacon/Samsung_AI_Challenge/Camera_Invariant_Domain_Adaptation_segmentation/Data/open/train_source_image/TRAIN_SOURCE_0000.png'
equiRect = cv2.imread(IMAGE_PATH)
# cv2.imshow('sdf', equiRect)

# Defining output shape
outShape = [512,512]

# Creating mapper object
mapper = fisheyeImgConv()

# Converting equirectangular to fisheye using Unified Camera model (UCM)
# fisheye = mapper.equirect2Fisheye_UCM(equiRect,outShape=outShape,xi=0.00000001)
# cv2.imshow("UCM Model Output",fisheye)
# cv2.waitKey(0)

# Converting equirectangular to fisheye using Extended UCM model
fisheye = mapper.equirect2Fisheye_EUCM(equiRect,outShape=[512,512],f=100,a_=0.000100,b_=0.000001,angles=[0,0,0])
cv2.imshow("EUCM Model Output",fisheye)
cv2.waitKey(0)

# # Converting equirectangular to fisheye using Field Of Vide (FOV) model
# fisheye = mapper.equirect2Fisheye_FOV(equiRect,outShape=[250,250],f=40,w_=0.5,angles=[0,0,0])
# cv2.imshow("FOV model Output",fisheye)
# cv2.waitKey(0)
#
# # Converting equirectangular to fisheye using Double Sphere (DS) model
# fisheye = mapper.equirect2Fisheye_DS(equiRect,outShape=[250,250],f=90,a_=0.4,xi_=0.8,angles=[0,0,0])
# cv2.imshow("DS Model Output",fisheye)
# cv2.waitKey(0)
#
# # Changing the distortion coefficient for (UCM)
# fisheye = mapper.equirect2Fisheye(equiRect,outShape=outShape,xi=0.2)
# cv2.imshow("fisheye",fisheye)
# cv2.waitKey(0)
#
# # Rotate the sphere
# fisheye = mapper.equirect2Fisheye(equiRect,outShape=outShape,angles=[90,0,0])
# cv2.imshow("fisheye",fisheye)
# cv2.waitKey(0)