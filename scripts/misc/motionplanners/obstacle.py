import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2

img = np.ones((512, 512, 3), np.uint8) 
 
# Make some example data
y = np.array([320,224,192])
x = np.array([288,192,320])

for xx,yy in zip(x,y):
    cv2.circle(img, (xx,yy), 16, (0,0,255) , thickness=-1)

cv2.circle(img, (256,256), 16, (255,0,0) , thickness=-1)
cv2.circle(img, (96,64), 12, (0,255,0) , thickness=-1)

cv2.imshow('yes',img)
cv2.imwrite('./obstacle.png',img)


