__author__ = 'nicky'

import  numpy
import cv2
from copy import copy

#Original Where's Waldo Image
original = cv2.imread('4cd61f6cf43ff9313a84d6c2df7cf5e074041a0ebb450cc13eb9259d.jpg')

#convert image from rgb to hsv (a form that openCV can work with) and grayscale
hsv = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayScale.jpg', gray)

#Limits for red colors (Waldo's red shirt actually does not tend to be that red...)(This is BGR formatted)
red_lower_bounds = numpy.array([0, 100, 100], dtype=numpy.uint8 )
red_upper_bounds = numpy.array ([5,255,255], dtype=numpy.uint8)

#Make colors in red range black and all others white to find red stripes
redsFound = 255-cv2.inRange(hsv, red_lower_bounds, red_upper_bounds)
cv2.imwrite('RedsFound.jpg',redsFound)

#Keep the white colors white and make everything else black to find white stripes
val, whitesFound = cv2.threshold(gray,240, 255,cv2.THRESH_TOZERO)
#we actually want to now make all of our blacks to white and all of our whites to black because the red image has turned everything non-red to white.
whitesFound = 255 - whitesFound
cv2.imwrite('WhitesFound.jpg',whitesFound)


#bring the two images together so that we can find the places that contain both red AND white
redAndWhite = copy(whitesFound)
redAndWhite[redsFound == 0] = 0
#we now need to invert the image back to normal
redAndWhite = 255-redAndWhite
cv2.imwrite('redAndWhite.jpg',redAndWhite)

#We need to try to get rid of as much 'noise' from the image as possible so that we do not end up with false positives
redAndWhite = cv2.blur(redAndWhite,(30,30))
val, redAndWhite = cv2.threshold(redAndWhite,40,255,cv2.THRESH_BINARY)
cv2.imwrite('redAndWhiteNoiseReduced.jpg',redAndWhite)


#Canny is an OpenCV function that detects the edges drawn and if there are enough edges drawn, it will connect the dots
#This will result in drawing a shape around the red and white images in the picture
edges = cv2.Canny(redAndWhite,100,200)
edges[edges != 255] = 0
edges = cv2.dilate(edges, None)
original[edges == 255] = (0, 0, 0)
cv2.imwrite("result.jpg", original)
