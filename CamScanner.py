import cv2
import numpy as np

#import the mapper function: finds the endpoints of a picture
import mapper


#loading the image: location of the image on your local computer
image = cv2.imread('sample.jpg')


#resize the image --> smaller images work well with opencv
image = cv2.resize(image, (1300,800))


#create a copy of the image
original = image.copy()


#convert to GRAY SCALE ---> cv2.COLOR_BGR2GRAY
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



#using gaussian blur technique to smoothen the image ---> takes in three paramters : image || kernel-size || sigma
#the image should not be blurred too much; as it may loose its edges
blurred = cv2.GaussianBlur(gray, (5,5), 0)



#Canny Edge detection ---> takes in two values : minimum and maximum value
edge = cv2.Canny(blurred, 30, 50)



#NOISE REMOVAL
#To extract the boundary we use:
#It returns 3-items(image, contours, hierarchy) in a tuple
#We use the (CHAIN_APPROX_SIMPLE) approximation method
contours, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


#we are interested in finding the boundary: so we use contours
contours = sorted(contours, key = cv2.contourArea, reverse=True)



#we use arc length function: Tries to find a square or a closed shape in the images
for c in contours:
    #True ---for closed shapes
    #False ---for lines and curves
    p = cv2.arcLength(c, True)
    #the image returned by the arcLength is not perfect square

    #so we use : approxPolyDP
    approx = cv2.approxPolyDP(c, 0.03*p, True)

    #if len(approx) is 4 then it means the function has retured a square/ rectangle

    if len(approx)==4:
        target = approx
        break



#target contains the final contours


#for the bird eye view: find the endponts of the images
#mapper.map --> finds the endpoint
approx = mapper.mapp(target)



#defining a BOUNDARY for the final image: [[top-left], [top-right], [bottom-right], [bottom-left]]
#NOTE: we have used float32 not float
pts = np.float32([[0,0],[800,0],[800,800],[0,800]])



#Using the get perspective transform function
output = cv2.getPerspectiveTransform(approx, pts)



#warpPerspective requires --> ORIGINAL image || output of getPerspectiveTransform || window-size
scanned = cv2.warpPerspective(original, output, (800,800))



#to show the images;
cv2.imshow('Image', scanned)


#save the image
cv2.imwrite('scanned.jpg',scanned)



#keeps the image intact ---> avoids sudden disappearing of the image
cv2.waitKey()
