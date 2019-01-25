import cv2
import numpy as np
from imutils.perspective import four_point_transform
import math
from imutils import contours as imCont


def euclidDistance(p1, p2):
    dist1 = abs((p1[0]-p2[0]))
    dist2 = abs((p1[1]-p2[1]))
    return math.sqrt(dist1**2 + dist2**2)

def getAdaptiveThresh(frame):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    adaptiveFrame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)
    #adaptiveFrame = canny = cv2.Canny(frame, 127, 255)
    return adaptiveFrame

def getFourPoints():
    
    squareContours = [] 
    
    _,contours, hie = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
	fourPoints = []
	i=0
	for cnt in contours:
	    
	    (x,y),(MA,ma),angle = cv2.minAreaRect(cnt)	    
		
	    epsilon = 0.04*cv2.arcLength(cnt,False)
	    approx = cv2.approxPolyDP(cnt,epsilon,True)
    
	    x,y,w,h = cv2.boundingRect(cnt)
	    aspect_ratio = float(w)/h
	    if len(approx) == 4 and aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
		M = cv2.moments(cnt)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		fourPoints.append((cx,cy))
		squareContours.append(cnt)
		i+=1
	return fourPoints, squareContours
    

def getOvalContours(adaptiveFrame, sqrAvrArea):
    _,contours,hierarchy = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ovalContours = []
    for cnt in contours:
	epsilon = 0.04*cv2.arcLength(cnt,False)
	approx = cv2.approxPolyDP(cnt,epsilon,True)	
    
	x,y,w,h = cv2.boundingRect(cnt)
	aspect_ratio = float(w)/h	
    
	area = cv2.contourArea(cnt)	
	    
	if len(approx) >= 6 and area > sqrAvrArea and aspect_ratio >= 0.8 and aspect_ratio <= 1.2:
		ovalContours.append(cnt)
	
    return ovalContours

ANSWER_KEY = { 0: 1, 1: 2, 2: 0, 3: 3, 4: 2,
              5: 2, 6: 2, 7: 4, 8: 0, 9: 3,
              10: 1, 11: 1, 12: 4, 13: 0, 14: 3,
              15: 1, 16: 1, 17: 4, 18: 0, 19: 1,
              20: 1, 21: 1, 22:3, 23: 4, 24: 0,
              25: 3, 26: 1, 27: 1, 28: 4, 29: 0,
              30: 3, 31: 1, 32: 1, 33: 4, 34: 0,
              35: 3, 36: 1, 37: 1, 38: 4, 39: 0,
              40: 3
              }


questionCount = 40
bubbleCount = 5
ovalCount = questionCount * bubbleCount

frame = cv2.imread('optik12.jpg')
w,h,c = frame.shape
gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
canny = cv2.Canny(gray, 127, 255)

fourPoints = np.array(getFourPoints()[0], dtype="float32")
fourCountours = getFourPoints()[1]

if len(fourPoints) >= 4:
    
    newFourPoints = []
    newFourPoints.append(fourPoints[0])
    newFourPoints.append(fourPoints[1])
    newFourPoints.append(fourPoints[len(fourPoints)-2])
    newFourPoints.append(fourPoints[len(fourPoints)-1])

    newSquareContours = []    
    newSquareContours.append(fourCountours[0])
    newSquareContours.append(fourCountours[1])
    newSquareContours.append(fourCountours[len(fourCountours)-2])
    newSquareContours.append(fourCountours[len(fourCountours)-1])
    

    sqrAvrArea = 0    
    for cnt in newSquareContours:
	area = cv2.contourArea(cnt)
    
	sqrAvrArea += area
	
    
    sqrAvrArea = int(sqrAvrArea/6)      
    
    newFourPoints = np.array(newFourPoints, dtype="float32")        
    
    warped = four_point_transform(frame, newFourPoints)
    adaptiveWarped = getAdaptiveThresh(warped)
    ovalContours = getOvalContours(adaptiveWarped,sqrAvrArea)

    print len(ovalContours)
    
    cv2.imshow('canny', adaptiveWarped);
    cv2.waitKey(0)
	
    if(len(ovalContours) == ovalCount):
	#cv2.drawContours(warped, ovalContours, -1, (0,255,0), -1)
	
	ovalContours = imCont.sort_cont(ovalContours)
	
	
	correct = 0
	k = 0
	for (q, i) in enumerate(np.arange(0, len(ovalContours), bubbleCount)):
	    
	    bubbles = ovalContours[i: i+bubbleCount];
	    bubbles = imCont.sort_contours(bubbles, method="left-to-right")[0]
	    
	    
	    total = 0
	    answer = -1
	    bubbled = False
	    emptyStartX = 0
	    emptyStartY = 0
	    emptyFinishX = 0
	    emptyFinishY = 0
	    startX = 0
	    startY = 0
	    finishX = 0
	    finishY = 0
	  
	    for (j, c) in enumerate(bubbles):
		M = cv2.moments(c)
		cx = int(M['m10']/M['m00'])		
		cy = int(M['m01']/M['m00'])
		
		
		if j == 0:
		    emptyStartX = cx
		    emptyStartY = cy
		    
		if j == bubbleCount-1:
		    emptyFinishX = cx
		    emptyFinishY = cy
		
		area = cv2.contourArea(c)
		
		mask = np.zeros(adaptiveWarped.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		mask = cv2.bitwise_and(adaptiveWarped, adaptiveWarped, mask=mask)
		total = cv2.countNonZero(mask)
	    
		
		areaRatio = (float) (total) / (float)(area)
		
		print(areaRatio, j, i)
		
		if(areaRatio >= 1.0):
		    if(bubbled == False):
			answer = j
			bubbled = True
			startX = cx
			startY = cy
		    else:
			answer = -2
			finishX = cx
			finishY = cy
		j+=1
	    if answer == -2:
		None
		
	    if answer == -1:
		cv2.line(warped, (emptyStartX, emptyStartY), (emptyFinishX, emptyFinishY), (0, 255, 255), 2 )
	    
	    if answer != -2 and answer != -1:
		if(ANSWER_KEY[j] == answer):
		    cv2.drawContours(warped, bubbles, answer, (0,255,0), 2)
		else:
		    cv2.drawContours(warped, bubbles, answer, (0,0,255), 2)
	
	      
	cv2.imshow('canny', warped);
	cv2.waitKey(0)
else:
    print('4 kare bulunamadi!')