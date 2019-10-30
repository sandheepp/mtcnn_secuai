from mtcnn.mtcnn import MTCNN
import cv2
import freenect
import time
import time
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import matplotlib.pyplot as plt

# For multiple faces
# cap = cv2.VideoCapture(0)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


previous = 0
speed =0

while True:
    #Capture frame-by-frame
    # __, image = cap.read()
    start = time.time()
    # print("start :" , start)

    # (depth, _) = freenect.sync_get_depth()
    (image, _) = freenect.sync_get_video()
    image = imutils.resize(image, width=min(400, image.shape[1]))
	# orig = image.copy()

	# detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    #  apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    end = time.time()
    time_taken = end- start

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        speed = (xA - previous)/time_taken/10
        speed = abs(speed)
        # Using cv2.putText() method 
        cv2.putText(image, str(speed) , (xA+yA, yA), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        print("Speed of person:", speed, "m/s")
        previous = xA
     
    # print("Time taken :", time_taken)
    cv2.imshow("Video Stream", image)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break#When everything's done, release capture

# cap.release()
cv2.destroyAllWindows()                         
