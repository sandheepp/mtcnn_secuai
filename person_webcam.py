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


# detector = MTCNN()
# image = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
# result = detector.detect_faces(image)
# bounding_box = result[0]['box']
# keypoints = result[0]['keypoints']

# cv2.rectangle(image,
#               (bounding_box[0], bounding_box[1]),
#               (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
#               (0,155,255),
#               2)

# For multiple faces
cap = cv2.VideoCapture(0)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


previous = 0

while True:
    #Capture frame-by-frame
    __, image = cap.read()
    start = time.time()

    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    
    
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
        # speed = (xA - previous)/time_taken/100
        # print("Velocity of face movement:", abs(speed), "m/s")
        # previous = xA

    # show the output images
    # cv2.imshow("Before NMS", orig)
    print("Time taken :", time_taken)
    cv2.imshow("Video Stream", image)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break#When everything's done, release capture

    # # print(bounding_box[0] - previous) 
    # previous = bounding_box[0];
    # # print(previous)
    # #display resulting frame
    # cv2.imshow('frame',frame)
    # # print(time.time()-start)
    # if cv2.waitKey(1) &0xFF == ord('q'):
    #     break#When everything's done, release capture
    # cap.release()


cv2.destroyAllWindows()                         
