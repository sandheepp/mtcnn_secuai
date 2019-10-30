from mtcnn.mtcnn import MTCNN
import cv2
import time
# import freenect

detector = MTCNN()
image = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image)
bounding_box = result[0]['box']
keypoints = result[0]['keypoints']

cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)

# For multiple faces
cap = cv2.VideoCapture(0)

previous =0

while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    start = time.time()
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    # print(result)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']

            frame = cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
    
            cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
            
            
    end = time.time()
    while( (end- start)< 0.3):
        end = time.time()
        # print(end - start)
    time_taken = end- start
    speed = (bounding_box[0] - previous)/time_taken/100
    print("Velocity of face movement:", abs(speed), "m/s")
    previous = bounding_box[0]
    
    # print(previous)
    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()
