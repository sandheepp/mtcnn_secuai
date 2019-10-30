from mtcnn.mtcnn import MTCNN
import cv2
import time
import argparse
# import freenect

parser = argparse.ArgumentParser(
        description='This script is used to detect faces in given camera. ')


parser.add_argument('--camera', default=0, type=int, help='Select the camera using the port with the command "ls -ltrh /dev/video*".')

parser.add_argument('--video', default= None, type=str, help='Select the video path')


args = parser.parse_args()

# Network initialisation
detector = MTCNN()

# Selection between camera and video, assign cap variable to it

if args.video == None:
    cap = cv2.VideoCapture(args.camera)
else:
    cap = cv2.VideoCapture(args.video)

previous = 0

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
            time_taken = end- start
            speed = (bounding_box[0] - previous)/time_taken/100
            print("Velocity of face movement:", abs(speed), "m/s")
            previous = bounding_box[0]
            speed = round(speed,2)
            display_text = str(speed)
            cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,155,255), 2, lineType=cv2.LINE_AA)

    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
    
#When everything's done, release capture    
cap.release()
cv2.destroyAllWindows()
