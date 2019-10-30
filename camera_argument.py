import argparse
import cv2

parser = argparse.ArgumentParser(
        description='This script is used to detect faces in given camera. ')



parser.add_argument('--camera', default=0, type=int, help='Select the camera using the port with the command "ls -ltrh /dev/video*".')

args = parser.parse_args()


cap = cv2.VideoCapture(args.camera)
_, frame = cap.read()
cv2.imshow('image',frame)
cv2.waitKey(3000)
cv2.destroyAllWindows()
