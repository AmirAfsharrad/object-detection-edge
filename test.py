import numpy as np
import imagezmq

# import imutils
import cv2
import time
from detect import init, detect_function

# initialize the ImageHub object
# imageHub = imagezmq.ImageHub()

print('here')

detector = init("config_edge.yaml", is_in_last_part=False)
detector2 = init("config_edge.yaml", is_in_last_part=True)

# start looping over all the frames
time.sleep(10)
start = time.time()
num_of_frames = 0
total_detect_time = 0
cap = cv2.VideoCapture(0)

# while cap.isOpened():
while True:
    # receive RPi name and frame from the RPi and acknowledge
    # the receipt
    # (rpiName, frame) = imageHub.recv_image()
    ret, frame = cap.read()
    # frame = cv2.imread("catdog.jpeg")

    print(time.time() - start)
    # print(frame.shape)
    # imageHub.send_reply(b'OK')

    # detect_start = time.time()
    detect_function(frame, detector, detector2)
    time.sleep(0.01)
    # total_detect_time = total_detect_time + time.time() - detect_start

    # frame = imutils.resize(frame, width=400)
    num_of_frames += 1

    # cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    # break

    if key == ord("q"):
        break

# do a bit of cleanup

fps = num_of_frames / (time.time() - start)
print("fps is: ")
print(fps)
cv2.destroyAllWindows()
