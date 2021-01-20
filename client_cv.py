# ## From https://stackoverflow.com/questions/30988033/sending-live-video-frame-over-network-in-python-opencv
# import socket
import numpy as np
from numpysocket import NumpySocket
import cv2
import time
from detect import init, detect_function



npSocket = NumpySocket()

npSocket.startClient(9999)

host_ip = '172.27.3.5'
send_socket = NumpySocket()
send_socket.startServer(host_ip, 9000)



# Read until video is completed
start = time.time()
fps = 0

detector = init("config_edge.yaml")

while(True):
    
    # Capture frame-by-frame
    frame = npSocket.recieveNumpy()
    send_socket.sendAck()
    # cv2.imshow('Frame', frame)
    detect_function(frame, detector)
    fps += 1
    # Press Q on keyboard to  exit
    #print(fps)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    
print(frame.shape)
print(fps/(time.time() - start))

npSocket.endServer()
print("Closing")


