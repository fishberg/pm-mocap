#!/usr/bin/python3

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(gray.shape)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# https://askubuntu.com/questions/348838/how-to-check-available-webcams-from-the-command-line
# https://stackoverflow.com/questions/4290834/how-to-get-a-list-of-video-capture-devices-web-cameras-on-linux-ubuntu-c
# https://superuser.com/questions/639738/how-can-i-list-the-available-video-modes-for-a-usb-webcam-in-linux

# sudo apt install v4l-utils
# v4l2-ctl --list-devices
# v4l2-ctl -d 0 -l
# v4l2-ctl -d 0 --list-formats
# v4l2-ctl -d 0 --list-framesizes=YUYV
# v4l2-ctl -d 0 --list-framesizes=MJPG
