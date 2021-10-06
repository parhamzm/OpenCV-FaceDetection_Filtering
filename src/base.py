import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while (True):
    # capture frame by frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


# when everything done ->

cap.release()
cv2.destroyAllWindows()