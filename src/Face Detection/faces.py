import numpy as np
import cv2
import pickle

faces_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

#### load the labels dictionary ...
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    # reverse
    labels = {v:k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame by frame
    ret, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faces_cascade.detectMultiScale(gray_scale, scaleFactor=1.5, minNeighbors=5) # detects the faces by ROI
    # ROI -> stands for Region Of Interest
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray_scale[y:y+h, x:x+w] # (ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]

        # recognize ? /// deep learn model predict / keras / tensorflow / pytorch / scikit learn /
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45: # and conf <= 85:
            # print(id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0) # BGR type ... why not RGB ???????
        stroke = 2 # thikness of the line
        end_cord_width = x + w
        end_cord_height = y + h
        cv2.rectangle(frame, (x, y), (end_cord_width, end_cord_width), color, stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


# when everything finished -> release the capture
cap.release()
cv2.destroyAllWindows()