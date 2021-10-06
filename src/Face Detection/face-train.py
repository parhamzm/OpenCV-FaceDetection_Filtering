import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIS = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIS, "images")

faces_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_trains = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() # root = os.path.dirname(path)
            # print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            # y_labels.append(label) # some number
            # x_trains.append(path) # verify this image, turn into a NUMPY array, GRAY
            pil_image = Image.open(path).convert("L") # grayscale
            # resize the images ...
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8") # we turning the grayscale image to a numpy array
            # print(image_array)
            faces = faces_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5) # face detection part

            for (x, y, w, h) in faces:
                roi = image_array[y: y+h, x: x+w]
                x_trains.append(roi)
                y_labels.append(id_)


# Saving the detected faces and persons ...
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_trains, np.array(y_labels))
recognizer.save("trainer.yml")