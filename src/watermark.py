import numpy as np
import cv2

from utils import ParhamVideoConfiguration, image_resize

cap = cv2.VideoCapture(0)


save_path = 'saved-media/watermark.mp4'
frames_per_second = 24
config = ParhamVideoConfiguration(cap, filepath=save_path, res='720p')
out = cv2.VideoWriter(save_path, config.video_type, frames_per_second, config.dims)
img_path = 'images/logo/pepsi.png'
logo = cv2.imread(img_path, -1)
# cv2.imshow('logo', logo)

watermark = image_resize(logo, height=50)
cv2.imshow('watermark', watermark)

while (True):
    # capture frame by frame
    ret, frame = cap.read()
    # frame.addimage(watermark)   -> does not work like this ...
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


# when everything done ->
cap.release()
out.release()
cv2.destroyAllWindows()