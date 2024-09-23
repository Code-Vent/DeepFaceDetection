import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model
import util
import numpy as np

facetracker = load_model('facetracker.h5')
cap = cv.VideoCapture(1) 
while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500, :]
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    classes, coords = facetracker.predict(np.expand_dims(resized/255, 0))
    if classes[0] > 0.9:
        util.render_bbox(img=frame, coords=coords[0], width=450, height=450, tag='face')
        cv.imshow('Face Tracker', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
           break
       
cap.release()
cv.destroyAllWindows()