import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np


blue = (255, 0, 0)
red = (0, 0, 255)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classes_dict = {0 : "man", 1 : "woman"}
gpu = tf.config.experimental.list_physical_devices('GPU') #delete these 2 lines
tf.config.experimental.set_memory_growth(gpu[0], True) #if running on CPU
classifier = keras.models.load_model('./gender_classifier.h5')

def resize_image(image, x, y, w, h):
    '''
    Uses co-ordinates passed as arguments,
    to crop and resize images to tensors of size (1, 100, 100, 3)
    '''
    if x - 0.5*w > 0:
        start_x = int(x - 0.5*w)
    else:
        start_x = x
    if y - 0.5*h > 0:
        start_y = int(y - 0.5*h)
    else:
        start_y = y

    end_x = int(x + (1 + 0.5)*w)
    end_y = int(y + (1 + 0.5)*h)

    face = image[start_y:end_y, start_x:end_x]
    face = tf.image.resize(face, [100, 100])
    face = np.expand_dims(face, axis=0)
    return face


class Camera:

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            grey,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            image = resize_image(frame, x, y, w, h)
            arr = classifier.predict(image)
            prediction = classes_dict[np.argmax(arr)]
            confidence = round(np.max(arr)*100)
            if prediction == "man":
                colour = blue
            else:
                colour = red

            cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
            cv2.putText(frame, "{0}: {1}%".format(prediction, confidence), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, colour, 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
