import cv2
from model import FaceModel
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX


def start_app(cnn):
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    print("[INFO] starting video stream...")
    rgb = VideoStream(usePiCamera=True).start()
    fps = FPS().start()
    facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        
        _, frame = rgb.read()
        img = frame
        height, width, channels = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(rgb, 1.3, 5)
        # Loop through all the faces detected 
        identities = []
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h
            face_image = faces[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] 
            roi = cv2.resize(face_image, (150, 150))
            pred = cnn.predict_face(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(frame, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            fps.update()

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Filter', fr)
    	cv2.destroyAllWindows()
        fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if __name__ == '__main__':
    model = FaceModel("model.json", "model_weights.h5")
    start_app(model)
