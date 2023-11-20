import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
# add
# import winsound
model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    video = cv2.VideoCapture('cars.mp4') 
    # for camera use 
    # video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # add
        if frame is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Rest of your code here

            roi = cv2.resize(gray_frame, (250, 250))

            pred, prob = model.predict_accident(roi[np.newaxis, :, :])
            if(pred == "Accident"):
                prob = (round(prob[0][0]*100, 2))
                
                # # to beep when alert:
                # if(prob > 90):
                #     # os.system("say beep")
                #     winsound. Beep(2000, 1500)

                cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)
                 # add here:
                # if(prob > 90):
                #     winsound. Beep(2000, 1000)

            if cv2.waitKey(33) & 0xFF == ord('q'):
                return
            cv2.imshow('Video', frame)  
        else:
            print("Error: Empty frame. Check your video source or image file.")
            return

if __name__ == '__main__':
    startapplication()