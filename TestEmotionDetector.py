import cv2
import csv 
import matplotlib as plt
from matplotlib import animation
from matplotlib import style
import pandas as pd
import numpy as np
import time 
from keras.models import model_from_json
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot



# emotion_dict = {0: "Happy", 1: "Disgusted", 2: "Fearful", 3: "Angry", 4: "surpised", 5: "Sad", 6: "Neutral"}
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

plotY = LivePlot(900, 600, [-25, 25], invert=True)

# stuff = pd.read_csv("graphy.csv")
fieldnames = ["x_Time", "y_EmoSum"]
x_Time = []
y_Emotion =[]
y_EmoSum = 0

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

#Blink rate Variabels
detector = FaceMeshDetector(maxFaces=1)
ratioList = []
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
blinkCounter = 0
counter = 0
color = (255, 0, 255)
final_count = 0

#Lie and truth Counters
lie_Counter = 0
truth_Counter = 0
emo_freq_dict = {
                    "Anger" : 0,
                    "Disgust" : 0,
                    "Fear" : 0,
                    "Happy" : 0,
                    "Neutral" : 0,
                    "Sad" : 0,
                    "Surprise" : 0
                 } # Dictionary for Emotions frequency Detections

endtime = int(time.time()) + 12
time_Count = 0

with open('EmotionsDetected.csv', 'a', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")
def emotion_test(): 
    global time_Count 
    global blinkCounter 
    global counter 
    global final_count 
    global lie_Counter
    global truth_Counter 
    global y_EmoSum
    color = (255, 0, 255)
    start_execution_time = time.time()
    end_execution_time = start_execution_time + 30

    while time.time() <= end_execution_time:

##############################_BLINK RATE code_#################################
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, draw=False)
    
        if faces:
            face = faces[0]
            # for id in idList:
            #     cv2.circle(img, face[id], 5)
    
            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            lenghtVer, _ = detector.findDistance(leftUp, leftDown)
            lenghtHor, _ = detector.findDistance(leftLeft, leftRight)
    
            cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
            cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)
    
            ratio = int((lenghtVer / lenghtHor) * 100)
            ratioList.append(ratio)
            if len(ratioList) > 3:
                ratioList.pop(0)
            ratioAvg = sum(ratioList) / len(ratioList)
    
            if ratioAvg < 35 and counter == 0:
                blinkCounter += 1
                final_count += 1
                # color = (0,200,0)
                counter = 1
            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0
                    # color = (255,0, 255)
    
            # cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100))
    
            # imgPlot = plotY.update(ratioAvg, color)
            # img = cv2.resize(img, (900, 600))
            # imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
        # else:
        #     img = cv2.resize(img, (900, 600))
        #     # imgStack = cvzone.stackImages([img, img], 2, 1)

############################_Emotion Detection Code_##############################
        # y_EmoSum = 0
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1080, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            serial = 1
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))

            if maxindex == 0: #Angry = -1
                y_EmoSum =+ -1
                truth_Counter =+ 1
                emo_freq_dict['Anger'] +=1
            elif maxindex == 1: #Disgusted = 1
                y_EmoSum += 1
                lie_Counter += 1
                emo_freq_dict['Disgust'] +=1
            elif maxindex == 2: #Fear = 2
                y_EmoSum += 2
                lie_Counter += 1
                emo_freq_dict['Fear'] +=1
            elif maxindex == 3: #Happy = -2
                y_EmoSum += -2
                truth_Counter += 1
                emo_freq_dict['Happy'] +=1
            elif maxindex == 4: #Neutral = 0
                y_EmoSum += 0
                emo_freq_dict['Neutral'] +=1
            elif maxindex == 5: #Sad = -3
                y_EmoSum += -3
                truth_Counter += 1
                emo_freq_dict['Sad'] +=1
            elif maxindex == 6: #Surprised = 3
                y_EmoSum += 3
                lie_Counter += 1
                emo_freq_dict['Surprise'] +=1

            x_Time.append(time_Count)
            y_Emotion.append(y_EmoSum)
            dataframe = pd.DataFrame(list(zip(x_Time, y_Emotion)), columns=['Time', 'Emotions'])
            dataframe.to_csv("EmotionsDetected.csv")   

            print(maxindex)
            print(final_count)
            # cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)
            
            time_Count =+ 1
            cv2.putText(frame, "Blink : " + str(blinkCounter) + " " + emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # imgPlot = plotY.update(maxindex, color)
            # frame = cv2.resize(img, (900, 600))
            # imgStack = cvzone.stackImages([frame, imgPlot], 2, 1)

        cv2.imshow('Emotion Detection', frame)
        # cv2.imshow("Image", imgStack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # file.close()
    cap.release()
    cv2.destroyAllWindows()


emotion_test()

print("Lied Percentage : ", int((lie_Counter/(lie_Counter + truth_Counter))*100))
print("Truth Percentage : ", int((truth_Counter/(lie_Counter + truth_Counter))*100))
print("Average Blink rate : ", final_count)
print(emo_freq_dict)