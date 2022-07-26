import boto3
import cv2
import dlib
import time
from datetime import datetime
import os
import numpy as np



video = cv2.VideoCapture('videoTest.mp4') # Capture frames from the video # We put 0 here for real timwe video feed.

car_Cas = cv2.CascadeClassifier('cars.xml')#Classifier for detecting cars

#Height and Width of the video
HEIGHT = 720 
WIDTH = 1280

startTrack = {} #To store Start time of cars
endTrack = {} # To store end End time of cars
cropBegin = 240 #CROP VIDEO FRAME FROM THIS POINT
mark1 = 100 #mark to start time time
mark2 = 360 #mark to end the time
markGap = 15 #Distance between markers in meters
fps = 3 # In order to controllow processing

def blackout(image):
    xBlack = 360
    yBlack = 300
    triangle_cnt = np.array( [[0,0], [xBlack,0], [0,yBlack]] )
    triangle_cnt2 = np.array( [[WIDTH,0], [WIDTH-xBlack,0], [WIDTH,yBlack]] )
    cv2.drawContours(image, [triangle_cnt], 0, (0,0,0), -1)#to draw any shape provided you have its boundary points
    cv2.drawContours(image, [triangle_cnt2], 0, (0,0,0), -1)

    return image


def car(speed,image): #This is a function to save the image, date, time and speed of a car
    now = datetime.today().now()
    nameTime = now.strftime("%d-%m-%Y-%H-%M-%S-%f")
    link = 'main/cars/'+nameTime+'.jpeg'
    try:
        cv2.imwrite(link,image)
    except:
        print("format error")

def estimateSpeed(carID): #This is to calculate speed
    diff_Time = endTrack[carID]-startTrack[carID]
    try:
        speed = round(markGap/diff_Time*fps*3.6,2) #Formula, 
    except:
        print("divide error")
    return speed





def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    carTracker = {} #To know what car it is. Storing in a dictionary

    while True:
        try:

            rc, image = video.read() # Reads the video
        except:
            print("Error in video reading")

        if type(image) == type(None): # Checking if the image has been processed.
            break

        frameTime = time.time()# Tells the particular time.
        try:

            image = cv2.resize(image, (WIDTH, HEIGHT))[cropBegin:720,0:1280] #To resize the image according to the width.
        except:
            print("Error in resize")
        resultImage = blackout(image)# for a 3 channel image. So what image we have we make a copy right now.
        cv2.line(resultImage,(0,mark1),(1280,mark1),(0,0,255),2)#cv2.line(image, start_point, end_point, color, thickness) 
        cv2.line(resultImage,(0,mark2),(1280,mark2),(0,0,255),2)

        frameCounter = frameCounter + 1 # incrementing frame by 1 because frame is changing.

        #DELETE CARIDs NOT IN FRAME
        carIDtoDelete = []

        for carID in carTracker.keys():# We need to delete the cars that have come out of the frame.
            trackingQuality = carTracker[carID].update(image)# Whst the threshold. Can we track it.
        

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            #print({"Removing"+ str(carID) + 'from list of trackers')
            carTracker.pop(carID, None) #It removes the lsst element in the list.

        #MAIN PROGRAM
        if not (frameCounter % 10 == 0):#If frame counter value not zero then the 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#gray scaling
            cars = car_Cas.detectMultiScale(gray, 1.1, 13, 18, (24, 24)) #DETECT CARS IN FRAME detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]

            for (_x, _y, _w, _h) in cars:
                #GET POSITION OF A CAR
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                #IF CENTROID OF CURRENT CAR NEAR THE CENTROID OF ANOTHER CAR IN PREVIOUS FRAME THEN THEY ARE THE SAME
                for carID in carTracker.keys():
                    trackerPosition = carTracker[carID].get_position()#Get positin of the id

                    t_x = int(trackerPosition.left()) 
                    t_y = int(trackerPosition.top())
                    t_w = int(trackerPosition.width())
                    t_h = int(trackerPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID


                if matchCarID is None:
                    

                    tracker = dlib.correlation_tracker()#to track objects that change in both (1) translation and (2) scaling throughout a video stream — and furthermore, we can perform this tracking in real-time.
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))#Here we track the rectangle around image

                    carTracker[currentCarID] = tracker#Allocating the tracker to the ids
                    currentCarID = currentCarID + 1#Incrementing by 1.


        for carID in carTracker.keys():
            trackerPosition = carTracker[carID].get_position() #Going from one loc to another
            speed=0
            t_x = int(trackerPosition.left()) 
            t_y = int(trackerPosition.top())
            t_w = int(trackerPosition.width())
            t_h = int(trackerPosition.height())

            
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 3)#Making a rectangle and continue tracking the position.
            cv2.putText(resultImage, str(carID), (t_x,t_y - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
            try:
                speed = estimateSpeed(carID)
                cv2.putText(resultImage, str(carID)+"  , "+str(speed), (t_x,t_y - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
            except:
                print("")
            #Here we estimate the speed
            if carID not in startTrack and mark2 > t_y + t_h > mark1 and t_y < mark1:
                startTrack[carID] = frameTime

            elif carID in startTrack and carID not in endTrack and mark2 < t_y + t_h:
                endTrack[carID] = frameTime
                speed = estimateSpeed(carID)
                cv2.putText(resultImage, str(carID)+str(speed), (t_x,t_y - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
                #saveCar(speed,image[t_y:t_y+t_h, t_x:t_x+t_w])
                print('CAR-ID : {} : {} kmph'.format(carID, speed))
           
        #DISPLAY EACH FRAME
        cv2.imshow('result', resultImage)

        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    trackMultipleObjects()


