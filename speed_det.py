import cv2


cap = cv2.VideoCapture('carsd.mp4')#Capture frames from the video'

car_cas = cv2.CascadeClassifier('cars.xml')# Will help in car detection

while True:
    ret, frames = cap.read() # This reads the frames from a video 
    
   # gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) #Converts to gray

    #detect cars and return a rectangle on them
    cars = car_cas.detectMultiScale( frames, 1.5, 2)
    for (x,y,a,b) in cars:
        cv2.rectangle(frames,(x,y),(x+a,y+b),(0,0,255),2)
        cv2.imshow('Car Detection', frames)# Display frames
    if cv2.waitKey(33) == 13:
        break

cv2.destroyAllWindows()   


