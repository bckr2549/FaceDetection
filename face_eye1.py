import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haar_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar_eye.xml')
smile_cascade = cv2.CascadeClassifier('haar_smile.xml')
img = cv2.imread("photoes/varun.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#if faces is ():
    #print("No faces found")

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("Face Detection", img)
    cv2.waitKey(0)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(100,255,220),2)
    smile = smile_cascade.detectMultiScale(roi_gray,1.3,2)
    for (sx,sy,sw,sh) in smile:
        cv2.rectangle(roi_color, (sx,sy),(sy+sh, sx+sw),(120,255,155),2)
        cv2.imshow("img", img)
        cv2.waitKey(0)
cv2.destroyAllWindows()
