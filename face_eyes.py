import cv2
import numpy as np
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(r"C:\Users\CHENNA KRISHNA\Desktop\open cv\.vscode\haar_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\CHENNA KRISHNA\Desktop\open cv\.vscode\haar_eye.xml")

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3,6)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img, 'Face',(x,y), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,250,0),2)

    eyes = eye_cascade.detectMultiScale(gray, 1.3,6)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(255,150,255),2)
        cv2.putText(img, 'Eye',(ex,ey), cv2.FONT_HERSHEY_COMPLEX,0.5,(250,0,250),2)
    

    cv2.imshow("img", img)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()