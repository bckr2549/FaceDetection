import cv2
img = cv2.imread('photoes/sriram.png')
cv2.imshow('Sriram', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray People', gray)
haar_cascade = cv2.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(f'Number of faces found = {len(faces_rect)}')
for (x,y,w,h) in faces_rect:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), thickness=3)
cv2.imshow('Detected Face', img)
cv2.waitKey(0)