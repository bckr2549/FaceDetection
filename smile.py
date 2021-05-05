import cv2

cap = cv2.VideoCapture(0)
smile_cascade = cv2.CascadeClassifier(r"C:\Users\CHENNA KRISHNA\Desktop\open cv\.vscode\haar_smile.xml")
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.3, 9)
    for (sx,sy,sw,sh) in smiles:
        cv2.rectangle(frame, (sx,sy),(sx+sw, sy+sh), (0,255,0), 2)
        cv2.putText(frame, 'smile', (sx,sy),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),3)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) 
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()