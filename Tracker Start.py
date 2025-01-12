import cv2


face_cascade = cv2.CascadeClassifier("C:/Users/Pelos/OneDrive/Desktop/Projects/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(frame_w)

while True:
    cap_success, frame = cap.read()
    if not cap_success:
        break
    grey_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey_frame,scaleFactor=1.1, minNeighbors=5,minSize=(30,30))
   
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (128, 0, 128), 2)
        center_x = int(x + (float(w)/2.0))
       
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
