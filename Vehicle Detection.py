import cv2

cap = cv2.VideoCapture(r'D:\Shailesh Programs\Git Hub repo\Vehicle detection\car1.mp4')

car_cascade = cv2.CascadeClassifier(r'D:\Shailesh Programs\Git Hub repo\Vehicle detection\cars.xml')

while True:
    ret, frames = cap.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('project', frames )
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
