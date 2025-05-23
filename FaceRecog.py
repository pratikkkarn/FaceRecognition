import cv2

face_cap = cv2.CascadeClassifier("C:/Users/PRATIK KUMAR/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Convert only the face region to gray and paste it back
        face_region_gray = gray[y:y+h, x:x+w]
        video_data[y:y+h, x:x+w] = cv2.cvtColor(face_region_gray, cv2.COLOR_GRAY2BGR)
        
        # Draw green rectangle
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Videotra", video_data)

    if cv2.waitKey(10) == ord("x"):
        break

video_cap.release()
cv2.destroyAllWindows()
