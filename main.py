import cv2

# Load the pre-trained model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face, detect emotions
    for (x,y,w,h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # Extract the face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect emotions
        emotions = emotion_cascade.detectMultiScale(roi_gray, 1.3, 5)

        # For each emotion, draw a rectangle around it
        for (ex,ey,ew,eh) in emotions:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the frame
    cv2.imshow('frame',frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
