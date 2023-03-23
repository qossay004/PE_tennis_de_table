import cv2
import numpy as np

# Initialize a background model from the first frame of the video
cap = cv2.VideoCapture("017.mp4")
ret, frame = cap.read()
background_model = np.float32(frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Subtract the current frame from the background model
    diff = cv2.absdiff(background_model, frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to separate the ball from the background
    threshold = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in any gaps in the ball
    kernel = np.ones((5, 5), np.uint8)
    threshold = cv2.dilate(threshold, kernel, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and draw a bounding box around the ball
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the processed frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
