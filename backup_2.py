from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

# Load the T-shirt image with alpha channel
tshirt = cv2.imread('Shirt.png', -1)
if tshirt is None:
    print("Error: T-shirt image not found.")
else:
    print(f"T-shirt image loaded with shape: {tshirt.shape}")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the Haar cascade for upper body detection as an alternative
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Desired display frame size
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect bodies in the frame with adjusted parameters
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Debug: Print number of bodies detected
    print(f"Number of bodies detected: {len(bodies)}")

    for (x, y, w, h) in bodies:
        # Calculate the position and size of the shirt based on the body
        shirt_width = int(1.5 * w)
        shirt_height = int(1.5 * h)
        shirt_x = x - int((shirt_width - w) / 2)
        shirt_y = y - int((shirt_height - h) / 2)

        # Resize the shirt to fit the body
        resized_tshirt = cv2.resize(tshirt, (shirt_width, shirt_height), interpolation=cv2.INTER_AREA)

        # Ensure the shirt doesn't go out of bounds
        if shirt_x < 0:
            resized_tshirt = resized_tshirt[:, -shirt_x:]
            shirt_x = 0
        if shirt_y < 0:
            resized_tshirt = resized_tshirt[-shirt_y:, :]
            shirt_y = 0
        if shirt_x + resized_tshirt.shape[1] > frame.shape[1]:
            resized_tshirt = resized_tshirt[:, :frame.shape[1] - shirt_x]
        if shirt_y + resized_tshirt.shape[0] > frame.shape[0]:
            resized_tshirt = resized_tshirt[:frame.shape[0] - shirt_y, :]

        # Extract the alpha channel from the resized T-shirt image
        alpha_tshirt = resized_tshirt[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_tshirt

        # Overlay the T-shirt onto the frame using alpha blending
        for c in range(0, 3):
            frame[shirt_y:shirt_y + resized_tshirt.shape[0], shirt_x:shirt_x + resized_tshirt.shape[1], c] = (
                alpha_tshirt * resized_tshirt[:, :, c] +
                alpha_frame * frame[shirt_y:shirt_y + resized_tshirt.shape[0], shirt_x:shirt_x + resized_tshirt.shape[1], c]
            )

        # Estimate additional dimensions
        arm_length = int(0.4 * shirt_height)
        chest_width = int(0.8 * shirt_width)
        neck_width = int(0.3 * shirt_width)

        # Print the dimensions of the fitted shirt on the screen
        cv2.putText(frame, f'Shirt: {shirt_width}x{shirt_height}', 
                    (shirt_x, shirt_y - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Arms: {arm_length}px', 
                    (shirt_x, shirt_y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Chest: {chest_width}px', 
                    (shirt_x, shirt_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Neck: {neck_width}px', 
                    (shirt_x, shirt_y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)

    # Resize the frame for display
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # Display the frame
    cv2.imshow('Virtual T-shirt Try-On', display_frame)

    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
