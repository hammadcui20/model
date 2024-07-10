from flask import Flask, render_template, request
import json
import numpy as np
import cv2
from math import floor
from flask_cors import CORS
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)

@app.route('/try-on', methods=['POST'])
def predict():
    try:
        shirtno = 3
        data = request.json
        image_url = data.get('image_path')
        print(f"Image URL: {image_url}")

        response = requests.get(image_url)
        if response.status_code != 200:
            return "Failed to download image from URL", 500

        image_array = np.array(bytearray(response.content), dtype=np.uint8)
        imgshirt = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        cv2.waitKey(1)
        cap = cv2.VideoCapture(0)
        ih = shirtno

        while True:
            if ih == 3:
                shirtgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)  # grayscale conversion
                ret, orig_masks_inv = cv2.threshold(shirtgray, 200, 255, cv2.THRESH_BINARY)  # thresholding
                orig_masks = cv2.bitwise_not(orig_masks_inv)
            else:
                shirtgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)  # grayscale conversion
                ret, orig_masks = cv2.threshold(shirtgray, 0, 255, cv2.THRESH_BINARY)  # thresholding
                orig_masks_inv = cv2.bitwise_not(orig_masks)
            
            origshirtHeight, origshirtWidth = imgshirt.shape[:2]
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            ret, img = cap.read()
            if not ret:
                return "Failed to capture image from camera", 500
            
            height = img.shape[0]
            width = img.shape[1]
            resizewidth = int(width * 3 / 2)
            resizeheight = int(height * 3 / 2)
            
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("img", (resizewidth, resizeheight))
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)

                shirtWidth = 3 * w  
                shirtHeight = shirtWidth * origshirtHeight / origshirtWidth  
                
                x1s = x - w
                x2s = x1s + 3 * w
                y1s = y + h + 16  # Start the shirt a bit lower (10 pixels below the chin)
                y2s = y1s + h * 4
                
                x1s = max(0, x1s)
                x2s = min(img.shape[1], x2s)
                y1s = max(0, y1s)
                y2s = min(img.shape[0], y2s)

                if y1s >= y2s or x1s >= x2s:
                    print("Invalid ROI dimensions, skipping frame.")
                    continue

                shirtWidth = int(abs(x2s - x1s))
                shirtHeight = int(abs(y2s - y1s))
                y1s, y2s, x1s, x2s = map(int, [y1s, y2s, x1s, x2s])

                shirt = cv2.resize(imgshirt, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(orig_masks, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)
                masks_inv = cv2.resize(orig_masks_inv, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)
                
                rois = img[y1s:y2s, x1s:x2s]

                print(f"ROI size: {rois.shape}, Shirt size: {shirt.shape}")
                
                if rois.shape[:2] == shirt.shape[:2]:
                    roi_bgs = cv2.bitwise_and(rois, rois, mask=masks_inv)
                    roi_fgs = cv2.bitwise_and(shirt, shirt, mask=mask)
                    dsts = cv2.add(roi_bgs, roi_fgs)
                    img[y1s:y2s, x1s:x2s] = dsts
                else:
                    print(f"Size mismatch: ROI size: {rois.shape}, Shirt size: {shirt.shape}")

                # Annotate the image with neck, arm, chest, and width dimensions
                neck_width = w
                arm_length = h * 2  # This is an estimated value
                chest_width = w * 2  # This is an estimated value
                image_width = img.shape[1]

                text_color = (255, 0, 0)  # Blue color in BGR

                cv2.putText(img, f"Neck Width: {neck_width}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                cv2.putText(img, f"Arm Length: {arm_length}px", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                cv2.putText(img, f"Chest Width: {chest_width}px", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                cv2.putText(img, f"Image Width: {image_width}px", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

                break

            cv2.imshow("img", img)
            if cv2.waitKey(100) == ord('q'):
                break

        cap.release()  # Release the video capture object
        cv2.destroyAllWindows()  # Close all OpenCV windows
        
        return "Prediction completed successfully", 200

    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
