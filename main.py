from flask import Flask, request, jsonify
import numpy as np
import cv2
from flask_cors import CORS
import base64
import requests
app = Flask(__name__)
CORS(app)

@app.route('/try-on', methods=['POST'])
def predict():
    try:
        # Load T-shirt image with transparency
        imgshirt_url = request.form.get('image_path')
        print(f"Image URL: {imgshirt_url}")
        
        # Convert JPEG to PNG
        response = requests.get(imgshirt_url)
        if response.status_code != 200:
            return "Failed to download image from URL", 500

        image_array = np.array(bytearray(response.content), dtype=np.uint8)
        imgshirt = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        if imgshirt is None:
            print("Error: T-shirt image not found.")
            return "T-shirt image not found", 500
        else:
            print(f"T-shirt image loaded with shape: {imgshirt.shape}")

        # Ensure T-shirt image has 4 channels (RGBA)
        if imgshirt.shape[2] != 4:
            print("Error: T-shirt image does not have an alpha channel.")
            return "T-shirt image does not have an alpha channel", 500

        # Extract the alpha mask from the T-shirt image
        alpha_mask = imgshirt[:, :, 3]
        rgb_shirt = imgshirt[:, :, :3]
        
        # Prepare the T-shirt mask
        orig_masks = alpha_mask
        orig_masks_inv = cv2.bitwise_not(orig_masks)

        origshirtHeight, origshirtWidth = rgb_shirt.shape[:2]
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Load the person's image (Replace with actual user's uploaded image handling)
        user_img_data = request.files['user_image'].read()
        nparr = np.frombuffer(user_img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        zoom_factor = 0.40  # Adjust this factor to zoom out more or less
        img = cv2.resize(img, (0, 0), fx=zoom_factor, fy=zoom_factor)
        height, width = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            shirtWidth = 10 * w  
            shirtHeight = (shirtWidth * origshirtHeight / origshirtWidth  ) 
            
            y1s = y + h  # Start the shirt just below the chin
            y2s = y1s + shirtHeight  # Ensure shirt extends below the calculated y1s

            x1s = max(0, x - int(shirtWidth / 2))  # Center the shirt horizontally under the face
            x2s = x1s + shirtWidth  # Ensure shirt width matches calculated x1s
            
            x1s = max(0, x1s)
            x2s = min(img.shape[1], x2s)
            y1s = max(0, y1s) - 8
            y2s = min(img.shape[0], y2s)

            if y1s >= y2s or x1s >= x2s:
                print("Invalid ROI dimensions, skipping frame.")
                continue

            shirtWidth = int(abs(x2s - x1s))
            shirtHeight = int(abs(y2s - y1s ))
            y1s, y2s, x1s, x2s = map(int, [y1s, y2s, x1s, x2s])

            shirt = cv2.resize(rgb_shirt, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)
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

            # Annotate the image with neck, arm, chest, and width dimensions in a single line
            text_color = (255, 0, 0)  # Blue color in BGR
            cv2.putText(img, f"Neck Width: {w}px | Arm Length: {h * 2}px | Chest Width: {w * 2}px | Image Width: {img.shape[1]}px", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 2)

            break

        # Convert image to base64 string
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # Return base64-encoded image in JSON response
        return jsonify({"image": img_base64})

    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)