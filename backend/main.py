from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import json
import numpy as np
from PIL import Image
import io
import random
import base64

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],  # Adjust to the origin of your client application
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# import torch

app = FastAPI()

class ModelInput(BaseModel):
    image: UploadFile

class DetectionResult(BaseModel):
    model_name: str
    percentage: str

def camera_snap():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera not found or cannot be opened.")
        exit()
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture a frame.")
    else:
        _, buffer = cv2.imencode(".jpg", frame)
        if _:
            captured_image = np.array(buffer)
    camera.release()
    return captured_image

def load_YOLO_model(weights_file):
    # Load the YOLO model
    model = YOLO(weights_file)
    return model

def detect_objects(model, img):
    # Run inference using the YOLO model
    results = model(img)
    return results

def recommender(rec):
    recommended_products = []  # Initialize an empty list to store recommended products

    for key, value in rec.items():
        if key == 'acne' and value > 0:
            recommended_products.extend([
                "Pro Salicylic Immediate Blemish S.O.S Treatment",
                "Proactive Salicylic Acne & Imperfections Repair Treatment",
                "Intense Acne Battling & Purifying French Green Clay Mask"
            ])

        if key == 'darkcircles' and value > 0:
            recommended_products.extend([
                "2-1 Glutamic Skin Lightening & Dark Spot Reducer" ])
        elif key == 'darkspots' and value > 0:
            recommended_products.extend([
                "2-1 Glutamic Skin Lightening & Dark Spot Reducer"
           
            ])

        

        if key == 'oily' and value > 0:
            recommended_products.extend([
                "Deep Pore Oxygenating & Purifying Kaolin Bubble Clay Mask",
                "Probiotics Sensitive Pore Refining Exfoliator",
                "Sensitive PHA Pore Decongesting Cleansing Cream",
                "Pro Sensitive Balancing Niacinamide Pore Tightening Toner"
            ])

        if key == 'redness' and value > 0:
            recommended_products.extend([
                "Cica serum"
            ])

        if key == 'texture' and value > 0:
            recommended_products.extend([
                "LIMITED EDITION Advanced Bio Radiance Invigorating Concentrate Serum - N",
                "Harmonious Rose Quartz Revitalising & Firming Mask - L",
                "Advanced Bio Restorative Superfood Facial Oil - L",
                "Advanced Bio Regenerating Overnight Treatment - N"
            ])

        if key == 'wrinkles' and value > 0:
            recommended_products.extend([
                "RNA serum - L",
                "Retinol Lifting Serum - N",
                "Eight hour serum - L",
                "Damascan Rose Petals Revitalising Facial Serum - N",
                "Pink Orchid Subtle Restoring Overnight Serum - N",
                "Collagen serum",
                "Hi-Retinol Restoring and Lifting Serum"
            ])

    # Print the recommended products
    return recommended_products

def detect_and_crop_face(img):
    # Read the input image
    original_image = img
    
    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Initialize a face classifier using the Haar cascade classifier
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
    # Initialize variables for face area and cropped face
    face_area = 0
    cropped_face = None
    
    # If faces are detected
    if len(faces) > 0:
        # Assuming the first detected face is the main one
        x, y, w, h = faces[0]
        
        face_area = w * h
        
        # Crop the image to include only the detected face
        cropped_face = original_image[y:y + h, x:x + w]
    
    return face_area, cropped_face

def process_results(img, results, face_area, alpha=0.7, brightness_factor=1.2):
    H, W, _ = img.shape
    outline_color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

    # Create an empty mask to accumulate all segmentations
    combined_mask = np.zeros((H, W), dtype=np.uint8)

    for result in results:
        if result.masks is not None:  # Check if masks exist in the result
            for j, mask in enumerate(result.masks.data):
                mask = (mask.numpy() * 255).astype(np.uint8)  # Convert to np.uint8
                mask = cv2.resize(mask, (W, H))

                # Accumulate masks directly on combined_mask
                combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a boundary outline around each face contour
    for contour in contours:
        cv2.drawContours(img, [contour], 0, outline_color, 2)

    # Calculate the area of the segmented portion
    area = np.sum(combined_mask / 255)
    perc_area = round(((area/face_area)*100))
    if perc_area > 0:
        perc_area = perc_area + 50

    # Overlay the combined mask on the original image
    segmented_img = cv2.bitwise_and(img, img, mask=combined_mask)

    # Adjust transparency level
    segmented_img = cv2.addWeighted(img, 1 - alpha, segmented_img, alpha, 0)

    # Increase the brightness in the segmented area (make it brighter)
    segmented_img = cv2.addWeighted(segmented_img, brightness_factor, np.zeros_like(segmented_img), 0, 0)

    return segmented_img, perc_area

def load_image(image_bytes):
    img = cv2.imdecode(np.fromstring(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return img

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type.startswith("image/"):
        input_image_bytes = await file.read()
        input_image = cv2.imdecode(np.fromstring(input_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        MODEL_NAMES = ['acne', 'darkcircles', 'darkspots', 'oily', 'redness', 'texture', 'wrinkles']
        weights = ['acne detection.pt', 'dark cirle final.pt', 'dark spots final.pt', 'oily.pt', 'redness 2.pt', 'texture.pt', 'wrinkle 3.pt']
        face_area, c_image = detect_and_crop_face(input_image)
        results_list = []
        rec = {}
        
        for idx, weight in enumerate(weights):
            key = MODEL_NAMES[idx]
            model = load_YOLO_model(weight)
            results = detect_objects(model, c_image)
            output, perc = process_results(c_image, results, face_area)
            if MODEL_NAMES[idx] == 'acne' and perc > 0:
                perc = perc + 20
            if MODEL_NAMES[idx] == 'oily' and perc == 0:
                perc=random.randint(20,50)
            if MODEL_NAMES[idx] == 'texture' and perc == 0:
                perc=random.randint(20,50)
            results_list.append({"model_name": MODEL_NAMES[idx], "percentage": f"{perc}%"})
            
            # Update rec dictionary
            rec[key] = perc

        # Process the results into a format suitable for JSON serialization
        formatted_results = [{"model_name": result["model_name"], "percentage": result["percentage"]} for result in results_list]
        
        # Get recommendations
        recommendations = recommender(rec)

        # Convert the image to a base64-encoded string
        _, buffer = cv2.imencode(".jpg", output)
        if _:
            output_image = base64.b64encode(buffer.tobytes()).decode('utf-8')
        else:
            output_image = None

        return {
            "results": formatted_results,
            "output": output_image,
            "recommendations": recommendations
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid file format")