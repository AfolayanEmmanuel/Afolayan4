import cv2
import pickle
import time
import numpy as np
import face_recognition
import csv
from datetime import datetime
from scipy.spatial import distance as dist
import os

# Load the trained ensemble model
with open("classifier/ensemble_model.pkl", 'rb') as f:
    ensemble_clf = pickle.load(f)

# Define helper functions

def detect_movement(prev_landmarks, curr_landmarks, threshold=5):
    """
    Detect movement in the eye landmarks.
    """
    prev_center = np.mean(prev_landmarks, axis=0)
    curr_center = np.mean(curr_landmarks, axis=0)
    movement = np.linalg.norm(curr_center - prev_center)
    return movement > threshold

def calculate_ear(eye):
    """
    Calculate Eye Aspect Ratio (EAR) for a given eye landmarks.
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def predict_faces(frame, movement_threshold=5):
    predictions = []
    face_locations = face_recognition.face_locations(frame)
    for top, right, bottom, left in face_locations:
        face_image = frame[top:bottom, left:right]
        # Convert BGR to RGB
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(face_image)
        if len(face_encoding) == 0:
            continue
        
        # Predict using ensemble model
        ensemble_pred = ensemble_clf.predict(face_encoding)[0]
        ensemble_confidence = ensemble_clf.predict_proba(face_encoding).max() * 100  # Confidence score for Ensemble in percentage
        
        # Detect facial landmarks for calculating EAR
        face_landmarks = face_recognition.face_landmarks(face_image)
        movement_detected = False
        if len(face_landmarks) > 1:  # Ensure both eyes are detected
            prev_left_eye = np.array(face_landmarks[0]['left_eye'])
            prev_right_eye = np.array(face_landmarks[0]['right_eye'])
            for landmarks in face_landmarks[1:]:
                curr_left_eye = np.array(landmarks['left_eye'])
                curr_right_eye = np.array(landmarks['right_eye'])
                movement_left_eye = detect_movement(prev_left_eye, curr_left_eye, threshold=movement_threshold)
                movement_right_eye = detect_movement(prev_right_eye, curr_right_eye, threshold=movement_threshold)
                if movement_left_eye or movement_right_eye:
                    movement_detected = True
                    break
                
            # Calculate EAR for both eyes
            left_eye_ear = calculate_ear(face_landmarks[0]['left_eye'])
            right_eye_ear = calculate_ear(face_landmarks[0]['right_eye'])
            
            print(f"Left Eye EAR: {left_eye_ear:.2f}")
            print(f"Right Eye EAR: {right_eye_ear:.2f}")

        predictions.append((ensemble_pred, ensemble_confidence, movement_detected, (top, right, bottom, left), face_landmarks))
    return predictions

def log_attendance(name, confidence, attendance_recorded):
    if name != "unknown" and not attendance_recorded.get(name, False):
        attendance_recorded[name] = True
        print(f"Attendance recorded for: {name}, Confidence: {confidence:.2f}%, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        with open('attendance.csv', 'a', newline='') as csvfile:
            fieldnames = ['Name', 'Time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow({'Name': name, 'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

def main():
    # Initialize webcam
    webcam = cv2.VideoCapture(0)  # Open webcam with index 0 (primary webcam)
    if not webcam.isOpened():  # Check if the webcam is opened successfully
        print("Failed to open webcam. Exiting...")
        return
    
    # Warm-up period
    print("Warm-up period: Starting...")
    for _ in range(10):  # Perform warm-up by reading and discarding initial frames
        rval, _ = webcam.read()
        if not rval:
            print("Failed to read frame from webcam during warm-up. Exiting...")
            return
        time.sleep(0.1)  # Sleep for a short duration between frames
    print("Warm-up period: Complete")
    
    # Initialize attendance record
    attendance_recorded = {}
    
    movement_threshold = 5
    
    last_capture_time = time.time()
    capture_interval = 3  # seconds
    
    while True:
        current_time = time.time()
        if current_time - last_capture_time < capture_interval:
            time.sleep(0.1)
            continue
        
        last_capture_time = current_time
        
        rval, frame = webcam.read()
        if not rval:
            print("Failed to read frame from webcam. Exiting...")
            break
        
        predictions = predict_faces(frame, movement_threshold)
        font = cv2.FONT_HERSHEY_DUPLEX
        
        for pred, confidence, _, (top, right, bottom, left), _ in predictions:
            log_attendance(pred, confidence, attendance_recorded)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, pred, (left, top - 10), font, 0.8, (255, 255, 255), 1)  # Write name
        
        cv2.imshow('Video', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("Exiting. Please wait...")
            time.sleep(1)
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
