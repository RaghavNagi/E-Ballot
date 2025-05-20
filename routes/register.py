from flask import Blueprint, request, jsonify, render_template
import os
import cv2
import numpy as np
import pickle
from config import Config

register_bp = Blueprint('register', __name__, url_prefix='/register')

@register_bp.route('/', methods=['GET'])
def register_form():
    return render_template('register.html')

@register_bp.route('/', methods=['POST'])
def register():
    aadhar_number = request.form.get('aadhar')

    if not aadhar_number:
        return jsonify({'status': 'error', 'message': 'Aadhar number is required'}), 400

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    frames_captured = 0
    max_frames = 10
    faces_data = []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while frames_captured < max_frames:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = gray[y:y + h, x:x + w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten()

            if resized_img.shape[0] != 2500:
                print(f"Unexpected shape: {resized_img.shape}")
                continue

            faces_data.append(resized_img)
            frames_captured += 1

            print(f"Captured frame {frames_captured}/{max_frames}")

    video_capture.release()
    cv2.destroyAllWindows()

    # If no faces were captured
    if len(faces_data) == 0:
        print("No face data captured.")
        return jsonify({'status': 'error', 'message': 'No face data captured'}), 400

    # Convert to numpy array
    faces_data = np.array(faces_data)

    # Save data
    data_path = Config.DATA_PATH
    os.makedirs(data_path, exist_ok=True)

    names_path = os.path.join(data_path, 'names.pkl')
    faces_path = os.path.join(data_path, 'faces_data.pkl')

    # Load existing data or initialize new
    if os.path.exists(names_path):
        try:
            with open(names_path, 'rb') as f:
                names = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print("Warning: names.pkl is empty or corrupt. Initializing empty list.")
            names = []
    else:
        names = []

    # Check if aadhar number already registered
    if aadhar_number in names:

        cv2.destroyAllWindows()
        return jsonify({'status': 'error', 'message': 'Aadhar number already registered'}), 400

    if os.path.exists(faces_path):
        try:
            with open(faces_path, 'rb') as f:
                faces = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print("Warning: faces_data.pkl is empty or corrupt. Initializing empty array.")
            faces = np.empty((0, 2500))
    else:
        faces = np.empty((0, 2500))

    # Check for face duplication:
    # Compute distances between each new face and all existing faces
    # If any distance is below threshold, reject as duplicate registration
    threshold = 3000  # You can tune this value (lower = stricter)
    for new_face in faces_data:
        distances = np.linalg.norm(faces - new_face, axis=1) if faces.shape[0] > 0 else []
        if len(distances) > 0 and np.any(distances < threshold):
            cv2.destroyAllWindows()
            return jsonify({'status': 'error', 'message': 'Face already registered'}), 400

    # Ensure data consistency
    print(f"Existing faces data shape: {faces.shape}")
    print(f"New faces data shape: {faces_data.shape}")

    if faces.shape[1] != faces_data.shape[1]:
        print(f"Shape mismatch. Expected {faces.shape[1]}, got {faces_data.shape[1]}")
        return jsonify({'status': 'error', 'message': 'Data shape mismatch'}), 400

    # Update data
    names += [aadhar_number] * len(faces_data)
    faces = np.vstack([faces, faces_data])

    # Save updated data
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)

    with open(faces_path, 'wb') as f:
        pickle.dump(faces, f)

    print(f"User {aadhar_number} successfully registered with {len(faces_data)} images.")

    return jsonify({'status': 'success', 'message': 'Registration successful'})