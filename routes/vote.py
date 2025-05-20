from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session
import os
import cv2
import numpy as np
import pickle
from config import Config
from sklearn.neighbors import KNeighborsClassifier

vote_bp = Blueprint('vote', __name__, url_prefix='/vote')

@vote_bp.route('/', methods=['GET'])
def vote_form():
    """ Voting form to submit Aadhar number. """
    return render_template('vote.html')


@vote_bp.route('/verify', methods=['POST'])
def verify_aadhar():
    """ Verify Aadhar and capture face using webcam. """
    aadhar_number = request.form.get('aadhar')

    if not aadhar_number:
        return jsonify({'status': 'error', 'message': 'Aadhar number is required'}), 400

    # Load registered data
    data_path = Config.DATA_PATH
    names_path = os.path.join(data_path, 'names.pkl')
    faces_path = os.path.join(data_path, 'faces_data.pkl')

    if not os.path.exists(names_path) or not os.path.exists(faces_path):
        return jsonify({'status': 'error', 'message': 'No registered data found'}), 404

    with open(names_path, 'rb') as f:
        names = pickle.load(f)

    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)

    if len(names) != faces.shape[0]:
        return jsonify({'status': 'error', 'message': 'Data inconsistency detected'}), 500

    # Check if the Aadhar is registered
    if aadhar_number not in names:
        return jsonify({'status': 'error', 'message': 'Aadhar number not registered'}), 400

    # Initialize webcam and face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0)
    face_verified = False

    while True:
        ret, frame = video.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_in_frame = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_in_frame:
            face_img = gray[y:y + h, x:x + w]
            resized_img = cv2.resize(face_img, (50, 50)).flatten().reshape(1, -1)

            # Train KNN model
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(faces, names)

            # Predict the Aadhar number
            predicted_aadhar = knn.predict(resized_img)[0]

            # Check if the predicted Aadhar matches the provided Aadhar
            if predicted_aadhar == aadhar_number:
                face_verified = True
                session['aadhar'] = aadhar_number  # Store Aadhar in session for voting
                video.release()
                cv2.destroyAllWindows()
                # Redirect to the voting page
                return redirect(url_for('vote.vote_page'))

        # -- COMMENTED OUT GUI display because it breaks Flask server --
        # cv2.imshow("Face Verification", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    video.release()
    cv2.destroyAllWindows()

    return jsonify({'status': 'error', 'message': 'Face verification failed'}), 400


@vote_bp.route('/vote', methods=['GET', 'POST'])
def vote_page():
    """ Voting page after successful verification. """
    aadhar = session.get('aadhar')

    if not aadhar:
        return redirect(url_for('vote.vote_form'))  # Redirect to Aadhar form if not verified

    if request.method == 'POST':
        selected_vote = request.form.get('vote')

        if not selected_vote:
            return jsonify({'status': 'error', 'message': 'Vote is required'}), 400

        # Check if user already voted
        data_path = Config.DATA_PATH
        voter_file = os.path.join(data_path, 'Voter.csv')

        if os.path.exists(voter_file):
            with open(voter_file, 'r') as f:
                for line in f:
                    line_aadhar = line.strip().split(',')[0]
                    if line_aadhar == aadhar:
                        cv2.destroyAllWindows()
                        return jsonify({'status': 'error', 'message': 'You have already voted'}), 403

        # Save vote to CSV
        from datetime import datetime
        with open(voter_file, 'a') as f:
            f.write(f"{aadhar},{selected_vote},{datetime.now()}\n")

        # Clear session after voting
        session.pop('aadhar', None)

        return redirect(url_for('home'))

    # Render voting page with voting options
    return render_template('voting.html')