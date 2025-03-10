from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import os
import csv
import time
from datetime import datetime
import subprocess  # For macOS built-in 'say' command
import pickle

def speak(text):
    """ Use macOS built-in 'say' command for speech synthesis. """
    subprocess.run(["say", text])

video = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists('data/'):
    os.makedirs('data/')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f) 

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")

COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']

while True:
    output = None
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)[0]  # Extract single value

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    if output is None:
        continue

    frame_resized = cv2.resize(frame, (640, 480))  
    imgBackground[370:370 + 480, 255:255 + 640] = frame_resized

    cv2.imshow('frame', imgBackground)
    k = cv2.waitKey(1)

    def check_if_exists(value):
        """Check if a voter has already voted."""
        try:
            with open("Voter.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row and row[0] == value:
                        return True
        except FileNotFoundError:
            print("File Not Found or unable to open the CSV file.")
        return False

    if check_if_exists(output):
        print("YOU HAVE ALREADY VOTED")
        speak("YOU HAVE ALREADY VOTED")
        break

    exist = os.path.isfile("Voter.csv")  

    # Voting Logic
    if k in [ord('1'), ord('2'), ord('3'), ord('4')]:
        vote_map = {'1': "BJP", '2': "CONGRESS", '3': "AAP", '4': "NOTA"}
        selected_vote = vote_map[chr(k)]

        speak("YOUR VOTE HAS BEEN RECORDED")
        time.sleep(3)

        with open("Voter.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)  
            writer.writerow([output, selected_vote, date, timestamp])  

        speak("THANK YOU FOR PARTICIPATING IN THE ELECTIONS")
        break

video.release()
cv2.destroyAllWindows()