# Install necessary libraries
#!pip install face_recognition pandas opencv-python

import face_recognition
import cv2
import pandas as pd
from datetime import datetime

# Function to recognize faces and update CSV
def recognize_and_update(image_path, csv_path):
    # Load the CSV file or create a new one if it doesn't exist
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name', 'LastSeen'])

    # Load the image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for encoding in face_encodings:
        # Compare with known faces in the CSV file
        matches = face_recognition.compare_faces(df['FaceEncoding'].tolist(), encoding)

        if any(matches):
            # If a match is found, update the timestamp for the first matching face
            match_index = matches.index(True)
            name = df.loc[match_index, 'Name']
            df.loc[match_index, 'LastSeen'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Welcome back, {name}!")
        else:
            # If no match is found, add a new entry to the CSV file
            name = f"Person_{len(df) + 1}"
            df = df.append({'Name': name, 'LastSeen': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'FaceEncoding': encoding}, ignore_index=True)
            print(f"New person detected: {name}!")

    # Save the updated CSV file
    df.to_csv(csv_path, index=False)

# Example usage:
image_path = 'path/to/your/image.jpg'
csv_path = 'path/to/your/face_records.csv'
recognize_and_update(image_path, csv_path)
