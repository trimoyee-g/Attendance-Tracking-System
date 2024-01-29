import cv2
import mediapipe as mp
import csv
from datetime import datetime
import numpy as np
import face_recognition

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

video_capture = cv2.VideoCapture(0)

# Load known faces and encodings
s1_image = face_recognition.load_image_file("photos/s1.jpg")
s1_encoding = face_recognition.face_encodings(s1_image)[0]

s2_image = face_recognition.load_image_file("photos/s2.jpg")
s2_encoding = face_recognition.face_encodings(s2_image)[0]

s3_image = face_recognition.load_image_file("photos/s3.jpg")
s3_encoding = face_recognition.face_encodings(s3_image)[0]

s4_image = face_recognition.load_image_file("photos/s4.jpg")
s4_encoding = face_recognition.face_encodings(s4_image)[0]

known_face_encodings = [s1_encoding, s2_encoding, s3_encoding, s4_encoding]
known_face_names = ["s1.jpg", "s2.jpg", "s3.jpg", "s4.jpg"]

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

# Set to store the names of faces already stored
stored_faces = set()

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:

    while True:
        _, frame = video_capture.read()

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe Face Detection
        results = face_detection.process(rgb_frame)

        # Check if any detections are present
        if results.detections:
            for detection in results.detections:
                # Check if the detection score is above a threshold
                if detection.score[0] > 0.5:
                    ih, iw, _ = frame.shape

                    # Extract face bounding box coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)

                    # Extract face encoding from the detected face
                    face_encoding = face_recognition.face_encodings(rgb_frame, [bbox])[0]

                    # Compare with known face encodings
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    if any(matches):
                        best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                        name = known_face_names[best_match_index]

                        # Check if the face has not been stored yet
                        if name not in stored_faces:
                            current_time = now.strftime("%H-%M-%S")
                            lnwriter.writerow([name, current_time])
                            stored_faces.add(name)  # Add the name to the set to mark it as stored

                    # Visual signal: Draw a green rectangle around the detected face
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

                    # Add text to indicate face detected and stored
                    cv2.putText(frame, f"Face Detected: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Attendance system", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
f.close()







# video = cv2.VideoCapture(0);
# students = []

# with open("1.csv", "r") as file:
#     reader = csv.reader(file)
#     for row in reader:
#         students.append(row[1])


# while True:
#     check, frame = video.read()
#     d = decode(frame)
#     try:
#         for obj in d:
#             name = d[0].data.decode()
#             if name in students:
#                 students.remove(name)
#                 print("deleted...")
#     except:
#         print("error")

#     cv2.imshow("Attendance", frame)
#     key = cv2.waitKey(1)
#     if key==ord('q'):
#         print(students)
#         break

# video.release()
# cv2.destroyAllWindows()