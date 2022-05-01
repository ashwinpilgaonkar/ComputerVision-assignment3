import face_recognition
import cv2
import numpy as np

# Get a reference to webcam #0
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
ashwin_image = face_recognition.load_image_file("ashwin_img.jpg")
ashwin_image_enc = face_recognition.face_encodings(ashwin_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    ashwin_image_enc,
]
known_face_names = [
    "Ashwin Pilgaonkar"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
top_lip = []
bottom_lip=[]
center_points = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 25%
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # BGR -> RGB conversion
    rgb_small_frame = small_frame[:, :, ::-1]

    # Process alternate frames
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
        face_names = []
        for index, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            if name == 'Ashwin Pilgaonkar':
                keys = list(face_landmarks_list[index].keys())
                top_lip = face_landmarks_list[index][keys[-2]]
                bottom_lip = face_landmarks_list[index][keys[-1]]
                top_lip = np.array(top_lip, dtype=np.int32)
                bottom_lip = np.array(bottom_lip, dtype=np.int32)
                top_lip = top_lip*4
                bottom_lip = bottom_lip*4
                center_top_lip = np.mean(top_lip, axis=0)
                center_top_lip = center_top_lip.astype('int')
                center_points.append(center_top_lip)
                # print(face_landmarks_list[index][keys[-2]])
            face_names.append(name)
    process_this_frame = not process_this_frame


    # Display results
    cv2.polylines(frame, np.array([top_lip]), 1, (255,255,255))
    cv2.polylines(frame,np.array([bottom_lip]), 1, (255,255,255))
    for i in range(1, len(center_points)):
        if center_points[i-1] is None or center_points[i] is None:
            continue
        cv2.line(frame, tuple(center_points[i-1]), tuple(center_points[i]), (0,0,255), 2)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
    #Display the output
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()