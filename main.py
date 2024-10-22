import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start capturing video
cap = cv2.VideoCapture(0)


def mainLoop(normal_th, up_th, down_th):
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # Visualize landmarks for eyes and eyebrows
            for n in range(36, 48):  # Draw eyes (left and right)
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Blue for eyes

            for n in range(17, 27):  # Draw eyebrows (left and right)
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green for eyebrows

            # Get left eyebrow and left eye points
            left_eyebrow = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(18, 23)])
            left_eye = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])

            # Get right eyebrow and right eye points
            right_eyebrow = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(23, 28)])
            right_eye = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])

            # Flip the subtraction to make the distance positive (from eyes to eyebrows)
            left_eyebrow_eye_dist = np.mean(left_eye[:, 1]) - np.mean(left_eyebrow[:, 1])
            right_eyebrow_eye_dist = np.mean(right_eye[:, 1]) - np.mean(right_eyebrow[:, 1])

            # Average the distances for both eyebrows
            avg_eyebrow_eye_dist = (left_eyebrow_eye_dist + right_eyebrow_eye_dist) / 2

            # Get the face height (distance between chin and forehead)
            face_height = abs(landmarks.part(8).y - landmarks.part(27).y)

            # Normalize the eyebrow-eye distance relative to face height
            eyebrow_eye_dist_normalized = avg_eyebrow_eye_dist / face_height

            # Determine the eyebrow position based on dynamic thresholds
            if eyebrow_eye_dist_normalized > up_th - normal_th/10:
                position = "Eyebrows Up"
            elif eyebrow_eye_dist_normalized < down_th + normal_th/8:
                position = "Eyebrows Down"
            else:
                position = "Eyebrows Normal"

            print(position)

            # Display the eyebrow position on the frame
            cv2.putText(frame, position, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with detected landmarks and eyebrow position
        ## cv2.imshow("Frame", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def captureThreshold(position_description):
    # Function should capture the threshold value for the given position
    # It should run for 5 seconds and capture the average value
    # Return the average value

    input("Press Enter to start capturing the threshold for " + position_description)
    start_time = cv2.getTickCount()
    threshold_values = []

    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 5:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # Visualize landmarks for eyes and eyebrows
            for n in range(36, 48):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            for n in range(17, 27):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            left_eyebrow = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(18, 23)])
            left_eye = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])
            
            right_eyebrow = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(23, 28)])
            right_eye = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])

            left_eyebrow_eye_dist = np.mean(left_eye[:, 1]) - np.mean(left_eyebrow[:, 1])
            right_eyebrow_eye_dist = np.mean(right_eye[:, 1]) - np.mean(right_eyebrow[:, 1])

            avg_eyebrow_eye_dist = (left_eyebrow_eye_dist + right_eyebrow_eye_dist) / 2
            face_height = abs(landmarks.part(8).y - landmarks.part(27).y)
            eyebrow_eye_dist_normalized = avg_eyebrow_eye_dist / face_height

            threshold_values.append(eyebrow_eye_dist_normalized)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return np.mean(threshold_values)

# Capture the threshold values for each position
normal_th = captureThreshold("Normal")
up_th = captureThreshold("Up")
down_th = captureThreshold("Down")

# Run the main loop with the threshold values
mainLoop(normal_th, up_th, down_th)

cap.release()
cv2.destroyAllWindows()
