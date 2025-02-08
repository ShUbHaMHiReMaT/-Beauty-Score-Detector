import cv2
import numpy as np

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_beauty_score(face_width, face_height, eye_width):
    """Calculates beauty based on the golden ratio (1.618)."""
    golden_ratio = 1.618
    
    # Face width to height ratio
    ratio1 = face_height / face_width
    
    # Eye width to face width ratio
    ratio2 = eye_width / face_width
    
    # Compute deviation from golden ratio
    deviation = (abs(ratio1 - golden_ratio) + abs(ratio2 - golden_ratio)) / 2
    score = max(0, 100 - (deviation * 100))  # Convert deviation to percentage
    return round(score, 2)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect eyes inside the face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        if len(eyes) >= 2:
            eye1, eye2 = eyes[:2]  # Take the first two detected eyes
            eye1_center = (eye1[0] + eye1[2] // 2, eye1[1] + eye1[3] // 2)
            eye2_center = (eye2[0] + eye2[2] // 2, eye2[1] + eye2[3] // 2)
            eye_width = calculate_distance(eye1_center, eye2_center)

            # Calculate beauty score
            score = calculate_beauty_score(w, h, eye_width)

            # Display score
            cv2.putText(frame, f"Beauty Score: {score}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
