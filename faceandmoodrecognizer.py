import cv2
import os
import numpy as np
from deepface import DeepFace

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
dataset_path = "Faces"
os.makedirs(dataset_path, exist_ok=True)

# Threshold for confidence to detect "Unknown"
CONFIDENCE_THRESHOLD = 80  # If confidence is below this, consider as "Unknown"

def capture_faces(user_name, num_images=20):
    """Capture faces for training."""
    print(f"Capturing faces for user: {user_name}")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Increased resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Increased resolution for better quality
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not access the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face, (100, 100))  # Standard size for recognition
            file_name = os.path.join(dataset_path, f"{user_name}_{count}.jpg")
            cv2.imwrite(file_name, resized_face)
            count += 1

            # Draw rectangle and display count
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{count}/{num_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 2)

        cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {user_name}.")

def train_faces():
    """Train the recognizer using captured faces."""
    print("Training the recognizer...")
    faces, labels = [], []
    label_map = {}

    for idx, file_name in enumerate(os.listdir(dataset_path)):
        if file_name.endswith(".jpg"):
            user_name = file_name.split('_')[0]
            label_map[user_name] = label_map.get(user_name, len(label_map))
            face = cv2.imread(os.path.join(dataset_path, file_name), cv2.IMREAD_GRAYSCALE)
            faces.append(face)
            labels.append(label_map[user_name])

    if faces:
        recognizer.train(faces, np.array(labels))
        print("Training completed successfully!")
        print(f"Label Mapping: {label_map}")
    else:
        print("No faces found in the dataset for training.")
    
    # Show the list of users added to the dataset
    print("Users in the dataset:")
    for user in label_map.keys():
        print(f"- {user}")
    
    return label_map

def recognize_faces(label_map):
    """Recognize faces and detect emotions."""
    print("Starting recognition and mood detection...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Increased resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Increased resolution for better quality

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not access the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face, (100, 100))

            try:
                # Face Recognition
                label, confidence = recognizer.predict(resized_face)
                
                # If confidence is low, consider it as "Unknown"
                if confidence < CONFIDENCE_THRESHOLD:
                    user_name = "Unknown"
                else:
                    user_name = [name for name, lbl in label_map.items() if lbl == label][0]
                
                recognition_text = f"User: {user_name} ({round(confidence, 2)})"

                # Mood Detection using DeepFace
                face_color = frame[y:y + h, x:x + w]  # Colored face for emotion analysis
                analysis = DeepFace.analyze(face_color, actions=["emotion"], enforce_detection=False)

                # Correctly access dominant emotion
                mood = analysis[0]["dominant_emotion"]
                mood_text = f"Mood: {mood}"

                # Draw rectangle and display info
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, recognition_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 255), 2)
                cv2.putText(frame, mood_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 255), 2)

            except Exception as e:
                print(f"Error processing face: {e}")
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 255), 2)

        cv2.imshow("Face Recognition and Mood Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Menu-driven execution
if __name__ == "__main__":
    label_map = {}
    while True:
        print("\nMenu:")
        print("1. Capture Faces")
        print("2. Train Model")
        print("3. Recognize Faces and Detect Mood")
        print("4. Exit")

        choice = input("Enter your choice: ")
        if choice == "1":
            user_name = input("Enter the user's name: ")
            capture_faces(user_name)
        elif choice == "2":
            label_map = train_faces()
        elif choice == "3":
            if not label_map:
                print("Please train the model first!")
            else:
                recognize_faces(label_map)
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
