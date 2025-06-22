import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def create_sunglasses_overlay(img_shape, left_eye_center, right_eye_center, eye_distance):
    overlay = np.zeros(img_shape, dtype=np.uint8)
    
    # Calculate sunglasses dimensions based on eye distance
    glasses_width = int(eye_distance * 1.4)
    glasses_height = int(eye_distance * 0.8)
    
    # Left lens
    cv2.ellipse(overlay, left_eye_center, (glasses_width//4, glasses_height//2), 0, 0, 360, (200, 200, 200), 3, cv2.LINE_AA)
  
    # Right lens
    cv2.ellipse(overlay, right_eye_center, (glasses_width//4, glasses_height//2), 0, 0, 360, (200, 200, 200), 3, cv2.LINE_AA)
    
    # Bridge
    bridge_start = (int(left_eye_center[0] + glasses_width//4), int(left_eye_center[1]))
    bridge_end = (int(right_eye_center[0] - glasses_width//4), int(right_eye_center[1]))
    cv2.line(overlay, bridge_start, bridge_end, (200, 200, 200), 3, cv2.LINE_AA)
    
    return overlay


def apply_face_filter(image):
  
    # Initialize face mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image)
        
        # Create a copy of the original image
        filtered_image = image.copy()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get image dimensions
                h, w, _c = image.shape
                
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append([x, y])
                
                landmarks = np.array(landmarks)
                
                LEFT_EYE_CENTER = 159
                RIGHT_EYE_CENTER = 386
                
                left_eye = landmarks[LEFT_EYE_CENTER]
                right_eye = landmarks[RIGHT_EYE_CENTER]
                
                eye_distance = np.linalg.norm(right_eye - left_eye)
                overlay = create_sunglasses_overlay(image.shape, left_eye, right_eye, eye_distance)

                filtered_image = cv2.add(filtered_image, overlay)
        
        return filtered_image


def start_real_time_capture():
    try:
        cap = cv2.VideoCapture(0)
  
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            filtered_frame = apply_face_filter(frame)

            cv2.imshow('Simple Face Filter', filtered_frame)            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
      
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Webcam access failed: {e}")


if __name__ == "__main__":
    start_real_time_capture()
