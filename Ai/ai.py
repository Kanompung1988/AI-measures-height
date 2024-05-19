import cv2
import mediapipe as mp
from ultralytics import YOLO
import math

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize YOLOv8 models (make sure you have YOLOv8 weights and ultralytics installed)
body_model = YOLO('yolov8n.pt')  # Replace with your YOLOv8 model and path for body detection
face_model = YOLO('yolov8n-face.pt')  # Replace with your YOLOv8 model and path for face detection

# ฟังก์ชันการคำนวณระยะห่างโดยใช้ความกว้างของใบหน้าในพิกเซล
def calculate_distance(face_width_in_pixels, known_face_width, focal_length):
    return ((known_face_width * focal_length) / face_width_in_pixels) * 1.5

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.dist(point1, point2)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    ab = [b[0] - a[0], b[1] - a[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    cosine_angle = (ab[0] * bc[0] + ab[1] * bc[1]) / (math.sqrt(ab[0]**2 + ab[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2))
    angle = math.degrees(math.acos(cosine_angle))
    return angle

# Open a connection to the webcam
cap = cv2.VideoCapture(1)
frame_width, frame_height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

KNOWN_FACE_WIDTH = 14.3  # ความกว้างเฉลี่ยของใบหน้ามนุษย์ในหน่วยเซนติเมตร
FOCAL_LENGTH = 700  # ค่าความยาวโฟกัส (จำเป็นต้องปรับค่านี้ให้ตรงกับกล้องที่ใช้)

HEAD_TO_FOOT_THRESHOLD = 0.1
WILLING = 0.1
HeightSub = 1
faceX = 1

is_sitting = False
is_sleeping = False
is_sit_on_chair = False
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Perform object detection with YOLOv8 for body detection
    body_results = body_model(frame)

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    head_coords = None
    neck_coords = None
    left_knee_coords = None
    right_knee_coords = None
    left_foot_coords = None
    right_foot_coords = None
    waist_coords = None

    if result.pose_landmarks:
        # Extract landmarks
        landmarks = result.pose_landmarks.landmark

        # Define indices for required landmarks
        head_idx = mp_pose.PoseLandmark.NOSE.value
        neck_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value  # Approximating neck position as midpoint of shoulders
        left_knee_idx = mp_pose.PoseLandmark.LEFT_KNEE.value
        right_knee_idx = mp_pose.PoseLandmark.RIGHT_KNEE.value
        left_foot_idx = mp_pose.PoseLandmark.LEFT_HEEL.value
        right_foot_idx = mp_pose.PoseLandmark.RIGHT_HEEL.value
        left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
        right_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value

        # Get the coordinates of the landmarks
        head_coords = (int(landmarks[head_idx].x * frame.shape[1]), int(landmarks[head_idx].y * frame.shape[0]))
        neck_coords = (int((landmarks[left_hip_idx].x + landmarks[right_hip_idx].x) / 2 * frame.shape[1]), int((landmarks[left_hip_idx].y + landmarks[right_hip_idx].y) / 2 * frame.shape[0]))
        waist_coords = (int((landmarks[left_hip_idx].x + landmarks[right_hip_idx].x) / 2 * frame.shape[1]), int((landmarks[left_hip_idx].y + landmarks[right_hip_idx].y) / 2 * frame.shape[0]))
        left_knee_coords = (int(landmarks[left_knee_idx].x * frame.shape[1]), int(landmarks[left_knee_idx].y * frame.shape[0]))
        right_knee_coords = (int(landmarks[right_knee_idx].x * frame.shape[1]), int(landmarks[right_knee_idx].y * frame.shape[0]))
        left_foot_coords = (int(landmarks[left_foot_idx].x * frame.shape[1]), int(landmarks[left_foot_idx].y * frame.shape[0]))
        right_foot_coords = (int(landmarks[right_foot_idx].x * frame.shape[1]), int(landmarks[right_foot_idx].y * frame.shape[0]))

        # Calculate the midpoint between left and right knee
        knee_midpoint = ((left_knee_coords[0] + right_knee_coords[0]) // 2, (left_knee_coords[1] + right_knee_coords[1]) // 2)

        # Calculate the midpoint between left and right foot
        foot_midpoint = ((left_foot_coords[0] + right_foot_coords[0]) // 2, (left_foot_coords[1] + right_foot_coords[1]) // 2)

        # Mark the midpoint between left and right foot
        cv2.circle(frame, foot_midpoint, 5, (0, 255, 255), -1)  # Yellow circle for midpoint

        # Mark the midpoint between left and right knee
        cv2.circle(frame, knee_midpoint, 5, (255, 255, 0), -1)  # Cyan circle for midpoint

        # Mark the points on the frame
        cv2.circle(frame, head_coords, 5, (255, 0, 0), -1)  # Red circle for head
        cv2.circle(frame, neck_coords, 5, (0, 255, 255), -1)  # Yellow circle for neck
        cv2.circle(frame, left_knee_coords, 5, (0, 255, 0), -1)  # Blue circle for left knee
        cv2.circle(frame, right_knee_coords, 5, (0, 0, 255), -1)  # Blue circle for right knee
        cv2.circle(frame, left_foot_coords, 5, (0, 255, 0), -1)  # Green circle for left foot
        cv2.circle(frame, right_foot_coords, 5, (0, 255, 0), -1)  # Green circle for right foot
        cv2.circle(frame, waist_coords, 5, (0, 0, 255), -1)  # Blue circle for waist

        # Draw lines to connect the points
        cv2.line(frame, head_coords, neck_coords, (255, 255, 255), 2)
        cv2.line(frame, neck_coords, waist_coords, (255, 255, 255), 2)
        cv2.line(frame, waist_coords, right_knee_coords, (255, 255, 255), 2)
        cv2.line(frame, waist_coords, left_knee_coords, (255, 255, 255), 2)
        cv2.line(frame, right_knee_coords, right_foot_coords, (255, 255, 255), 2)
        cv2.line(frame, left_knee_coords, left_foot_coords, (255, 255, 255), 2)

        # Calculate total pixel distance for the line from foot midpoint to knee midpoint
        distance_foot_to_knee = euclidean_distance(foot_midpoint, knee_midpoint)
        distance_knee_to_waist = euclidean_distance(knee_midpoint, waist_coords)
        distance_waist_to_neck = euclidean_distance(waist_coords, neck_coords)
        distance_neck_to_head = euclidean_distance(neck_coords, head_coords)

        # Calculate angle at knees to determine sitting position
        angle_waist_left_knee = calculate_angle(waist_coords, left_knee_coords, left_foot_coords)
        angle_waist_right_knee = calculate_angle(waist_coords, right_knee_coords, right_foot_coords)

        is_sitting = angle_waist_left_knee > 100 and angle_waist_right_knee > 100
        is_sit_on_chair = 80 >= angle_waist_left_knee >= 55 and 80 >= angle_waist_right_knee >= 55

        height_difference = abs(head_coords[1] - foot_midpoint[1])
        is_sleeping = height_difference / frame_height < HEAD_TO_FOOT_THRESHOLD

        if is_sitting:
            cv2.putText(frame, "Sitting", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            total_body_distance = distance_foot_to_knee + distance_knee_to_waist + distance_waist_to_neck + distance_neck_to_head
        elif is_sit_on_chair:
            cv2.putText(frame, "On Chair", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            total_body_distance = distance_foot_to_knee + distance_knee_to_waist + distance_waist_to_neck + distance_neck_to_head
        elif is_sleeping:
            cv2.putText(frame, "Sleeping", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            total_body_distance = distance_foot_to_knee + distance_knee_to_waist + distance_waist_to_neck + distance_neck_to_head
        else:
            cv2.putText(frame, "Standing", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            total_body_distance = distance_foot_to_knee + distance_knee_to_waist + distance_waist_to_neck + distance_neck_to_head

    faces = face_model(frame)  # ใช้ YOLOv8 สำหรับการตรวจจับใบหน้า
    for result in faces:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            faceX = x1
            # วาดกรอบรอบใบหน้า
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # คำนวณระยะห่าง
            face_width = x2 - x1
            distance_f = calculate_distance(face_width, KNOWN_FACE_WIDTH, FOCAL_LENGTH)

            if is_sitting:
                cv2.putText(frame, f"is_sitting: {is_sitting:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if 340 <= distance_f <= 370: # 350cm
                    HeightSub = 2.4
                elif 315 <= distance_f < 340: # 325cm
                    HeightSub = 2.63
                elif 290 <= distance_f < 315: # 300cm
                    HeightSub = 2.6
                elif 265 <= distance_f < 290: # 275cm
                    HeightSub = 3
                else:
                    1
            elif is_sit_on_chair:
                HeightSub = 3.18 #300cms
            elif is_sleeping:
                cv2.putText(frame, f"is_sleeping: {is_sleeping:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                HeightSub = 4.782
            else:
                if 340 <= distance_f <= 370: # 350cm
                    HeightSub = 2.55
                elif 315 <= distance_f < 340: # 325cm
                    HeightSub = 2.75
                elif 290 <= distance_f < 315: # 300cm
                    HeightSub = 2.98
                elif 265 <= distance_f < 290: # 275cm
                    HeightSub = 3.3
                elif 230 <= distance_f < 265: # 250cm
                    HeightSub = 3.53
                else:
                    1

            # แสดงระยะห่างบนภาพ
            cv2.putText(frame, f"Distance: {distance_f:.2f} cm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Process YOLO results for body detection
    total_Height = None
    for result in body_results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates directly
                conf = box.conf[0]           # Extract confidence
                cls = box.cls[0]             # Extract class label

                # Filter for people (class index 0) and confident detections
                if cls == 0 and conf > 0.5:
                    # Calculate the pixel height of the bounding box
                    pixel_height = y2 - y1

                    # Calculate the height of the person using similar triangles
                    person_height = pixel_height / 2

                    if head_coords is not None:
                        # Calculate distance from head to top of the bounding box
                        head_to_box_top = head_coords[1] - y1
                        head_to_box_side = head_coords[1] - x1
                        head_to_box_N = head_coords[1] - faceX
                        
                        if is_sitting: #นั่ง
                            total_Height = (total_body_distance + head_to_box_top) / HeightSub
                        elif is_sleeping: #นอน
                            total_Height = (total_body_distance + head_to_box_side) / HeightSub
                        else: #ยืน
                            total_Height = (total_body_distance + head_to_box_top) / HeightSub

                    # Draw bounding box and height on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if total_Height is not None:
        cv2.putText(frame, f"Total Height: {total_Height:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

    # Display the frame
    cv2.imshow("Height Measurement", frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
