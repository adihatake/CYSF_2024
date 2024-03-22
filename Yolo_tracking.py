import cv2
import torch
import random
from angle_pairs import joint_pairs
from ultralytics import YOLO




# Load the YOLOv8 model
#model = YOLO('yolov8n.pt')
model = YOLO('yolov8n-pose.pt')

# Open the video file
#video_path = "/Users/catherinebalajadia/Downloads/CYSF_2024/Testing_Videos/Point_Sparring/IMG_3933.MOV"
video_path = "/Users/catherinebalajadia/Downloads/CYSF_2024/Testing_Videos/Comp_team/testing_series.mov"
video_path = "/Users/catherinebalajadia/Downloads/CYSF_2024/Testing_Videos/Comp_team/testing_series.mov"
video_path = "/Users/catherinebalajadia/Downloads/CYSF_2024/Testing_Videos/Comp_team/testing5.mov"

cap = cv2.VideoCapture(video_path)

import numpy as np


def angle_between_points(point1, point2):
    # Convert points to numpy arrays for easier computation
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Compute dot product of the two vectors
    dot_product = np.dot(point1, point2)

    # Compute magnitudes of the vectors
    magnitude1 = np.linalg.norm(point1)
    magnitude2 = np.linalg.norm(point2)

    # Compute the angle in radians
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    # Convert angle from radians to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


class ATHLETE:
    def __init__(self):
        self.ID = 0
        self.on_image_keypoints = None
        self.normalized_keypoints = None
        self.joint_angles = []
        self.colour = (0,0,0)
    def check_valid_move(self, keypoints_normalized):
        # keras model to classify what move it is
        pass

    def draw_self(self):
        #print(self.on_image_keypoints)
        if self.ID == 1:
            self.colour = (255, 0, 0)
        if self.ID == 2:
            self.colour = (0, 0, 255)

        joint_num = 0
        for point in self.on_image_keypoints:
            noise_x = 0
            noise_y = 0
            if point[0] != 0:
                noise_x = random.randint(-5, 5)
                noise_y = random.randint(-5, 5)
            draw_coord = (int(point[0].item())+noise_x, int(point[1].item()+noise_x))

            if draw_coord[0] != 0 and draw_coord[1] != 0:
                cv2.circle(frame, draw_coord, radius=10, color=self.colour, thickness=-1)
            joint_num += 1

    def check_if_opposite(self):
        right_hand = self.on_image_keypoints[10]
        left_hand = self.on_image_keypoints[9]

        if right_hand[0].item() < left_hand[0].item():
            print(f"NOT FACING RIGHT: {self.colour}")


    def angle_calculation(self):
        for pair in joint_pairs:
            point1 = self.on_image_keypoints[pair[0]]
            point2 = self.on_image_keypoints[pair[1]]

            # Convert points to numpy arrays for easier computation
            point1 = np.array(point1)
            point2 = np.array(point2)

            # Compute dot product of the two vectors
            dot_product = np.dot(point1, point2)

            # Compute magnitudes of the vectors
            magnitude1 = np.linalg.norm(point1)
            magnitude2 = np.linalg.norm(point2)

            # Compute the angle in radians
            cosine_angle = dot_product / (magnitude1 * magnitude2)
            angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

            # Convert angle from radians to degrees
            angle_deg = np.degrees(angle_rad)

            self.joint_angles.append(angle_deg)
        print(self.joint_angles)

def check_if_hit(keypoints):
    pass


iteration = 0
lag = 10

karateka_1 = ATHLETE()
karateka_2 = ATHLETE()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    print(iteration)

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.8, iou=0.8, max_det=2)

        # Visualize the results on the frame
        #annotated_frame = results[0].plot()

        keypoints_list = results[0].keypoints.xy  # change to xyn later
        NORM_keypoints = results[0].keypoints.xyn  # change to xyn later
        ID_list = results[0].boxes.id
        Any = results[0].boxes.id

        print(ID_list)

        if iteration == lag:
            karateka_1.ID = ID_list[0]
            karateka_2.ID = ID_list[1]

        elif iteration > lag and Any != None:

            if karateka_1.ID in ID_list:
                karateka_1.on_image_keypoints = []
                index_ID_1 = torch.where(ID_list == karateka_1.ID)[0]

                karateka_1.on_image_keypoints = keypoints_list[index_ID_1][0]
                karateka_1.normalized_keypoints = NORM_keypoints[index_ID_1][0]
            karateka_1.draw_self()
            karateka_1.angle_calculation()


            if karateka_2.ID in ID_list:
                karateka_2.on_image_keypoints = []
                index_ID_2 = torch.where(ID_list == karateka_2.ID)[0]

                karateka_2.on_image_keypoints = keypoints_list[index_ID_2][0]
                karateka_2.normalized_keypoints = NORM_keypoints[index_ID_2][0]

            karateka_2.draw_self()
            karateka_2.angle_calculation()

            karateka_1.check_if_opposite()
            karateka_2.check_if_opposite()

        cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break
    iteration += 1



# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()