import cv2
import torch
import keras
import random
import numpy as np
import math
from angle_pairs import joint_pairs
from ultralytics import YOLO




# Load the YOLOv8 model
pose_model = YOLO('yolov8n-pose.pt')
valid_move_classifer = keras.models.load_model("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/augment_5_model.keras")


# Open the video file
video_path = "/Users/catherinebalajadia/Downloads/CYSF_2024/Testing_Videos/Comp_team/test_scorer.mov"
#video_path = "/Users/catherinebalajadia/Downloads/CYSF_2024/Testing_Videos/Regular_Sparring/IMG_0007.MOV"
# 11
# 9

cap = cv2.VideoCapture(video_path)

valid_moves = ['front', 'side', 'round', 'punch', 'nothing']



def distance_to_head(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if distance < 100:
        return True
    else:
        return False


class ATHLETE:
    def __init__(self):
        self.ID = 0
        self.on_image_keypoints = None
        self.normalized_keypoints = None
        self.joint_angles = []
        self.colour = (0,0,0)
        self.current_move = ' '
        self.have_point = False

    def check_valid_move(self):
        if len(self.joint_angles) >= 30:
            input_angles = np.array([self.joint_angles])
            input_angles = np.nan_to_num(input_angles, nan=0)
            predicted_scores = valid_move_classifer(input_angles)

            max_label_index = np.argmax(predicted_scores)
            self.current_move = valid_moves[max_label_index]
            #if self.current_move != "nothing":
            print(f'CURRENT MOVEMENT of PLAYER {self.colour}: {self.current_move} with scores: {predicted_scores}')

    def draw_self(self):
        #print(self.on_image_keypoints)
        if self.have_point == False:
            if self.ID == 1:
                self.colour = (255, 0, 0)
            elif self.ID == 2:
                self.colour = (0, 0, 255)

        elif self.have_point == True:
            self.colour = (0, 255, 0)

        joint_num = 0
        for point in self.on_image_keypoints:
            noise_x = 0
            noise_y = 0
            if point[0] != 0:
                noise_x = random.randint(-5, 5)
                noise_y = random.randint(-5, 5)
            draw_coord = (int(point[0].item())+noise_x, int(point[1].item()+noise_x))

            if draw_coord[0] != 0 and draw_coord[1] != 0:
                if joint_num == 10 or joint_num == 9:
                    pass
                    # draw_coord = (draw_coord[0], draw_coord[1])
                    # cv2.circle(frame, draw_coord, radius=5, color=(0,255,0), thickness=-1)
                else:
                    cv2.circle(frame, draw_coord, radius=5, color=self.colour, thickness=-1)

            joint_num += 1


    def angle_calculation(self):
        total_angle_list = []
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
            cosine_angle = 0

            if (magnitude1 * magnitude2) == 0:
                cosine_angle = 0

            cosine_angle = dot_product / (magnitude1 * magnitude2)

            angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

            # Convert angle from radians to degrees
            angle_deg = np.degrees(angle_rad)
            total_angle_list.append(angle_deg)

        if len(self.joint_angles) == 30:
            self.joint_angles.pop(0)
        else:
            self.joint_angles.append(total_angle_list)

        #print(self.joint_angles)

    def check_if_hit(self):
        if self.current_move != 'nothing':
            if self.ID == 1:
                shoulder_right = karateka_2.on_image_keypoints[5]
                shoulder_left = karateka_2.on_image_keypoints[6]
                hip_right = karateka_2.on_image_keypoints[11]
                hip_left = karateka_2.on_image_keypoints[12]
                head_opponent = karateka_2.on_image_keypoints[0]

            elif self.ID == 2:
                shoulder_right = karateka_1.on_image_keypoints[5]
                shoulder_left = karateka_1.on_image_keypoints[6]
                hip_right = karateka_1.on_image_keypoints[11]
                hip_left = karateka_1.on_image_keypoints[12]
                head_opponent = karateka_1.on_image_keypoints[0]

            opponent_body_polygon = [shoulder_right, shoulder_left, hip_right, hip_left, head_opponent]


            hit_checks = [15, 16, 10, 9]

            for body_part in hit_checks:
                check_coords = self.on_image_keypoints[body_part]

                for opp_point in opponent_body_polygon:
                    is_point_awarded = distance_to_head(x1=check_coords[0], y1=check_coords[1],x2=opp_point[0], y2=opp_point[1])
                    if is_point_awarded:
                        print(f'PLAYER {self.colour} has scored with {self.current_move}')
                        self.have_point = True





iteration = 0
lag = 10

karateka_1 = ATHLETE()
karateka_2 = ATHLETE()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    print(f'FRAME {iteration}')

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = pose_model.track(frame, persist=True, conf=0.8, iou=0.8, max_det=2, verbose=False)

        # Visualize the results on the frame
        #annotated_frame = results[0].plot()

        keypoints_list = results[0].keypoints.xy  # change to xyn later
        NORM_keypoints = results[0].keypoints.xyn  # change to xyn later
        ID_list = results[0].boxes.id
        Any = results[0].boxes.id


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

            karateka_1.draw_self()
            karateka_2.draw_self()

            karateka_1.angle_calculation()
            karateka_2.angle_calculation()

            karateka_1.check_valid_move()
            karateka_2.check_valid_move()

            karateka_1.check_if_hit()
            karateka_2.check_if_hit()

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