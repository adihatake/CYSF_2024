import cv2
import numpy as np
import random
import os
from angle_pairs import joint_pairs
from ultralytics import YOLO

def angle_calculation(on_image_keypoints):
    joint_angles = []
    for pair in joint_pairs:
        #noise = random.randint(-5, 5)

        point1 = on_image_keypoints[pair[0]]
        point1 = (point1[0], point1[1])

        point2 = on_image_keypoints[pair[1]]
        point2 = (point2[0], point2[1])

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

        else:
            cosine_angle = dot_product / (magnitude1 * magnitude2)

        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        # Convert angle from radians to degrees
        angle_deg = np.degrees(angle_rad)
        joint_angles.append(angle_deg)

    return joint_angles




lag = 10
FPS_setting = 30
augmentation_methods =['flip', 'rotate', 'rotate_flip']


def list_files_in_directory(directory):
    return [file for file in os.listdir(directory) if file != '.DS_Store']

main_directory = "/Users/catherinebalajadia/Downloads/CYSF_2024/Testing_Videos/Valid_moves"

# main_listing = list_files_in_directory(main_directory)
# main_listing.pop(1)
# main_listing.pop(-1)
# main_listing.pop(1)

main_listing = ['Front_kick', 'Side_kick', 'Round_house', 'Punch']
main_listing = ['idle']

for classes in main_listing:
    directory = "/Users/catherinebalajadia/Downloads/CYSF_2024/Testing_Videos/Valid_moves/" + classes
    listing = list_files_in_directory(directory)
    total_files = len(listing)
    num_classes = 5
    print(total_files)

    dataset_inputs = np.empty((total_files, 30, 20))
    dataset_targets = np.empty((total_files, num_classes))
    errors = []

    file_num = 0
    for file in listing:
        angle_history = []
        print(f'Processing {file}')
        video_path = directory + "/" + file
        print(video_path)
        cap = cv2.VideoCapture(video_path)

        # Load the YOLOv8 model
        model = YOLO('yolov8n-pose.pt')

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                pose_angle_data = []
                for i in range(19):
                    pose_angle_data.append(0)
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True, conf=0.8, iou=0.8, max_det=1, verbose=False)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                Any = results[0].boxes.id

                if Any != None:
                    NORM_keypoints = results[0].keypoints.xyn  # change to xyn later
                    image_keypoints = results[0].keypoints.xy
                    pose_angle_data = None

                    pose_angle_data = angle_calculation(NORM_keypoints[0])

                    if len(angle_history) == FPS_setting:
                        angle_history.pop(0)
                    else:
                        angle_history.append(pose_angle_data)
            else:
                break


            # Display the annotated frame
            #cv2.imshow("YOLOv8 Tracking", annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break



        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

        while True:
            if len(angle_history) < 30:
                angle_history.append(angle_history[-1])
            else:
                break

        angle_history = np.array(angle_history)

        target_class = np.zeros(num_classes)
        target_class[2] = 1

        dataset_inputs[file_num] = angle_history
        dataset_targets[file_num] = target_class
        print(f"Inputs shape: {angle_history.shape}\n\n")
        file_num += 1

    print(f'Following files do not work: {errors}')
    save_directory ="/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/"

    train_file = save_directory + str(classes) + "_inputs"
    np.save(train_file, dataset_inputs)

    targets_file = save_directory + str(classes) + "_targets"
    np.save(targets_file, dataset_targets)



