import numpy as np

# label 1
front_kick_data = np.load('/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Front_kick_inputs.npy')

# label 2
punch_data = np.load('/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Punch_inputs.npy')

# label 3
round_house_data = np.load('/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Round_house_inputs.npy')

# label 4
side_kick_data = np.load('/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Side_kick_inputs.npy')

# label 5
nothing_data = np.load('/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Nothing_inputs.npy')

# [front_kick_data, side_kick_data, round_house_data, punch_data, nothing_data]

total_dataset = [front_kick_data, side_kick_data, round_house_data, punch_data, nothing_data]
total_targets = []
num_classes = len(total_dataset)
index = 0


augmentation_settings = [5, 2.5]

for classes in total_dataset:
    #Generate random noise within the specified range
    noisy_data = []
    for setting in augmentation_settings:
        max_noise = setting
        min_noise = -setting
        noise = np.random.uniform(min_noise, max_noise, size=classes.shape)
        augmented_classes = classes.copy() + noise
        noisy_data.append(augmented_classes)


    final_data = np.vstack((classes.copy(), noisy_data[0], noisy_data[1]))

    # add target datapoints
    # total_files = classes.shape[0]
    total_files = final_data.shape[0]
    class_targets = np.zeros((total_files, num_classes))

    for i in range(total_files):
        new_target = np.zeros(num_classes)
        new_target[index] = 1
        class_targets[i] = new_target

    total_targets.append(class_targets)
    total_dataset[index] = final_data

    print(total_dataset[index].shape)

    index += 1

dataset_targets = np.vstack((total_targets[0], total_targets[1], total_targets[2], total_targets[3], total_targets[4]))
dataset_inputs = np.vstack((total_dataset[0], total_dataset[1], total_dataset[2], total_dataset[3], total_dataset[4]))

print(f'Final dataset input {dataset_inputs.shape}')
print(f'Final dataset targets {dataset_targets.shape}')

save_directory = "/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_targets_augmented_5.npy"
np.save(save_directory, dataset_targets)
save_directory ="/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_inputs_augmented_5.npy"

np.save(save_directory, dataset_inputs)

# 1: 5
# 2: 5, 10, 15
# 3: 5, 10
# 4: 5, 2.5
# 5: 5, 2.5 with nothing class