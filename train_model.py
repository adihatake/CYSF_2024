import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


num_classes = 4
num_angles = 20
sequence_length = 30
LSTM_units = 20
LSTM_layers = 10
dropout_rate = 0.3
l2_lambda = 0.001  # Strength of L2 regularization
batch_size = 32
epochs = 300


model = Sequential([
    LSTM(units=256, input_shape=(30, 20), return_sequences=True, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
    Dropout(dropout_rate),
    BatchNormalization(),
    LSTM(units=128, return_sequences=True, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
    Dropout(dropout_rate),
    BatchNormalization(),
    LSTM(units=64, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
    Dropout(dropout_rate),
    BatchNormalization(),
    Dense(units=128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(dropout_rate),
    BatchNormalization(),
    Dense(units=num_classes, activation='softmax')


    # LSTM(units=128, input_shape=(30, 20), return_sequences=True, kernel_regularizer=l2(0.001)),
    # Dropout(0.3),
    # LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.001)),
    # Dropout(0.3),
    # LSTM(units=32, kernel_regularizer=l2(0.001)),
    # Dense(units=64, activation='relu', kernel_regularizer=l2(0.001)),
    # Dropout(0.3),
    # Dense(units=num_classes, activation='softmax')


    # LSTM(units=128, input_shape=(30, 20), return_sequences=True),
    # Dropout(0.2),
    # LSTM(units=64, return_sequences=True),
    # Dropout(0.2),
    # LSTM(units=32),
    # Dense(units=64, activation='relu'),
    # Dense(units=num_classes, activation='softmax')
])

# model = Sequential()
# model.add(LSTM(units=LSTM_units, return_sequences=True, input_shape=(30, 20))) # units define the output space. the LSTM knows that it must process 1293 sequences
#
# for i in range(LSTM_layers):
#   model.add(LSTM(units=LSTM_units, return_sequences=True))  # Next LSTM layer with return_sequences=True
# model.add(LSTM(units=LSTM_units, return_sequences=False))  # Fourth LSTM layer with return_sequences=False
#
#
# model .add(Dense(units=num_classes, activation='softmax'))


# train_inputs_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/total_inputs.npy.npy")
# train_labels_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/total_targets.npy")

train_inputs_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_inputs_augmented.npy")
train_labels_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_targets_augmented.npy")

# train_inputs_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_inputs_augmented_2.npy")
# train_labels_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_targets_augmented_2.npy")

# train_inputs_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_inputs_augmented_3.npy")
# train_labels_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_targets_augmented_3.npy")

# train_inputs_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_inputs_augmented_4.npy")
# train_labels_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_targets_augmented_4.npy")

# train_inputs_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_inputs_augmented_5.npy")
# train_inputs_data = np.nan_to_num(train_inputs_data, nan=0)

# train_inputs_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_inputs_augmented_5.npy")
# train_inputs_data = np.nan_to_num(train_inputs_data, nan=0)
# train_inputs_data = train_inputs_data[:60, :, :]
#
# train_labels_data = np.load("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/Data_augmentation/total_targets_augmented_5.npy")
# train_labels_data = train_labels_data[:60, :, :]


#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Changed loss function for classification
opt = SGD(learning_rate=0.001)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])

history = model.fit(
    x=train_inputs_data,
    y=train_labels_data,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
)


model.save("/Users/catherinebalajadia/Downloads/CYSF_2024/datasets/augment_6_model.keras")





#burrowed from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('LSTM Accuracy (no augmentation)')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1.2
plt.show()



# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM Loss (no augmentation)')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

