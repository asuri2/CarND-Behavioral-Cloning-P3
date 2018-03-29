import numpy as np
import csv
import cv2
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D

# Constants
data_path = "data/"
image_path = data_path + "IMG/"
left_image_angle_correction = 0.20
right_image_angle_correction = -0.20
csv_data = []
processed_csv_data = []

# Reading the content of csv file
with open(data_path + 'driving_log.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    # Skipping the headers
    next(csv_reader, None)
    for each_line in csv_reader:
        csv_data.append(each_line)


# Method to pre-process the input image
def pre_process_image(image):
    # Since cv2 reads the image in BGR format and the simulator will send the image in RGB format
    # Hence changing the image color space from BGR to RGB
    colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Cropping the image
    cropped_image = colored_image[60:140, :]
    # Downscaling the cropped image
    resized_image = cv2.resize(cropped_image, None, fx=0.25, fy=0.4, interpolation=cv2.INTER_CUBIC)
    return resized_image


def generator(input_data, batch_size=64):
    # Since we are augmenting 3 more images for a given input image, so dividing the batch size by 4
    processing_batch_size = int(batch_size / 4)
    number_of_entries = len(input_data)
    # Shuffling the csv entries
    input_data = sklearn.utils.shuffle(input_data)
    while True:
        for offset in range(0, number_of_entries, processing_batch_size):
            # Splitting the data set into required batch size
            batch_data = input_data[offset:offset + processing_batch_size]
            image_data = []
            steering_angle = []

            # Iterating over each image in batch_data
            for each_entry in batch_data:
                center_image_path = image_path + each_entry[0].split('/')[-1]
                center_image = cv2.imread(center_image_path)
                steering_angle_for_centre_image = float(each_entry[3])
                if center_image is not None:
                    # Pre-processing the image
                    processed_center_image = pre_process_image(center_image)
                    image_data.append(processed_center_image)
                    steering_angle.append(steering_angle_for_centre_image)
                    # Flipping the image
                    image_data.append(cv2.flip(processed_center_image, 1))
                    steering_angle.append(- steering_angle_for_centre_image)

                # Processing the left image
                left_image_path = image_path + each_entry[1].split('/')[-1]
                left_image = cv2.imread(left_image_path)
                if left_image is not None:
                    image_data.append(pre_process_image(left_image))
                    steering_angle.append(steering_angle_for_centre_image + left_image_angle_correction)

                # Processing the right image
                right_image_path = image_path + each_entry[2].split('/')[-1]
                right_image = cv2.imread(right_image_path)
                if right_image is not None:
                    image_data.append(pre_process_image(right_image))
                    steering_angle.append(steering_angle_for_centre_image + right_image_angle_correction)

            # Shuffling and returning the image data back to the calling function
            yield sklearn.utils.shuffle(np.array(image_data), np.array(steering_angle))


# Splitting the csv data set into train and validation data
train_data, validation_data = train_test_split(csv_data, test_size=0.2)
# Creating generator instances for train and validation data set
train_generator_instance = generator(train_data)
validation_generator_instance = generator(validation_data)


# Getting shape of processed image
first_img_path = image_path + csv_data[0][0].split('/')[-1]
first_image = cv2.imread(first_img_path)
processed_image_shape = pre_process_image(first_image).shape


# My final model architecture
model = Sequential()
# Normalizing the input image data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=processed_image_shape))
# First Convolution2D layer
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
# Second Convolution2D layer
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), activation="relu"))
model.add(MaxPooling2D())
# Flattening the output of 2nd Convolution2D layer
model.add(Flatten())
# First Dense layer
model.add(Dense(32))
model.add(Dropout(0.20))
# Second Dense Layer
model.add(Dense(16))
# Third and Final Dense Layer
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator_instance, samples_per_epoch=len(train_data) * 4, verbose=1, validation_data=validation_generator_instance, nb_val_samples=len(validation_data)*4, nb_epoch=3)
model.save('model.h5')
