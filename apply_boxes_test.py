import os
import tarfile
import urllib.request
import pickle
import numpy as np
from PIL import Image, ImageDraw
# from tensorflow.keras.datasets import cifar10

import tqdm


# 1. DOWNLOAD AND EXTRACT THE CIFAR-10 DATASET

# https://www.cs.toronto.edu/~kriz/cifar.html
# URL of the CIFAR-10 dataset
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# directory to download the dataset to
download_dir = "cifar10_dataset/"

# # File path to save the dataset
# file_path = os.path.join(download_dir, "cifar-10-python.tar.gz")

# # make the directory if it does not exist
# os.makedirs(download_dir, exist_ok=True)

# # download the dataset
# if not os.path.isfile(file_path):
#     print("Downloading CIFAR-10 dataset...")
#     urllib.request.urlretrieve(url, file_path)

# # extract the dataset
# with tarfile.open(file_path) as tar:
#     tar.extractall(path=download_dir)


# 2. LOAD THE CIFAR-10 DATASET TO MEMORY

# Now you can load the dataset from the downloaded and extracted files
def load_cifar10_data():
    files = [
        #'cifar-10-batches-py/data_batch_1', 
        #'cifar-10-batches-py/data_batch_2', 
        #'cifar-10-batches-py/data_batch_3', 
        #'cifar-10-batches-py/data_batch_4', 
        #cifar-10-batches-py/data_batch_5'
        'cifar-10-batches-py/test_batch'
    ]
    
    data = []
    labels = []

    for file in files:
        with open(os.path.join(download_dir, file), 'rb') as f:
            cifar10_data_dict = pickle.load(f, encoding='bytes')
            data.append(cifar10_data_dict[b'data'])
            labels.append(cifar10_data_dict[b'labels'])

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    # Reshape the data
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return data, labels

train_images, train_labels = load_cifar10_data()



# 3. SAVE THE RAW IMAGES TO DISK

# # save unprocessed images to the disk, so like downloading images to disk
# unprocessed_images_dir = "unprocessed_images"
# os.makedirs(unprocessed_images_dir, exist_ok=True)

# for i, image in enumerate(train_images):
#     im = Image.fromarray(image)
#     im.save(os.path.join(unprocessed_images_dir, f"image_{i}.png"))
    




# 4. PROCESS THE IMAGES WITH BOXES, THEN SAVE TO DISK

# define the directory to save the images with black boxes
processed_images_dir = "processed_images_boxes_test"
os.makedirs(processed_images_dir, exist_ok=True)

# define the dimensions of the grid and the size of the black boxes
grid_size = 3
img_size = train_images.shape[1]
box_sizes = [10, 11, 11]

# run through all the images
for i in tqdm.tqdm(range(train_images.shape[0])):
    # create the 9 permutations of black boxes
    for j in range(grid_size):
        for k in range(grid_size):
            # create a copy of the image for each black box
            image_copy = train_images[i].copy()
            image = Image.fromarray(image_copy)
            draw = ImageDraw.Draw(image)
            
            top_left = (sum(box_sizes[:j]), sum(box_sizes[:k]))
            bottom_right = (top_left[0] + box_sizes[j], top_left[1] + box_sizes[k])

            draw.rectangle([top_left, bottom_right], fill="black")
            image.save(os.path.join(processed_images_dir, f"image_{i}_{j}_{k}.png"))



# 5. OR SAVE THE PROCESSED IMAGES TO PICKLE FILES

# change the processed images back to NumPy arrays
processed_images = []
for i in tqdm.tqdm(range(train_images.shape[0])):
    for j in range(grid_size):
        for k in range(grid_size):
            image_path = os.path.join(processed_images_dir, f"image_{i}_{j}_{k}.png")
            image = Image.open(image_path)
            image_array = np.array(image)
            processed_images.append(image_array)

processed_images = np.array(processed_images)

# split the processed data into 5 batches like the original CIFAR-10 dataset

num_batches = 5
batch_size = processed_images.shape[0] // num_batches
processed_batches = np.array_split(processed_images, num_batches)

# assume the labels stay the same, so we can split the original labels in the same way
label_batches = np.array_split(train_labels, num_batches)

# saving the processed batches into a directory
processed_batches_dir = "processed_batches_boxes_test" #"processed_batches_boxes"
os.makedirs(processed_batches_dir, exist_ok=True)

# save the processed batches and the labels into pickle files
for i, (data_batch, label_batch) in enumerate(zip(processed_batches, label_batches)):
    output_dict = {
        'batch_label': f'processed_batch_{i+1}',
        'data': data_batch,
        'labels': label_batch
    }
    with open(os.path.join(processed_batches_dir, f'processed_batch_{i+1}.pickle'), 'wb') as f:
        pickle.dump(output_dict, f)



# # run cnn to see if it can recognize occluded images of cifar10

# # 6. LOAD THE PROCESSED BATCHES

# # load the processed batches
# def load_processed_batches():
#     files = [
#         'processed_batch_1.pickle', 
#         'processed_batch_2.pickle', 
#         'processed_batch_3.pickle', 
#         'processed_batch_4.pickle', 
#         'processed_batch_5.pickle'
#     ]
    
#     data = []
#     labels = []

#     for file in files:
#         with open(os.path.join(processed_batches_dir, file), 'rb') as f:
#             processed_batch_dict = pickle.load(f, encoding='bytes')
#             data.append(processed_batch_dict[b'data'])
#             labels.append(processed_batch_dict[b'labels'])

#     data = np.concatenate(data)
#     labels = np.concatenate(labels)

#     return data, labels

# processed_images, processed_labels = load_processed_batches()

# # split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(processed_images, processed_labels, test_size=0.2, random_state=42)

# # normalize the data
# X_train = X_train / 255.0
# X_test = X_test / 255.0
# # why 255?

# # one-hot encode the labels
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# # 7. BUILD THE CNN

# # define the model
# model = Sequential()

# # add the first convolutional layer
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))

# # add the max pooling layer
# model.add(MaxPooling2D((2, 2)))

# # add the second convolutional layer
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

# # add another max pooling layer
# model.add(MaxPooling2D((2, 2)))

# # add the third convolutional layer
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

# # add another max pooling layer
# model.add(MaxPooling2D((2, 2)))

# # add the flatten layer
# model.add(Flatten())

# # add the first dense layer
# model.add(Dense(64, activation='relu'))

# # add the output layer
# model.add(Dense(10, activation='softmax'))

# # compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # 8. TRAIN THE CNN

# # train the model
# model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(X_test, y_test))

# # 9. EVALUATE THE CNN

# # evaluate the model
# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
# print(f'Test loss: {test_loss}')
# print(f'Test accuracy: {test_acc}')

# # 10. MAKE PREDICTIONS

# # make predictions
# predictions = model.predict(X_test)

# # get the index of the highest probability
# y_pred = np.argmax(predictions, axis=1)

# # get the index of the highest probability
# y_true = np.argmax(y_test, axis=1)

# # get the confusion matrix
# confusion_matrix(y_true, y_pred)

# # get the classification report
# print(classification_report(y_true, y_pred))
