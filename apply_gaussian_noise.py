import os
import tarfile
import urllib.request
import pickle
import numpy as np
from PIL import Image, ImageDraw
# from tensorflow.keras.datasets import cifar10




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
        'cifar-10-batches-py/data_batch_1', 
        'cifar-10-batches-py/data_batch_2', 
        'cifar-10-batches-py/data_batch_3', 
        'cifar-10-batches-py/data_batch_4', 
        'cifar-10-batches-py/data_batch_5'
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

def generate_gaussian_noise(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(0, 1, shape)

# define the directory to save the images with noise boxes
processed_images_dir = "processed_images_gaussian_noise"
os.makedirs(processed_images_dir, exist_ok=True)

# define the dimensions of the grid and the size of the noise boxes
grid_size = 3
img_size = train_images.shape[1]
box_sizes = [10, 11, 11]

# run through all the images
for i in range(train_images.shape[0]):
    # create the 9 permutations of noise boxes
    for j in range(grid_size):
        for k in range(grid_size):
            # create a copy of the image for each noise box
            image_copy = train_images[i].copy().astype(np.int32) # convert to int32 here

            # generate gaussian noise with the same size as the box
            gaussian_noise = generate_gaussian_noise((box_sizes[j], box_sizes[k], 3), 42)

            # scale the noise and convert to integer
            gaussian_noise = (gaussian_noise * 30).astype(np.int32)

            # add the noise to the image
            image_copy[sum(box_sizes[:j]):sum(box_sizes[:j])+box_sizes[j], sum(box_sizes[:k]):sum(box_sizes[:k])+box_sizes[k]] += gaussian_noise

            # clip the pixel values
            image_copy = np.clip(image_copy, 0, 255)

            # save the image
            Image.fromarray(image_copy.astype('uint8')).save(os.path.join(processed_images_dir, f"image_{i}_{j}_{k}.png"))






# 5. OR SAVE THE PROCESSED IMAGES TO PICKLE FILES

# # change the processed images back to NumPy arrays
# processed_images = []
# for i in range(train_images.shape[0]):
#     for j in range(grid_size):
#         for k in range(grid_size):
#             image_path = os.path.join(processed_images_dir, f"image_{i}_{j}_{k}.png")
#             image = Image.open(image_path)
#             image_array = np.array(image)
#             processed_images.append(image_array)

# processed_images = np.array(processed_images)

# # split the processed data into 5 batches like the original CIFAR-10 dataset

# num_batches = 5
# batch_size = processed_images.shape[0] // num_batches
# processed_batches = np.array_split(processed_images, num_batches)

# # assume the labels stay the same, so we can split the original labels in the same way
# label_batches = np.array_split(train_labels, num_batches)

# # saving the processed batches into a directory
# processed_batches_dir = "processed_batches_gaussian_noise"
# os.makedirs(processed_batches_dir, exist_ok=True)

# # save the processed batches and the labels into pickle files
# for i, (data_batch, label_batch) in enumerate(zip(processed_batches, label_batches)):
#     output_dict = {
#         'batch_label': f'processed_batch_{i+1}',
#         'data': data_batch,
#         'labels': label_batch
#     }
#     with open(os.path.join(processed_batches_dir, f'processed_batch_{i+1}.pickle'), 'wb') as f:
#         pickle.dump(output_dict, f)
