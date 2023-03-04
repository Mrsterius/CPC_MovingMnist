import tensorflow_datasets as tfds
import numpy as np
import cv2

# Load the Moving MNIST dataset from TFDS
mnist_dataset, mnist_info = tfds.load('moving_mnist', with_info=True)
num_frames = 20
# Convert the dataset to NumPy arrays
data = np.zeros((mnist_info.splits['test'].num_examples, num_frames, 64, 64, 1), dtype=np.float32)
for i, example in enumerate(tfds.as_numpy(mnist_dataset['test'])):
    for j in range(num_frames):
        # Extract the j-th frame of the i-th video sequence
        frame = example['image_sequence'][j, :, :, 0]
        # Resize the frame to 64x64 and normalize pixel values to [0, 1]
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)
        frame = frame.astype(np.float32) / 255.0
        data[i, j, :, :, 0] = frame

# Save the preprocessed test sequence to disk
np.save('moving_mnist_test_preprocessed.npy', data)