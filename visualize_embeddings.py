from keras.models import load_model
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the trained model and encoder
from data2 import get_batch

# model = load_model('models/64x64/cpc.h5')
encoder = load_model('models/64x64/encoder.h5', compile=False)

# Get a batch of data
batch_size = 2
dataset = np.load('resources/moving_mnist_test_preprocessed.npy')
validation_data_array = dataset[int(0.8 * len(dataset)):]
data, labels = get_batch(video_dataset=validation_data_array, batch_size=batch_size, seq_len=8)
print(data.shape)
data = data[0]
embeddings = encoder.predict(data)
# Reduce the dimensionality of the embeddings using PCA or t-SNE
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# log_dir = "logs/image_embeddings"
plt.figure()
plt.plot(embeddings_2d[:,0],embeddings_2d[:,1], 'ro', alpha = 0.5)
# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],'ro')
for i in range(embeddings_2d.shape[0]):
    plt.text(embeddings_2d[i,0], embeddings_2d[i,1], str(i))
plt.show()

