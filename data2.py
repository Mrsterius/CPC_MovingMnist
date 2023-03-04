import matplotlib.pyplot as plt
import numpy as np

def get_batch(video_dataset, batch_size, seq_len):
    # Get a random set of videos from the dataset
    videos = []
    for _ in range(batch_size//2):
        video = video_dataset[np.random.randint(len(video_dataset))]
        videos.append(video)

    # Generate positive samples by selecting consecutive frames
    pos_samples = []
    pos_labels = []
    for video in videos:
        start_frame = np.random.randint(len(video) - seq_len)
        pos_sample = video[start_frame:start_frame+seq_len]
        pos_samples.append(pos_sample)
        pos_labels.append(1)

    # Generate negative samples by selecting non-consecutive frames
    neg_samples = []
    neg_labels = []
    # for i, video in enumerate(videos):
    #     other_videos = videos[:i] + videos[i+1:]
    #     other_video = other_videos[np.random.randint(len(other_videos))]
    #     start_frame = np.random.randint(len(other_video) - seq_len)
    #     neg_sample = other_video[start_frame:start_frame+seq_len]
    #     neg_samples.append(neg_sample)
    #     neg_labels.append(0)
    for video in videos:
        num_images = video.shape[0]
        num_samples = seq_len
        # generate a random set of indices to select the images
        sample_indices = np.random.choice(num_images, size=num_samples, replace=False)

        # use the selected indices to retrieve the corresponding images from the image sequence
        neg_sample = video[sample_indices]
        neg_samples.append(neg_sample)
        neg_labels.append(0)
    # Stack the samples and convert to numpy arrays
    pos_samples = np.stack(pos_samples)
    pos_labels = np.array(pos_labels)
    neg_samples = np.stack(neg_samples)
    neg_labels = np.array(neg_labels)

    # Concatenate positive and negative samples with their respective labels
    samples = np.concatenate((pos_samples, neg_samples), axis=0)
    labels = np.concatenate((pos_labels, neg_labels), axis=0)

    return samples, labels



def plot_sequences(samples, labels=None, output_path=None):

    ''' Draws a plot where sequences of numbers can be studied conveniently '''

    # images = np.concatenate([x, y], axis=1)
    images = samples
    n_batches = images.shape[0]
    n_terms = images.shape[1]
    counter = 1
    for n_b in range(n_batches):
        for n_t in range(n_terms):
            plt.subplot(n_batches, n_terms, counter)
            plt.imshow(images[n_b, n_t, :, :, :])
            plt.axis('off')
            counter += 1
        if labels is not None:
            plt.title(labels[n_b])

    if output_path is not None:
        plt.savefig(output_path, dpi=600)
    else:
        plt.show()




video_dataset = np.load('resources/moving_mnist_test_preprocessed.npy')
batch_size = 8
seq_len = 8
samples,labels = get_batch(video_dataset, batch_size, seq_len)
print(samples.shape, labels.shape)
plot_sequences(samples,labels, output_path=r'resources/batch_sample.png')