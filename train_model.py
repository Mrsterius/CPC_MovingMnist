'''
This module describes the contrastive predictive coding model from DeepMind:

Oord, Aaron van den, Yazhe Li, and Oriol Vinyals.
"Representation Learning with Contrastive Predictive Coding."
arXiv preprint arXiv:1807.03748 (2018).
'''
from os.path import join, basename, dirname, exists
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
from data2 import get_batch
def network_encoder(x, code_size):

    ''' Define the network mapping images to embeddings '''

    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x


def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)

    return x


def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def network_cpc(image_shape, terms, predict_terms, code_size, learning_rate):

    ''' Define the CPC network combining encoder and autoregressive model '''

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # Define encoder model
    encoder_input = keras.layers.Input(image_shape)
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()

    # Define rest of model
    x_input = keras.layers.Input((terms, image_shape[0], image_shape[1], image_shape[2]))
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)
    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = keras.layers.Input((predict_terms, image_shape[0], image_shape[1], image_shape[2]))
    y_encoded = keras.layers.TimeDistributed(encoder_model)(y_input)

    # Loss
    dot_product_probs = CPCLayer()([preds, y_encoded])

    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=dot_product_probs)

    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    cpc_model.summary()

    return cpc_model


# def train_model(epochs, batch_size, output_dir, code_size, lr=1e-4, terms=4, predict_terms=4, image_size=28, color=False):
#
#     # Prepare data
#     # Define the positive and negative samples
#     # def get_samples(images, window_size):
#     #     # Generate positive samples
#     #     pos_samples = []
#     #     for i in range(len(images) - window_size):
#     #         pos_sample = images[i:i + window_size]
#     #         pos_samples.append(pos_sample)
#     #
#     #     # Generate negative samples
#     #     neg_samples = []
#     #     for i in range(len(images) - window_size):
#     #         j = np.random.choice(np.delete(np.arange(len(images)), np.arange(i, i + window_size)), 1)[0]
#     #         neg_sample = tf.stack([images[i], images[j]])
#     #         neg_samples.append(neg_sample)
#     #
#     #     return tf.data.Dataset.from_tensor_slices(
#     #         (pos_samples + neg_samples, [1] * len(pos_samples) + [0] * len(neg_samples)))
#     #
#     # # Create the training dataset
#     # data = np.load('resources/moving_mnist_test_preprocessed.npy')
#     # dataset = tf.data.Dataset.from_tensor_slices(data)
#     # dataset = dataset.map(lambda x: get_samples(x, window_size=2))
#     # dataset = dataset.shuffle(buffer_size=10000)
#     # train_size = int(0.8*(len(dataset)))
#     # train_dataset = dataset.take(train_size)
#     # val_size = int(0.2 * (len(dataset)))
#     # validation_dataset = dataset.skip(train_size).take(val_size)
#     # train_dataset = train_dataset.batch(batch_size=64)
#     # validation_dataset = validation_dataset.batch(batch_size=64)
#     # train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
#     #                                    positive_samples=batch_size // 2, predict_terms=predict_terms,
#     #                                    image_size=image_size, color=color, rescale=True)
#     #
#     # validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
#     #                                         positive_samples=batch_size // 2, predict_terms=predict_terms,
#     #                                         image_size=image_size, color=color, rescale=True)
#     import numpy as np
#     import tensorflow as tf
#
#     # Load the video sequence dataset
#     # data = np.load('resources/moving_mnist_test_preprocessed.npy')
#
#     # Define the function for generating positive and negative samples
#     # def generate_samples(batch_size, num_frames):
#     #     while True:
#     #         # Choose random frames for each sample
#     #         frames = np.random.randint(num_frames, size=batch_size)
#     #
#     #         # Choose random videos for each sample
#     #         videos = np.random.randint(data.shape[0], size=batch_size)
#     #
#     #         # Generate positive and negative samples
#     #         pos_samples = []
#     #         neg_samples = []
#     #         for i in range(batch_size):
#     #             # Positive sample: two consecutive frames from the same video
#     #             pos_samples.append(data[videos[i], frames[i]:frames[i] + 2])
#     #
#     #             # Negative sample: two consecutive frames from different videos
#     #             other_video = np.random.choice(np.delete(np.arange(data.shape[0]), videos[i]))
#     #             neg_samples.append(np.vstack((data[videos[i], frames[i]], data[other_video, frames[i] + 1])))
#     #
#     #         # Convert samples to TensorFlow tensors
#     #         pos_samples = tf.convert_to_tensor(np.stack(pos_samples), dtype=tf.float32)
#     #         neg_samples = tf.convert_to_tensor(np.stack(neg_samples), dtype=tf.float32)
#     #
#     #         yield pos_samples, neg_samples
#     #
#     # # Define the batch size and number of frames per sample
#     # batch_size = 64
#     # num_frames = 20
#     #
#     # # Create the dataset
#     # dataset = tf.data.Dataset.from_generator(
#     #     generate_samples, args=[batch_size, num_frames], output_types=(tf.float32, tf.float32))
#     dataset = VideoSequenceDataset(dataset_path='resources/moving_mnist_test_preprocessed.npy', seq_len=20, num_negatives=2, K=4, batch_size=64)
#     validation_dataset = VideoSequenceDataset(dataset_path='resources/moving_mnist_test_preprocessed.npy', seq_len=20, num_negatives=2, K=4, batch_size=64)
#     # Prepares the model
#     model = network_cpc(image_shape=(image_size, image_size, 1), terms=terms, predict_terms=predict_terms,
#                         code_size=code_size, learning_rate=lr)
#
#     # Callbacks
#     callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]
#
#     # Trains the model
#     model.fit(
#         dataset,
#         steps_per_epoch=len(dataset),
#         validation_data=validation_dataset,
#         validation_steps=len(validation_dataset),
#         epochs=epochs,
#         verbose=1,
#         callbacks=callbacks
#     )
#
#     # Saves the model
#     # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
#     model.save(join(output_dir, 'cpc.h5'))
#
#     # Saves the encoder alone
#     encoder = model.layers[1].layer
#     encoder.save(join(output_dir, 'encoder.h5'))

def train_model(epochs, batch_size, output_dir, code_size, lr=1e-4, terms=4, predict_terms=4, image_size=64, color=False):

    # Prepares the model
    model = network_cpc(image_shape=(image_size, image_size, 1), terms=terms, predict_terms=predict_terms,
                        code_size=code_size, learning_rate=lr)

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]


    # Loop over the number of epochs and generate batches for training and validation
    for epoch in range(epochs):
        # Generate training and validation data
        train_data, _ = get_batch(video_dataset=train_data_array, batch_size=batch_size, seq_len=terms+predict_terms)
        np.random.shuffle(train_data)
        validation_data, _ = get_batch(video_dataset=validation_data_array, batch_size=batch_size, seq_len=terms+predict_terms)
        np.random.shuffle(validation_data)
        # print(train_data.shape)
        # print(batch_size)
        # Trains the model on the current epoch
        model.fit(x=[train_data[:, :terms], train_data[:, terms:]], y=np.ones((batch_size, 1)),
                  validation_data=([validation_data[:, :terms], validation_data[:, terms:]], np.ones((batch_size, 1))),
                  epochs=1, batch_size=batch_size, callbacks=callbacks, verbose=1)

    # Saves the model
    model.save(join(output_dir, 'cpc.h5'))

    # Saves the encoder alone
    encoder = model.layers[1].layer
    encoder.save(join(output_dir, 'encoder.h5'))

if __name__ == "__main__":
    # Define the video dataset as a numpy array or a list of numpy arrays
    dataset = np.load('resources/moving_mnist_test_preprocessed.npy')
    train_data_array = dataset[:int(0.8 * len(dataset))]
    validation_data_array = dataset[int(0.8 * len(dataset)):]
    train_model(
        epochs=10,
        batch_size=32,
        output_dir='models/64x64',
        code_size=128,
        lr=1e-3,
        terms=4,
        predict_terms=4,
        image_size=64,
        color=True
    )


# if __name__ == "__main__":
#
#     train_model(
#         epochs=10,
#         batch_size=32,
#         output_dir='models/64x64',
#         code_size=128,
#         lr=1e-3,
#         terms=4,
#         predict_terms=4,
#         image_size=64,
#         color=True
#     )

