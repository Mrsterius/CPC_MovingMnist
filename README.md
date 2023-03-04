### Representation Learning with Contrastive Predictive Coding

This repository contains a Keras implementation of the algorithm presented in the paper [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) modified from here https://github.com/davidtellez/contrastive-predictive-coding.

The goal of unsupervised representation learning is to capture semantic information about the world, recognizing patterns in the data without using annotations. This paper presents a new method called Contrastive Predictive Coding (CPC) that can do so across multiple applications. The main ideas of the paper are:
* Contrastive: it is trained using a contrastive approach, that is, the main model has to discern between *right* and *wrong* data sequences.
* Predictive: the model has to predict future patterns given the current context.
* Coding: the model performs this prediction in a latent space, transforming code vectors into other code vectors (in contrast with predicting high-dimensional data directly).

CPC has to predict the next item in a sequence using only an embedded representation of the data, provided by an encoder. In order to solve the task, this encoder has to learn a meaningful representation of the data space. After training, this encoder can be used for other downstream tasks like supervised classification.

To train the CPC algorithm, I have used the Moving Mnist dataset. This dataset consists of sequences of modified MNIST numbers (64x64 RGB). Positive sequence samples contain *sorted* sequence, and negative ones *random* sequence. For example, let's assume that the context sequence length is S=4, and CPC is asked to predict the next P=2 numbers. A positive sample could look like consecutive 6 frames, whereas a negative sequence could be random sequence of 6 frames.

### Results

After 10 training epochs, CPC reports a 100% validation accuracy on the contrastive task. 

### Usage

- Execute ```python train_model.py``` to train the CPC model.
- Execute ```visualize_embeddings.py``` to view the final generated embeddings.
- Execute ```generate_data.py``` to get the moving_mnist_test_preprocessed.npy file in the resources directory.
- Use ```data2.py``` to use the get_batch() method to get the dataset.
### Requisites

- [Anaconda Python 3.5.3](https://www.continuum.io/downloads)
- [Keras 2.0.6](https://keras.io/)
- [Tensorflow 1.4.0](https://www.tensorflow.org/)
- GPU for fast training.

### References

- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
