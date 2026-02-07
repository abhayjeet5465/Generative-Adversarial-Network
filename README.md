# Generative-Adversarial-Network

---

## Abstract

Generative Adversarial Networks (GANs) are powerful generative models capable of learning complex data distributions through adversarial training. This project presents the implementation of a basic GAN from scratch to generate handwritten digit images using the MNIST dataset. A fully connected Generator and Discriminator are trained simultaneously, where the Generator learns to synthesize realistic digit images from random noise, and the Discriminator learns to distinguish between real and generated samples. The results demonstrate that even a simple GAN architecture can successfully capture the structure of handwritten digits and produce visually recognizable outputs.

---

## I. Introduction

Generative modeling is a fundamental problem in machine learning, where the objective is to learn the underlying distribution of a dataset and generate new samples that resemble the original data. Generative Adversarial Networks address this problem using an adversarial framework that trains two neural networks in competition with each other. Due to their effectiveness, GANs have been widely adopted in image synthesis, data augmentation, and representation learning.

This project focuses on understanding the core principles of GANs by implementing a basic architecture and evaluating its ability to generate handwritten digit images.

---

## II. Dataset Description

The MNIST dataset is used for training and evaluation. It consists of grayscale images of handwritten digits ranging from 0 to 9, with each image having a resolution of 28 × 28 pixels. The images are normalized and flattened before being provided as input to the fully connected neural networks used in this project.

---

## III. Model Architecture

The GAN architecture consists of two multilayer perceptron networks:

### A. Generator

The Generator network takes a random noise vector as input and transforms it through multiple fully connected layers with nonlinear activation functions. The output layer produces a flattened vector, which is reshaped into a 28 × 28 grayscale image. The Generator is trained to produce images that are indistinguishable from real MNIST samples.

### B. Discriminator

The Discriminator network acts as a binary classifier. It takes either a real image from the dataset or a generated image from the Generator and outputs a probability indicating whether the input is real or fake. The Discriminator provides feedback that guides the Generator during training.

---

## IV. Training Methodology

The training process follows an adversarial learning strategy:

1. A batch of real images is sampled from the MNIST dataset.
2. Random noise vectors are generated and passed through the Generator to create fake images.
3. The Discriminator is trained to correctly classify real images as real and generated images as fake.
4. The Generator is trained to fool the Discriminator by maximizing the probability that generated images are classified as real.
5. This process is repeated for a fixed number of epochs until stable training behavior is observed.

Binary cross-entropy loss and mini-batch gradient descent are used for optimization.

---

## V. Results and Discussion

After the completion of training, the Generator network was evaluated independently by sampling random noise vectors and generating handwritten digit images.

The generated outputs exhibit clear digit structures, coherent stroke patterns, and appropriate contrast between foreground and background. Most samples are visually recognizable as valid digits, indicating that the Generator successfully learned the essential features of the MNIST data distribution. Training losses for both the Generator and Discriminator showed stable convergence, suggesting balanced adversarial learning. No severe mode collapse was observed.

---

## VI. Conclusion

This project demonstrates that a simple fully connected Generative Adversarial Network can effectively learn the distribution of handwritten digit images and generate realistic samples from random noise. The results validate the effectiveness of adversarial learning even with basic architectures and provide a strong foundation for exploring more advanced GAN variants.

---

## VII. Future Work

Future enhancements to this project may include:
- Implementing convolutional GAN architectures (DCGAN)
- Using quantitative evaluation metrics such as FID
- Conditional digit generation
- Training on higher-resolution image datasets
