# Beta-VAE
Tensorflow implementation of Beta-Variational-AutoEncoder for CelebA dataset

This work is aimed to extract disentangled representations from CelebA image dataset using beta-variational-autoencoders.
For more on VAE's and Beta-VAE's refer these works:

1. [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)
2. [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)
3. [β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK](https://openreview.net/pdf?id=Sy2fzU9gl)
4. [Understanding disentangling in β-VAE](https://arxiv.org/pdf/1804.03599.pdf)

Code compatibility:
python>=3.6
Tensorflow==1.14.0

## Dataset

`python download_celebA.py 0B7EVK8r0v71pZjFTYXZWM3FlRnM CelebA.zip`

## Usage

For training:
`python vae.py --train`

For generating new samples:
`python vae.py --generate`

For latent space traversal:
`python vae.py --traverse`

# Results
Random generation during course of training
![training-result](results/vae_training.gif)

Random Generation after 26 Epochs:
![results_26epoch](results/fakes_epoch26_batch05000.jpg)




