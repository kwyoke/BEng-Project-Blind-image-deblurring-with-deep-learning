# BEng-Project-Blind-image-deblurring-with-deep-learning

## Abstract
This project explores and experiments with both kernel-based and end-to-end image deblurring algorithms based on Convolutional Neural Networks (CNN). It focuses on the analysis and implementation of patch level motion kernel estimation using a CNN (referred to as patchCNN). Using Keras, I implemented and trained a CNN to distinguish different types of linear motion blur present in 30x30 pixel patches. It was observed that CNNs are indeed capable of recognizing different types of blur, and that a training dataset of patches containing more edge information speeds up the training process significantly. Experimentation with an end-to-end deblurring algorithm based on Generative Adversarial Networks (DeblurGAN) reveals the suitability of GANs in generating perceptually sharp results and its strong generalising power. The deblurred results of three deblurring algorithms, patchCNN, DeblurGAN and SRN-DeblurNet are then quantitatively compared on various benchmark datasets and evaluation measures, leading to the conclusion that patchCNN only performs well under specific conditions and is inferior to end-to-end deblurring algorithms.

## Directory structure
1. patchCNN
Folder training_data contains original sharp images from PASCAL VOC2010 dataset and GOPRO dataset and code for generating training dataset from 73 blur kernels.

Contains code used for  training patchCNN model, resultant model weights and evaluating model.

Contains code for partial kernel deconvolution with GMM prior.

Contains Matlab code uploaded by Sun et al used as reference in this project.

2. evaluation_results
Contains test benchmarks with blur, sharp and deblurred results for both GOPRO and synthetic non-uniform dataset
Deblurred results from three algorithms: patchCNN, DeblurGAN, SRN-DeblurNet

deblurgan.zip contains model weights per epoch from training deblurGAN. Finally evaluation results on test benchmarks.
Full training codes for DeblurGAN from Raphael Meudec's repository: https://github.com/RaphaelMeudec/deblur-gan
