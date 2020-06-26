# BEng-Project-Blind-image-deblurring-with-deep-learning

## Abstract
This project explores and experiments with both kernel-based and end-to-end image deblurring algorithms based on Convolutional Neural Networks (CNN). It focuses on the analysis and implementation of patch level motion kernel estimation using a CNN (referred to as patchCNN). Using Keras, I implemented and trained a CNN to distinguish different types of linear motion blur present in 30x30 pixel patches. It was observed that CNNs are indeed capable of recognizing different types of blur, and that a training dataset of patches containing more edge information speeds up the training process significantly. Experimentation with an end-to-end deblurring algorithm based on Generative Adversarial Networks (DeblurGAN) reveals the suitability of GANs in generating perceptually sharp results and its strong generalising power. The deblurred results of three deblurring algorithms, patchCNN, DeblurGAN and SRN-DeblurNet are then quantitatively compared on various benchmark datasets and evaluation measures, leading to the conclusion that patchCNN only performs well under specific conditions and is inferior to end-to-end deblurring algorithms.

## Directory structure
1. patchCNN
Contains python scripts for generating training dataset, model architecture, training and evaluating CNN model, and partial code for kernel deconvolution with GMM prior.

Access https://drive.google.com/drive/folders/1vAtmjor7JjmDoQILXNKsV99Xo2cPqbsw?usp=sharing for
- original sharp images used for training (GOPRO and PASCAL VOC2010 datasets)
- model weights resultant from training CNN to recognise 73 kernels
- Matlab code from Sun et al (official implementation) used as reference 
- evaluation results of patchCNN, DeblurGAN and SRN-DeblurNet on two test benchmarks
- model weights from training DeblurGAN with only horizontal blur dataset after 5, 10, 15, 20, 25, 30, 35 epochs. (Full training codes for DeblurGAN from Raphael Meudec's repository: https://github.com/RaphaelMeudec/deblur-gan)

## Implementations of patchCNN, DeblurGAN and SRN-DeblurNet used for testing:

patchCNN: http://gr.xjtu.edu.cn/c/document_library/get_file?folderId=2076150&name=DLFE-78101.zip

DeblurGAN: https://github.com/RaphaelMeudec/deblur-gan

SRN-DeblurNet: https://github.com/jiangsutx/SRN-Deblur
