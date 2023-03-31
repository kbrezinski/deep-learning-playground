# Repository Summary

This repository contains original code implementations for various Deep Learning and Machine Learning algorithms for learning purposes.

- [Repository Summary](#repository-summary)
- [Deep Learning Architectures](#deep-learning-architectures)
- [Deep Learning Uncertainty Estimation](#deep-learning-uncertainty-estimation)
  - [Useful resources](#useful-resources)
- [Graph Neural Networks](#graph-neural-networks)
- [Machine Learning Architectures](#machine-learning-architectures)
- [MLops](#mlops)


# Deep Learning Architectures

This repository contains original code implementations for the following implementations of deep learning architectures found in `/deep_learning`:
* **Multilayer Perceptrons**: logistic regression, logistic regression
* **Convolutional Neural Networks**: `batchnorm`, `adaptive lr`, `dropout`, `xavier`, `adam`, `alexnet`, `custom kernels`
*  **Recurrent Neural Networks**: `character rnn`, `word rnn`, `lstm`, `attention`
*  **Auto-Encoders**: `vanilla`, `variational`, `generative adversarial networks`

# Deep Learning Uncertainty Estimation

This folder contains code for the following implementations of uncertainty estimation in deep learning models found in `~/dl_uncertainty`. Example implementations include `Gaussian`, `Gaussian Mixture`, `Quantile Regression`, `Bayesian`, `Monte Carlo Dropout` and `Ensemble`. Final test results are shown below with 95% confidence intervals.

Single Gaussian             |  Gaussian Mixture Model 
:-:|:-:|
Single Gaussian | <img src=dl_uncertainty/images/uncertainty.png width='200'> 
Quantile Regression w/ `pinball_loss` | <img src=dl_uncertainty/images/uncertainty_quantile.png width='200'>
Gaussian Mixture Model | <img src=dl_uncertainty/images/uncertainty_mixture.png width='200'> 
Bayesian Neural Network | <img src=dl_uncertainty/images/uncertainty_bayesian.png width='200'>
Monte Carlo Dropout | <img src=dl_uncertainty/images/uncertainty_dropout.png width='200'> 
Ensemble | <img src=dl_uncertainty/images/uncertainty_de.png width='200'>


## Useful resources
* [Bayesian Methods for Hackers](https://rohanvarma.me/Regularization/)
* [Regularization as bayesian prior](https://rohanvarma.me/Regularization/)
* [Variational Inference Blogpost](http://krasserm.github.io/2019/03/14/bayesian-neural-networks/#appendix)




# Graph Neural Networks

# Machine Learning Architectures

This repository contains original code implementations for the following implementations of deep learning architectures found in `/machine_learning`:
* k-NN, logistic regression, adaboost, decision trees


# MLops
Contains `torch_lightning` scripts.