# Denoising and Derivative Estimation from Noisy Data in the SINDy Algorithm

*Author: Jiaqi Yao, School of Computer and Mathematical Sciences, University of Adelaide*

This repository contains the code used for the experimental part of my Honours project.

In the project, we identify one research problem of the SINDy algorithm. That is. in real-world situations, the data collected is often noisy and derivative data is inaccessible. In this case, to enable the algorithm to operate, it is necessary to estimate the denoised state and derivative data from the noisy state observations. Previous approaches to achieve this can be broadly categorised into traditional methods and implicit neural representation based deep learning methods. However, both come with their respective limitations. Traditional methods we consider consists of Savitzky-Golay Filter [1] and Total Variation Regularisation [2]. Implicit neural representation based methods consists of standard training and Sobolev training [3]. **The main contribution of this work is the design of Runge-Kutta training, a novel algorithm proposed to address this research problem.** 

Through extensive experiments, we demonstrate its superiority over existing methods. The experiment consists of two parts. First, at a fixed noise level, we test the performance of the five methods on different dynamic systems. Second, we test how the estimated errors of the five algorithms vary with the noise level for specific dynamic systems.

This code framework is developed based on [Ali-Forootani's code](https://github.com/Ali-Forootani/iNeural_SINDy_paper) for I-NeuralSINDy [4].

## References
[1] Schafer, Ronald W. "What is a savitzky-golay filter?[lecture notes]." IEEE Signal processing magazine 28.4 (2011): 111-117.

[2] Vogel, Curtis R. Computational methods for inverse problems. Society for Industrial and Applied Mathematics, 2002.

[3] Czarnecki, Wojciech M., et al. "Sobolev training for neural networks." Advances in neural information processing systems 30 (2017).

[4] Forootani, Ali, Pawan Goyal, and Peter Benner. "A robust SINDy approach by combining neural networks and an integral form." arXiv preprint arXiv:2309.07193 (2023).
