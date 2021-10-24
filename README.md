# Physics Assisted GANs for Inverse Scattering Problems

### Inverse Problems

A forward problem is defined as <img src="https://render.githubusercontent.com/render/math?math=y=Ax+n"> where `x` is the input, `A` is the model describing the system and `y` is the vector of measurements obtained when `x` is acted upon by the model `A` . An inverse problem is then defined as the problem of minimizing <img src="https://render.githubusercontent.com/render/math?math=||y%20-%20Ax||^2"> over `x` where `x` is the reconstruction and `y` is the set of measurements obtained from the forward problem. Thus, the objective of the inverse problem is to obtain `x` given `A` and `y`.

### Inverse Scattering Problem and the role of GANs

In case of inverse scattering, this problem is highly underdetermined since `A` is a highly ill-posed problem, i.e. the system can have infinite solutions. A more detailed description of the problem can be found in [1]. In such cases, we add regularization to shrink the domain of possible solutions of the optimization problem and obtain more stable solutions. The simpliest way to do this is to add handcrafted priors to the objective function, like enforcing sparsity using Lasso (applying penalty to the <img src="https://render.githubusercontent.com/render/math?math=l_1"> norm of `x`) or applying a high penalty to outliers using Ridge (applying penalty to <img src="https://render.githubusercontent.com/render/math?math=l_2"> norm of `x`). However, the assumptions made while designing these priors might not always be valid. A better technique then is to use data-driven priors to adds constraints to the solution space, thus removing the need to handcraft the priors altogether. This also leads to a lot more flexibility in terms of the information captured by these priors rather than simple conditions like sparsity or continuity in the obtained solutions. In this project, we use GANs for reconstructing `x` in which the generator acts as a reconstruction network that minimizes the data term <img src="https://render.githubusercontent.com/render/math?math=y=Ax+n"> while the discriminator acts as a prior and imposes constraints on the solution space.


### Architecture used

A conditional GAN [2] model is used which takes the input `A'y` instead of a random input vector in some latent space where `A'` is the pseudoinverse of the model `A`. The generator model is a UNet which takes as input `A'y` and produces the desired reconstruction of x. The discriminator model is a PatchGAN architecture which penalizes structure in the output across multiple pixels. Since the model `A` contains information about the physics of wave scattering in the inverse scattering problem, we call the model **Physics Assisted GANs**. We train it using Wasserstein distance as the loss and with gradient penalty. For every training step of the generator, the discriminator takes 5 training steps.

### Data
The data used for training is generated using this code: https://github.com/dsamruddhi/Inverse-Scattering-Problem

### Results

Ground truth images followed by the output of the generator for successive epochs. The GAN is trained for a total of 50 epochs.


<img align="left" title="Ground Truth" src="https://user-images.githubusercontent.com/5306916/138594685-57c88fa7-9ec5-4349-8620-48dad6419e91.jpg" width="165" height="150">
<img align="left" title="Generator output" src="https://user-images.githubusercontent.com/5306916/138595125-ffc155af-54a5-450e-b52b-4ebf4c50dd3a.gif" width="200" height="150">

<img align="left" title="Ground Truth" src="https://user-images.githubusercontent.com/5306916/138594567-6151eb2d-174e-4cc3-8d6e-2fd54feddafb.jpg" width="165" height="150">
<img align="center" title="Generator output" src="https://user-images.githubusercontent.com/5306916/138595121-e4dc9144-53d1-4c73-9153-35e4add57c72.gif" width="200" height="150">

<img align="left" title="Ground Truth" src="https://user-images.githubusercontent.com/5306916/138594591-3423dc37-e526-4cb1-8b77-04ed66537129.jpg" width="165" height="150">
<img align="left" title="Generator output" src="https://user-images.githubusercontent.com/5306916/138595123-380324bf-9a8a-4b3b-bef8-4e52ae35e438.gif" width="200" height="150">

<img align="left" title="Ground Truth" src="https://user-images.githubusercontent.com/5306916/138594511-2f02032f-382b-4cf5-9857-cd8713010b29.jpg" width="165" height="150">
<img align="center" title="Generator output" src="https://user-images.githubusercontent.com/5306916/138595126-340ba512-b086-4524-aba1-4e5814300fcd.gif" width="200" height="150">

### References
[1] Dubey, Amartansh, et al. "An Enhanced Approach to Imaging the Indoor Environment Using WiFi RSSI Measurements." IEEE Transactions on Vehicular Technology 70.9 (2021): 8415-8430.

[2] Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
