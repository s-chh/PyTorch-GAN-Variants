
# Pytorch-CycleGAN-Digits
Unofficial Pytorch implementation of [CycleGAN](https://arxiv.org/abs/1703.10593) for MNIST, USPS, SVHN, MNIST-M and SyntheticDigits datasets.

<br>

Change the DS variables to change the dataset.
For using the saved model to generate images, set LOAD_MODEL to True and EPOCHS to 0.
## Generated Samples
### MNIST &#10231; SVHN
MNIST &#10230; SVHN             |  SVHN &#10230; MNIST
:-------------------------:|:-------------------------:
![MNIST_SVHN.](./Results/SVHN_MNIST/MNIST_SVHN.png)  |  ![SVHN_MNIST](./Results/SVHN_MNIST/SVHN_MNIST.png)

### MNIST &#10231; MNIST-M
MNIST &#10230; MNIST-M             |  MNIST-M &#10230; MNIST
:-------------------------:|:-------------------------:
![MNIST_MNISTM.](Results/MNIST_MNISTM/MNIST_MNISTM.png)  |  ![MNISTM_MNIST](./Results/MNIST_MNISTM/MNISTM_MNIST.png)

### MNIST &#10231; USPS
MNIST &#10230; USPS             |  USPS &#10230; MNIST
:-------------------------:|:-------------------------:
![MNIST_USPS.](./Results/MNIST_USPS/MNIST_USPS.png)  |  ![MNISTM_MNIST](./Results/MNIST_USPS/USPS_MNIST.png)

### SyDigits &#10231; SVHN
SyDigits &#10230; SVHN             |  SVHN &#10230; SyDigits
:-------------------------:|:-------------------------:
![SyDigits_SVHN.](./Results/SyDigits_SVHN/SyDigits_SVHN.png)  |  ![SVHN_SyDigits](./Results/SyDigits_SVHN/SVHN_SyDigits.png)
