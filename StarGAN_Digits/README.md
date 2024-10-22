# Pytorch-StarGAN-Digits
Unofficial Pytorch implementation of StarGAN for generating Digit-5 datasets (MNIST, SVHN, SynDigits, MNIST-M, and USPS).

## Generated Samples
Input | SYN | MNIST | MNIST-M | SVHN | USPS 
--- | --- | --- | --- | --- | ---
![Input](/Results/Input.png) | ![SynDigits](/Results/SynDigits.png) | ![MNIST](/Results/MNIST.png) | ![MNIST-M](/Results/MNISTM.png) | ![SVHN](/Results/SVHN.png) | ![USPS](/Results/USPS.png) 

### Data layout
    .
    ├── MNIST 
    |	└MNIST Images
    |
    |── MNIST-M
    |	└MNIST-M Images
    |
    |── SVHN  
    |	└SVHN Images
    |
    |── SynDigits
    |	└SynDigits Images (Use only 50-100k images due to data imbalance)
    |
    └── USPS
    	└USPS Images

<img src="/Results/Digits.png" width="500"></img>