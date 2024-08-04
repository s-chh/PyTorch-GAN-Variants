# Pytorch-Tiny GAN
Pytorch implementation of a small GAN network for MNIST, FashionMNIST, and USPS datasets.

### Parameters
Image Size | Generator | Discriminator
--- | --- | ---
28x28 | 6,897 | 3,001
16x16 | 3,729 | 2,473

<br>

## Run commands (also available in <a href="scripts.sh">scripts.sh</a>): <br>

<table>
  <tr>
    <th>Dataset</th>
    <th>Run command</th>
  </tr>
  <tr>
    <td>MNIST</td>
    <td>python main.py --dataset mnist</td>
  </tr>
  <tr>
    <td>Fashion MNIST</td>
    <td>python main.py --dataset fashionmnist</td>
  </tr>
  <tr>
    <td>USPS</td>
    <td>python main.py --dataset usps  --image_size 16</td>
  </tr>
</table>

<br>

## Generated Samples
#### MNIST
<img src="./Results/MNIST.png" width="500"></img>
#### FashionMNIST
<img src="./Results/FashionMNIST.png" width="500"></img>
#### USPS
<img src="./Results/USPS.png" width="500"></img>
