# Pytorch-LSGAN-MNIST
Pytorch implementation of LSGAN for generating MNIST images.

<br>

## Run commands (also available in <a href="scripts.sh">scripts.sh</a>): <br>

<table>
  <tr>
    <th>Dataset</th>
    <th>Run command</th>
  </tr>
  <tr>
    <td>MNIST</td>
    <td>python main.py --dataset mnist 		  --epochs 50</td>
  </tr>
  <tr>
    <td>Fashion MNIST</td>
    <td>python main.py --dataset fashionmnist  --epochs 100</td>
  </tr>
  <tr>
    <td>USPS</td>
    <td>python main.py --dataset usps --epochs 100  --image_size 16</td>
  </tr>
</table>

<br>

## Generated Samples
#### MNIST
<img src="./Results/MNIST.png" width="500"></img>
#### FashionMNIST
<img src="./Results/FashionMNIST.png" width="500"></img>